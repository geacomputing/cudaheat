"""Bounded host-memory NetCDF input for CUDAHeat."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


class NetCDFReader:
    """Open a NetCDF variable and materialize complete-time spatial batches.

    With ``lazy=True`` the SST variable is backed by Dask. Native NetCDF chunks
    are retained for efficient disk reads, then the stacked point dimension is
    rechunked for bounded spatial selections. CUDAHeat still materializes a
    complete time series before every GPU transfer.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        variable: str,
        lazy: bool = False,
        time_dim: str | None = "time",
        spatial_chunk: int = 256,
    ) -> None:
        if spatial_chunk < 1:
            raise ValueError("lazy_spatial_chunk must be positive")

        self.lazy = lazy
        self.variable = variable
        self._spatial_chunk = spatial_chunk
        self._dataset: xr.Dataset | None = None
        self._optimized_point_layout = False
        if not lazy:
            self._dataset = xr.open_dataset(source)
            return

        try:
            import dask.array  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ImportError("lazy=True requires Dask; reinstall cudaheat.") from exc

        # Metadata inspection does not materialize the SST variable. It lets
        # us preserve native on-disk chunks where they exist, preventing Dask
        # from splitting compressed NetCDF chunks during every read.
        metadata = xr.open_dataset(source)
        try:
            if variable not in metadata:
                raise KeyError(f"variable {variable!r} is not present in the dataset")
            data = metadata[variable]
            resolved_time_dim = resolve_time_dim(data, time_dim)
            self._optimized_point_layout = is_optimized_point_layout(data, resolved_time_dim)
            spatial_dims = tuple(dim for dim in data.dims if dim != resolved_time_dim)
            if not spatial_dims:
                raise ValueError("SST variable needs at least one spatial dimension besides time")
            stored_chunks = data.encoding.get("chunksizes")
            if _has_native_chunks(stored_chunks, data.ndim):
                chunks: dict[str, int] = {}
            else:
                # Contiguous/un-chunked files have no native layout to honor.
                # Use full time and bounded spatial chunks as the safe fallback.
                chunks = {resolved_time_dim: -1}
                chunks.update({dim: 1 for dim in spatial_dims[:-1]})
                chunks[spatial_dims[-1]] = spatial_chunk
        finally:
            metadata.close()

        self._dataset = xr.open_dataset(source, chunks=chunks)

    @property
    def dataset(self) -> xr.Dataset:
        """The open Xarray dataset, without materializing its data variables."""
        if self._dataset is None:
            raise RuntimeError("NetCDF reader is closed")
        return self._dataset

    def materialize_spatial_batch(
        self,
        stacked: xr.DataArray,
        point_start: int,
        point_stop: int,
        *,
        dtype: np.dtype[Any],
    ) -> np.ndarray:
        """Load ``all time × points[start:stop]`` as a contiguous NumPy array."""
        selected = stacked.isel(point=slice(point_start, point_stop)).load()
        try:
            return np.ascontiguousarray(selected.data, dtype=dtype)
        finally:
            del selected

    def rechunk_stacked(self, stacked: xr.DataArray) -> xr.DataArray:
        """Apply the lazy-reader spatial chunk target after native-chunk open."""
        if not self.lazy or self._optimized_point_layout:
            return stacked
        return stacked.chunk({"point": self._spatial_chunk})

    def close(self) -> None:
        """Close the underlying NetCDF handle."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None


def resolve_time_dim(source: xr.DataArray, requested: str | None) -> str:
    """Resolve the time dimension using CUDAHeat's existing conventions."""
    if requested is not None and requested in source.dims:
        return requested
    named = [dim for dim in source.dims if dim.lower() == "time"]
    if len(named) == 1:
        return named[0]
    datetime_dims = [
        dim
        for dim in source.dims
        if dim in source.coords and np.issubdtype(source[dim].dtype, np.datetime64)
    ]
    if len(datetime_dims) == 1:
        return datetime_dims[0]
    raise ValueError("could not identify one time dimension; pass time_dim explicitly")


def resolve_spatial_dimensions(
    source: xr.DataArray, dataset: xr.Dataset, *, time_dim: str
) -> tuple[str, str, str, str]:
    """Find 1-D CF-style latitude and longitude coordinates for a grid."""
    candidates = tuple(dimension for dimension in source.dims if dimension != time_dim)
    if len(candidates) != 2:
        raise ValueError("preprocessing requires exactly two spatial dimensions besides time")
    latitude = _find_geographic_coordinate(dataset, candidates, "latitude")
    longitude = _find_geographic_coordinate(dataset, candidates, "longitude")
    if latitude is None or longitude is None:
        raise ValueError("could not identify latitude and longitude coordinates")
    latitude_name, latitude_dim = latitude
    longitude_name, longitude_dim = longitude
    if latitude_dim == longitude_dim:
        raise ValueError("latitude and longitude must use distinct spatial dimensions")
    return latitude_dim, longitude_dim, latitude_name, longitude_name


def is_optimized_point_layout(source: xr.DataArray, time_dim: str) -> bool:
    """Return whether a variable is already CUDAHeat's ``time × point`` layout."""
    return tuple(dimension for dimension in source.dims if dimension != time_dim) == ("point",)


def _find_geographic_coordinate(
    dataset: xr.Dataset, dimensions: tuple[str, ...], kind: str
) -> tuple[str, str] | None:
    aliases = {"latitude": {"latitude", "lat"}, "longitude": {"longitude", "lon"}}[kind]
    for name, coordinate in dataset.variables.items():
        if coordinate.ndim != 1 or coordinate.dims[0] not in dimensions:
            continue
        lowered = name.lower()
        attrs = coordinate.attrs
        standard_name = str(attrs.get("standard_name", "")).lower()
        axis = str(attrs.get("axis", "")).upper()
        units = str(attrs.get("units", "")).lower()
        expected_axis = "Y" if kind == "latitude" else "X"
        expected_units = "degrees_north" if kind == "latitude" else "degrees_east"
        if (
            lowered in aliases
            or standard_name == kind
            or axis == expected_axis
            or expected_units in units
        ):
            return name, coordinate.dims[0]
    return None


def _has_native_chunks(chunks: Any, ndim: int) -> bool:
    """Return whether NetCDF metadata supplies one positive chunk per axis."""
    return (
        isinstance(chunks, (tuple, list))
        and len(chunks) == ndim
        and all(isinstance(size, Integral) and size > 0 for size in chunks)
    )
