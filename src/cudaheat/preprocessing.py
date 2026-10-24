"""One-time, analytical NetCDF layout optimization for CUDAHeat.

The writer deliberately has no timing or candidate-layout loop.  It inspects
one source dataset, derives a deterministic ``(time, point)`` chunk shape,
and writes one compact NetCDF file suitable for CUDAHeat's spatial batching.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import os
from pathlib import Path
import time
from collections.abc import Iterator
from typing import Any, Literal
from uuid import uuid4

import h5netcdf
import h5py
import numpy as np
import xarray as xr

from .cache_paths import optimized_netcdf_path
from .io import is_optimized_point_layout, resolve_spatial_dimensions, resolve_time_dim


DEFAULT_TARGET_CHUNK_BYTES = 4 * 1024 * 1024
"""Default raw HDF5 chunk target: four MiB before compression."""

_PREPROCESSING_SCHEMA_VERSION = 2

_MIB = 1024 * 1024
_PACKING_ATTRIBUTES = frozenset(
    {
        "_FillValue",
        "missing_value",
        "scale_factor",
        "add_offset",
        "valid_min",
        "valid_max",
        "valid_range",
        "coordinates",
    }
)


@dataclass(frozen=True)
class NetCDFInspection:
    """Storage and semantic metadata needed to plan an optimized file."""

    variable: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    time_dim: str
    latitude_dim: str
    longitude_dim: str
    latitude_coordinate: str
    longitude_coordinate: str
    ntime: int
    grid_shape: tuple[int, int]
    dtype: str
    dtype_itemsize: int
    fill_value: Any
    missing_value: Any
    scale_factor: Any
    add_offset: Any
    source_chunks: tuple[int, ...] | None
    compression: str | None


@dataclass(frozen=True)
class PreprocessingReport:
    """Deterministic summary of a completed preprocessing operation."""

    input_path: Path
    output_path: Path
    variable: str
    original_shape: tuple[int, ...]
    original_spatial_points: int
    retained_points: int
    removed_points: int
    complete_points: int
    points_with_gaps: int
    missing_samples: int
    original_dtype: str
    output_dtype: str
    source_chunks: tuple[int, ...] | None
    output_chunks: tuple[int, int]
    target_chunk_bytes: int
    actual_chunk_bytes: int
    source_compression: str | None
    output_compression: str | None
    input_size_bytes: int
    output_size_bytes: int
    max_dtype_conversion_error: float
    elapsed_seconds: float
    validation_passed: bool

    def __str__(self) -> str:
        return (
            "CUDAHeat NetCDF preprocessing\n"
            "=============================\n"
            f"Variable: {self.variable}\n"
            f"Input shape: {' x '.join(map(str, self.original_shape))}\n"
            f"Retained / removed points: {self.retained_points} / {self.removed_points}\n"
            f"Partial-gap points / samples: {self.points_with_gaps} / {self.missing_samples}\n"
            f"Output layout: time x point, chunks {self.output_chunks}\n"
            f"Raw chunk size: {self.actual_chunk_bytes / _MIB:.2f} MiB "
            f"(target {self.target_chunk_bytes / _MIB:.2f} MiB)\n"
            "Selection: analytical; benchmarking: none\n"
            f"Validation: {'passed' if self.validation_passed else 'failed'}\n"
            f"Elapsed: {self.elapsed_seconds:.2f} s"
        )


@dataclass(frozen=True)
class PreparedNetCDF:
    """The actual file selected for a CUDAHeat path input.

    ``auto`` preparation never overwrites a researcher's source file.  For an
    ordinary grid it creates and validates a deterministic sibling cache; for
    an already compact ``time × point`` file it simply returns the original.
    """

    source_path: Path
    active_path: Path
    was_already_optimized: bool
    reused_cached_optimization: bool
    report: PreprocessingReport | None = None
    source_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "active_path": str(self.active_path),
            "source_sha256": self.source_sha256,
            "source_checksum_algorithm": "sha256",
            "was_already_optimized": self.was_already_optimized,
            "reused_cached_optimization": self.reused_cached_optimization,
            "report": None if self.report is None else {
                "output_chunks": self.report.output_chunks,
                "retained_points": self.report.retained_points,
                "removed_points": self.report.removed_points,
                "elapsed_seconds": self.report.elapsed_seconds,
            },
        }


def inspect_netcdf(
    input_path: str | Path,
    *,
    variable: str,
    time_dim: str | None = "time",
) -> NetCDFInspection:
    """Inspect a compatible gridded NetCDF without materializing its SST cube."""
    path = Path(input_path)
    with xr.open_dataset(path, decode_cf=False, mask_and_scale=False) as dataset:
        if variable not in dataset:
            raise KeyError(f"variable {variable!r} is not present in {path}")
        data = dataset[variable]
        resolved_time = resolve_time_dim(data, time_dim)
        latitude_dim, longitude_dim, latitude_name, longitude_name = resolve_spatial_dimensions(
            data, dataset, time_dim=resolved_time
        )
        source_chunks = _source_chunks(data)
        return NetCDFInspection(
            variable=variable,
            dimensions=tuple(data.dims),
            shape=tuple(int(data.sizes[dimension]) for dimension in data.dims),
            time_dim=resolved_time,
            latitude_dim=latitude_dim,
            longitude_dim=longitude_dim,
            latitude_coordinate=latitude_name,
            longitude_coordinate=longitude_name,
            ntime=int(data.sizes[resolved_time]),
            grid_shape=(int(data.sizes[latitude_dim]), int(data.sizes[longitude_dim])),
            dtype=np.dtype(data.dtype).name,
            dtype_itemsize=np.dtype(data.dtype).itemsize,
            fill_value=data.attrs.get("_FillValue"),
            missing_value=data.attrs.get("missing_value"),
            scale_factor=data.attrs.get("scale_factor"),
            add_offset=data.attrs.get("add_offset"),
            source_chunks=source_chunks,
            compression=_source_compression(path, variable, data.encoding),
        )


def prepare_netcdf_input(
    input_path: str | Path,
    *,
    variable: str,
    dtype: str | np.dtype[Any] = "float32",
    strategy: Literal["auto", "skip", "require"] = "auto",
    time_dim: str | None = "time",
) -> PreparedNetCDF:
    """Return a CUDAHeat-ready NetCDF without ever replacing ``input_path``.

    ``auto`` accepts an existing compact file or creates/reuses
    ``<stem>.cudaheat<suffix>`` beside an ordinary gridded input.  A cache is
    reused only when its stored source file identity still matches, preventing
    an older optimization from being used after the source was replaced.
    """
    if strategy not in {"auto", "skip", "require"}:
        raise ValueError("input_optimization must be 'auto', 'skip', or 'require'")
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"NetCDF input does not exist: {source}")
    source_sha256 = _sha256_file(source)
    if _is_cudaheat_point_file(source, variable=variable, time_dim=time_dim):
        return PreparedNetCDF(
            source,
            source,
            was_already_optimized=True,
            reused_cached_optimization=False,
            source_sha256=source_sha256,
        )
    if strategy == "skip":
        return PreparedNetCDF(
            source,
            source,
            was_already_optimized=False,
            reused_cached_optimization=False,
            source_sha256=source_sha256,
        )
    if strategy == "require":
        raise ValueError(
            "input_optimization='require' needs an optimized CUDAHeat time × point NetCDF"
        )

    if source.stem.endswith(".cudaheat"):
        raise ValueError(
            "input is named like a CUDAHeat-derived file but is not a valid CUDAHeat point layout; "
            "provide the original NetCDF instead"
        )
    destination = _automatic_optimized_path(source, source_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _cached_optimization_matches(
        destination,
        source,
        source_sha256=source_sha256,
        variable=variable,
        dtype=np.dtype(dtype),
        time_dim=time_dim,
    ):
        return PreparedNetCDF(
            source,
            destination,
            was_already_optimized=False,
            reused_cached_optimization=True,
            source_sha256=source_sha256,
        )
    report = optimize_netcdf(
        source,
        destination,
        variable=variable,
        dtype=dtype,
        time_dim=time_dim,
        overwrite=destination.exists(),
        _source_sha256=source_sha256,
    )
    return PreparedNetCDF(
        source,
        destination,
        was_already_optimized=False,
        reused_cached_optimization=False,
        report=report,
        source_sha256=source_sha256,
    )


def choose_chunk_shape(
    ntime: int,
    npoints: int,
    dtype: str | np.dtype[Any],
    *,
    source_chunks: tuple[int, ...] | None = None,
    target_chunk_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
    expected_batch_points: int | None = None,
) -> tuple[int, int]:
    """Choose a deterministic physical ``(time, point)`` chunk shape.

    The policy uses metadata and arithmetic only.  Temporal chunks grow with
    the time horizon (week/month/quarter/year scale) so full histories do not
    touch an excessive number of chunks.  Point chunks then fill the selected
    raw-byte budget and are aligned modestly to source spatial tiles where
    that information is available.
    """
    if ntime < 1 or npoints < 1:
        raise ValueError("ntime and npoints must both be positive")
    value_dtype = np.dtype(dtype)
    if value_dtype.itemsize < 1:
        raise ValueError("dtype must have a positive item size")
    if target_chunk_bytes < value_dtype.itemsize:
        raise ValueError("target_chunk_bytes must fit at least one value")
    if expected_batch_points is not None and expected_batch_points < 1:
        raise ValueError("expected_batch_points must be positive when supplied")

    # Long histories benefit from month/quarter/year-sized time tiles; short
    # histories stay whole.  These are storage intervals, not scientific bins.
    if ntime < 24:
        time_chunk = ntime
    elif ntime <= 31 * 24:
        time_chunk = min(ntime, 7 * 24)
    elif ntime <= 5 * 366 * 24:
        time_chunk = 31 * 24
    elif ntime <= 30 * 366 * 24:
        time_chunk = 92 * 24
    else:
        time_chunk = 366 * 24

    source_time, source_spatial = _source_chunk_hints(source_chunks)
    if source_time >= 24:
        # Preserve useful source cadence as a bounded hint, never as a layout
        # constraint.  It cannot grow the chosen interval by more than 2x.
        hinted = _hour_multiple(min(source_time, time_chunk * 2))
        if hinted >= max(24, time_chunk // 2):
            time_chunk = hinted
    time_chunk = min(ntime, max(1, time_chunk))

    # A single time axis is extremely unlikely to consume the budget, but cap
    # it nevertheless so the byte target remains a real invariant where able.
    max_time_for_one_point = max(1, target_chunk_bytes // value_dtype.itemsize)
    if time_chunk > max_time_for_one_point:
        time_chunk = _hour_multiple(max_time_for_one_point) if max_time_for_one_point >= 24 else max_time_for_one_point
        time_chunk = max(1, min(ntime, time_chunk))

    ideal_points = max(1, target_chunk_bytes // (time_chunk * value_dtype.itemsize))
    point_chunk = min(npoints, ideal_points)
    if expected_batch_points is not None:
        point_chunk = min(point_chunk, expected_batch_points)

    # Aligning no more than 256 points keeps point-range reads tidy without
    # hard-wiring CUDAHeat's current GPU batch size.
    alignment = _alignment_hint(source_spatial, point_chunk)
    if alignment > 1 and point_chunk > alignment:
        point_chunk = _closest_aligned_point_count(
            point_chunk, alignment, npoints, time_chunk, value_dtype.itemsize, target_chunk_bytes
        )
    return (max(1, min(ntime, time_chunk)), max(1, min(npoints, point_chunk)))


def optimize_netcdf(
    input_path: str | Path,
    output_path: str | Path,
    *,
    variable: str,
    dtype: str | np.dtype[Any] = "float32",
    time_dim: str | None = "time",
    chunking: Literal["auto"] = "auto",
    time_chunk_size: int | None = None,
    point_chunk_size: int | None = None,
    target_chunk_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
    expected_batch_points: int | None = None,
    compression: Literal["auto", "gzip", "none"] | None = "auto",
    compression_level: int = 1,
    overwrite: bool = False,
    _source_sha256: str | None = None,
) -> PreprocessingReport:
    """Create one compact, physically rechunked ``time × point`` NetCDF.

    Permanent-invalid points are removed; valid points with partial gaps are
    retained unchanged.  Data is decoded once by Xarray and written directly
    as the requested floating-point dtype.  No benchmarking or autotuning is
    performed.
    """
    started = time.perf_counter()
    source_path = Path(input_path)
    destination = Path(output_path)
    if source_path.resolve() == destination.resolve():
        raise ValueError("input_path and output_path must be different files")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}; pass overwrite=True")
    if not destination.parent.exists():
        raise FileNotFoundError(f"output directory does not exist: {destination.parent}")
    if chunking != "auto":
        raise ValueError("chunking must be 'auto'; use explicit chunk-size overrides when needed")
    output_dtype = np.dtype(dtype)
    if output_dtype.kind != "f":
        raise ValueError("dtype must be a floating-point type so missing values remain representable")
    if compression_level < 0 or compression_level > 9:
        raise ValueError("compression_level must be between 0 and 9")
    source_sha256 = _source_sha256 or _sha256_file(source_path)
    source_signature = _source_stat_signature(source_path)

    inspection = inspect_netcdf(source_path, variable=variable, time_dim=time_dim)
    nlat, nlon = inspection.grid_shape
    original_points = nlat * nlon
    source_chunks_time_order = _chunks_in_time_order(inspection)
    resolved_compression = _resolve_compression(compression, inspection.compression)

    # Decoding/masking happens here exactly once per source block.  Keeping the
    # source opened rather than calling .values avoids a full-cube materialize.
    decoded = xr.open_dataset(source_path, decode_cf=True, mask_and_scale=True)
    raw = xr.open_dataset(source_path, decode_cf=False, mask_and_scale=False)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    max_conversion_error = 0.0
    try:
        data = decoded[variable]
        ordered = data.transpose(
            inspection.time_dim, inspection.latitude_dim, inspection.longitude_dim
        )
        valid_samples, missing_samples_by_point = _classify_points(
            ordered, original_points, output_dtype.itemsize
        )
        retained_flat = np.flatnonzero(valid_samples > 0).astype(np.intp, copy=False)
        if retained_flat.size == 0:
            raise ValueError("input variable has no spatial point with a valid time sample")
        partial = (valid_samples > 0) & (missing_samples_by_point > 0)
        complete = valid_samples == inspection.ntime

        automatic_chunks = choose_chunk_shape(
            inspection.ntime,
            int(retained_flat.size),
            output_dtype,
            source_chunks=source_chunks_time_order,
            target_chunk_bytes=target_chunk_bytes,
            expected_batch_points=expected_batch_points,
        )
        time_chunk, point_chunk = _apply_chunk_overrides(
            automatic_chunks,
            inspection.ntime,
            int(retained_flat.size),
            time_chunk_size,
            point_chunk_size,
        )

        latitude_values = np.asarray(decoded[inspection.latitude_coordinate].values)
        longitude_values = np.asarray(decoded[inspection.longitude_coordinate].values)
        lat_index = retained_flat // nlon
        lon_index = retained_flat % nlon
        max_conversion_error = _write_optimized_file(
            temporary,
            raw=raw,
            ordered=ordered,
            inspection=inspection,
            variable=variable,
            output_dtype=output_dtype,
            retained_flat=retained_flat,
            lat_index=lat_index,
            lon_index=lon_index,
            latitude_values=latitude_values,
            longitude_values=longitude_values,
            chunks=(time_chunk, point_chunk),
            compression=resolved_compression,
            compression_level=compression_level,
            source_path=source_path,
            source_sha256=source_sha256,
        )
        _validate_output(
            temporary,
            variable=variable,
            expected_shape=(inspection.ntime, int(retained_flat.size)),
            expected_dtype=output_dtype,
            expected_chunks=(time_chunk, point_chunk),
            expected_points=int(retained_flat.size),
        )
        if _source_stat_signature(source_path) != source_signature:
            raise RuntimeError("source NetCDF changed while CUDAHeat was optimizing it")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        raw.close()
        decoded.close()

    actual_bytes = time_chunk * point_chunk * output_dtype.itemsize
    return PreprocessingReport(
        input_path=source_path,
        output_path=destination,
        variable=variable,
        original_shape=inspection.shape,
        original_spatial_points=original_points,
        retained_points=int(retained_flat.size),
        removed_points=int(original_points - retained_flat.size),
        complete_points=int(np.count_nonzero(complete)),
        points_with_gaps=int(np.count_nonzero(partial)),
        missing_samples=int(missing_samples_by_point[partial].sum()),
        original_dtype=inspection.dtype,
        output_dtype=output_dtype.name,
        source_chunks=inspection.source_chunks,
        output_chunks=(time_chunk, point_chunk),
        target_chunk_bytes=target_chunk_bytes,
        actual_chunk_bytes=actual_bytes,
        source_compression=inspection.compression,
        output_compression=resolved_compression,
        input_size_bytes=source_path.stat().st_size,
        output_size_bytes=destination.stat().st_size,
        max_dtype_conversion_error=max_conversion_error,
        elapsed_seconds=time.perf_counter() - started,
        validation_passed=True,
    )


def _classify_points(
    ordered: xr.DataArray, npoints: int, output_itemsize: int
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.zeros(npoints, dtype=np.int64)
    missing = np.zeros(npoints, dtype=np.int64)
    for start, stop in _time_blocks(ordered.sizes[ordered.dims[0]], npoints, output_itemsize):
        values = np.asarray(ordered.isel({ordered.dims[0]: slice(start, stop)}).load().data)
        flat = values.reshape(stop - start, npoints)
        invalid = ~np.isfinite(flat)
        missing += invalid.sum(axis=0, dtype=np.int64)
        valid += (~invalid).sum(axis=0, dtype=np.int64)
    return valid, missing


def _write_optimized_file(
    path: Path,
    *,
    raw: xr.Dataset,
    ordered: xr.DataArray,
    inspection: NetCDFInspection,
    variable: str,
    output_dtype: np.dtype[Any],
    retained_flat: np.ndarray,
    lat_index: np.ndarray,
    lon_index: np.ndarray,
    latitude_values: np.ndarray,
    longitude_values: np.ndarray,
    chunks: tuple[int, int],
    compression: str | None,
    compression_level: int,
    source_path: Path,
    source_sha256: str,
) -> float:
    ntime = inspection.ntime
    npoints = int(retained_flat.size)
    maximum_conversion_error = 0.0
    with h5netcdf.File(path, "w") as target:
        target.dimensions["time"] = ntime
        target.dimensions["point"] = npoints
        target.dimensions["original_latitude"] = latitude_values.size
        target.dimensions["original_longitude"] = longitude_values.size
        target.attrs.update(
            {
                "cudaheat_preprocessed": "true",
                "cudaheat_preprocessing_schema_version": _PREPROCESSING_SCHEMA_VERSION,
                "cudaheat_layout": "time, point",
                "cudaheat_flattening": "C-order: original_flat_index = lat_index * nlon + lon_index",
                "cudaheat_original_time_dimension": inspection.time_dim,
                "cudaheat_original_latitude_dimension": inspection.latitude_dim,
                "cudaheat_original_longitude_dimension": inspection.longitude_dim,
                "cudaheat_chunk_selection": "analytical; benchmarking=none",
                # These attributes make the automatic sibling cache safe to
                # reuse while keeping manual preprocessing fully transparent.
                "cudaheat_source_path": str(source_path.resolve()),
                "cudaheat_source_size_bytes": int(source_path.stat().st_size),
                "cudaheat_source_mtime_ns": int(source_path.stat().st_mtime_ns),
                "cudaheat_source_sha256": source_sha256,
                "cudaheat_source_checksum_algorithm": "sha256",
                "cudaheat_source_variable": variable,
                "cudaheat_output_dtype": output_dtype.name,
            }
        )
        _write_time_coordinate(target, raw, inspection.time_dim)
        _write_coordinate(
            target,
            "point",
            ("point",),
            np.arange(npoints, dtype=np.int64),
            {"long_name": "CUDAHeat compact point index"},
        )
        _write_coordinate(
            target,
            "latitude",
            ("point",),
            latitude_values[lat_index],
            _safe_attrs(ordered[inspection.latitude_coordinate].attrs),
        )
        _write_coordinate(
            target,
            "longitude",
            ("point",),
            longitude_values[lon_index],
            _safe_attrs(ordered[inspection.longitude_coordinate].attrs),
        )
        _write_coordinate(
            target,
            "original_latitude",
            ("original_latitude",),
            latitude_values,
            _safe_attrs(ordered[inspection.latitude_coordinate].attrs),
        )
        _write_coordinate(
            target,
            "original_longitude",
            ("original_longitude",),
            longitude_values,
            _safe_attrs(ordered[inspection.longitude_coordinate].attrs),
        )
        _write_coordinate(
            target,
            "original_lat_index",
            ("point",),
            lat_index.astype(np.int64, copy=False),
            {"long_name": "zero-based index into original_latitude"},
        )
        _write_coordinate(
            target,
            "original_lon_index",
            ("point",),
            lon_index.astype(np.int64, copy=False),
            {"long_name": "zero-based index into original_longitude"},
        )
        _write_coordinate(
            target,
            "original_flat_index",
            ("point",),
            retained_flat.astype(np.int64, copy=False),
            {"long_name": "C-order flattened index in original latitude x longitude grid"},
        )

        variable_kwargs: dict[str, Any] = {
            "chunks": chunks,
            "fillvalue": np.array(np.nan, dtype=output_dtype).item(),
        }
        if compression == "gzip":
            variable_kwargs.update(
                compression="gzip", compression_opts=compression_level, shuffle=True
            )
        output = target.create_variable(
            variable, ("time", "point"), dtype=output_dtype, **variable_kwargs
        )
        output.attrs.update(_safe_attrs(ordered.attrs))
        output.attrs.update(
            {
                "coordinates": "latitude longitude original_lat_index original_lon_index original_flat_index",
                "cudaheat_source_variable": variable,
                "cudaheat_decoded_once": "true",
            }
        )
        for start, stop in _time_blocks(ntime, inspection.grid_shape[0] * inspection.grid_shape[1], output_dtype.itemsize):
            values = np.asarray(ordered.isel({inspection.time_dim: slice(start, stop)}).load().data)
            compact = values.reshape(stop - start, -1)[:, retained_flat]
            converted = np.ascontiguousarray(compact, dtype=output_dtype)
            finite = np.isfinite(compact)
            if finite.any():
                narrowed = converted.astype(np.float64, copy=False)
                difference = np.max(np.abs(compact[finite].astype(np.float64) - narrowed[finite]))
                maximum_conversion_error = max(maximum_conversion_error, float(difference))
            output[start:stop, :] = converted
    return maximum_conversion_error


def _write_time_coordinate(target: h5netcdf.File, raw: xr.Dataset, source_time_dim: str) -> None:
    source_time = raw[source_time_dim]
    values = np.asarray(source_time.values)
    if values.dtype.kind in "OUS":
        values = values.astype(str)
        dtype: Any = str
    else:
        dtype = values.dtype
    coordinate = target.create_variable("time", ("time",), dtype=dtype)
    coordinate[:] = values
    coordinate.attrs.update(_safe_attrs(source_time.attrs))


def _write_coordinate(
    target: h5netcdf.File,
    name: str,
    dimensions: tuple[str, ...],
    values: np.ndarray,
    attrs: dict[str, Any],
) -> None:
    variable = target.create_variable(name, dimensions, dtype=values.dtype)
    variable[:] = values
    variable.attrs.update(attrs)


def _validate_output(
    path: Path,
    *,
    variable: str,
    expected_shape: tuple[int, int],
    expected_dtype: np.dtype[Any],
    expected_chunks: tuple[int, int],
    expected_points: int,
) -> None:
    """Verify physical storage after closing the one output file."""
    with h5py.File(path, "r") as reopened:
        if variable not in reopened:
            raise RuntimeError("validation failed: output variable is missing")
        data = reopened[variable]
        if data.shape != expected_shape:
            raise RuntimeError(f"validation failed: shape {data.shape} != {expected_shape}")
        if np.dtype(data.dtype) != expected_dtype:
            raise RuntimeError(f"validation failed: dtype {data.dtype} != {expected_dtype}")
        if data.chunks != expected_chunks:
            raise RuntimeError(
                f"validation failed: physical chunks {data.chunks} != {expected_chunks}"
            )
        for name in (
            "latitude",
            "longitude",
            "original_lat_index",
            "original_lon_index",
            "original_flat_index",
        ):
            if name not in reopened or reopened[name].shape != (expected_points,):
                raise RuntimeError(f"validation failed: topology mapping {name!r} is invalid")
        if reopened.attrs.get("cudaheat_preprocessed") != "true":
            raise RuntimeError("validation failed: CUDAHeat preprocessing marker is absent")


def _automatic_optimized_path(source: Path, source_sha256: str) -> Path:
    """Return the central non-destructive cache path used by auto preparation."""
    return optimized_netcdf_path(source, source_sha256)


def _is_cudaheat_point_file(path: Path, *, variable: str, time_dim: str | None) -> bool:
    """Inspect only metadata to determine whether CUDAHeat can consume a file directly."""
    with xr.open_dataset(path, decode_cf=False, mask_and_scale=False) as dataset:
        if variable not in dataset:
            raise KeyError(f"variable {variable!r} is not present in {path}")
        data = dataset[variable]
        return is_optimized_point_layout(data, resolve_time_dim(data, time_dim))


def _cached_optimization_matches(
    cached: Path,
    source: Path,
    *,
    source_sha256: str,
    variable: str,
    dtype: np.dtype[Any],
    time_dim: str | None,
) -> bool:
    """Return whether a sibling optimization is both valid and from this source revision."""
    if not cached.is_file():
        return False
    try:
        source_stat = source.stat()
        with xr.open_dataset(cached, decode_cf=False, mask_and_scale=False) as dataset:
            if variable not in dataset:
                return False
            data = dataset[variable]
            if not is_optimized_point_layout(data, resolve_time_dim(data, time_dim)):
                return False
            attrs = dataset.attrs
            return (
                attrs.get("cudaheat_preprocessed") == "true"
                and attrs.get("cudaheat_preprocessing_schema_version")
                == _PREPROCESSING_SCHEMA_VERSION
                and attrs.get("cudaheat_source_path") == str(source.resolve())
                and int(attrs.get("cudaheat_source_size_bytes", -1)) == source_stat.st_size
                and int(attrs.get("cudaheat_source_mtime_ns", -1)) == source_stat.st_mtime_ns
                and attrs.get("cudaheat_source_sha256") == source_sha256
                and attrs.get("cudaheat_source_checksum_algorithm") == "sha256"
                and attrs.get("cudaheat_source_variable") == variable
                and attrs.get("cudaheat_output_dtype") == dtype.name
            )
    except (OSError, TypeError, ValueError):
        # A damaged or legacy sibling cache is simply rebuilt by ``auto``.
        return False


def _source_stat_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return (int(stat.st_size), int(stat.st_mtime_ns))


def _sha256_file(path: Path) -> str:
    """Return a stable content fingerprint, refusing a file changed mid-read."""
    before = _source_stat_signature(path)
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if _source_stat_signature(path) != before:
        raise RuntimeError(f"source NetCDF changed while its checksum was being calculated: {path}")
    return digest.hexdigest()


def _source_chunks(data: xr.DataArray) -> tuple[int, ...] | None:
    chunks = data.encoding.get("chunksizes")
    if isinstance(chunks, (tuple, list)) and len(chunks) == data.ndim:
        normalized = tuple(int(size) for size in chunks)
        if all(size > 0 for size in normalized):
            return normalized
    return None


def _source_compression(path: Path, variable: str, encoding: dict[str, Any]) -> str | None:
    try:
        with h5py.File(path, "r") as source:
            if variable in source and source[variable].compression:
                return str(source[variable].compression)
    except OSError:
        pass  # A NetCDF3 file has no HDF5 compression metadata.
    if encoding.get("zlib"):
        return "gzip"
    return None


def _chunks_in_time_order(inspection: NetCDFInspection) -> tuple[int, ...] | None:
    if inspection.source_chunks is None:
        return None
    chunks = dict(zip(inspection.dimensions, inspection.source_chunks, strict=True))
    return (
        chunks[inspection.time_dim],
        chunks[inspection.latitude_dim],
        chunks[inspection.longitude_dim],
    )


def _source_chunk_hints(source_chunks: tuple[int, ...] | None) -> tuple[int, int]:
    if not source_chunks:
        return (0, 0)
    source_time = int(source_chunks[0])
    spatial = math.prod(int(size) for size in source_chunks[1:]) if len(source_chunks) > 1 else 0
    return (source_time, spatial)


def _hour_multiple(value: int) -> int:
    return max(24, (value // 24) * 24)


def _alignment_hint(source_spatial: int, point_chunk: int) -> int:
    limit = min(256, source_spatial, point_chunk)
    if limit < 2:
        return 1
    return 1 << (limit.bit_length() - 1)


def _closest_aligned_point_count(
    ideal: int,
    alignment: int,
    npoints: int,
    time_chunk: int,
    itemsize: int,
    target_bytes: int,
) -> int:
    lower = max(alignment, (ideal // alignment) * alignment)
    upper = min(npoints, lower + alignment)
    candidates = [count for count in (lower, upper, npoints) if 1 <= count <= npoints]
    return min(candidates, key=lambda count: (abs(time_chunk * count * itemsize - target_bytes), -count))


def _apply_chunk_overrides(
    automatic: tuple[int, int],
    ntime: int,
    npoints: int,
    time_chunk_size: int | None,
    point_chunk_size: int | None,
) -> tuple[int, int]:
    time_chunk, point_chunk = automatic
    for name, value, maximum in (
        ("time_chunk_size", time_chunk_size, ntime),
        ("point_chunk_size", point_chunk_size, npoints),
    ):
        if value is not None:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            if value > maximum:
                raise ValueError(f"{name} cannot exceed its dimension ({maximum})")
    return (time_chunk_size or time_chunk, point_chunk_size or point_chunk)


def _resolve_compression(
    requested: Literal["auto", "gzip", "none"] | None, source_compression: str | None
) -> str | None:
    if requested == "auto":
        return "gzip" if source_compression else None
    if requested in (None, "none"):
        return None
    if requested == "gzip":
        return "gzip"
    raise ValueError("compression must be 'auto', 'gzip', 'none', or None")


def _time_blocks(ntime: int, npoints: int, itemsize: int) -> Iterator[tuple[int, int]]:
    # Keep each source block around 64 MiB, including an intermediate decoded
    # array.  This bounds host memory independently of the complete cube.
    block = max(1, (64 * _MIB) // max(1, npoints * max(itemsize, 8)))
    block = min(ntime, block)
    return ((start, min(ntime, start + block)) for start in range(0, ntime, block))


def _safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in attrs.items() if name not in _PACKING_ATTRIBUTES}
