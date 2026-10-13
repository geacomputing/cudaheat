"""Sparse event results and memory-safe dense event-cube reconstruction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import xarray as xr

from .climatology import (
    CALENDAR_MODE_MONTH_DAY_HOUR,
    baseline_rows,
    climatology_bin_count,
    cpu_pre_event_fields,
    hourly_climatology_index,
)


@dataclass(frozen=True)
class EventGrid:
    """Small CPU metadata needed to expand an event catalogue."""

    time: np.ndarray
    point_index: Any
    point_indices: np.ndarray
    time_dim: str
    point_coordinates: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.time.size, self.point_indices.size)


@dataclass
class EventDetectionResult:
    """Sparse event catalogue plus GPU-independent dense reconstruction tools."""

    data: xr.Dataset
    telemetry: tuple[Any, ...]
    metadata: dict[str, Any]
    _grid: EventGrid

    @property
    def event_count(self) -> xr.DataArray:
        """Number of retained events per spatial point."""
        return self.data["event_count"]

    def to_xarray(self, *, time_chunk_hours: int = 168) -> xr.Dataset:
        """Return a chunked dense event-ID cube reconstructed from the catalogue.

        This method never creates a full in-memory ``(time, space)`` array.
        The returned Dask-backed Dataset has one ``event_id`` variable: zero
        means no event and positive values are the sparse catalogue IDs.
        Compute a spatial or temporal subset, or write it to a chunked store;
        do not call ``.load()`` on the complete production-sized Dataset.
        """
        _validate_time_chunk(time_chunk_hours)
        started = perf_counter()
        dense = _event_cube(self.data, self._grid, time_chunk_hours)
        dense.attrs.update(
            {
                "cudaheat_representation": "chunked dense EventDetectionResult event labels",
                "source_variable": self.metadata.get("variable"),
                "dtype": self.metadata.get("dtype"),
                "baseline": self.metadata.get("baseline"),
                "percentile": self.metadata.get("percentile"),
                "smooth_window_hours": self.metadata.get("smooth_window_hours"),
                "smooth_min_periods": self.metadata.get("smooth_min_periods"),
                "calendar_mode": self.metadata.get("calendar_mode"),
                "min_duration_hours": self.metadata.get("min_duration_hours"),
                "max_internal_gap_hours": self.metadata.get("max_internal_gap_hours"),
                "join_gap_hours": self.metadata.get("join_gap_hours"),
                "event_id_convention": (
                    "0 means no retained event; positive values are sparse event IDs."
                ),
                "stop_index_semantics": "exclusive; each event occupies [start_index, stop_index).",
                "time_chunk_hours": time_chunk_hours,
                "dense_reconstruction_seconds": perf_counter() - started,
            }
        )
        dense["event_id"].attrs["description"] = dense.attrs["event_id_convention"]
        return dense

    def to_dense(self, *, time_chunk_hours: int = 168) -> xr.Dataset:
        """Alias for :meth:`to_xarray`."""
        return self.to_xarray(time_chunk_hours=time_chunk_hours)

    def select_events(
        self,
        *,
        start: Any | None = None,
        end: Any | None = None,
        min_duration_hours: int | None = None,
        point_indices: Iterable[int] | None = None,
    ) -> xr.Dataset:
        """Filter the sparse catalogue on CPU and recalculate event counts.

        Time filtering keeps events that overlap the closed interval.  This is
        independent of CUDA, CuPy, and the source NetCDF file.
        """
        if min_duration_hours is not None:
            if not isinstance(min_duration_hours, int) or isinstance(min_duration_hours, bool):
                raise TypeError("min_duration_hours must be an integer or None")
            if min_duration_hours < 1:
                raise ValueError("min_duration_hours must be positive")

        event_total = self.data.sizes["event"]
        selected = np.ones(event_total, dtype=bool)
        if start is not None:
            selected &= np.asarray(self.data["end_time"].values) >= np.datetime64(start)
        if end is not None:
            selected &= np.asarray(self.data["start_time"].values) <= np.datetime64(end)
        if min_duration_hours is not None:
            selected &= np.asarray(self.data["duration_hours"].values) >= min_duration_hours
        if point_indices is not None:
            requested = np.asarray(tuple(point_indices), dtype=np.intp)
            selected &= np.isin(np.asarray(self.data["point_index"].values), requested)

        catalogue = self.data.isel(event=np.flatnonzero(selected)).copy()
        catalogue["event_count"] = _event_counts(catalogue, self._grid)
        return catalogue

    def point_timeseries(self, **spatial_indexers: int) -> xr.Dataset:
        """Return CPU-computed SST and event fields for one positional grid point.

        Keyword arguments use :meth:`xarray.DataArray.isel` positional
        semantics, for example ``latitude=33, longitude=30``.  Only one full
        time series is read from the source NetCDF file, so this is safe for
        plotting production-size grids.
        """
        local_point = _local_point_from_isel(self._grid, spatial_indexers)
        return _source_point_timeseries(self.data, self.metadata, self._grid, local_point)

    def point_events(self, **spatial_indexers: int) -> xr.Dataset:
        """Return the sparse event catalogue for one positional grid point."""
        local_point = _local_point_from_isel(self._grid, spatial_indexers)
        return self.select_events(point_indices=[int(self._grid.point_indices[local_point])])

    def event_timeseries(self, event_id: int, *, pad_hours: int = 24) -> xr.Dataset:
        """Return a plot-ready CPU time window around one retained event.

        The result contains ``sst``, ``climatology``, ``threshold``,
        ``exceedance``, and ``event_id``.  No CUDA work is performed.
        """
        if not isinstance(event_id, int) or isinstance(event_id, bool):
            raise TypeError("event_id must be an integer")
        _validate_pad_hours(pad_hours)
        event_positions = np.flatnonzero(np.asarray(self.data["event"].values) == event_id)
        if event_positions.size != 1:
            raise KeyError(f"event ID {event_id} is not present in this result")
        event_position = int(event_positions[0])
        global_point = int(np.asarray(self.data["point_index"].values)[event_position])
        local_point = _local_point_from_global(self._grid, global_point)
        history = _source_point_timeseries(self.data, self.metadata, self._grid, local_point)
        start = int(np.asarray(self.data["start_index"].values)[event_position])
        stop = int(np.asarray(self.data["stop_index"].values)[event_position])
        return history.isel(
            {
                self._grid.time_dim: slice(
                    max(0, start - pad_hours),
                    min(history.sizes[self._grid.time_dim], stop + pad_hours),
                )
            }
        )


def _validate_pad_hours(pad_hours: int) -> None:
    if not isinstance(pad_hours, int) or isinstance(pad_hours, bool):
        raise TypeError("pad_hours must be an integer")
    if pad_hours < 0:
        raise ValueError("pad_hours must be non-negative")


def _local_point_from_isel(grid: EventGrid, spatial_indexers: dict[str, int]) -> int:
    """Resolve source-grid positional indexers to one retained point."""
    if not _has_stacked_grid(grid):
        if set(spatial_indexers) != {"point"}:
            raise ValueError("provide exactly this positional indexer: ('point',)")
        position = spatial_indexers["point"]
        if not isinstance(position, int) or isinstance(position, bool):
            raise TypeError("point index must be an integer")
        if position < 0 or position >= grid.point_indices.size:
            raise IndexError(f"point index {position} is outside [0, {grid.point_indices.size})")
        return position
    dimensions = tuple(grid.point_index.names)
    if set(spatial_indexers) != set(dimensions):
        raise ValueError(f"provide exactly these positional indexers: {dimensions!r}")
    matches = np.ones(grid.point_indices.size, dtype=bool)
    for dimension, position in spatial_indexers.items():
        if not isinstance(position, int) or isinstance(position, bool):
            raise TypeError(f"{dimension} index must be an integer")
        level_number = dimensions.index(dimension)
        level = grid.point_index.levels[level_number]
        if position < 0 or position >= len(level):
            raise IndexError(f"{dimension} index {position} is outside [0, {len(level)})")
        matches &= np.asarray(grid.point_index.get_level_values(dimension) == level[position])
    local_points = np.flatnonzero(matches)
    if local_points.size != 1:
        raise ValueError("positional indexers do not identify one retained spatial point")
    return int(local_points[0])


def _local_point_from_global(grid: EventGrid, global_point: int) -> int:
    local_points = np.flatnonzero(grid.point_indices == global_point)
    if local_points.size != 1:
        raise RuntimeError(f"event catalogue references unavailable point index {global_point}")
    return int(local_points[0])


def _source_point_timeseries(
    sparse: xr.Dataset, metadata: dict[str, Any], grid: EventGrid, local_point: int
) -> xr.Dataset:
    """Read and process one source point entirely on the CPU."""
    source_path = metadata.get("source_path")
    variable = metadata.get("variable")
    spatial_dims = tuple(metadata.get("spatial_dimensions", ()))
    if not source_path or not variable or not spatial_dims:
        raise RuntimeError(
            "point timeseries reconstruction requires CUDAHeat to be created from a NetCDF path"
        )

    global_point = int(grid.point_indices[local_point])
    time_dim = grid.time_dim
    with xr.open_dataset(source_path) as source:
        if variable not in source:
            raise KeyError(f"source variable {variable!r} is not present in {source_path!r}")
        if spatial_dims == ("point",):
            stacked = source[variable].transpose(time_dim, "point")
        else:
            stacked = (
                source[variable]
                .transpose(time_dim, *spatial_dims)
                .stack(point=spatial_dims)
                .transpose(time_dim, "point")
            )
        selected = stacked.isel(point=global_point).load()
        values = np.ascontiguousarray(selected.data, dtype=np.dtype(metadata["dtype"]))
        time = np.array(selected[time_dim].values, copy=True)
        if spatial_dims == ("point",):
            point_coords = {
                name: coordinate.item()
                for name, coordinate in selected.coords.items()
                if name not in {time_dim, "point"} and coordinate.ndim == 0
            }
        else:
            point_coords = {
                dimension: selected.coords[dimension].item()
                for dimension in spatial_dims
                if dimension in selected.coords
            }

    time_coordinate = xr.DataArray(time, dims=(time_dim,), coords={time_dim: time})
    calendar_mode = metadata.get("calendar_mode", CALENDAR_MODE_MONTH_DAY_HOUR)
    bins = hourly_climatology_index(time_coordinate, calendar_mode=calendar_mode)
    rows = baseline_rows(time_coordinate, metadata.get("baseline"))
    nbins = climatology_bin_count(bins, rows, calendar_mode=calendar_mode)
    fields = cpu_pre_event_fields(
        values[:, None],
        bins,
        rows,
        percentile=float(metadata["percentile"]),
        smooth_window_hours=int(metadata["smooth_window_hours"]),
        nbins=nbins,
        smooth_min_periods=int(metadata.get("smooth_min_periods", 1)),
        calendar_mode=calendar_mode,
    )
    climatology_by_time = np.full(values.size, np.nan, dtype=fields.climatology.dtype)
    valid_bins = (bins >= 0) & (bins < nbins)
    climatology_by_time[valid_bins] = fields.climatology[bins[valid_bins], 0]
    event_id = _point_event_ids(sparse, global_point, values.size)
    coordinates: dict[str, Any] = {time_dim: time}
    coordinates.update(point_coords)
    return xr.Dataset(
        {
            "sst": (time_dim, values),
            "climatology": (time_dim, climatology_by_time),
            "threshold": (time_dim, fields.threshold[:, 0]),
            "exceedance": (time_dim, fields.exceedance[:, 0]),
            "event_id": (time_dim, event_id),
        },
        coords=coordinates,
    )


def _point_event_ids(sparse: xr.Dataset, global_point: int, ntime: int) -> np.ndarray:
    """Expand only the retained events for one point into a 1-D label array."""
    event_count = int(sparse.sizes["event"])
    dtype = np.uint32 if event_count <= np.iinfo(np.uint32).max else np.uint64
    labels = np.zeros(ntime, dtype=dtype)
    event_ids = np.asarray(sparse["event"].values, dtype=dtype)
    points = np.asarray(sparse["point_index"].values, dtype=np.intp)
    starts = np.asarray(sparse["start_index"].values, dtype=np.intp)
    stops = np.asarray(sparse["stop_index"].values, dtype=np.intp)
    for event_id, start, stop in zip(
        event_ids[points == global_point],
        starts[points == global_point],
        stops[points == global_point],
        strict=True,
    ):
        labels[start:stop] = event_id
    return labels


def _validate_time_chunk(time_chunk_hours: int) -> None:
    if not isinstance(time_chunk_hours, int) or isinstance(time_chunk_hours, bool):
        raise TypeError("time_chunk_hours must be an integer")
    if time_chunk_hours < 1:
        raise ValueError("time_chunk_hours must be positive")


def _event_cube(sparse: xr.Dataset, grid: EventGrid, time_chunk_hours: int) -> xr.Dataset:
    """Create a Dask-backed event-ID cube from compact event intervals."""
    try:
        import dask.array as da
    except ModuleNotFoundError as exc:  # pragma: no cover - declared dependency
        raise ImportError(
            "Dask is required for dense event reconstruction; reinstall cudaheat."
        ) from exc

    event_id = _lazy_event_id_array(sparse, grid, da, time_chunk_hours)
    return xr.Dataset(
        {
            "event_id": _time_point_array(event_id, "event_id", grid),
            "event_count": sparse["event_count"],
        }
    )


def _time_point_array(values: Any, name: str, grid: EventGrid) -> xr.DataArray:
    """Restore source spatial dimensions from CUDAHeat's point MultiIndex."""
    array = xr.DataArray(
        values,
        dims=(grid.time_dim, "point"),
        coords={grid.time_dim: grid.time, "point": grid.point_index},
        name=name,
    )
    return _restore_point_coordinates(array, grid)


def _lazy_event_id_array(
    sparse: xr.Dataset, grid: EventGrid, da: Any, time_chunk_hours: int
) -> Any:
    """Represent dense labels as independently reconstructed time blocks."""
    event_count = int(sparse.sizes["event"])
    dtype = np.uint32 if event_count <= np.iinfo(np.uint32).max else np.uint64
    event_ids = np.asarray(sparse["event"].values, dtype=dtype)
    points = np.asarray(sparse["point_index"].values, dtype=np.intp)
    starts = np.asarray(sparse["start_index"].values, dtype=np.intp)
    stops = np.asarray(sparse["stop_index"].values, dtype=np.intp)
    local_points = {int(point): index for index, point in enumerate(grid.point_indices)}
    ntime, npoints = grid.shape
    time_chunk = min(time_chunk_hours, ntime)
    block_events = _partition_event_blocks(starts, stops, time_chunk, ntime)
    template = da.empty((ntime, npoints), chunks=(time_chunk, npoints), dtype=dtype)
    return da.map_blocks(
        _event_id_block,
        template,
        dtype=dtype,
        event_ids=event_ids,
        points=points,
        starts=starts,
        stops=stops,
        local_points=local_points,
        block_events=block_events,
    )


def _partition_event_blocks(
    starts: np.ndarray, stops: np.ndarray, time_chunk: int, ntime: int
) -> tuple[np.ndarray, ...]:
    """Index each sparse run only in the time blocks it overlaps."""
    nblocks = (ntime + time_chunk - 1) // time_chunk
    block_events: list[list[int]] = [[] for _ in range(nblocks)]
    for event_index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        first_block = int(start) // time_chunk
        last_block = (int(stop) - 1) // time_chunk
        for block_index in range(first_block, last_block + 1):
            block_events[block_index].append(event_index)
    return tuple(np.asarray(indices, dtype=np.intp) for indices in block_events)


def _event_id_block(
    block: np.ndarray,
    *,
    event_ids: np.ndarray,
    points: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    local_points: dict[int, int],
    block_events: tuple[np.ndarray, ...],
    block_info: dict[Any, Any] | None = None,
) -> np.ndarray:
    """Materialize labels only for one Dask time block."""
    labels = np.zeros(block.shape, dtype=event_ids.dtype)
    if block_info is None:
        return labels
    time_start, time_stop = block_info[None]["array-location"][0]
    event_indices = block_events[block_info[None]["chunk-location"][0]]
    for event_id, point, start, stop in zip(
        event_ids[event_indices],
        points[event_indices],
        starts[event_indices],
        stops[event_indices],
        strict=True,
    ):
        labels[
            max(start, time_start) - time_start : min(stop, time_stop) - time_start,
            local_points[int(point)],
        ] = event_id
    return labels


def _event_counts(sparse: xr.Dataset, grid: EventGrid) -> xr.DataArray:
    """Count catalogue rows per point without constructing a time-by-space array."""
    counts = np.zeros(grid.point_indices.size, dtype=np.uint64)
    local_points = {int(point): index for index, point in enumerate(grid.point_indices)}
    for point in np.asarray(sparse["point_index"].values, dtype=np.intp):
        counts[local_points[int(point)]] += 1
    array = xr.DataArray(
        counts,
        dims=("point",),
        coords={"point": grid.point_index},
        name="event_count",
    )
    return _restore_point_coordinates(array, grid)


def _restore_point_coordinates(array: xr.DataArray, grid: EventGrid) -> xr.DataArray:
    """Unstack ordinary grids or retain compact optimized-point metadata."""
    if _has_stacked_grid(grid):
        return array.unstack("point")
    return array.assign_coords(
        {
            name: ("point", values)
            for name, values in grid.point_coordinates.items()
            if name not in array.coords
        }
    )


def _has_stacked_grid(grid: EventGrid) -> bool:
    """Distinguish Xarray's even one-level MultiIndex from a plain point index."""
    return hasattr(grid.point_index, "levels") and hasattr(grid.point_index, "get_level_values")
