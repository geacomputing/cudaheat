"""Validated NetCDF/Zarr access shared by plotting and enrichment execution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import xarray as xr

from cudaheat.cache_paths import climatology_zarr_path, optimized_netcdf_path
from cudaheat.climatology import hourly_climatology_index

from .models import Event, EventWindow, RunReference


def resolve_cache_path(run: RunReference, override: Path | None) -> Path:
    """Resolve the persisted climatology cache or its deterministic fallback."""
    if override is not None:
        return override.expanduser().resolve()
    candidates: list[Path] = []
    recorded = run.metadata.get("climatology_cache_path")
    if isinstance(recorded, str) and recorded:
        candidates.append(Path(recorded).expanduser())
    candidates.append(climatology_zarr_path(run.source_path, run.source_sha256, run.climatology_hash))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("could not find the run's Zarr cache: " + ", ".join(map(str, candidates)))


def open_and_validate_cache(path: Path, run: RunReference) -> Any:
    """Open only the completed cache that belongs to the selected run."""
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ImportError("install CUDAHeat's base dependencies, including zarr") from exc
    group = zarr.open_group(str(path), mode="r")
    expected = {
        "cudaheat_climatology_cache": "true",
        "cudaheat_complete": True,
        "cudaheat_climatology_hash": run.climatology_hash,
        "cudaheat_source_sha256": run.source_sha256,
    }
    mismatches = {key: value for key, value in expected.items() if group.attrs.get(key) != value}
    if mismatches:
        raise ValueError(f"Zarr cache does not match run {run.run_id}: {mismatches}")
    if not {"climatology", "threshold", "point"}.issubset(group):
        raise KeyError("Zarr cache must contain climatology, threshold, and point arrays")
    return group


def resolve_sst_path(run: RunReference, override: Path | None) -> Path:
    """Find only the optimized NetCDF cache associated with this persisted run."""
    if override is not None:
        return override.expanduser().resolve()
    candidates: list[Path] = []
    preparation = run.metadata.get("input_preparation")
    if isinstance(preparation, Mapping):
        active_path = preparation.get("active_path")
        if isinstance(active_path, str) and active_path:
            candidates.append(Path(active_path).expanduser())
    candidates.append(optimized_netcdf_path(run.source_path, run.source_sha256))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "could not find the run's optimized NetCDF cache: " + ", ".join(map(str, candidates))
    )


def open_and_validate_sst(path: Path, run: RunReference) -> xr.Dataset:
    """Open a compatible optimized source without loading SST eagerly."""
    source = xr.open_dataset(path, engine="h5netcdf", cache=False)
    time_dimension = str(run.metadata.get("time_dimension", "time"))
    data = source[run.variable] if run.variable in source else None
    if (
        source.attrs.get("cudaheat_preprocessed") != "true"
        or data is None
        or time_dimension not in data.dims
        or "point" not in data.dims
        or "point" not in source
        or str(source.attrs.get("cudaheat_source_sha256")) != run.source_sha256
    ):
        source.close()
        raise ValueError("SST input must be this run's CUDAHeat optimized (time, point) NetCDF cache")
    return source


def cache_columns(group: Any, grid_point_indices: np.ndarray) -> np.ndarray:
    """Map PostgreSQL point IDs to positional Zarr cache columns."""
    points = np.asarray(group["point"][:], dtype=np.int64)
    order = np.argsort(points)
    locations = np.searchsorted(points[order], grid_point_indices)
    found = locations < points.size
    found[found] &= points[order][locations[found]] == grid_point_indices[found]
    if not np.all(found):
        raise LookupError(f"event points are absent from the Zarr cache: {np.unique(grid_point_indices[~found])[:10].tolist()}")
    return order[locations]


def source_columns(source: xr.Dataset, grid_point_indices: np.ndarray) -> np.ndarray:
    """Map PostgreSQL point IDs to positional optimized-NetCDF columns."""
    points = np.asarray(source["point"].values, dtype=np.int64)
    order = np.argsort(points)
    locations = np.searchsorted(points[order], grid_point_indices)
    found = locations < points.size
    found[found] &= points[order][locations[found]] == grid_point_indices[found]
    if not np.all(found):
        raise LookupError(f"event points are absent from the NetCDF source: {np.unique(grid_point_indices[~found])[:10].tolist()}")
    return order[locations]


def source_time_context(source: xr.Dataset, run: RunReference) -> tuple[np.ndarray, np.ndarray]:
    """Return source timestamps and their CUDAHeat climatology-cycle bins."""
    time_dimension = str(run.metadata.get("time_dimension", "time"))
    times = np.asarray(source[time_dimension].values, dtype="datetime64[ns]")
    coordinate = xr.DataArray(times, dims=("event_time",))
    return times, hourly_climatology_index(coordinate, calendar_mode=run.calendar_mode)


def aligned_cycle(array: Any, bins: np.ndarray, column: int) -> np.ndarray:
    """Align one climatology-cycle point to explicit timestamps."""
    cycle = np.asarray(array[:, column], dtype=np.float64)
    values = np.full(bins.size, np.nan, dtype=np.float64)
    available = (bins >= 0) & (bins < cycle.size)
    values[available] = cycle[bins[available]]
    return values


def source_location(source: xr.Dataset, column: int) -> tuple[float | None, float | None]:
    """Return point latitude/longitude where the optimized source provides them."""
    def coordinate(name: str) -> float | None:
        if name not in source or source[name].dims != ("point",):
            return None
        value = float(np.asarray(source[name].isel(point=column).values))
        return value if np.isfinite(value) else None

    return coordinate("latitude"), coordinate("longitude")


def fetch_event_window(
    source: xr.Dataset,
    zarr_group: Any,
    run: RunReference,
    event: Event,
    *,
    source_column: int,
    cache_column: int,
) -> EventWindow:
    """Fetch one aligned, timed event window for the optional benchmark plot."""
    time_dimension = str(run.metadata.get("time_dimension", "time"))
    source_slice = slice(event.start_index, event.stop_index)
    latitude, longitude = source_location(source, source_column)
    started = perf_counter()
    time = np.asarray(source[time_dimension].isel({time_dimension: source_slice}).values)
    bins = hourly_climatology_index(xr.DataArray(time, dims=("event_time",)), calendar_mode=run.calendar_mode)
    timing_bins = perf_counter() - started
    started = perf_counter()
    climatology = aligned_cycle(zarr_group["climatology"], bins, cache_column)
    timing_climatology = perf_counter() - started
    started = perf_counter()
    threshold = aligned_cycle(zarr_group["threshold"], bins, cache_column)
    timing_threshold = perf_counter() - started
    started = perf_counter()
    sst = np.asarray(
        source[run.variable].isel({time_dimension: source_slice, "point": source_column}).values,
        dtype=np.float64,
    )
    timing_sst = perf_counter() - started
    if time.size != event.duration_hours or sst.size != time.size:
        raise ValueError(f"event {event.event_id} window disagrees with persisted duration")
    return EventWindow(
        event=event,
        time=time,
        climatology=climatology,
        threshold=threshold,
        sst=sst,
        latitude=latitude,
        longitude=longitude,
        timings_seconds={
            "time_and_cycle_bins": timing_bins,
            "climatology_fetch_and_align": timing_climatology,
            "threshold_fetch_and_align": timing_threshold,
            "sst_fetch": timing_sst,
            "total_fetch_and_compute": timing_bins + timing_climatology + timing_threshold + timing_sst,
        },
    )
