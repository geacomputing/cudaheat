"""The small numerical interface made available to every Python enrichment rule."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import xarray as xr

from .access import source_columns
from .models import Event, RunReference


@dataclass
class RuleContext:
    """One spatial chunk plus vectorized sparse-event helpers.

    Rules use regular NumPy against the lazy properties below.  Each property
    is loaded only if the rule actually accesses it, preserving the former
    threshold-only I/O path while avoiding per-rule input declarations.
    """

    events: Sequence[Event]
    cache_columns: np.ndarray
    active_cache_columns: np.ndarray
    cache_start: int
    cache_stop: int
    source: xr.Dataset
    zarr_group: Any
    run: RunReference
    source_times: np.ndarray
    time_bins: np.ndarray
    timings: dict[str, float]
    starts: np.ndarray = field(init=False)
    stops: np.ndarray = field(init=False)
    columns: np.ndarray = field(init=False)
    duration_hours: np.ndarray = field(init=False)
    event_ids: np.ndarray = field(init=False)
    grid_point_indices: np.ndarray = field(init=False)
    start_times: np.ndarray = field(init=False)
    end_times: np.ndarray = field(init=False)
    _source_indices: np.ndarray | None = field(default=None, init=False, repr=False)
    _values: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.starts = np.asarray([event.start_index for event in self.events], dtype=np.intp)
        self.stops = np.asarray([event.stop_index for event in self.events], dtype=np.intp)
        self.columns = np.asarray(
            np.searchsorted(self.active_cache_columns, self.cache_columns), dtype=np.intp
        )
        self.duration_hours = np.asarray(
            [event.duration_hours for event in self.events], dtype=np.intp
        )
        self.event_ids = np.asarray([event.event_id for event in self.events], dtype=np.int64)
        self.grid_point_indices = np.asarray(
            [event.grid_point_index for event in self.events], dtype=np.int64
        )
        self.start_times = np.asarray([event.start_time for event in self.events])
        self.end_times = np.asarray([event.end_time for event in self.events])

    @property
    def shape(self) -> tuple[int, int]:
        return self.source_times.size, self.active_cache_columns.size

    @property
    def timestamps(self) -> np.ndarray:
        return self.source_times

    @property
    def utc_hour(self) -> np.ndarray:
        return np.mod(self.source_times.astype("datetime64[h]").astype(np.int64), 24)

    @property
    def sst(self) -> np.ndarray:
        return self.netcdf(self.run.variable)

    @property
    def threshold(self) -> np.ndarray:
        return self.zarr_cycle("threshold")

    @property
    def climatology(self) -> np.ndarray:
        return self.zarr_cycle("climatology")

    @property
    def longitude(self) -> np.ndarray:
        return self.coordinate("longitude")

    @property
    def latitude(self) -> np.ndarray:
        return self.coordinate("latitude")

    @property
    def local_hour(self) -> np.ndarray:
        return np.mod(self.utc_hour[:, None] + self.longitude[None, :] / 15.0, 24.0)

    def count(self, mask: np.ndarray) -> np.ndarray:
        """Count truthy samples inside every half-open event interval at once."""
        field = np.broadcast_to(np.asarray(mask, dtype=bool), self.shape)
        prefix = np.empty((field.shape[0] + 1, field.shape[1]), dtype=np.int32)
        prefix[0] = 0
        np.cumsum(field, axis=0, dtype=np.int32, out=prefix[1:])
        return prefix[self.stops, self.columns] - prefix[self.starts, self.columns]

    def _source_point_indices(self) -> np.ndarray:
        if self._source_indices is None:
            points = np.asarray(self.zarr_group["point"][self.active_cache_columns], dtype=np.int64)
            self._source_indices = source_columns(self.source, points)
        return self._source_indices

    def netcdf(self, variable: str) -> np.ndarray:
        """Return any optimized source variable with ``(time, point)`` dimensions."""
        key = f"netcdf:{variable}"
        if key in self._values:
            return self._values[key]
        if variable not in self.source:
            raise KeyError(f"optimized source does not contain rule-requested variable {variable!r}")
        time_dimension = str(self.run.metadata.get("time_dimension", "time"))
        data = self.source[variable]
        if time_dimension not in data.dims or "point" not in data.dims:
            raise ValueError(f"source variable {variable!r} must use the (time, point) layout")
        started = perf_counter()
        indices = self._source_point_indices()
        first, last = int(indices.min()), int(indices.max()) + 1
        block = np.asarray(
            data.isel({time_dimension: slice(None), "point": slice(first, last)}).values,
            dtype=np.float32,
        )[:, indices - first]
        self.timings["source_read"] += perf_counter() - started
        self._values[key] = block
        return block

    def zarr_cycle(self, array_name: str) -> np.ndarray:
        """Return any cycle-by-point Zarr array aligned to source timestamps."""
        key = f"zarr:{array_name}"
        if key in self._values:
            return self._values[key]
        if array_name not in self.zarr_group:
            raise KeyError(f"climatology cache does not contain {array_name!r}")
        started = perf_counter()
        cycle = np.asarray(
            self.zarr_group[array_name][:, self.cache_start : self.cache_stop], dtype=np.float32
        )
        local_columns = self.active_cache_columns - self.cache_start
        values = cycle[self.time_bins, :][:, local_columns]
        self.timings["zarr_read"] += perf_counter() - started
        self._values[key] = values
        return values

    def coordinate(self, name: str) -> np.ndarray:
        """Return any one-dimensional point coordinate for the active spatial chunk."""
        key = f"coordinate:{name}"
        if key in self._values:
            return self._values[key]
        if name not in self.source or self.source[name].dims != ("point",):
            raise KeyError(f"optimized source has no point coordinate {name!r}")
        started = perf_counter()
        values = np.asarray(
            self.source[name].isel(point=self._source_point_indices()).values, dtype=np.float64
        )
        self.timings["source_read"] += perf_counter() - started
        self._values[key] = values
        return values
