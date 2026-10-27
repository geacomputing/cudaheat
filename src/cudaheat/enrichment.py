"""Vectorized, pre-persistence enrichment of sparse CUDAHeat event catalogues."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import xarray as xr

from .climatology import hourly_climatology_index
from .result import EventDetectionResult


class RuleValidationError(ValueError):
    """Raised when a YAML enrichment rule is not safe to execute."""


RULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "id",
        "version",
        "short_name",
        "long_name",
        "category",
        "description",
        "formula_latex",
        "symbols",
        "output",
        "python",
    ],
    "properties": {
        "id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
        "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
        "short_name": {"type": "string", "minLength": 1},
        "long_name": {"type": "string", "minLength": 1},
        "category": {
            "type": "object",
            "required": ["primary", "secondary"],
            "properties": {
                "primary": {"type": "string", "minLength": 1},
                "secondary": {"type": "string", "minLength": 1},
            },
        },
        "description": {"type": "string", "minLength": 1},
        "formula_latex": {"type": "string", "minLength": 1},
        "symbols": {"type": "object", "additionalProperties": {"type": "string"}},
        "output": {
            "type": "object",
            "required": ["name", "unit", "range", "dimensions"],
            "properties": {
                "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                "unit": {"type": "string"},
                "range": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "number"},
                },
                "dimensions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "enum": ["event"]},
                },
            },
        },
        "python": {"type": "string", "minLength": 1},
    },
}


@dataclass(frozen=True)
class EnrichmentRule:
    """Validated rule metadata and the Python function embedded in its YAML file."""

    source_path: Path
    metadata: Mapping[str, Any]
    calculate: Callable[["RuleContext"], Mapping[str, Any]]
    sha256: str

    @property
    def id(self) -> str:
        return str(self.metadata["id"])

    @property
    def version(self) -> str:
        return str(self.metadata["version"])

    @property
    def output_name(self) -> str:
        return str(self.metadata["output"]["name"])


@dataclass
class RuleContext:
    """One storage-local spatial batch shared by every selected rule.

    ``sst`` and the threshold cycle are read once per spatial batch.  Aligned
    dense fields are created lazily and then shared by every rule in that
    batch.  Event interval reductions use one prefix sum, not event loops.
    """

    variable: str
    timestamps: np.ndarray
    time_bins: np.ndarray
    _sst: np.ndarray
    _threshold_cycle: np.ndarray
    _climatology_loader: Callable[[], np.ndarray] | None
    _coordinate_loader: Callable[[str], np.ndarray]
    starts: np.ndarray
    stops: np.ndarray
    columns: np.ndarray
    duration_hours: np.ndarray
    event_ids: np.ndarray
    grid_point_indices: np.ndarray
    _values: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        shape = self._sst.shape
        if self._sst.ndim != 2 or self._threshold_cycle.ndim != 2:
            raise ValueError("rule context fields must have shape (time, point)")
        if self.time_bins.shape != (shape[0],):
            raise ValueError("time_bins must have one entry per source time step")
        if self._threshold_cycle.shape[1] != shape[1]:
            raise ValueError("threshold cycle is not aligned to the SST batch")
        event_count = self.starts.size
        for name in (
            "stops",
            "columns",
            "duration_hours",
            "event_ids",
            "grid_point_indices",
        ):
            if np.asarray(getattr(self, name)).shape != (event_count,):
                raise ValueError(f"{name} must have one value per event")
        if np.any(self.starts < 0) or np.any(self.stops > shape[0]) or np.any(self.stops < self.starts):
            raise ValueError("event intervals are outside the shared source batch")
        if np.any(self.columns < 0) or np.any(self.columns >= shape[1]):
            raise ValueError("event columns are outside the shared source batch")

    @property
    def shape(self) -> tuple[int, int]:
        return self._sst.shape

    @property
    def xp(self) -> Any:
        """Array backend used by this context (NumPy for standalone enrichment)."""
        return np

    @property
    def sst(self) -> np.ndarray:
        return self._sst

    @property
    def threshold(self) -> np.ndarray:
        return self._aligned_cycle("threshold", self._threshold_cycle)

    @property
    def climatology(self) -> np.ndarray:
        if self._climatology_loader is None:
            raise KeyError("the climatology cache has no climatology field")
        return self._aligned_cycle("climatology", self._climatology_loader())

    @property
    def latitude(self) -> np.ndarray:
        return self.coordinate("latitude")

    @property
    def longitude(self) -> np.ndarray:
        return self.coordinate("longitude")

    @property
    def utc_hour(self) -> np.ndarray:
        return np.mod(self.timestamps.astype("datetime64[h]").astype(np.int64), 24)

    @property
    def local_hour(self) -> np.ndarray:
        return np.mod(self.utc_hour[:, None] + self.longitude[None, :] / 15.0, 24.0)

    def netcdf(self, variable: str) -> np.ndarray:
        """Return the batch SST field under its persisted variable name."""
        if variable != self.variable:
            raise KeyError(f"rule requested unavailable optimized source variable {variable!r}")
        return self.sst

    def coordinate(self, name: str) -> np.ndarray:
        key = f"coordinate:{name}"
        if key not in self._values:
            values = np.asarray(self._coordinate_loader(name))
            if values.shape != (self.shape[1],):
                raise ValueError(f"coordinate {name!r} is not aligned to the SST batch")
            self._values[key] = values
        return self._values[key]

    def count(self, mask: np.ndarray) -> np.ndarray:
        """Count a boolean field inside every half-open event interval at once."""
        field_values = np.broadcast_to(np.asarray(mask, dtype=bool), self.shape)
        prefix = np.empty((field_values.shape[0] + 1, field_values.shape[1]), dtype=np.int32)
        prefix[0] = 0
        np.cumsum(field_values, axis=0, dtype=np.int32, out=prefix[1:])
        return prefix[self.stops, self.columns] - prefix[self.starts, self.columns]

    def sum(self, values: np.ndarray) -> np.ndarray:
        """Sum a numeric field inside every half-open event interval at once."""
        field_values = np.broadcast_to(np.asarray(values), self.shape)
        if field_values.dtype.kind not in "biuf":
            raise TypeError("rule sums require numeric or boolean values")
        prefix = np.empty((field_values.shape[0] + 1, field_values.shape[1]), dtype=field_values.dtype)
        prefix[0] = 0
        np.cumsum(field_values, axis=0, dtype=field_values.dtype, out=prefix[1:])
        return prefix[self.stops, self.columns] - prefix[self.starts, self.columns]

    def _aligned_cycle(self, name: str, cycle: np.ndarray) -> np.ndarray:
        if name in self._values:
            return self._values[name]
        values = np.full(self.shape, np.nan, dtype=cycle.dtype)
        valid = (self.time_bins >= 0) & (self.time_bins < cycle.shape[0])
        values[valid] = cycle[self.time_bins[valid]]
        self._values[name] = values
        return values


@dataclass
class DeviceRuleContext:
    """GPU-resident equivalent of :class:`RuleContext` for one spatial sub-batch.

    It deliberately exposes the same scientific fields and half-open interval
    reduction API, while leaving file access to the standalone context above.
    Rules can use ``context.xp`` for backend-explicit NumPy/CuPy operations.
    """

    xp: Any
    variable: str
    timestamps: np.ndarray
    _sst: Any
    _threshold: Any
    _bins: Any
    _utc_hours: Any
    _climatology_loader: Callable[[], Any] | None
    _coordinate_loader: Callable[[str], Any]
    starts: Any
    stops: Any
    columns: Any
    duration_hours: Any
    event_ids: Any
    grid_point_indices: Any
    _values: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._sst.shape)

    @property
    def sst(self) -> Any:
        return self._sst

    @property
    def time_bins(self) -> Any:
        return self._bins

    @property
    def threshold(self) -> Any:
        return self._threshold

    @property
    def climatology(self) -> Any:
        if self._climatology_loader is None:
            raise KeyError("the detection batch has no climatology field")
        if "climatology" not in self._values:
            cycle = self._climatology_loader()
            values = self.xp.full(self.shape, self.xp.nan, dtype=cycle.dtype)
            valid = (self._bins >= 0) & (self._bins < cycle.shape[0])
            values[valid] = cycle[self._bins[valid]]
            self._values["climatology"] = values
        return self._values["climatology"]

    @property
    def latitude(self) -> Any:
        return self.coordinate("latitude")

    @property
    def longitude(self) -> Any:
        return self.coordinate("longitude")

    @property
    def utc_hour(self) -> Any:
        return self._utc_hours

    @property
    def local_hour(self) -> Any:
        return self.xp.mod(self.utc_hour[:, None] + self.longitude[None, :] / 15.0, 24.0)

    def netcdf(self, variable: str) -> Any:
        if variable != self.variable:
            raise KeyError(f"rule requested unavailable optimized source variable {variable!r}")
        return self.sst

    def coordinate(self, name: str) -> Any:
        key = f"coordinate:{name}"
        if key not in self._values:
            values = self._coordinate_loader(name)
            if tuple(values.shape) != (self.shape[1],):
                raise ValueError(f"coordinate {name!r} is not aligned to the SST batch")
            self._values[key] = values
        return self._values[key]

    def count(self, mask: Any) -> Any:
        """GPU prefix-sum interval counts, with no loop over events."""
        field_values = self.xp.broadcast_to(self.xp.asarray(mask, dtype=bool), self.shape)
        prefix = self.xp.empty((self.shape[0] + 1, self.shape[1]), dtype=self.xp.int32)
        prefix[0] = 0
        self.xp.cumsum(field_values, axis=0, dtype=self.xp.int32, out=prefix[1:])
        return prefix[self.stops, self.columns] - prefix[self.starts, self.columns]

    def sum(self, values: Any) -> Any:
        """GPU prefix-sum interval reductions, with no loop over events."""
        field_values = self.xp.broadcast_to(self.xp.asarray(values), self.shape)
        if field_values.dtype.kind not in "biuf":
            raise TypeError("rule sums require numeric or boolean values")
        prefix = self.xp.empty(
            (self.shape[0] + 1, self.shape[1]),
            dtype=field_values.dtype,
        )
        prefix[0] = 0
        self.xp.cumsum(field_values, axis=0, dtype=field_values.dtype, out=prefix[1:])
        return prefix[self.stops, self.columns] - prefix[self.starts, self.columns]


@dataclass(frozen=True)
class DeviceEnrichmentOutput:
    """Sparse CPU outputs transferred after GPU enrichment of one detection batch."""

    values: dict[tuple[str, str], dict[str, np.ndarray]]
    gpu_seconds: float
    spatial_subbatches: int


def enrich_device_batch(
    rules: Sequence[EnrichmentRule],
    *,
    xp: Any,
    variable: str,
    timestamps: np.ndarray,
    sst: Any,
    threshold: Any,
    bins: Any,
    utc_hours: Any,
    events: Any,
    event_index_start: int,
    point_indices: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    enrichment_batch_size: int,
    climatology_cycle: Any | Callable[[], Any] | None = None,
) -> DeviceEnrichmentOutput:
    """Apply all rules to final GPU events without any source or Zarr reread."""
    if enrichment_batch_size < 1:
        raise ValueError("enrichment_batch_size must be positive")
    if not rules:
        return DeviceEnrichmentOutput({}, 0.0, 0)
    event_count = int(events.point.size)
    parts: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {
        (rule.id, rule.version): {} for rule in rules
    }
    started = xp.cuda.Event()
    finished = xp.cuda.Event()
    started.record()
    spatial_subbatches = 0
    for point_start in range(0, int(sst.shape[1]), enrichment_batch_size):
        point_stop = min(point_start + enrichment_batch_size, int(sst.shape[1]))
        event_start = int(xp.searchsorted(events.point, point_start, side="left"))
        event_stop = int(xp.searchsorted(events.point, point_stop, side="left"))
        if event_start == event_stop:
            continue
        spatial_subbatches += 1
        columns = events.point[event_start:event_stop] - point_start
        batch_point_indices = xp.asarray(point_indices[point_start:point_stop])

        def coordinate_loader(name: str, *, start: int = point_start, stop: int = point_stop) -> Any:
            try:
                return xp.asarray(coordinates[name][start:stop])
            except KeyError as exc:
                raise KeyError(f"no point coordinate {name!r} is available to enrichment rules") from exc

        def climatology_loader(
            *,
            value: Any | Callable[[], Any] | None = climatology_cycle,
            start: int = point_start,
            stop: int = point_stop,
        ) -> Any:
            if value is None:
                raise KeyError("the detection batch has no climatology field")
            cycle = value() if callable(value) else value
            return cycle[:, start:stop]

        context = DeviceRuleContext(
            xp=xp,
            variable=variable,
            timestamps=timestamps,
            _sst=sst[:, point_start:point_stop],
            _threshold=threshold[:, point_start:point_stop],
            _bins=bins,
            _utc_hours=utc_hours,
            _climatology_loader=None if climatology_cycle is None else climatology_loader,
            _coordinate_loader=coordinate_loader,
            starts=events.start[event_start:event_stop],
            stops=events.stop[event_start:event_stop],
            columns=columns,
            duration_hours=events.duration[event_start:event_stop],
            event_ids=xp.arange(
                event_index_start + event_start + 1,
                event_index_start + event_stop + 1,
                dtype=xp.int64,
            ),
            grid_point_indices=batch_point_indices[columns],
        )
        for rule in rules:
            values = _normalise_device_rule_values(rule, context, rule.calculate(context))
            destination = parts[(rule.id, rule.version)]
            for name, values_array in values.items():
                destination.setdefault(name, []).append(xp.asnumpy(values_array))
    finished.record()
    finished.synchronize()
    values: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for rule in rules:
        key = (rule.id, rule.version)
        values[key] = {
            name: np.concatenate(arrays) for name, arrays in parts[key].items()
        }
        if "value" not in values[key]:
            values[key]["value"] = np.empty(event_count, dtype=np.float64)
    return DeviceEnrichmentOutput(
        values=values,
        gpu_seconds=xp.cuda.get_elapsed_time(started, finished) / 1000,
        spatial_subbatches=spatial_subbatches,
    )


def discover_rules(rule_directory: str | Path | None = None) -> tuple[EnrichmentRule, ...]:
    """Discover, schema-validate, compile, and load every YAML rule."""
    directory = _rule_directory(rule_directory)
    paths = tuple(sorted((*directory.glob("*.yaml"), *directory.glob("*.yml"))))
    if not paths:
        raise RuleValidationError(f"no YAML enrichment rules found in {directory}")
    rules: list[EnrichmentRule] = []
    identifiers: set[tuple[str, str]] = set()
    outputs: set[str] = set()
    for path in paths:
        metadata = _load_rule_yaml(path)
        _validate_rule_schema(path, metadata)
        calculate = _compile_calculate(path, str(metadata["python"]))
        identifier = (str(metadata["id"]), str(metadata["version"]))
        output_name = str(metadata["output"]["name"])
        if identifier in identifiers:
            raise RuleValidationError(f"duplicate enrichment rule {identifier[0]}@{identifier[1]}")
        if output_name in outputs:
            raise RuleValidationError(f"duplicate enrichment output name {output_name!r}")
        identifiers.add(identifier)
        outputs.add(output_name)
        rules.append(
            EnrichmentRule(path, metadata, calculate, sha256(path.read_bytes()).hexdigest())
        )
    return tuple(rules)


def enrich(
    result: EventDetectionResult,
    *,
    rule_directory: str | Path | None = None,
) -> EventDetectionResult:
    """Evaluate YAML rules over a new sparse result before PostgreSQL persistence."""
    if not isinstance(result, EventDetectionResult):
        raise TypeError("result must be an EventDetectionResult")
    rules = discover_rules(rule_directory)
    data, provenance, timing = _enrich_catalogue(result.data, result.metadata, rules)
    metadata = dict(result.metadata)
    existing = metadata.get("enrichment_rules", ())
    if existing:
        raise ValueError("result already contains enrichment provenance")
    metadata["enrichment_rules"] = provenance
    metadata["enrichment_timing"] = timing
    return replace(result, data=data, metadata=metadata)


def _enrich_catalogue(
    catalogue: xr.Dataset,
    metadata: Mapping[str, Any],
    rules: Sequence[EnrichmentRule],
) -> tuple[xr.Dataset, list[dict[str, Any]], dict[str, Any]]:
    """Evaluate rules from a sparse catalogue without PostgreSQL input access."""
    event_count, event_values = _catalogue_values(catalogue)
    outputs: dict[tuple[str, str], dict[str, np.ndarray]] = {
        (rule.id, rule.version): {} for rule in rules
    }
    timing: dict[str, Any] = {
        "event_count": event_count,
        "spatial_batches": 0,
        "source_reads": 0,
        "threshold_reads": 0,
        "climatology_reads": 0,
    }
    started = perf_counter()
    if event_count:
        _evaluate_catalogue_batches(catalogue, metadata, rules, event_values, outputs, timing)
    else:
        for rule in rules:
            outputs[(rule.id, rule.version)]["value"] = np.empty(0, dtype=np.float64)
    timing["wall_seconds"] = perf_counter() - started
    data, provenance = attach_enrichment_outputs(catalogue, rules, outputs)
    return data, provenance, timing


def attach_enrichment_outputs(
    catalogue: xr.Dataset,
    rules: Sequence[EnrichmentRule],
    outputs: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    """Attach sparse rule outputs and provenance to an event catalogue."""
    data = catalogue.copy(deep=False)
    provenance: list[dict[str, Any]] = []
    for rule in rules:
        key = (rule.id, rule.version)
        variables: dict[str, str] = {}
        for result_name, values in outputs.get(key, {}).items():
            variable_name = rule.output_name if result_name == "value" else f"{rule.output_name}_{result_name}"
            if variable_name in data:
                raise ValueError(f"enrichment output conflicts with result field {variable_name!r}")
            data[variable_name] = ("event", values)
            variables[result_name] = variable_name
        if "value" not in variables:
            raise RuleValidationError(f"rule {rule.id}@{rule.version} did not return 'value'")
        provenance.append(
            {
                "id": rule.id,
                "version": rule.version,
                "sha256": rule.sha256,
                "source_path": str(rule.source_path),
                "metadata": {name: value for name, value in rule.metadata.items() if name != "python"},
                "result_variables": variables,
            }
        )
    return data, provenance


def _evaluate_catalogue_batches(
    catalogue: xr.Dataset,
    metadata: Mapping[str, Any],
    rules: Sequence[EnrichmentRule],
    event_values: Mapping[str, np.ndarray],
    outputs: dict[tuple[str, str], dict[str, np.ndarray]],
    timing: dict[str, Any],
) -> None:
    source_path = _required_metadata_path(metadata, "source_path")
    cache_path = _required_metadata_path(metadata, "climatology_cache_path")
    variable_name = _required_metadata_text(metadata, "variable")
    time_dimension = str(metadata.get("time_dimension", "time"))
    calendar_mode = str(metadata.get("calendar_mode", "month_day_hour"))
    source = xr.open_dataset(source_path, engine="h5netcdf", cache=False)
    try:
        if variable_name not in source:
            raise KeyError(f"optimized source has no SST variable {variable_name!r}")
        source_values = source[variable_name]
        if tuple(source_values.dims) != (time_dimension, "point"):
            raise ValueError("enrichment source must use CUDAHeat's (time, point) layout")
        if "point" not in source or time_dimension not in source:
            raise ValueError("enrichment source is missing its point or time coordinate")
        group = _open_cache(cache_path, metadata)
        cache_points = np.asarray(group["point"][:], dtype=np.int64)
        source_points = np.asarray(source["point"].values, dtype=np.int64)
        cache_columns = _point_columns(cache_points, event_values["point_index"], "climatology cache")
        chunk_width = int(group["threshold"].chunks[1])
        order = np.argsort(cache_columns // chunk_width, kind="stable")
        chunk_ids = cache_columns[order] // chunk_width
        boundaries = np.r_[0, np.flatnonzero(np.diff(chunk_ids)) + 1, order.size]
        timestamps = np.asarray(source[time_dimension].values, dtype="datetime64[ns]")
        time_bins = hourly_climatology_index(
            xr.DataArray(timestamps, dims=(time_dimension,)), calendar_mode=calendar_mode
        )
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            positions = order[left:right]
            chunk_id = int(chunk_ids[left])
            cache_start = chunk_id * chunk_width
            cache_stop = min(cache_start + chunk_width, int(group["threshold"].shape[1]))
            active_cache_columns = np.unique(cache_columns[positions])
            active_points = cache_points[active_cache_columns]
            source_columns = _point_columns(source_points, active_points, "optimized source")
            source_first = int(source_columns.min())
            source_stop = int(source_columns.max()) + 1
            sst = np.asarray(
                source_values.isel(
                    {time_dimension: slice(None), "point": slice(source_first, source_stop)}
                ).values
            )[:, source_columns - source_first]
            threshold_cycle = np.asarray(
                group["threshold"][:, cache_start:cache_stop]
            )[:, active_cache_columns - cache_start]
            timing["spatial_batches"] += 1
            timing["source_reads"] += 1
            timing["threshold_reads"] += 1

            def climatology_loader(
                *,
                group: Any = group,
                cache_start: int = cache_start,
                cache_stop: int = cache_stop,
                active_columns: np.ndarray = active_cache_columns,
            ) -> np.ndarray:
                timing["climatology_reads"] += 1
                return np.asarray(group["climatology"][:, cache_start:cache_stop])[  # type: ignore[index]
                    :, active_columns - cache_start
                ]

            def coordinate_loader(
                name: str,
                *,
                group: Any = group,
                source: xr.Dataset = source,
                active_columns: np.ndarray = active_cache_columns,
                source_columns: np.ndarray = source_columns,
            ) -> np.ndarray:
                if name in group:
                    return np.asarray(group[name][active_columns])
                if name in source and source[name].dims == ("point",):
                    return np.asarray(source[name].isel(point=source_columns).values)
                raise KeyError(f"no point coordinate {name!r} is available to enrichment rules")

            context = RuleContext(
                variable=variable_name,
                timestamps=timestamps,
                time_bins=time_bins,
                _sst=sst,
                _threshold_cycle=threshold_cycle,
                _climatology_loader=climatology_loader if "climatology" in group else None,
                _coordinate_loader=coordinate_loader,
                starts=event_values["start_index"][positions],
                stops=event_values["stop_index"][positions],
                columns=np.searchsorted(active_cache_columns, cache_columns[positions]),
                duration_hours=event_values["duration_hours"][positions],
                event_ids=event_values["event"][positions],
                grid_point_indices=event_values["point_index"][positions],
            )
            for rule in rules:
                values = _normalise_rule_values(rule, context, rule.calculate(context))
                destination = outputs[(rule.id, rule.version)]
                for name, array in values.items():
                    if name not in destination:
                        destination[name] = np.empty(event_values["event"].size, dtype=array.dtype)
                    destination[name][positions] = array
    finally:
        source.close()


def _catalogue_values(catalogue: xr.Dataset) -> tuple[int, dict[str, np.ndarray]]:
    required = ("point_index", "start_index", "stop_index", "duration_hours")
    missing = [name for name in required if name not in catalogue]
    if missing:
        raise ValueError(f"event catalogue is missing required field(s): {missing!r}")
    if "event" not in catalogue.coords:
        raise ValueError("event catalogue requires an event coordinate")
    event = np.asarray(catalogue["event"].values, dtype=np.int64)
    values = {
        "event": event,
        "point_index": np.asarray(catalogue["point_index"].values, dtype=np.int64),
        "start_index": np.asarray(catalogue["start_index"].values, dtype=np.intp),
        "stop_index": np.asarray(catalogue["stop_index"].values, dtype=np.intp),
        "duration_hours": np.asarray(catalogue["duration_hours"].values, dtype=np.intp),
    }
    sizes = {array.size for array in values.values()}
    if len(sizes) != 1:
        raise ValueError("event catalogue columns must have the same length")
    if np.any(values["stop_index"] - values["start_index"] != values["duration_hours"]):
        raise ValueError("event durations must use half-open [start_index, stop_index) intervals")
    return event.size, values


def _normalise_rule_values(
    rule: EnrichmentRule, context: RuleContext, values: Any
) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping) or "value" not in values:
        raise RuleValidationError(f"rule {rule.id}@{rule.version} must return a mapping containing 'value'")
    normalised: dict[str, np.ndarray] = {}
    for name, value in values.items():
        array = np.asarray(value)
        if array.ndim == 0:
            array = np.full(context.starts.size, array.item())
        if array.shape != (context.starts.size,):
            raise RuleValidationError(
                f"rule {rule.id}@{rule.version} output {name!r} has shape {array.shape}; "
                f"expected {(context.starts.size,)}"
            )
        if array.dtype.kind not in "biuf":
            raise RuleValidationError(
                f"rule {rule.id}@{rule.version} output {name!r} must be numeric or boolean"
            )
        normalised[str(name)] = array
    return normalised


def _normalise_device_rule_values(
    rule: EnrichmentRule, context: DeviceRuleContext, values: Any
) -> dict[str, Any]:
    if not isinstance(values, Mapping) or "value" not in values:
        raise RuleValidationError(f"rule {rule.id}@{rule.version} must return a mapping containing 'value'")
    normalised: dict[str, Any] = {}
    event_count = int(context.starts.size)
    for name, value in values.items():
        array = context.xp.asarray(value)
        if array.ndim == 0:
            array = context.xp.full(event_count, array.item())
        if tuple(array.shape) != (event_count,):
            raise RuleValidationError(
                f"rule {rule.id}@{rule.version} output {name!r} has shape {array.shape}; "
                f"expected {(event_count,)}"
            )
        if array.dtype.kind not in "biuf":
            raise RuleValidationError(
                f"rule {rule.id}@{rule.version} output {name!r} must be numeric or boolean"
            )
        normalised[str(name)] = array
    return normalised


def _rule_directory(rule_directory: str | Path | None) -> Path:
    directory = (
        Path(rule_directory)
        if rule_directory is not None
        else Path(__file__).resolve().parents[2] / "rules"
    )
    if not directory.is_dir():
        raise FileNotFoundError(f"enrichment rule directory not found: {directory}")
    return directory


def _load_rule_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - declared dependency
        raise ImportError("YAML enrichment rules require PyYAML") from exc
    try:
        metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuleValidationError(f"could not parse YAML enrichment rule {path}: {exc}") from exc
    if not isinstance(metadata, Mapping):
        raise RuleValidationError(f"YAML enrichment rule {path} must be a mapping")
    return metadata


def _validate_rule_schema(path: Path, metadata: Mapping[str, Any]) -> None:
    try:
        import jsonschema
    except ModuleNotFoundError as exc:  # pragma: no cover - declared dependency
        raise ImportError("YAML enrichment rules require jsonschema") from exc
    try:
        jsonschema.validate(metadata, RULE_SCHEMA)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path)
        suffix = f" at {location}" if location else ""
        raise RuleValidationError(f"invalid enrichment rule {path}{suffix}: {exc.message}") from exc


def _compile_calculate(path: Path, source: str) -> Callable[[RuleContext], Mapping[str, Any]]:
    try:
        compiled = compile(source, str(path), "exec")
    except SyntaxError as exc:
        raise RuleValidationError(f"invalid Python in enrichment rule {path}: {exc}") from exc
    namespace: dict[str, Any] = {"__file__": str(path), "__name__": f"cudaheat_rule_{sha256(str(path).encode()).hexdigest()}"}
    try:
        exec(compiled, namespace)
    except Exception as exc:
        raise RuleValidationError(f"could not load Python in enrichment rule {path}: {exc}") from exc
    calculate = namespace.get("calculate")
    if not callable(calculate):
        raise RuleValidationError(f"enrichment rule {path} must define calculate(context)")
    return calculate


def _open_cache(path: Path, metadata: Mapping[str, Any]) -> Any:
    try:
        import zarr
    except ModuleNotFoundError as exc:  # pragma: no cover - declared dependency
        raise ImportError("enrichment requires zarr") from exc
    group = zarr.open_group(str(path), mode="r")
    if not {"point", "threshold"}.issubset(group):
        raise KeyError("climatology cache must contain point and threshold arrays")
    expected_hash = metadata.get("climatology_hash")
    actual_hash = group.attrs.get("cudaheat_climatology_hash")
    if expected_hash and actual_hash != expected_hash:
        raise ValueError("climatology cache does not match the detection result")
    if group.attrs.get("cudaheat_complete") is False:
        raise ValueError("climatology cache is incomplete")
    return group


def _point_columns(point_values: np.ndarray, requested: np.ndarray, label: str) -> np.ndarray:
    order = np.argsort(point_values)
    locations = np.searchsorted(point_values[order], requested)
    found = locations < point_values.size
    found[found] &= point_values[order][locations[found]] == requested[found]
    if not np.all(found):
        missing = np.unique(requested[~found])[:10].tolist()
        raise LookupError(f"event points are absent from the {label}: {missing}")
    return order[locations]


def _required_metadata_path(metadata: Mapping[str, Any], name: str) -> Path:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"result metadata must contain a non-empty {name!r}")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"result {name} does not exist: {path}")
    return path


def _required_metadata_text(metadata: Mapping[str, Any], name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"result metadata must contain a non-empty {name!r}")
    return value


__all__ = [
    "DeviceEnrichmentOutput",
    "DeviceRuleContext",
    "EnrichmentRule",
    "RULE_SCHEMA",
    "RuleContext",
    "RuleValidationError",
    "attach_enrichment_outputs",
    "discover_rules",
    "enrich",
    "enrich_device_batch",
]
