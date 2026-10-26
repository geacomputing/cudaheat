"""Ordered PostgreSQL batching and bulk persistence for Python enrichment rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import xarray as xr

from cudaheat.climatology import hourly_climatology_index

from .context import RuleContext
from .models import Event, RunReference
from .rules import EnrichmentRule, RuleValidationError

_EVENT_RESULT_COLUMNS = frozenset(
    {"event_id", "grid_point_index", "start_index", "stop_index", "start_time", "end_time", "duration_hours"}
)
_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS event_enrichments (
        event_enrichment_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        event_id BIGINT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
        rule_id TEXT NOT NULL,
        rule_version TEXT NOT NULL,
        rule_sha256 CHAR(64) NOT NULL,
        result JSONB NOT NULL,
        applied_at TIMESTAMPTZ NOT NULL,
        UNIQUE (event_id, rule_id, rule_version)
    )
    """,
    "CREATE INDEX IF NOT EXISTS event_enrichments_rule_idx ON event_enrichments (rule_id, rule_version)",
)


def load_run(connection: Any, run_id: str) -> RunReference:
    """Load immutable run identity and cache provenance without mutating PostgreSQL."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT run_id, source_path, source_sha256, climatology_hash, variable, metadata FROM runs WHERE run_id = %s",
            (run_id,),
        )
        row = cursor.fetchone()
    if row is None:
        raise LookupError(f"run does not exist: {run_id}")
    persisted_id, source_path, source_sha256, climatology_hash, variable, metadata = row
    if not climatology_hash or not isinstance(metadata, Mapping):
        raise ValueError("run is missing climatology provenance or metadata")
    return RunReference(
        run_id=str(persisted_id),
        source_path=Path(str(source_path)),
        source_sha256=str(source_sha256),
        climatology_hash=str(climatology_hash),
        variable=str(variable),
        metadata=metadata,
        calendar_mode=str(metadata.get("calendar_mode", "month_day_hour")),
    )


def load_events(connection: Any, run_id: str, limit: int) -> list[Event]:
    """Load a small sample for optional benchmark plots."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT e.event_id, p.grid_point_index, e.start_time, e.end_time,
                   e.start_index, e.stop_index, e.duration_hours
            FROM events AS e JOIN points AS p ON p.point_id = e.point_id
            WHERE e.run_id = %s ORDER BY e.event_id LIMIT %s
            """,
            (run_id, limit),
        )
        rows = cursor.fetchall()
    return [Event(int(row[0]), int(row[1]), row[2], row[3], int(row[4]), int(row[5]), int(row[6])) for row in rows]


def apply_rules(
    connection: Any,
    rules: Sequence[EnrichmentRule],
    run: RunReference,
    source: xr.Dataset,
    zarr_group: Any,
    *,
    batch_size: int,
    write_results: bool,
) -> list[dict[str, Any]]:
    """Run selector-compatible rules together in ordered spatial batches."""
    if batch_size < 1:
        raise ValueError("--rule-batch-size must be positive")
    if write_results:
        _ensure_schema(connection)
        from psycopg.types.json import Jsonb
        for rule in rules:
            _assert_immutable_version(connection, rule)

    time_dimension = str(run.metadata.get("time_dimension", "time"))
    source_times = np.asarray(source[time_dimension].values, dtype="datetime64[ns]")
    time_bins = hourly_climatology_index(
        xr.DataArray(source_times, dims=("event_time",)), calendar_mode=run.calendar_mode
    )
    summaries: list[dict[str, Any]] = []
    for selector_rules in _selector_groups(rules):
        group_started = perf_counter()
        selector_rule = selector_rules[0]
        last_point, last_event = -1, 0
        evaluated = batches = chunks = 0
        shared_timings = {
            name: 0.0 for name in ("postgres_read", "point_mapping", "source_read", "zarr_read")
        }
        rule_timings = {
            _rule_label(rule): {name: 0.0 for name in ("numpy_compute", "postgres_write")}
            for rule in selector_rules
        }
        while True:
            started = perf_counter()
            events = _events_for_rule(
                connection, selector_rule, run.run_id, last_point, last_event, batch_size
            )
            shared_timings["postgres_read"] += perf_counter() - started
            if not events:
                break
            started = perf_counter()
            points = np.asarray([event.grid_point_index for event in events], dtype=np.int64)
            cache_columns = _cache_columns(zarr_group, points)
            shared_timings["point_mapping"] += perf_counter() - started
            point_chunk = int(zarr_group["threshold"].chunks[1])
            for chunk_id in np.unique(cache_columns // point_chunk):
                positions = np.flatnonzero(cache_columns // point_chunk == chunk_id)
                cache_start = int(chunk_id * point_chunk)
                cache_stop = min(cache_start + point_chunk, int(zarr_group["threshold"].shape[1]))
                active = np.unique(cache_columns[positions])
                context = RuleContext(
                    events=[events[int(position)] for position in positions],
                    cache_columns=cache_columns[positions],
                    active_cache_columns=active,
                    cache_start=cache_start,
                    cache_stop=cache_stop,
                    source=source,
                    zarr_group=zarr_group,
                    run=run,
                    source_times=source_times,
                    time_bins=time_bins,
                    timings=shared_timings,
                )
                for rule in selector_rules:
                    reads_before = shared_timings["source_read"] + shared_timings["zarr_read"]
                    started = perf_counter()
                    results = _normalise_results(rule, context, rule.calculate(context))
                    elapsed = perf_counter() - started
                    reads_during_calculation = (
                        shared_timings["source_read"] + shared_timings["zarr_read"] - reads_before
                    )
                    rule_timings[_rule_label(rule)]["numpy_compute"] += max(
                        0.0, elapsed - reads_during_calculation
                    )
                    if write_results:
                        applied_at = datetime.now(timezone.utc)
                        rows = [
                            {
                                "event_id": event.event_id,
                                "rule_id": rule.id,
                                "rule_version": rule.version,
                                "rule_sha256": rule.sha256,
                                "result": Jsonb(results[index]),
                                "applied_at": applied_at,
                            }
                            for index, event in enumerate(context.events)
                        ]
                        started = perf_counter()
                        with connection.cursor() as cursor:
                            cursor.executemany(str(rule.metadata["postgres"]["write_sql"]), rows)
                        rule_timings[_rule_label(rule)]["postgres_write"] += perf_counter() - started
                chunks += 1
            if write_results:
                connection.commit()
            evaluated += len(events)
            batches += 1
            last_point, last_event = events[-1].grid_point_index, events[-1].event_id
            rendered_rules = ", ".join(f"{rule.id}" for rule in selector_rules)
            print(
                f"[selector group: {rendered_rules}]: "
                f"{evaluated:,} events in {batches} shared spatial batch(es)"
            )
        summaries.append(
            {
                "rule_ids": [rule.id for rule in selector_rules],
                "events_evaluated": evaluated,
                "spatial_batches": batches,
                "zarr_point_chunks_read": chunks,
                "selector_group_wall_seconds": perf_counter() - group_started,
                "shared_timing_seconds": shared_timings,
                "rule_timing_seconds": rule_timings,
                "write_results": write_results,
            }
        )
    return summaries


def _selector_groups(rules: Sequence[EnrichmentRule]) -> tuple[tuple[EnrichmentRule, ...], ...]:
    """Group only rules whose selector text is exactly identical and therefore safe to share."""
    groups: dict[str, list[EnrichmentRule]] = {}
    for rule in rules:
        selector = str(rule.metadata["postgres"]["select_sql"])
        groups.setdefault(selector, []).append(rule)
    return tuple(tuple(group) for group in groups.values())


def _rule_label(rule: EnrichmentRule) -> str:
    """Keep timing entries distinct when historical rule versions are co-loaded."""
    return f"{rule.id}@{rule.version}"


def _normalise_results(
    rule: EnrichmentRule, context: RuleContext, values: Any
) -> list[dict[str, Any]]:
    if not isinstance(values, Mapping) or rule.metadata["output"]["primary"] not in values:
        raise RuleValidationError(f"rule {rule.id}@{rule.version} must return its declared primary output")
    columns: dict[str, np.ndarray] = {}
    for name, value in values.items():
        array = np.asarray(value)
        if array.ndim == 0:
            array = np.full(len(context.events), array.item())
        if array.shape != (len(context.events),):
            raise RuleValidationError(
                f"rule {rule.id}@{rule.version} output {name!r} has shape {array.shape}; expected one value per event"
            )
        columns[str(name)] = array
    return [{name: _json_value(column[index]) for name, column in columns.items()} for index in range(len(context.events))]


def _json_value(value: Any) -> Any:
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, (float, np.floating)):
        return float(scalar) if np.isfinite(scalar) else None
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    if scalar is None:
        return None
    raise TypeError(f"rule output is not JSON-compatible: {type(scalar).__name__}")


def _events_for_rule(
    connection: Any, rule: EnrichmentRule, run_id: str, after_point: int, after_event: int, batch_size: int
) -> list[Event]:
    from psycopg.rows import dict_row

    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            str(rule.metadata["postgres"]["select_sql"]),
            {"run_id": run_id, "after_grid_point_index": after_point, "after_event_id": after_event, "batch_size": batch_size},
        )
        rows = cursor.fetchall()
    events = []
    for row in rows:
        missing = _EVENT_RESULT_COLUMNS - set(row)
        if missing:
            raise RuleValidationError(f"rule {rule.id}@{rule.version} selector omits {sorted(missing)}")
        events.append(Event(**{name: row[name] for name in _EVENT_RESULT_COLUMNS}))
    keys = [(event.grid_point_index, event.event_id) for event in events]
    if keys != sorted(keys) or (keys and keys[0] <= (after_point, after_event)):
        raise RuleValidationError(f"rule {rule.id}@{rule.version} selector must be strictly spatially ordered")
    return events


def _cache_columns(group: Any, points: np.ndarray) -> np.ndarray:
    cache_points = np.asarray(group["point"][:], dtype=np.int64)
    order = np.argsort(cache_points)
    locations = np.searchsorted(cache_points[order], points)
    found = locations < cache_points.size
    found[found] &= cache_points[order][locations[found]] == points[found]
    if not np.all(found):
        raise LookupError("event points are absent from the Zarr cache")
    return order[locations]


def _ensure_schema(connection: Any) -> None:
    with connection.cursor() as cursor:
        for statement in _SCHEMA_STATEMENTS:
            cursor.execute(statement)


def _assert_immutable_version(connection: Any, rule: EnrichmentRule) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT DISTINCT rule_sha256 FROM event_enrichments WHERE rule_id = %s AND rule_version = %s",
            (rule.id, rule.version),
        )
        hashes = {str(row[0]).strip() for row in cursor}
    if hashes and hashes != {rule.sha256}:
        raise RuleValidationError(
            f"rule {rule.id}@{rule.version} differs from its persisted file; increase the version"
        )
