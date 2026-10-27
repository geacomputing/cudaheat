"""Evaluate candidate enrichment rules for a persisted CUDAHeat run.

This laboratory is read-only: it never adds or changes PostgreSQL rows.
PostgreSQL supplies the chosen run's sparse event catalogue and paths to its
optimized SST and climatology cache. Candidate rules are evaluated only for
the selected events.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import xarray as xr

from cudaheat import discover_rules, load_postgres_config
from cudaheat.climatology import hourly_climatology_index
from cudaheat.enrichment import _enrich_catalogue


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSTGRES_CONFIG = PROJECT_ROOT / "config/postgres.ini"
DEFAULT_RULE_DIRECTORY = Path(__file__).with_name("rules")
DEFAULT_PLOT_DIRECTORY = Path(__file__).with_name("plots")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, type=UUID, help="persisted CUDAHeat run UUID")
    parser.add_argument("--postgres-config", type=Path, default=DEFAULT_POSTGRES_CONFIG)
    parser.add_argument(
        "--rule-directory",
        type=Path,
        default=DEFAULT_RULE_DIRECTORY,
        help="directory containing candidate YAML rules",
    )
    parser.add_argument(
        "--event-id",
        action="append",
        type=int,
        help="CUDAHeat event index to evaluate; repeat this option as needed",
    )
    parser.add_argument(
        "--point-index",
        action="append",
        type=int,
        help="grid point index to evaluate; repeat this option as needed",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=5,
        help="maximum selected events (default: 5; use 0 for all)",
    )
    parser.add_argument(
        "--plot-directory",
        type=Path,
        default=DEFAULT_PLOT_DIRECTORY,
        help="base directory for event plots (default: examples/rule_laboratory/plots)",
    )
    parser.add_argument(
        "--plot-padding-hours",
        type=int,
        default=24,
        help="hours of context before and after each event plot (default: 24)",
    )
    parser.add_argument("--output",
                        type=Path,
                        help="optional JSON result path")
    arguments = parser.parse_args()
    if arguments.max_events < 0:
        parser.error("--max-events must be non-negative")
    if arguments.plot_padding_hours < 0:
        parser.error("--plot-padding-hours must be non-negative")
    if not arguments.rule_directory.is_dir():
        parser.error(f"rule directory does not exist: {arguments.rule_directory}")
    return arguments


def load_stored_catalogue(
    run_id: UUID,
    *,
    postgres_config: Path,
    event_ids: Sequence[int] | None,
    point_indices: Sequence[int] | None,
    max_events: int,
) -> tuple[dict[str, Any], xr.Dataset]:
    """Load only the requested sparse events and the run's stored metadata."""
    try:
        import psycopg
    except ModuleNotFoundError as exc:
        raise ImportError("install CUDAHeat with the postgres extra: pip install -e '.[postgres]'") from exc

    config = load_postgres_config(postgres_config)
    with psycopg.connect(**config.connection_kwargs()) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT metadata FROM runs WHERE run_id = %s", (run_id,))
            row = cursor.fetchone()
            if row is None:
                raise LookupError(f"run does not exist: {run_id}")
            if not isinstance(row[0], Mapping):
                raise ValueError("persisted run metadata is malformed")
            metadata = dict(row[0])

            clauses = ["e.run_id = %s"]
            values: list[Any] = [run_id]
            if event_ids:
                clauses.append("e.cudaheat_event_index = ANY(%s)")
                values.append(list(event_ids))
            if point_indices:
                clauses.append("p.grid_point_index = ANY(%s)")
                values.append(list(point_indices))
            limit = "" if max_events == 0 else " LIMIT %s"
            if max_events:
                values.append(max_events)
            cursor.execute(
                f"""
                SELECT e.cudaheat_event_index, p.grid_point_index, e.start_index, e.stop_index,
                       e.duration_hours, e.start_time, e.end_time
                FROM events AS e
                JOIN points AS p ON p.point_id = e.point_id
                WHERE {" AND ".join(clauses)}
                ORDER BY e.cudaheat_event_index{limit}
                """,
                values,
            )
            rows = cursor.fetchall()
    if not rows:
        raise LookupError("no events matched the requested run and filters")
    return metadata, _catalogue_from_rows(rows)


def _catalogue_from_rows(rows: Sequence[Sequence[Any]]) -> xr.Dataset:
    columns = list(zip(*rows, strict=True))
    return xr.Dataset(
        {
            "point_index": ("event", np.asarray(columns[1], dtype=np.int64)),
            "start_index": ("event", np.asarray(columns[2], dtype=np.intp)),
            "stop_index": ("event", np.asarray(columns[3], dtype=np.intp)),
            "duration_hours": ("event", np.asarray(columns[4], dtype=np.intp)),
            "start_time": ("event", np.asarray(columns[5], dtype="datetime64[ns]")),
            "end_time": ("event", np.asarray(columns[6], dtype="datetime64[ns]")),
        },
        coords={"event": np.asarray(columns[0], dtype=np.int64)},
    )


def result_payload(
    run_id: UUID,
    catalogue: xr.Dataset,
    enriched: xr.Dataset,
    *,
    rules: Sequence[Any],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    output_names = sorted(set(enriched.data_vars) - set(catalogue.data_vars))
    events = [
        {
            "event_id": int(event_id),
            "point_index": int(catalogue["point_index"].values[position]),
            "start_time": str(catalogue["start_time"].values[position]),
            "end_time": str(catalogue["end_time"].values[position]),
            "duration_hours": int(catalogue["duration_hours"].values[position]),
            **{
                name: _json_value(enriched[name].values[position])
                for name in output_names
            },
        }
        for position, event_id in enumerate(catalogue["event"].values)
    ]
    return {
        "run_id": str(run_id),
        "event_count": len(events),
        "rules": [{"id": rule.id, "version": rule.version, "file": str(rule.source_path)} for rule in rules],
        "timing": dict(timing),
        "events": events,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_event_plots(
    run_id: UUID,
    catalogue: xr.Dataset,
    enriched: xr.Dataset,
    metadata: Mapping[str, Any],
    *,
    rules: Sequence[Any],
    directory: Path,
    padding_hours: int,
) -> list[Path]:
    """Plot event fields with a text panel for run and rule details."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import zarr
    except ModuleNotFoundError as exc:
        raise ImportError(
            "event plots require matplotlib; install CUDAHeat with: pip install -e '.[plot]'"
        ) from exc

    source_path = _metadata_path(metadata, "source_path")
    cache_path = _metadata_path(metadata, "climatology_cache_path")
    variable = _metadata_text(metadata, "variable")
    time_dimension = str(metadata.get("time_dimension", "time"))
    calendar_mode = str(metadata.get("calendar_mode", "month_day_hour"))
    directory = directory / str(run_id)
    directory.mkdir(parents=True, exist_ok=True)

    source = xr.open_dataset(source_path, engine="h5netcdf", cache=False)
    try:
        if variable not in source:
            raise KeyError(f"optimized source has no SST variable {variable!r}")
        if "point" not in source or time_dimension not in source:
            raise ValueError("optimized source is missing its point or time coordinate")
        source_values = source[variable]
        if tuple(source_values.dims) != (time_dimension, "point"):
            raise ValueError("optimized source must use CUDAHeat's (time, point) layout")

        cache = zarr.open_group(str(cache_path), mode="r")
        if "point" not in cache or "threshold" not in cache or "climatology" not in cache:
            raise ValueError("climatology cache is missing point, threshold, or climatology fields")
        timestamps = np.asarray(source[time_dimension].values, dtype="datetime64[ns]")
        time_bins = hourly_climatology_index(
            xr.DataArray(timestamps, dims=(time_dimension,)),
            calendar_mode=calendar_mode,
        )
        event_points = np.asarray(catalogue["point_index"].values, dtype=np.int64)
        source_columns = _point_columns(
            np.asarray(source["point"].values, dtype=np.int64),
            event_points,
            "optimized source",
        )
        cache_columns = _point_columns(
            np.asarray(cache["point"][:], dtype=np.int64),
            event_points,
            "climatology cache",
        )
        paths: list[Path] = []
        for position, event_id in enumerate(catalogue["event"].values):
            event_start = int(catalogue["start_index"].values[position])
            event_stop = int(catalogue["stop_index"].values[position])
            left = max(0, event_start - padding_hours)
            right = min(timestamps.size, event_stop + padding_hours)
            bins = time_bins[left:right]
            valid_bins = (bins >= 0) & (bins < cache["threshold"].shape[0])
            threshold = np.full(right - left, np.nan, dtype=np.float64)
            climatology = np.full(right - left, np.nan, dtype=np.float64)
            threshold_cycle = np.asarray(cache["threshold"][:, cache_columns[position]])
            climatology_cycle = np.asarray(cache["climatology"][:, cache_columns[position]])
            threshold[valid_bins] = threshold_cycle[bins[valid_bins]]
            climatology[valid_bins] = climatology_cycle[bins[valid_bins]]
            sst = np.asarray(
                source_values.isel(
                    {time_dimension: slice(left, right), "point": int(source_columns[position])}
                ).values
            )
            path = directory / f"event-{int(event_id):06d}_point-{event_points[position]}.png"
            figure, (axis, panel) = plt.subplots(
                1,
                2,
                figsize=(16, 5),
                constrained_layout=True,
                gridspec_kw={"width_ratios": (3, 1)},
            )
            plot_times = timestamps[left:right]
            axis.plot(plot_times, sst, label="SST", color="black", linewidth=1.5)
            axis.plot(plot_times, threshold, label="Threshold", color="tab:red", linewidth=1.25)
            axis.plot(plot_times, climatology, label="Climatology", color="tab:blue", linewidth=1.25)
            event_times = timestamps[event_start:event_stop]
            axis.axvspan(event_times[0], event_times[-1], color="tab:orange", alpha=0.15, label="Event")
            axis.set(
                title=(
                    f"CUDAHeat event {int(event_id)} | point {event_points[position]} | "
                    f"{event_stop - event_start} h ({(event_stop - event_start)/24} decimal days)"
                ),
                xlabel="Time",
                ylabel="Temperature (°C)",
            )
            axis.grid(alpha=0.25)
            axis.legend()
            panel.axis("off")
            panel.text(
                0,
                1,
                _event_panel_text(run_id, position, metadata, enriched, rules),
                transform=panel.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                linespacing=1.5,
            )
            figure.savefig(path, dpi=160)
            plt.close(figure)
            paths.append(path)
    finally:
        source.close()
    return paths


def _event_panel_text(
    run_id: UUID,
    position: int,
    metadata: Mapping[str, Any],
    enriched: xr.Dataset,
    rules: Sequence[Any],
) -> str:
    """Format the legible text-only panel displayed beside every event."""
    source_name = Path(str(metadata.get("source_path", "unknown"))).name
    climatology_hash = str(metadata.get("climatology_hash", "unknown"))
    if len(climatology_hash) > 12:
        climatology_hash = f"{climatology_hash[:12]}…"
    lines = [
        "RUN CONFIGURATION",
        f"Input: {source_name}",
        f"Variable: {metadata.get('variable', 'unknown')}",
        f"Baseline: {_format_baseline(metadata.get('baseline'))}",
        f"Percentile: {_format_percentile(metadata.get('percentile'))}",
        f"Min duration: {_format_hours(metadata.get('min_duration_hours'))}",
        f"Internal gap: {_format_hours(metadata.get('max_internal_gap_hours'))}",
        f"Join gap: {_format_hours(metadata.get('join_gap_hours'))}",
        "",
        "RULE RESULTS",
    ]
    for rule in rules:
        if rule.output_name not in enriched:
            continue
        lines.append(f"{rule.id}: {_format_value(enriched[rule.output_name].values[position])}")
    return "\n".join(lines)


def _format_baseline(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{value[0]}–{value[1]}"
    return "unknown"


def _format_percentile(value: Any) -> str:
    try:
        return f"{float(value) * 100:g}%"
    except (TypeError, ValueError):
        return "unknown"


def _format_hours(value: Any) -> str:
    try:
        return f"{int(value)} h"
    except (TypeError, ValueError):
        return "unknown"


def _format_value(value: Any) -> str:
    value = _json_value(value)
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def _metadata_path(metadata: Mapping[str, Any], name: str) -> Path:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"persisted run metadata has no {name!r}")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"persisted {name} does not exist: {path}")
    return path


def _metadata_text(metadata: Mapping[str, Any], name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"persisted run metadata has no {name!r}")
    return value


def _point_columns(points: np.ndarray, requested: np.ndarray, label: str) -> np.ndarray:
    order = np.argsort(points)
    locations = np.searchsorted(points[order], requested)
    found = locations < points.size
    found[found] &= points[order][locations[found]] == requested[found]
    if not np.all(found):
        missing = np.unique(requested[~found])[:10].tolist()
        raise LookupError(f"event points are absent from the {label}: {missing}")
    return order[locations]


def main() -> None:
    arguments = parse_arguments()
    metadata, catalogue = load_stored_catalogue(
        arguments.run_id,
        postgres_config=arguments.postgres_config,
        event_ids=arguments.event_id,
        point_indices=arguments.point_index,
        max_events=arguments.max_events,
    )
    rules = discover_rules(arguments.rule_directory)
    enriched, _, timing = _enrich_catalogue(catalogue, metadata, rules)
    payload = result_payload(arguments.run_id, catalogue, enriched, rules=rules, timing=timing)
    plot_paths = write_event_plots(
        arguments.run_id,
        catalogue,
        enriched,
        metadata,
        rules=rules,
        directory=arguments.plot_directory,
        padding_hours=arguments.plot_padding_hours,
    )
    payload["plots"] = [str(path) for path in plot_paths]
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    print(encoded)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(f"{encoded}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
