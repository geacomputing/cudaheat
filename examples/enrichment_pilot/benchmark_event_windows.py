#!/usr/bin/env python3
"""Benchmark event-window access and execute discovered Python enrichment rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from cudaheat.enrichment.access import (
    cache_columns,
    fetch_event_window,
    open_and_validate_cache,
    open_and_validate_sst,
    resolve_cache_path,
    resolve_sst_path,
    source_columns,
)
from cudaheat.enrichment.execution import apply_rules, load_events, load_run
from cudaheat.enrichment.models import EventWindow, RunReference
from cudaheat.enrichment.rules import discover_rules
from cudaheat.postgres import load_postgres_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config" / "postgres.ini"
DEFAULT_PLOT_DIRECTORY = Path(__file__).resolve().parent / "plots"
RULE_DIRECTORY = Path(__file__).resolve().parent / "enrichment_rules"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="UUID from the runs table")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--zarr-path", type=Path, help="Override the recorded Zarr cache path")
    parser.add_argument("--sst-path", type=Path, help="Override with a CUDAHeat optimized NetCDF")
    parser.add_argument("--limit", type=int, default=5, help="Optional plotting sample size")
    parser.add_argument("--plot-directory", type=Path, default=DEFAULT_PLOT_DIRECTORY)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-json", action=argparse.BooleanOptionalAction, default=False)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--apply-rules", action="store_true", help="Persist rule results")
    execution.add_argument("--dry-run-rules", action="store_true", help="Evaluate rules without writes")
    parser.add_argument("--rule-batch-size", type=int, default=50_000)
    return parser.parse_args()


def run_parameters_title(run: RunReference) -> str:
    metadata = run.metadata
    baseline = metadata.get("baseline")
    baseline_text = (
        f"baseline {baseline[0]}–{baseline[1]}"
        if isinstance(baseline, (list, tuple)) and len(baseline) == 2
        else "baseline all available years"
    )
    percentile = metadata.get("percentile")
    percentile_text = f"p{100 * float(percentile):g}" if percentile is not None else "p?"
    return (
        f"{baseline_text} · {percentile_text}\n"
        f"event rules: duration ≥{metadata.get('min_duration_hours')}h · "
        f"internal gap ≤{metadata.get('max_internal_gap_hours')}h · "
        f"join gap ≤{metadata.get('join_gap_hours')}h"
    )


def write_plot(window: EventWindow, run: RunReference, directory: Path) -> Path:
    """Render an untimed Celsius plot of SST, threshold, and climatology."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ImportError("install plotting support with: pip install -e '.[plot]'") from exc
    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 4.8), layout="constrained")
    axis.plot(window.time, window.sst - 273.15, color="#1f77b4", label="SST", zorder=3)
    axis.plot(window.time, window.threshold - 273.15, color="#d62728", label="Threshold", zorder=2)
    axis.plot(
        window.time,
        window.climatology - 273.15,
        color="#2ca02c",
        alpha=0.4,
        label="Climatology",
        zorder=1,
    )
    location = (
        f"lat {window.latitude:.2f}°, lon {window.longitude:.2f}°"
        if window.latitude is not None and window.longitude is not None
        else "lat/lon unavailable"
    )
    axis.set(
        xlabel="Time",
        ylabel="Temperature (°C)",
        title=(
            f"Event {window.event.event_id}: point {window.event.grid_point_index} · {location}\n"
            f"{window.event.duration_hours} hours ({window.event.duration_hours / 24:.1f} days)\n"
            f"{run_parameters_title(run)}"
        ),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    destination = directory / f"event_{window.event.event_id}.png"
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def summary(windows: list[EventWindow], plot_paths: list[Path]) -> dict[str, Any]:
    timing_names = tuple(windows[0].timings_seconds) if windows else ()
    return {
        "events_processed": len(windows),
        "hours_processed": sum(window.time.size for window in windows),
        "timing_seconds_total": {
            name: sum(window.timings_seconds[name] for window in windows) for name in timing_names
        },
        "timing_seconds_mean_per_event": {
            name: mean(window.timings_seconds[name] for window in windows) for name in timing_names
        },
        "plots": [str(path) for path in plot_paths],
    }


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.rule_batch_size < 1:
        raise ValueError("--limit and --rule-batch-size must be positive")
    rules = discover_rules(RULE_DIRECTORY)
    try:
        import psycopg
    except ModuleNotFoundError as exc:
        raise ImportError("install the optional driver with: pip install -e '.[postgres]'") from exc

    config = load_postgres_config(args.config)
    read_kwargs = config.connection_kwargs()
    read_kwargs["options"] = "-c default_transaction_read_only=on"
    benchmark_requested = args.plots or args.print_json or not (args.apply_rules or args.dry_run_rules)
    with psycopg.connect(**read_kwargs) as connection:
        run = load_run(connection, args.run_id)
        events = load_events(connection, run.run_id, args.limit) if benchmark_requested else []
    if benchmark_requested and not events:
        raise LookupError(f"run {run.run_id} has no events")

    cache_path = resolve_cache_path(run, args.zarr_path)
    zarr_group = open_and_validate_cache(cache_path, run)
    sst_path = resolve_sst_path(run, args.sst_path)
    source = open_and_validate_sst(sst_path, run)
    try:
        windows: list[EventWindow] = []
        if benchmark_requested:
            points = np.asarray([event.grid_point_index for event in events], dtype=np.int64)
            cache_point_columns = cache_columns(zarr_group, points)
            sst_point_columns = source_columns(source, points)
            windows = [
                fetch_event_window(
                    source,
                    zarr_group,
                    run,
                    event,
                    source_column=int(sst_point_columns[index]),
                    cache_column=int(cache_point_columns[index]),
                )
                for index, event in enumerate(events)
            ]
        execution_summary = None
        if args.apply_rules or args.dry_run_rules:
            execution_kwargs = config.connection_kwargs()
            if args.dry_run_rules:
                execution_kwargs["options"] = "-c default_transaction_read_only=on"
            with psycopg.connect(**execution_kwargs) as connection:
                execution_summary = apply_rules(
                    connection,
                    rules,
                    run,
                    source,
                    zarr_group,
                    batch_size=args.rule_batch_size,
                    write_results=args.apply_rules,
                )
    finally:
        source.close()

    plots = [write_plot(window, run, args.plot_directory) for window in windows] if args.plots else []
    output = {
        "mode": "enrichment_apply" if args.apply_rules else "enrichment_dry_run" if args.dry_run_rules else "benchmark",
        "run_id": run.run_id,
        "zarr_path": str(cache_path),
        "sst_path": str(sst_path),
        "plotting_timed": False,
        "rules_discovered": [
            {"rule_id": rule.id, "rule_version": rule.version, "rule_sha256": rule.sha256}
            for rule in rules
        ],
        "rule_execution": execution_summary,
        **summary(windows, plots),
    }
    if args.print_json:
        print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
