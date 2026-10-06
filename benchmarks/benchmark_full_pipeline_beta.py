"""CUDAHeat Full-Pipeline Beta Benchmark.

Measure the integrated numerical pipeline only: NetCDF ingestion, bounded
host-memory spatial batches, host-to-device transfer, GPU computation,
result transfer, and numerical agreement. MHW taxonomy is intentionally out
of scope for this campaign.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping

import numpy as np

from cudaheat import CUDAHeat, __version__
from cudaheat.benchmark import CUDAHeatBenchmark
from cudaheat.benchmark.metrics import BenchmarkRecord, records_to_rows


BENCHMARK_NAME = "CUDAHeat Full-Pipeline Beta Benchmark"
BENCHMARK_ID = "full_pipeline_beta"
PRESENTATION_DPI = 360

# Okabe-Ito-inspired, colour-vision-friendly palette.
COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#E69F00",
    "grey": "#4D4D4D",
    "light_blue": "#56B4E9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--baseline", type=int, nargs=2, metavar=("START", "END"), required=True)
    parser.add_argument("--percentile", type=float, default=0.90)
    parser.add_argument("--smooth-window-hours", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results") / BENCHMARK_ID)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--validation-points", type=int, default=512)
    parser.add_argument("--lazy-spatial-chunk", type=int, default=256)
    parser.add_argument("--dask-chunk-sizes", type=int, nargs="+")
    parser.add_argument("--chunk-repeats", type=int, default=2)
    parser.add_argument("--max-points", type=int, help="Cap campaign spatial points; omit for all points.")
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--quick", action="store_true", help="Use a compact smoke-test matrix.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in the output directory.")
    parser.add_argument("--skip-lazy-eager", action="store_true")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Regenerate tables, report, and figures from retained raw measurements.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=("lazy", "eager"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        _run_io_worker(args)
        return
    if args.render_only:
        _render_existing_campaign(args.output_dir)
        return
    _run_campaign(args)


def _run_campaign(args: argparse.Namespace) -> None:
    _validate_campaign_args(args)
    layout = _create_layout(args.output_dir, overwrite=args.overwrite)
    baseline = tuple(args.baseline)
    campaign_started = datetime.now(timezone.utc).isoformat()

    with CUDAHeat(
        args.dataset,
        variable=args.variable,
        dtype="float32",
        lazy=True,
        lazy_spatial_chunk=args.lazy_spatial_chunk,
    ) as engine:
        campaign_points = min(engine.npoints, args.max_points or engine.npoints)
        point_schedule = _point_schedule(campaign_points, quick=args.quick)
        batch_schedule = _batch_schedule(
            campaign_points,
            engine.batch_size,
            args.batch_sizes,
            quick=args.quick,
        )
        common = {
            "cpu_dtype": "float64",
            "baseline": baseline,
            "percentile": args.percentile,
            "smooth_window_hours": args.smooth_window_hours,
        }
        repeats = 1 if args.quick else args.repeats
        warmup = 0 if args.quick else 1
        benchmark = CUDAHeatBenchmark(engine)
        scenarios: list[tuple[str, list[BenchmarkRecord]]] = []

        scenarios.append(
            (
                "performance_scaling",
                _benchmark_records(
                    benchmark,
                    repeats=repeats,
                    warmup=warmup,
                    point_counts=point_schedule,
                    batch_sizes=(None,),
                    gpu_dtypes=("float32",),
                    **common,
                ),
            )
        )
        scenarios.append(
            (
                "batch_scaling",
                _benchmark_records(
                    benchmark,
                    repeats=repeats,
                    warmup=warmup,
                    point_counts=(campaign_points,),
                    batch_sizes=(None, *batch_schedule),
                    gpu_dtypes=("float32",),
                    **common,
                ),
            )
        )
        scenarios.append(
            (
                "precision_performance",
                _benchmark_records(
                    benchmark,
                    repeats=repeats,
                    warmup=warmup,
                    point_counts=(campaign_points,),
                    batch_sizes=(None,),
                    gpu_dtypes=("float32", "float64"),
                    **common,
                ),
            )
        )

        performance_rows = _performance_rows(scenarios)
        validation_payload = _run_precision_validation(
            engine,
            points=min(args.validation_points, campaign_points),
            baseline=baseline,
            percentile=args.percentile,
            smooth_window_hours=args.smooth_window_hours,
            quick=args.quick,
        )
        metadata = _metadata(
            engine,
            args,
            campaign_started=campaign_started,
            campaign_points=campaign_points,
            point_schedule=point_schedule,
            batch_schedule=batch_schedule,
            repeats=repeats,
        )

    lazy_eager_rows = [] if args.skip_lazy_eager else _run_lazy_eager_workers(args, campaign_points)
    dask_chunk_rows = _run_dask_chunk_sensitivity(
        args,
        campaign_points,
        baseline=baseline,
        quick=args.quick,
    )
    precision_rows = _precision_rows(validation_payload)
    memory_rows = _memory_rows(performance_rows)
    performance_summary = _performance_summary(performance_rows)
    memory_summary = _memory_summary(memory_rows)
    batch_summary = _batch_summary(performance_rows)
    repeatability_rows = _repeatability_rows(performance_rows, dask_chunk_rows)
    pipeline_rows = _pipeline_rows(performance_summary)
    throughput_rows = _throughput_rows(performance_summary)
    memory_efficiency_rows = _memory_efficiency_rows(memory_summary)
    dask_chunk_summary = _dask_chunk_summary(dask_chunk_rows)
    large_dataset_summary = _large_dataset_summary(metadata, performance_summary, lazy_eager_rows)

    _write_artifacts(
        layout,
        metadata=metadata,
        performance_rows=performance_rows,
        memory_rows=memory_rows,
        precision_rows=precision_rows,
        lazy_eager_rows=lazy_eager_rows,
        dask_chunk_rows=dask_chunk_rows,
        performance_summary=performance_summary,
        memory_summary=memory_summary,
        batch_summary=batch_summary,
        repeatability_rows=repeatability_rows,
        pipeline_rows=pipeline_rows,
        throughput_rows=throughput_rows,
        memory_efficiency_rows=memory_efficiency_rows,
        dask_chunk_summary=dask_chunk_summary,
        large_dataset_summary=large_dataset_summary,
        validation_payload=validation_payload,
    )
    _write_figures(
        layout,
        metadata,
        performance_summary,
        memory_summary,
        batch_summary,
        precision_rows,
        lazy_eager_rows,
        dask_chunk_summary,
        repeatability_rows,
        pipeline_rows,
        memory_efficiency_rows,
        large_dataset_summary,
    )
    _write_report(
        layout,
        metadata,
        performance_summary,
        precision_rows,
        lazy_eager_rows,
        dask_chunk_summary,
        repeatability_rows,
        large_dataset_summary,
    )
    print(f"{BENCHMARK_NAME}: {args.output_dir}")


def _render_existing_campaign(root: Path) -> None:
    """Rebuild derivative artifacts without rerunning an expensive measurement."""
    layout = _create_layout(root, overwrite=True)
    metadata = _read_json(layout["metadata"] / "benchmark_metadata.json")
    performance_payload = _read_json(layout["raw"] / "performance_runs.json")
    performance_rows = performance_payload["records"]
    dask_chunk_rows = _read_json(layout["raw"] / "dask_chunk_runs.json")
    lazy_eager_rows = _read_json(layout["raw"] / "lazy_eager_runs.json")
    validation_payload = _read_json(layout["raw"] / "validation_runs.json")
    precision_rows = _precision_rows(validation_payload)
    memory_rows = _memory_rows(performance_rows)
    performance_summary = _performance_summary(performance_rows)
    memory_summary = _memory_summary(memory_rows)
    batch_summary = _batch_summary(performance_rows)
    repeatability_rows = _repeatability_rows(performance_rows, dask_chunk_rows)
    pipeline_rows = _pipeline_rows(performance_summary)
    throughput_rows = _throughput_rows(performance_summary)
    memory_efficiency_rows = _memory_efficiency_rows(memory_summary)
    dask_chunk_summary = _dask_chunk_summary(dask_chunk_rows)
    large_dataset_summary = _large_dataset_summary(metadata, performance_summary, lazy_eager_rows)
    _write_artifacts(
        layout,
        metadata=metadata,
        performance_rows=performance_rows,
        memory_rows=memory_rows,
        precision_rows=precision_rows,
        lazy_eager_rows=lazy_eager_rows,
        dask_chunk_rows=dask_chunk_rows,
        performance_summary=performance_summary,
        memory_summary=memory_summary,
        batch_summary=batch_summary,
        repeatability_rows=repeatability_rows,
        pipeline_rows=pipeline_rows,
        throughput_rows=throughput_rows,
        memory_efficiency_rows=memory_efficiency_rows,
        dask_chunk_summary=dask_chunk_summary,
        large_dataset_summary=large_dataset_summary,
        validation_payload=validation_payload,
    )
    _write_figures(
        layout,
        metadata,
        performance_summary,
        memory_summary,
        batch_summary,
        precision_rows,
        lazy_eager_rows,
        dask_chunk_summary,
        repeatability_rows,
        pipeline_rows,
        memory_efficiency_rows,
        large_dataset_summary,
    )
    _write_report(
        layout,
        metadata,
        performance_summary,
        precision_rows,
        lazy_eager_rows,
        dask_chunk_summary,
        repeatability_rows,
        large_dataset_summary,
    )
    print(f"Regenerated {BENCHMARK_NAME}: {root}")


def _validate_campaign_args(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.validation_points < 1:
        raise ValueError("validation_points must be positive")
    if args.max_points is not None and args.max_points < 1:
        raise ValueError("max_points must be positive")
    if args.lazy_spatial_chunk < 1:
        raise ValueError("lazy_spatial_chunk must be positive")
    if args.chunk_repeats < 1:
        raise ValueError("chunk_repeats must be positive")
    if args.dask_chunk_sizes and any(size < 1 for size in args.dask_chunk_sizes):
        raise ValueError("dask_chunk_sizes must be positive")


def _create_layout(root: Path, *, overwrite: bool) -> dict[str, Path]:
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{root} already contains benchmark output; use --overwrite or choose --output-dir."
        )
    paths = {
        "root": root,
        "metadata": root / "metadata",
        "raw": root / "raw",
        "tables": root / "tables",
        "performance_figures": root / "figures" / "performance",
        "memory_figures": root / "figures" / "memory",
        "precision_figures": root / "figures" / "precision",
        "io_figures": root / "figures" / "io",
        "summary_figures": root / "figures" / "summary",
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def _point_schedule(limit: int, *, quick: bool) -> tuple[int, ...]:
    seeds = (32, 128, 512) if quick else (32, 128, 512, 2_000, 8_000, 32_000, 128_000)
    return tuple(sorted({point for point in (*seeds, limit) if point <= limit}))


def _batch_schedule(
    points: int,
    automatic_batch: int,
    requested: list[int] | None,
    *,
    quick: bool,
) -> tuple[int, ...]:
    if requested:
        values = requested
    elif quick:
        values = (min(64, points), min(256, points), min(automatic_batch, points))
    else:
        values = (
            min(64, points),
            min(256, points),
            min(512, points),
            min(1_024, points),
            min(automatic_batch, points),
        )
    return tuple(sorted({int(value) for value in values if value > 0}))


def _benchmark_records(benchmark: CUDAHeatBenchmark, **kwargs: Any) -> list[BenchmarkRecord]:
    before = len(benchmark.records)
    benchmark.run(**kwargs)
    return benchmark.records[before:]


def _performance_rows(scenarios: Iterable[tuple[str, Iterable[BenchmarkRecord]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario, records in scenarios:
        for row in records_to_rows(records):
            rows.append(
                {
                    "benchmark_name": BENCHMARK_NAME,
                    "benchmark_id": BENCHMARK_ID,
                    "scenario": scenario,
                    **row,
                }
            )
    return rows


def _run_precision_validation(
    engine: CUDAHeat,
    *,
    points: int,
    baseline: tuple[int, int],
    percentile: float,
    smooth_window_hours: int,
    quick: bool,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for dtype in ("float32", "float64"):
        validation = engine.validate(
            points=points,
            random_seed=17,
            gpu_dtype=dtype,
            baseline=baseline,
            percentile=percentile,
            smooth_window_hours=smooth_window_hours,
            validation_batch_size=64 if quick else 256,
        )
        results[dtype] = validation.to_dict()
    return results


def _precision_rows(validation: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dtype, payload in validation.items():
        common = {
            "benchmark_name": BENCHMARK_NAME,
            "benchmark_id": BENCHMARK_ID,
            "gpu_dtype": dtype,
            "validation_level": payload["level"],
            "validation_points": len(payload["point_indices"]),
            "ntime": payload["ntime"],
        }
        for field, metrics in payload["float_fields"].items():
            rows.append({"record_type": "float_field", "field": field, **common, **metrics})
        rows.append({"record_type": "mask", "field": "exceedance", **common, **payload["mask"]})
        rows.append(
            {
                "record_type": "count",
                "field": "exceedance_count",
                **common,
                **payload["exceedance_count"],
            }
        )
    return rows


def _memory_rows(performance_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in performance_rows:
        ntime = int(record["ntime"])
        batch = int(record["batch_size"])
        itemsize = 4 if record["dtype"] == "float32" else 8
        rows.append(
            {
                "benchmark_name": BENCHMARK_NAME,
                "benchmark_id": BENCHMARK_ID,
                "scenario": record["scenario"],
                "npoints": record["npoints"],
                "dtype": record["dtype"],
                "requested_batch_size": record["requested_batch_size"],
                "actual_batch_size": batch,
                "logical_input_gib": ntime * int(record["npoints"]) * itemsize / 2**30,
                "host_batch_gib": ntime * batch * itemsize / 2**30,
                "peak_vram_gib": float(record["peak_used_vram_bytes"]) / 2**30,
                "pool_used_gib": float(record.get("pool_used_bytes") or 0) / 2**30,
                "free_vram_gib": float(record["free_vram_bytes"]) / 2**30,
                "oom_retries": record["oom_retries"],
                "status": record["status"],
            }
        )
    return rows


def _performance_summary(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["status"] != "complete":
            continue
        key = (row["scenario"], row["npoints"], row["dtype"], row["requested_batch_size"])
        groups.setdefault(key, []).append(row)
    summaries = []
    metrics = (
        "cpu_compute_seconds",
        "gpu_compute_seconds",
        "gpu_end_to_end_seconds",
        "pipeline_wall_seconds",
        "host_read_seconds",
        "host_prepare_seconds",
        "transfer_to_device_seconds",
        "transfer_from_device_seconds",
        "gpu_compute_values_per_second",
        "gpu_end_to_end_values_per_second",
        "compute_speedup",
        "end_to_end_speedup",
        "pipeline_speedup",
        "peak_used_vram_bytes",
    )
    for key, values in sorted(
        groups.items(),
        key=lambda item: (*item[0][:3], -1 if item[0][3] is None else int(item[0][3])),
    ):
        scenario, npoints, dtype, requested_batch_size = key
        summary = {
            "benchmark_name": BENCHMARK_NAME,
            "benchmark_id": BENCHMARK_ID,
            "scenario": scenario,
            "npoints": npoints,
            "dtype": dtype,
            "requested_batch_size": requested_batch_size,
            "samples": len(values),
            "actual_batch_size_mean": _mean(values, "batch_size"),
        }
        for metric in metrics:
            summary[f"{metric}_mean"] = _mean(values, metric)
            summary[f"{metric}_min"] = _min(values, metric)
            summary[f"{metric}_max"] = _max(values, metric)
        summaries.append(summary)
    return summaries


def _memory_summary(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["scenario"], row["npoints"], row["dtype"], row["requested_batch_size"])
        groups.setdefault(key, []).append(row)
    result = []
    for key, values in sorted(
        groups.items(),
        key=lambda item: (*item[0][:3], -1 if item[0][3] is None else int(item[0][3])),
    ):
        scenario, npoints, dtype, requested_batch_size = key
        result.append(
            {
                "benchmark_name": BENCHMARK_NAME,
                "benchmark_id": BENCHMARK_ID,
                "scenario": scenario,
                "npoints": npoints,
                "dtype": dtype,
                "requested_batch_size": requested_batch_size,
                "samples": len(values),
                "actual_batch_size_mean": _mean(values, "actual_batch_size"),
                "logical_input_gib": _mean(values, "logical_input_gib"),
                "host_batch_gib": _mean(values, "host_batch_gib"),
                "peak_vram_gib": _mean(values, "peak_vram_gib"),
                "pool_used_gib": _mean(values, "pool_used_gib"),
                "oom_retries": _mean(values, "oom_retries"),
            }
        )
    return result


def _batch_summary(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in _performance_summary(rows) if row["scenario"] == "batch_scaling"]


def _large_dataset_summary(
    metadata: Mapping[str, Any],
    performance: Iterable[Mapping[str, Any]],
    lazy_eager: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = [
        row
        for row in performance
        if row["scenario"] == "performance_scaling" and row["dtype"] == "float32"
    ]
    largest = max(records, key=lambda row: row["npoints"], default=None)
    row = {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_id": BENCHMARK_ID,
        "dataset_path": metadata["dataset_path"],
        "ntime": metadata["ntime"],
        "dataset_npoints": metadata["dataset_npoints"],
        "campaign_npoints": metadata["campaign_npoints"],
        "logical_float32_input_gib": metadata["ntime"] * metadata["campaign_npoints"] * 4 / 2**30,
        "lazy_spatial_chunk": metadata["lazy_spatial_chunk"],
        "automatic_batch_size": metadata["automatic_batch_size"],
    }
    if largest:
        row.update(
            {
                "pipeline_wall_seconds_mean": largest["pipeline_wall_seconds_mean"],
                "gpu_end_to_end_seconds_mean": largest["gpu_end_to_end_seconds_mean"],
                "pipeline_speedup_mean": largest["pipeline_speedup_mean"],
                "peak_vram_gib": largest["peak_used_vram_bytes_mean"] / 2**30,
            }
        )
    for record in lazy_eager:
        mode = record["mode"]
        row[f"{mode}_status"] = record["status"]
        for field in (
            "peak_rss_gib",
            "peak_smaps_rss_gib",
            "peak_private_working_gib",
            "peak_private_mapped_gib",
            "peak_file_backed_rss_gib",
            "materialization_seconds",
            "pipeline_wall_seconds",
        ):
            row[f"{mode}_{field}"] = record.get(field)
    return [row]


def _metadata(
    engine: CUDAHeat,
    args: argparse.Namespace,
    *,
    campaign_started: str,
    campaign_points: int,
    point_schedule: tuple[int, ...],
    batch_schedule: tuple[int, ...],
    repeats: int,
) -> dict[str, Any]:
    import cupy as cp
    import dask
    import xarray as xr

    return {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_id": BENCHMARK_ID,
        "campaign_started_utc": campaign_started,
        "cudaheat_version": __version__,
        "dataset_path": str(args.dataset),
        "dataset_file_bytes": args.dataset.stat().st_size,
        "variable": args.variable,
        "ntime": engine.ntime,
        "dataset_npoints": engine.npoints,
        "campaign_npoints": campaign_points,
        "spatial_dimensions": list(engine.spatial_dims),
        "time_dimension": engine.time_dim,
        "baseline": list(args.baseline),
        "percentile": args.percentile,
        "smooth_window_hours": args.smooth_window_hours,
        "lazy_input": True,
        "lazy_spatial_chunk": args.lazy_spatial_chunk,
        "dask_chunk_sensitivity_sizes": list(
            sorted({*(args.dask_chunk_sizes or (64, 256, 1_024)), args.lazy_spatial_chunk})
        ),
        "dask_chunk_sensitivity_repeats": 1 if args.quick else args.chunk_repeats,
        "automatic_batch_size": engine.batch_size,
        "vram_safety": engine.vram_safety,
        "point_schedule": list(point_schedule),
        "batch_schedule": list(batch_schedule),
        "repeats": repeats,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "xarray_version": xr.__version__,
        "dask_version": dask.__version__,
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "gpu": engine.gpu_info.as_dict(),
        "scope": "Numerical pipeline only; MHW taxonomy and classification are excluded.",
        "time_was_split": False,
    }


def _run_lazy_eager_workers(args: argparse.Namespace, points: int) -> list[dict[str, Any]]:
    rows = []
    for mode in ("lazy", "eager"):
        output = args.output_dir / "raw" / f"{mode}_loading_worker.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(args.dataset),
            "--variable",
            args.variable,
            "--baseline",
            str(args.baseline[0]),
            str(args.baseline[1]),
            "--percentile",
            str(args.percentile),
            "--smooth-window-hours",
            str(args.smooth_window_hours),
            "--lazy-spatial-chunk",
            str(args.lazy_spatial_chunk),
            "--max-points",
            str(points),
            "--worker",
            "--worker-mode",
            mode,
            "--worker-output",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if output.exists():
            payload = json.loads(output.read_text(encoding="utf-8"))
        else:
            payload = {
                "status": "failed",
                "mode": mode,
                "error": completed.stderr.strip() or f"worker exited with status {completed.returncode}",
            }
        rows.append(
            {
                "benchmark_name": BENCHMARK_NAME,
                "benchmark_id": BENCHMARK_ID,
                "worker_returncode": completed.returncode,
                **payload,
            }
        )
    return rows


def _run_io_worker(args: argparse.Namespace) -> None:
    if args.worker_mode is None or args.worker_output is None:
        raise ValueError("worker mode and output are required")
    started = time.perf_counter()
    payload: dict[str, Any] = {"mode": args.worker_mode, "status": "failed"}
    memory = _ProcessMemorySampler()
    try:
        lazy = args.worker_mode == "lazy"
        with memory:
            if lazy:
                engine_started = time.perf_counter()
                with CUDAHeat(
                    args.dataset,
                    variable=args.variable,
                    dtype="float32",
                    lazy=True,
                    lazy_spatial_chunk=args.lazy_spatial_chunk,
                    point_indices=slice(0, args.max_points),
                ) as engine:
                    initialized = time.perf_counter()
                    payload.update(
                        _worker_pipeline_payload(
                            engine,
                            args,
                            started=started,
                            initialized=initialized,
                            source_open_seconds=None,
                            materialization_seconds=0.0,
                            engine_initialization_seconds=initialized - engine_started,
                        )
                    )
            else:
                import xarray as xr

                source_open_started = time.perf_counter()
                with xr.open_dataset(args.dataset) as dataset:
                    source_open_seconds = time.perf_counter() - source_open_started
                    materialization_started = time.perf_counter()
                    dataset.load()
                    materialization_seconds = time.perf_counter() - materialization_started
                    engine_started = time.perf_counter()
                    with CUDAHeat(
                        dataset,
                        variable=args.variable,
                        dtype="float32",
                        point_indices=slice(0, args.max_points),
                    ) as engine:
                        initialized = time.perf_counter()
                        payload.update(
                            _worker_pipeline_payload(
                                engine,
                                args,
                                started=started,
                                initialized=initialized,
                                source_open_seconds=source_open_seconds,
                                materialization_seconds=materialization_seconds,
                                engine_initialization_seconds=initialized - engine_started,
                            )
                        )
    except BaseException as exc:  # Worker must persist OOM failures as evidence.
        payload["error"] = f"{type(exc).__name__}: {exc}"
    payload.update(memory.peak)
    payload["peak_rss_gib"] = _peak_rss_gib()
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _worker_pipeline_payload(
    engine: CUDAHeat,
    args: argparse.Namespace,
    *,
    started: float,
    initialized: float,
    source_open_seconds: float | None,
    materialization_seconds: float,
    engine_initialization_seconds: float | None = None,
) -> dict[str, Any]:
    result = engine.run(
        baseline=tuple(args.baseline),
        percentile=args.percentile,
        smooth_window_hours=args.smooth_window_hours,
    )
    finished = time.perf_counter()
    counts = np.ascontiguousarray(result.exceedance_count.values)
    return {
        "status": "complete",
        "initialization_seconds": initialized - started,
        "source_open_seconds": source_open_seconds,
        "materialization_seconds": materialization_seconds,
        "engine_initialization_seconds": engine_initialization_seconds,
        "pipeline_wall_seconds": finished - initialized,
        "total_wall_seconds": finished - started,
        "npoints": engine.npoints,
        "ntime": engine.ntime,
        **_count_summary(counts),
    }


def _peak_rss_gib() -> float | None:
    try:
        import resource

        maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        bytes_used = maximum if sys.platform == "darwin" else maximum * 1024
        return float(bytes_used) / 2**30
    except Exception:  # pragma: no cover - platform-specific diagnostic
        return None


class _ProcessMemorySampler:
    """Capture Linux memory peaks, separating anonymous RAM from mapped files."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self.peak = _process_memory_snapshot()

    def __enter__(self) -> "_ProcessMemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        for key, value in _process_memory_snapshot().items():
            self.peak[key] = max(self.peak.get(key, 0.0), value)


def _process_memory_snapshot() -> dict[str, float]:
    """Return Linux smaps-rollup values in GiB; fall back cleanly elsewhere."""
    rollup = Path("/proc/self/smaps_rollup")
    if not rollup.exists():
        return {}
    values: dict[str, float] = {}
    for line in rollup.read_text(encoding="utf-8").splitlines():
        name, separator, amount = line.partition(":")
        if not separator:
            continue
        fields = amount.split()
        if len(fields) >= 2 and fields[1] == "kB":
            values[name] = float(fields[0]) / 2**20
    rss = values.get("Rss", 0.0)
    private_mapped = values.get("Private_Clean", 0.0) + values.get("Private_Dirty", 0.0)
    anonymous = values.get("Anonymous", 0.0)
    return {
        "peak_smaps_rss_gib": rss,
        # Anonymous pages are the useful bounded-working-RAM measure here.
        # Private_Clean can still be a privately mapped NetCDF source file.
        "peak_private_working_gib": anonymous,
        "peak_private_mapped_gib": private_mapped,
        "peak_anonymous_gib": anonymous,
        "peak_file_backed_rss_gib": max(0.0, rss - anonymous),
    }


def _count_summary(counts: np.ndarray) -> dict[str, Any]:
    """Return a NaN-stable signature for a spatial count field.

    Xarray represents masked land cells as NaNs after unstacking.  They are
    not measurements, so exclude them from the sum while retaining their
    positions in an explicit, stable checksum sentinel.
    """
    values = np.asarray(counts)
    if np.issubdtype(values.dtype, np.floating):
        finite = np.isfinite(values)
        normalized = np.where(finite, values, -1).astype(np.int64, copy=False)
        count_sum = int(np.nansum(values, dtype=np.float64))
        missing = int((~finite).sum())
    else:
        normalized = values.astype(np.int64, copy=False)
        count_sum = int(normalized.sum(dtype=np.int64))
        missing = 0
    digest = hashlib.sha256(np.ascontiguousarray(normalized).view(np.uint8)).hexdigest()
    return {
        "count_sha256": digest,
        "count_sum": count_sum,
        "count_valid_points": int(normalized.size - missing),
        "count_missing_points": missing,
    }


def _run_dask_chunk_sensitivity(
    args: argparse.Namespace,
    points: int,
    *,
    baseline: tuple[int, int],
    quick: bool,
) -> list[dict[str, Any]]:
    """Measure the post-open Dask point-chunk target without changing GPU batches."""
    requested = args.dask_chunk_sizes or (64, 256, 1_024)
    chunks = tuple(sorted({int(size) for size in (*requested, args.lazy_spatial_chunk)}))
    repeats = 1 if quick else args.chunk_repeats
    rows: list[dict[str, Any]] = []
    for spatial_chunk in chunks:
        for repeat in range(repeats):
            started = time.perf_counter()
            with CUDAHeat(
                args.dataset,
                variable=args.variable,
                dtype="float32",
                lazy=True,
                lazy_spatial_chunk=spatial_chunk,
                point_indices=slice(0, points),
            ) as engine:
                initialized = time.perf_counter()
                result = engine.run(
                    baseline=baseline,
                    percentile=args.percentile,
                    smooth_window_hours=args.smooth_window_hours,
                )
                finished = time.perf_counter()
                telemetry = result.telemetry
                counts = np.ascontiguousarray(result.exceedance_count.values)
                count_summary = _count_summary(counts)
                rows.append(
                    {
                        "benchmark_name": BENCHMARK_NAME,
                        "benchmark_id": BENCHMARK_ID,
                        "lazy_spatial_chunk": spatial_chunk,
                        "repeat": repeat,
                        "status": "complete",
                        "npoints": engine.npoints,
                        "ntime": engine.ntime,
                        "initialization_seconds": initialized - started,
                        "pipeline_wall_seconds": finished - initialized,
                        "total_wall_seconds": finished - started,
                        "batches": len(telemetry),
                        "actual_batch_size": max(item.batch_size for item in telemetry),
                        "host_read_seconds": sum(item.host_read_seconds for item in telemetry),
                        "host_prepare_seconds": sum(item.host_prepare_seconds for item in telemetry),
                        "transfer_to_device_seconds": sum(item.transfer_to_device_seconds for item in telemetry),
                        "gpu_compute_seconds": sum(item.gpu_compute_seconds for item in telemetry),
                        "transfer_from_device_seconds": sum(item.transfer_from_device_seconds for item in telemetry),
                        "result_assembly_seconds": sum(item.result_assembly_seconds for item in telemetry),
                        "peak_used_vram_bytes": max(item.peak_used_bytes for item in telemetry),
                        "pool_used_bytes": max(item.pool_used_bytes for item in telemetry),
                        **count_summary,
                    }
                )
    if rows:
        reference = rows[0]["count_sha256"]
        for row in rows:
            row["matches_reference_counts"] = row["count_sha256"] == reference
    return rows


def _dask_chunk_summary(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row["lazy_spatial_chunk"]), []).append(row)
    summaries = []
    metrics = (
        "initialization_seconds",
        "pipeline_wall_seconds",
        "host_read_seconds",
        "transfer_to_device_seconds",
        "gpu_compute_seconds",
        "transfer_from_device_seconds",
        "peak_used_vram_bytes",
    )
    for chunk, values in sorted(groups.items()):
        summary = {
            "benchmark_name": BENCHMARK_NAME,
            "benchmark_id": BENCHMARK_ID,
            "lazy_spatial_chunk": chunk,
            "samples": len(values),
            "all_count_hashes_match": all(row.get("matches_reference_counts") for row in values),
        }
        for metric in metrics:
            summary[f"{metric}_mean"] = _mean(values, metric)
            summary[f"{metric}_min"] = _min(values, metric)
            summary[f"{metric}_max"] = _max(values, metric)
        summaries.append(summary)
    return summaries


def _repeatability_rows(
    performance_rows: Iterable[Mapping[str, Any]], dask_chunk_rows: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in performance_rows:
        if row["status"] == "complete":
            requested = row["requested_batch_size"]
            batch = "automatic" if requested is None else str(requested)
            key = (
                f"{row['scenario']};points={row['npoints']};dtype={row['dtype']};"
                f"requested_batch={batch};actual_batch={row['batch_size']}",
                "pipeline_wall_seconds",
            )
            groups.setdefault(key, []).append(row)
    for row in dask_chunk_rows:
        if row["status"] == "complete":
            key = (f"dask_chunk={row['lazy_spatial_chunk']};points={row['npoints']}", "pipeline_wall_seconds")
            groups.setdefault(key, []).append(row)
    rows = []
    for (configuration, metric), values in sorted(groups.items()):
        samples = [_number(item.get(metric), default=np.nan) for item in values]
        samples = [value for value in samples if np.isfinite(value)]
        average = float(np.mean(samples)) if samples else np.nan
        standard_deviation = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
        rows.append(
            {
                "benchmark_name": BENCHMARK_NAME,
                "benchmark_id": BENCHMARK_ID,
                "configuration": configuration,
                "metric": metric,
                "samples": len(samples),
                "mean": average,
                "standard_deviation": standard_deviation,
                "coefficient_of_variation_percent": 100 * standard_deviation / average if average else np.nan,
                "minimum": float(np.min(samples)) if samples else np.nan,
                "maximum": float(np.max(samples)) if samples else np.nan,
            }
        )
    return rows


def _pipeline_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "host_read_seconds_mean",
        "host_prepare_seconds_mean",
        "transfer_to_device_seconds_mean",
        "gpu_compute_seconds_mean",
        "transfer_from_device_seconds_mean",
        "pipeline_wall_seconds_mean",
    )
    result = []
    for row in rows:
        if row["scenario"] != "performance_scaling" or row["dtype"] != "float32":
            continue
        result.append(
            {
                "benchmark_name": BENCHMARK_NAME,
                "benchmark_id": BENCHMARK_ID,
                "npoints": row["npoints"],
                **{field.removesuffix("_mean"): row.get(field) for field in fields},
            }
        )
    return result


def _throughput_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row["scenario"] != "performance_scaling":
            continue
        result.append(
            {
                "benchmark_name": BENCHMARK_NAME,
                "benchmark_id": BENCHMARK_ID,
                "npoints": row["npoints"],
                "dtype": row["dtype"],
                "gpu_compute_values_per_second": row.get("gpu_compute_values_per_second_mean"),
                "gpu_end_to_end_values_per_second": row.get("gpu_end_to_end_values_per_second_mean"),
                "compute_speedup": row.get("compute_speedup_mean"),
                "pipeline_speedup": row.get("pipeline_speedup_mean"),
            }
        )
    return result


def _memory_efficiency_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        peak = _number(row.get("peak_vram_gib"), default=np.nan)
        logical = _number(row.get("logical_input_gib"), default=np.nan)
        host = _number(row.get("host_batch_gib"), default=np.nan)
        result.append(
            {
                **row,
                "logical_to_peak_vram_ratio": logical / peak if peak else np.nan,
                "logical_to_host_batch_ratio": logical / host if host else np.nan,
            }
        )
    return result


def _write_artifacts(
    layout: Mapping[str, Path],
    *,
    metadata: Mapping[str, Any],
    performance_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    precision_rows: list[dict[str, Any]],
    lazy_eager_rows: list[dict[str, Any]],
    dask_chunk_rows: list[dict[str, Any]],
    performance_summary: list[dict[str, Any]],
    memory_summary: list[dict[str, Any]],
    batch_summary: list[dict[str, Any]],
    repeatability_rows: list[dict[str, Any]],
    pipeline_rows: list[dict[str, Any]],
    throughput_rows: list[dict[str, Any]],
    memory_efficiency_rows: list[dict[str, Any]],
    dask_chunk_summary: list[dict[str, Any]],
    large_dataset_summary: list[dict[str, Any]],
    validation_payload: Mapping[str, Any],
) -> None:
    _write_json(layout["metadata"] / "benchmark_metadata.json", metadata)
    _write_csv(layout["raw"] / "performance_runs.csv", performance_rows)
    _write_json(layout["raw"] / "performance_runs.json", {"metadata": metadata, "records": performance_rows})
    _write_csv(layout["raw"] / "memory_runs.csv", memory_rows)
    _write_json(layout["raw"] / "memory_runs.json", memory_rows)
    _write_csv(layout["raw"] / "precision_runs.csv", precision_rows)
    _write_json(layout["raw"] / "precision_runs.json", precision_rows)
    _write_csv(layout["raw"] / "lazy_eager_runs.csv", lazy_eager_rows)
    _write_json(layout["raw"] / "lazy_eager_runs.json", lazy_eager_rows)
    _write_csv(layout["raw"] / "dask_chunk_runs.csv", dask_chunk_rows)
    _write_json(layout["raw"] / "dask_chunk_runs.json", dask_chunk_rows)
    _write_json(layout["raw"] / "validation_runs.json", validation_payload)

    _write_csv(layout["tables"] / "performance_summary.csv", performance_summary)
    _write_json(layout["tables"] / "performance_summary.json", performance_summary)
    _write_csv(layout["tables"] / "numerical_consistency.csv", precision_rows)
    _write_json(layout["tables"] / "numerical_consistency.json", precision_rows)
    _write_csv(layout["tables"] / "memory_scaling.csv", memory_summary)
    _write_json(layout["tables"] / "memory_scaling.json", memory_summary)
    _write_csv(layout["tables"] / "memory_efficiency.csv", memory_efficiency_rows)
    _write_json(layout["tables"] / "memory_efficiency.json", memory_efficiency_rows)
    _write_csv(layout["tables"] / "batch_scaling.csv", batch_summary)
    _write_json(layout["tables"] / "batch_scaling.json", batch_summary)
    _write_csv(layout["tables"] / "pipeline_breakdown.csv", pipeline_rows)
    _write_json(layout["tables"] / "pipeline_breakdown.json", pipeline_rows)
    _write_csv(layout["tables"] / "throughput_summary.csv", throughput_rows)
    _write_json(layout["tables"] / "throughput_summary.json", throughput_rows)
    _write_csv(layout["tables"] / "repeatability.csv", repeatability_rows)
    _write_json(layout["tables"] / "repeatability.json", repeatability_rows)
    _write_csv(layout["tables"] / "dask_chunk_sensitivity.csv", dask_chunk_summary)
    _write_json(layout["tables"] / "dask_chunk_sensitivity.json", dask_chunk_summary)
    _write_csv(layout["tables"] / "eager_vs_lazy.csv", lazy_eager_rows)
    _write_json(layout["tables"] / "eager_vs_lazy.json", lazy_eager_rows)
    _write_csv(layout["tables"] / "large_dataset_summary.csv", large_dataset_summary)
    _write_json(layout["tables"] / "large_dataset_summary.json", large_dataset_summary)


def _write_figures(
    layout: Mapping[str, Path],
    metadata: Mapping[str, Any],
    performance: list[dict[str, Any]],
    memory: list[dict[str, Any]],
    batch: list[dict[str, Any]],
    precision: list[dict[str, Any]],
    lazy_eager: list[dict[str, Any]],
    dask_chunk: list[dict[str, Any]],
    repeatability: list[dict[str, Any]],
    pipeline: list[dict[str, Any]],
    memory_efficiency: list[dict[str, Any]],
    large: list[dict[str, Any]],
) -> None:
    _plot_cpu_gpu_scaling(performance, metadata, layout["performance_figures"] / "cpu_gpu_scaling.png")
    _plot_speedup(performance, metadata, layout["performance_figures"] / "speedup_vs_dataset_size.png")
    _plot_pipeline_breakdown(performance, metadata, layout["performance_figures"] / "pipeline_time_breakdown.png")
    _plot_throughput(performance, metadata, layout["performance_figures"] / "throughput_scaling.png")
    _plot_io_compute_balance(pipeline, metadata, layout["performance_figures"] / "io_vs_compute_scaling.png")
    _plot_pipeline_efficiency(performance, metadata, layout["performance_figures"] / "pipeline_efficiency.png")
    _plot_batch_pipeline_time(batch, metadata, layout["performance_figures"] / "batch_size_vs_pipeline_time.png")
    _plot_batch_throughput(batch, metadata, layout["performance_figures"] / "batch_size_vs_throughput.png")
    _plot_repeatability(repeatability, metadata, layout["performance_figures"] / "repeatability_coefficient_variation.png")
    _plot_logical_memory(memory, metadata, layout["memory_figures"] / "logical_size_vs_peak_memory.png")
    _plot_batch_memory(batch, metadata, layout["memory_figures"] / "vram_vs_batch_size.png")
    _plot_host_batch_memory(memory_efficiency, metadata, layout["memory_figures"] / "host_batch_memory_scaling.png")
    _plot_memory_efficiency(memory_efficiency, metadata, layout["memory_figures"] / "logical_to_working_memory_ratio.png")
    _plot_vram_utilisation(batch, metadata, layout["memory_figures"] / "vram_utilisation_by_batch_size.png")
    _plot_dtype_performance(performance, metadata, layout["precision_figures"] / "float32_float64_performance.png")
    _plot_numerical_consistency(precision, metadata, layout["precision_figures"] / "numerical_consistency.png")
    _plot_threshold_distance(precision, metadata, layout["precision_figures"] / "threshold_disagreement_distance.png")
    _plot_maximum_error(precision, metadata, layout["precision_figures"] / "maximum_absolute_error.png")
    _plot_mask_agreement(precision, metadata, layout["precision_figures"] / "mask_agreement.png")
    _plot_lazy_eager_memory(lazy_eager, metadata, layout["io_figures"] / "eager_vs_lazy_memory.png")
    _plot_lazy_eager_time(lazy_eager, metadata, layout["io_figures"] / "eager_vs_lazy_performance.png")
    _plot_lazy_eager_breakdown(lazy_eager, metadata, layout["io_figures"] / "eager_vs_lazy_ingestion_breakdown.png")
    _plot_lazy_eager_equivalence(lazy_eager, metadata, layout["io_figures"] / "eager_vs_lazy_count_equivalence.png")
    _plot_dask_chunk_time(dask_chunk, metadata, layout["io_figures"] / "dask_chunk_performance.png")
    _plot_dask_chunk_breakdown(dask_chunk, metadata, layout["io_figures"] / "dask_chunk_timing_breakdown.png")
    _plot_dask_chunk_stability(dask_chunk, metadata, layout["io_figures"] / "dask_chunk_stability.png")
    _plot_large_summary(large, metadata, layout["summary_figures"] / "large_dataset_scalability.png")
    _plot_full_pipeline_dashboard(performance, memory_efficiency, precision, lazy_eager, metadata, layout["summary_figures"] / "full_pipeline_beta_dashboard.png")


def _plot_cpu_gpu_scaling(rows, metadata, target: Path) -> None:
    rows = _scaling_rows(rows)
    figure, axis = _figure()
    _line(axis, rows, "npoints", "cpu_compute_seconds_mean", "CPU float64 compute", COLORS["grey"], "o")
    _line(axis, rows, "npoints", "gpu_compute_seconds_mean", "GPU float32 compute", COLORS["green"], "s")
    _line(axis, rows, "npoints", "pipeline_wall_seconds_mean", "GPU pipeline wall", COLORS["orange"], "D")
    axis.set_yscale("log")
    _finish(axis, metadata, "CPU and GPU execution scaling", "Spatial points", "Time [s, log scale]", legend=True)
    _save(figure, target)


def _plot_speedup(rows, metadata, target: Path) -> None:
    rows = _scaling_rows(rows)
    figure, axis = _figure()
    _line(axis, rows, "npoints", "compute_speedup_mean", "Compute-only", COLORS["green"], "o")
    _line(axis, rows, "npoints", "end_to_end_speedup_mean", "Device end-to-end", COLORS["blue"], "s")
    _line(axis, rows, "npoints", "pipeline_speedup_mean", "Pipeline wall", COLORS["orange"], "D")
    axis.axhline(1.0, color=COLORS["grey"], linestyle="--", linewidth=1.2, label="1× parity")
    _finish(axis, metadata, "Measured CPU-to-GPU speedup", "Spatial points", "Speedup [×]", legend=True)
    _save(figure, target)


def _plot_pipeline_breakdown(rows, metadata, target: Path) -> None:
    rows = _scaling_rows(rows)
    figure, axis = _figure(width=13)
    points = np.arange(len(rows))
    components = (
        ("host_read_seconds_mean", "Disk → CPU", COLORS["blue"]),
        ("transfer_to_device_seconds_mean", "CPU → GPU", COLORS["light_blue"]),
        ("gpu_compute_seconds_mean", "GPU compute", COLORS["green"]),
        ("transfer_from_device_seconds_mean", "GPU → CPU", COLORS["orange"]),
    )
    bottom = np.zeros(len(rows))
    for key, label, color in components:
        values = np.asarray([_number(row.get(key)) for row in rows])
        axis.bar(points, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set_xticks(points, [f"{row['npoints']:,}" for row in rows])
    _finish(axis, metadata, "Measured full-pipeline time breakdown", "Spatial points", "Time [s]", legend=True)
    _save(figure, target)


def _plot_throughput(rows, metadata, target: Path) -> None:
    rows = _scaling_rows(rows)
    figure, axis = _figure()
    for key, label, color, marker in (
        ("gpu_compute_values_per_second_mean", "GPU compute", COLORS["green"], "o"),
        ("gpu_end_to_end_values_per_second_mean", "GPU device end-to-end", COLORS["blue"], "s"),
    ):
        axis.plot(
            [row["npoints"] for row in rows],
            [_number(row.get(key)) / 1e6 for row in rows],
            marker=marker,
            color=color,
            linewidth=2.5,
            label=label,
        )
    _finish(axis, metadata, "GPU throughput scaling", "Spatial points", "Throughput [million values/s]", legend=True)
    _save(figure, target)


def _plot_io_compute_balance(rows, metadata, target: Path) -> None:
    figure, axis = _figure()
    rows = sorted(rows, key=lambda row: row["npoints"])
    io_seconds = [
        _number(row.get("host_read_seconds"))
        + _number(row.get("host_prepare_seconds"))
        + _number(row.get("transfer_to_device_seconds"))
        + _number(row.get("transfer_from_device_seconds"))
        for row in rows
    ]
    axis.plot([row["npoints"] for row in rows], io_seconds, "o-", color=COLORS["blue"], linewidth=2.5, label="I/O + transfers")
    axis.plot([row["npoints"] for row in rows], [_number(row.get("gpu_compute_seconds")) for row in rows], "s-", color=COLORS["green"], linewidth=2.5, label="GPU compute")
    _finish(axis, metadata, "I/O and transfer time versus GPU compute", "Spatial points", "Time [s]", legend=True)
    _save(figure, target)


def _plot_pipeline_efficiency(rows, metadata, target: Path) -> None:
    rows = _scaling_rows(rows)
    figure, axis = _figure()
    efficiency = [
        100 * _number(row.get("gpu_compute_seconds_mean")) / _number(row.get("pipeline_wall_seconds_mean"), default=np.nan)
        for row in rows
    ]
    axis.plot([row["npoints"] for row in rows], efficiency, "o-", color=COLORS["yellow"], linewidth=2.5)
    axis.set_ylim(bottom=0)
    _finish(axis, metadata, "GPU-compute share of measured pipeline wall time", "Spatial points", "GPU compute / pipeline [%]")
    _save(figure, target)


def _plot_batch_pipeline_time(rows, metadata, target: Path) -> None:
    rows = sorted(rows, key=lambda row: _number(row.get("actual_batch_size_mean")))
    figure, axis = _figure()
    _line(axis, rows, "actual_batch_size_mean", "pipeline_wall_seconds_mean", "Pipeline wall", COLORS["orange"], "o")
    _line(axis, rows, "actual_batch_size_mean", "gpu_end_to_end_seconds_mean", "GPU device E2E", COLORS["blue"], "s")
    _finish(axis, metadata, "Batch-size sensitivity of pipeline time", "Actual batch size [points]", "Time [s]", legend=True)
    _save(figure, target)


def _plot_batch_throughput(rows, metadata, target: Path) -> None:
    rows = sorted(rows, key=lambda row: _number(row.get("actual_batch_size_mean")))
    figure, axis = _figure()
    axis.plot(
        [row["actual_batch_size_mean"] for row in rows],
        [_number(row.get("gpu_end_to_end_values_per_second_mean")) / 1e6 for row in rows],
        "o-",
        color=COLORS["green"],
        linewidth=2.5,
    )
    _finish(axis, metadata, "Batch-size sensitivity of end-to-end throughput", "Actual batch size [points]", "Throughput [million values/s]")
    _save(figure, target)


def _plot_repeatability(rows, metadata, target: Path) -> None:
    figure, axis = _figure(width=14)
    rows = [row for row in rows if row["samples"] > 1]
    if rows:
        labels = [row["configuration"].replace(";", "\n") for row in rows]
        values = [_number(row.get("coefficient_of_variation_percent"), default=np.nan) for row in rows]
        axis.bar(range(len(rows)), values, color=COLORS["blue"])
        axis.set_xticks(range(len(rows)), labels, rotation=25, ha="right")
    else:
        _no_data(axis, "At least two repeats are required")
    _finish(axis, metadata, "Repeatability of full-pipeline wall time", "Configuration", "Coefficient of variation [%]")
    _save(figure, target)


def _plot_logical_memory(rows, metadata, target: Path) -> None:
    rows = [row for row in rows if row["scenario"] == "performance_scaling" and row["dtype"] == "float32"]
    figure, axis = _figure()
    axis.plot([row["logical_input_gib"] for row in rows], [row["peak_vram_gib"] for row in rows], "o-", color=COLORS["purple"], linewidth=2.5, label="Peak VRAM")
    axis.plot([row["logical_input_gib"] for row in rows], [row["host_batch_gib"] for row in rows], "s-", color=COLORS["blue"], linewidth=2.5, label="One host batch")
    _finish(axis, metadata, "Logical input size versus bounded working memory", "Logical float32 input [GiB]", "Memory [GiB]", legend=True)
    _save(figure, target)


def _plot_batch_memory(rows, metadata, target: Path) -> None:
    rows = sorted(rows, key=lambda row: _number(row.get("actual_batch_size_mean")))
    figure, axis = _figure()
    axis.plot([row["actual_batch_size_mean"] for row in rows], [row["peak_used_vram_bytes_mean"] / 2**30 for row in rows], "o-", color=COLORS["purple"], linewidth=2.5)
    _finish(axis, metadata, "VRAM use by spatial batch size", "Actual batch size [points]", "Peak VRAM [GiB]")
    _save(figure, target)


def _plot_host_batch_memory(rows, metadata, target: Path) -> None:
    rows = [row for row in rows if row["scenario"] == "batch_scaling" and row["dtype"] == "float32"]
    rows = sorted(rows, key=lambda row: _number(row.get("actual_batch_size_mean")))
    figure, axis = _figure()
    axis.plot([row["actual_batch_size_mean"] for row in rows], [_number(row.get("host_batch_gib")) for row in rows], "o-", color=COLORS["blue"], linewidth=2.5, label="One host batch")
    axis.plot([row["actual_batch_size_mean"] for row in rows], [_number(row.get("peak_vram_gib")) for row in rows], "s-", color=COLORS["purple"], linewidth=2.5, label="Peak VRAM")
    _finish(axis, metadata, "Host and GPU working memory by batch size", "Actual batch size [points]", "Memory [GiB]", legend=True)
    _save(figure, target)


def _plot_memory_efficiency(rows, metadata, target: Path) -> None:
    rows = [row for row in rows if row["scenario"] == "performance_scaling" and row["dtype"] == "float32"]
    figure, axis = _figure()
    axis.plot([row["logical_input_gib"] for row in rows], [_number(row.get("logical_to_peak_vram_ratio"), default=np.nan) for row in rows], "o-", color=COLORS["green"], linewidth=2.5, label="Logical input / peak VRAM")
    axis.plot([row["logical_input_gib"] for row in rows], [_number(row.get("logical_to_host_batch_ratio"), default=np.nan) for row in rows], "s-", color=COLORS["blue"], linewidth=2.5, label="Logical input / one host batch")
    _finish(axis, metadata, "Memory-bounded scaling efficiency", "Logical float32 input [GiB]", "Compression ratio [×]", legend=True)
    _save(figure, target)


def _plot_vram_utilisation(rows, metadata, target: Path) -> None:
    rows = sorted(rows, key=lambda row: _number(row.get("actual_batch_size_mean")))
    figure, axis = _figure()
    values = [
        100 * _number(row.get("peak_used_vram_bytes_mean")) / (metadata["gpu"]["total_bytes"] or np.nan)
        for row in rows
    ]
    axis.plot([row["actual_batch_size_mean"] for row in rows], values, "o-", color=COLORS["yellow"], linewidth=2.5)
    axis.set_ylim(bottom=0)
    _finish(axis, metadata, "Observed VRAM utilisation by batch size", "Actual batch size [points]", "GPU memory used [%]")
    _save(figure, target)


def _plot_dtype_performance(rows, metadata, target: Path) -> None:
    rows = [row for row in rows if row["scenario"] == "precision_performance"]
    figure, axes = _figure(panels=2, width=15)
    dtypes = ["float32", "float64"]
    available = {row["dtype"]: row for row in rows}
    for axis, metric, title in (
        (axes[0], "gpu_compute_seconds_mean", "GPU compute"),
        (axes[1], "pipeline_wall_seconds_mean", "Pipeline wall"),
    ):
        values = [_number(available[dtype].get(metric)) if dtype in available else np.nan for dtype in dtypes]
        bars = axis.bar(dtypes, values, color=[COLORS["green"], COLORS["purple"]])
        axis.bar_label(bars, labels=[f"{value:.2f} s" if np.isfinite(value) else "n/a" for value in values], padding=4)
        _finish(axis, metadata, title, "GPU dtype", "Time [s]")
    _save(figure, target)


def _plot_numerical_consistency(rows, metadata, target: Path) -> None:
    rows = [row for row in rows if row["record_type"] == "float_field"]
    figure, axis = _figure(width=14)
    fields = sorted({row["field"] for row in rows})
    dtypes = sorted({row["gpu_dtype"] for row in rows})
    x = np.arange(len(fields))
    width = 0.36
    for index, dtype in enumerate(dtypes):
        values = []
        for field in fields:
            match = next((row for row in rows if row["gpu_dtype"] == dtype and row["field"] == field), None)
            values.append(max(_number(match.get("rmse")) if match else np.nan, np.finfo(float).tiny))
        axis.bar(x + (index - (len(dtypes) - 1) / 2) * width, values, width, label=f"GPU {dtype}", color=COLORS["green"] if dtype == "float32" else COLORS["purple"])
    axis.set_xticks(x, fields, rotation=25, ha="right")
    axis.set_yscale("log")
    _finish(axis, metadata, "CPU float64 versus GPU numerical consistency", "Field", "RMSE [°C]", legend=True)
    _save(figure, target)


def _plot_threshold_distance(rows, metadata, target: Path) -> None:
    rows = [row for row in rows if row["record_type"] == "mask"]
    figure, axis = _figure()
    dtypes = [row["gpu_dtype"] for row in rows]
    values = [_number(row.get("threshold_distance_mean")) for row in rows]
    bars = axis.bar(dtypes, values, color=[COLORS["green"] if dtype == "float32" else COLORS["purple"] for dtype in dtypes])
    labels = [f"{int(row.get('disagreements', 0)):,} disagreements" for row in rows]
    axis.bar_label(bars, labels=labels, padding=4, fontsize=10)
    _finish(axis, metadata, "Threshold-boundary distance for mask disagreements", "GPU dtype", "Mean distance [°C]")
    _save(figure, target)


def _plot_maximum_error(rows, metadata, target: Path) -> None:
    rows = [row for row in rows if row["record_type"] == "float_field"]
    figure, axis = _figure(width=14)
    fields = sorted({row["field"] for row in rows})
    dtypes = sorted({row["gpu_dtype"] for row in rows})
    x = np.arange(len(fields))
    width = 0.36
    for index, dtype in enumerate(dtypes):
        values = []
        for field in fields:
            match = next((row for row in rows if row["gpu_dtype"] == dtype and row["field"] == field), None)
            value = _number(match.get("max_absolute_error"), default=np.nan) if match else np.nan
            values.append(max(value, np.finfo(float).tiny) if np.isfinite(value) else np.nan)
        axis.bar(x + (index - (len(dtypes) - 1) / 2) * width, values, width, label=f"GPU {dtype}", color=COLORS["green"] if dtype == "float32" else COLORS["purple"])
    axis.set_xticks(x, fields, rotation=25, ha="right")
    axis.set_yscale("log")
    _finish(axis, metadata, "Maximum absolute CPU/GPU field difference", "Field", "Maximum absolute error [°C]", legend=True)
    _save(figure, target)


def _plot_mask_agreement(rows, metadata, target: Path) -> None:
    rows = [row for row in rows if row["record_type"] == "mask"]
    figure, axis = _figure()
    dtypes = [row["gpu_dtype"] for row in rows]
    values = [_number(row.get("agreement_percent"), default=np.nan) for row in rows]
    bars = axis.bar(dtypes, values, color=[COLORS["green"] if dtype == "float32" else COLORS["purple"] for dtype in dtypes])
    axis.bar_label(bars, labels=[f"{value:.8f}%" for value in values], padding=4, fontsize=10)
    axis.set_ylim(min(99.0, min(values, default=100.0) - 0.1), 100.01)
    _finish(axis, metadata, "Exceedance-mask agreement with CPU float64", "GPU dtype", "Agreement [%]")
    _save(figure, target)


def _plot_lazy_eager_memory(rows, metadata, target: Path) -> None:
    figure, axis = _figure()
    complete = [row for row in rows if row.get("status") == "complete"]
    if complete:
        labels = [row["mode"].title() for row in complete]
        values = [_number(row.get("peak_private_working_gib"), default=np.nan) for row in complete]
        bars = axis.bar(labels, values, color=[COLORS["green"] if row["mode"] == "lazy" else COLORS["orange"] for row in complete])
        labels = [
            f"private {value:.2f} GiB\nRSS {_number(row.get('peak_rss_gib')):.2f} GiB"
            for row, value in zip(complete, values, strict=True)
        ]
        axis.bar_label(bars, labels=labels, padding=4, fontsize=10)
    else:
        _no_data(axis, "No completed isolated I/O measurements")
    _finish(axis, metadata, "Lazy versus eager private working memory", "Input mode", "Peak private working memory [GiB]")
    _save(figure, target)


def _plot_lazy_eager_time(rows, metadata, target: Path) -> None:
    figure, axis = _figure()
    complete = [row for row in rows if row.get("status") == "complete"]
    if complete:
        labels = [row["mode"].title() for row in complete]
        values = [_number(row.get("pipeline_wall_seconds")) for row in complete]
        bars = axis.bar(labels, values, color=[COLORS["green"] if row["mode"] == "lazy" else COLORS["orange"] for row in complete])
        axis.bar_label(bars, labels=[f"{value:.2f} s" for value in values], padding=4)
    else:
        _no_data(axis, "No completed isolated I/O measurements")
    _finish(axis, metadata, "Lazy versus eager pipeline time", "Input mode", "Pipeline wall time [s]")
    _save(figure, target)


def _plot_lazy_eager_breakdown(rows, metadata, target: Path) -> None:
    figure, axis = _figure(width=13)
    complete = [row for row in rows if row.get("status") == "complete"]
    if complete:
        x = np.arange(len(complete))
        bottom = np.zeros(len(complete))
        for key, label, color in (
            ("source_open_seconds", "Source open", COLORS["blue"]),
            ("materialization_seconds", "Eager materialization", COLORS["orange"]),
            ("engine_initialization_seconds", "Engine/Dask setup", COLORS["purple"]),
            ("pipeline_wall_seconds", "Pipeline execution", COLORS["green"]),
        ):
            values = np.asarray([_number(row.get(key), default=0.0) for row in complete])
            axis.bar(x, values, bottom=bottom, label=label, color=color)
            bottom += values
        axis.set_xticks(x, [row["mode"].title() for row in complete])
        _finish(axis, metadata, "Lazy and eager ingestion-to-result breakdown", "Input mode", "Time [s]", legend=True)
    else:
        _no_data(axis, "No completed isolated I/O measurements")
        _finish(axis, metadata, "Lazy and eager ingestion-to-result breakdown", "Input mode", "Time [s]")
    _save(figure, target)


def _plot_lazy_eager_equivalence(rows, metadata, target: Path) -> None:
    figure, axis = _figure(width=12, height=5)
    lazy = next((row for row in rows if row.get("mode") == "lazy" and row.get("status") == "complete"), None)
    eager = next((row for row in rows if row.get("mode") == "eager" and row.get("status") == "complete"), None)
    if lazy and eager:
        matches = lazy.get("count_sha256") == eager.get("count_sha256")
        color = COLORS["green"] if matches else COLORS["orange"]
        message = "Identical exceedance-count SHA-256" if matches else "Count checksums differ"
    elif lazy:
        color = COLORS["blue"]
        message = "Lazy run completed; eager run did not complete"
    else:
        color = COLORS["orange"]
        message = "No completed lazy/eager equivalence comparison"
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=19, color=color, fontweight="bold", bbox={"facecolor": "#F4F7F9", "edgecolor": color, "pad": 16})
    _finish(axis, metadata, "Lazy versus eager numerical equivalence", "", "")
    _save(figure, target)


def _plot_dask_chunk_time(rows, metadata, target: Path) -> None:
    figure, axis = _figure()
    rows = sorted(rows, key=lambda row: row["lazy_spatial_chunk"])
    axis.plot([row["lazy_spatial_chunk"] for row in rows], [_number(row.get("pipeline_wall_seconds_mean")) for row in rows], "o-", color=COLORS["blue"], linewidth=2.5, label="Pipeline wall")
    axis.plot([row["lazy_spatial_chunk"] for row in rows], [_number(row.get("host_read_seconds_mean")) for row in rows], "s-", color=COLORS["orange"], linewidth=2.5, label="Disk → CPU")
    _finish(axis, metadata, "Dask point-chunk sensitivity", "Lazy spatial chunk [points]", "Time [s]", legend=True)
    _save(figure, target)


def _plot_dask_chunk_breakdown(rows, metadata, target: Path) -> None:
    figure, axis = _figure(width=13)
    rows = sorted(rows, key=lambda row: row["lazy_spatial_chunk"])
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for key, label, color in (
        ("host_read_seconds_mean", "Disk → CPU", COLORS["blue"]),
        ("transfer_to_device_seconds_mean", "CPU → GPU", COLORS["light_blue"]),
        ("gpu_compute_seconds_mean", "GPU compute", COLORS["green"]),
        ("transfer_from_device_seconds_mean", "GPU → CPU", COLORS["orange"]),
    ):
        values = np.asarray([_number(row.get(key)) for row in rows])
        axis.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set_xticks(x, [f"{row['lazy_spatial_chunk']:,}" for row in rows])
    _finish(axis, metadata, "Dask point-chunk timing breakdown", "Lazy spatial chunk [points]", "Time [s]", legend=True)
    _save(figure, target)


def _plot_dask_chunk_stability(rows, metadata, target: Path) -> None:
    figure, axis = _figure()
    rows = sorted(rows, key=lambda row: row["lazy_spatial_chunk"])
    values = []
    for row in rows:
        mean = _number(row.get("pipeline_wall_seconds_mean"), default=np.nan)
        minimum = _number(row.get("pipeline_wall_seconds_min"), default=np.nan)
        maximum = _number(row.get("pipeline_wall_seconds_max"), default=np.nan)
        values.append(100 * (maximum - minimum) / mean if mean else np.nan)
    axis.bar([str(row["lazy_spatial_chunk"]) for row in rows], values, color=COLORS["yellow"])
    _finish(axis, metadata, "Dask point-chunk run-to-run spread", "Lazy spatial chunk [points]", "Range / mean [%]")
    _save(figure, target)


def _plot_large_summary(rows, metadata, target: Path) -> None:
    figure, axis = _figure(width=14, height=8)
    axis.axis("off")
    row = rows[0] if rows else {}
    details = [
        f"Dataset: {row.get('campaign_npoints', 0):,} spatial points × {row.get('ntime', 0):,} hours",
        f"Logical float32 input: {_number(row.get('logical_float32_input_gib')):.2f} GiB",
        f"Automatic spatial batch: {row.get('automatic_batch_size', 0):,}",
        f"Pipeline wall: {_number(row.get('pipeline_wall_seconds_mean')):.2f} s",
        f"Pipeline speedup: {_number(row.get('pipeline_speedup_mean')):.2f}×",
        f"Peak VRAM: {_number(row.get('peak_vram_gib')):.2f} GiB",
        f"Lazy loading: {row.get('lazy_status', 'not measured')}",
        f"Eager loading: {row.get('eager_status', 'not measured')}",
    ]
    axis.text(0.05, 0.9, "\n\n".join(details), va="top", fontsize=17, color="#222222", bbox={"facecolor": "#F4F7F9", "edgecolor": COLORS["blue"], "pad": 14})
    _finish(axis, metadata, "Large-dataset full-pipeline summary", "", "")
    _save(figure, target)


def _plot_full_pipeline_dashboard(performance, memory, precision, lazy_eager, metadata, target: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    scaling = _scaling_rows(performance)
    axes[0, 0].plot([row["npoints"] for row in scaling], [_number(row.get("pipeline_wall_seconds_mean")) for row in scaling], "o-", color=COLORS["orange"], linewidth=2.5)
    _style_axis(axes[0, 0], "Pipeline scaling", "Spatial points", "Pipeline wall [s]")

    axes[0, 1].plot([row["logical_input_gib"] for row in memory if row["scenario"] == "performance_scaling"], [_number(row.get("peak_vram_gib")) for row in memory if row["scenario"] == "performance_scaling"], "o-", color=COLORS["purple"], linewidth=2.5)
    _style_axis(axes[0, 1], "Bounded VRAM", "Logical float32 input [GiB]", "Peak VRAM [GiB]")

    mask = [row for row in precision if row["record_type"] == "mask"]
    axes[1, 0].bar([row["gpu_dtype"] for row in mask], [_number(row.get("agreement_percent")) for row in mask], color=[COLORS["green"] if row["gpu_dtype"] == "float32" else COLORS["purple"] for row in mask])
    axes[1, 0].set_ylim(min(99.0, min((_number(row.get("agreement_percent")) for row in mask), default=100.0) - 0.1), 100.01)
    _style_axis(axes[1, 0], "Numerical agreement", "GPU dtype", "Mask agreement [%]")

    axes[1, 1].axis("off")
    lazy = next((row for row in lazy_eager if row.get("mode") == "lazy"), {})
    eager = next((row for row in lazy_eager if row.get("mode") == "eager"), {})
    text = (
        f"{metadata['campaign_npoints']:,} points × {metadata['ntime']:,} hours\n\n"
        f"Lazy: {lazy.get('status', 'not measured')}\n"
        f"Eager: {eager.get('status', 'not measured')}\n\n"
        "Complete time series per point\nSpatial batching only"
    )
    axes[1, 1].text(0.08, 0.88, text, va="top", fontsize=16, linespacing=1.4, bbox={"facecolor": "#F4F7F9", "edgecolor": COLORS["blue"], "pad": 12})
    figure.suptitle(BENCHMARK_NAME, fontsize=19, fontweight="bold")
    figure.text(0.5, 0.01, f"{BENCHMARK_ID} · numerical pipeline only; taxonomy excluded", ha="center", fontsize=10, color=COLORS["grey"])
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=PRESENTATION_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(target.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _figure(*, panels: int = 1, width: float = 12, height: float = 7):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, panels, figsize=(width, height), layout="constrained")
    return figure, axes


def _finish(axis, metadata: Mapping[str, Any], title: str, xlabel: str, ylabel: str, *, legend: bool = False) -> None:
    _style_axis(axis, title, xlabel, ylabel)
    if legend:
        axis.legend(frameon=True, fontsize=10)
    axis.figure.suptitle(BENCHMARK_NAME, fontsize=14, fontweight="bold")


def _style_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.set_title(title, fontsize=17, pad=14)
    axis.tick_params(labelsize=11)
    axis.grid(True, alpha=0.22, linewidth=0.8)


def _line(axis, rows, x_key: str, y_key: str, label: str, color: str, marker: str) -> None:
    axis.plot(
        [row[x_key] for row in rows],
        [_number(row.get(y_key)) for row in rows],
        marker=marker,
        linewidth=2.5,
        color=color,
        label=label,
    )


def _no_data(axis, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes, fontsize=14)


def _save(figure, target: Path) -> None:
    import matplotlib.pyplot as plt

    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=PRESENTATION_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(target.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_report(
    layout: Mapping[str, Path],
    metadata: Mapping[str, Any],
    performance: list[dict[str, Any]],
    precision: list[dict[str, Any]],
    lazy_eager: list[dict[str, Any]],
    dask_chunk: list[dict[str, Any]],
    repeatability: list[dict[str, Any]],
    large: list[dict[str, Any]],
) -> None:
    scaling = [
        row
        for row in performance
        if row["scenario"] == "performance_scaling" and row["dtype"] == "float32"
    ]
    largest = max(scaling, key=lambda row: row["npoints"], default={})
    mask_rows = [row for row in precision if row["record_type"] == "mask"]
    io_lines = "\n".join(
        f"- {row['mode']}: **{row['status']}**; peak private working memory "
        f"{row.get('peak_private_working_gib', 'n/a')} GiB; process RSS "
        f"{row.get('peak_rss_gib', 'n/a')} GiB; {row.get('error', '')}"
        for row in lazy_eager
    ) or "- Isolated lazy/eager comparison was skipped."
    chunk_lines = "\n".join(
        f"- {row['lazy_spatial_chunk']:,} points: {row['pipeline_wall_seconds_mean']:.3f} s pipeline wall; "
        f"count hashes match: {row['all_count_hashes_match']}."
        for row in dask_chunk
    ) or "- Dask chunk sensitivity was not measured."
    stable = [row for row in repeatability if row["samples"] > 1]
    stability_line = (
        f"The largest observed pipeline-wall coefficient of variation was "
        f"{max(row['coefficient_of_variation_percent'] for row in stable):.3f}%."
        if stable
        else "Only one repeat was recorded per configuration."
    )
    text = f"""# {BENCHMARK_NAME}

`benchmark_id`: `{BENCHMARK_ID}`

## Scope

This campaign evaluates the integrated numerical mechanism only: lazy NetCDF ingestion, bounded host-memory spatial batching, H2D, GPU computation, D2H/results, and numerical validation. MHW taxonomy and event classification are excluded.

## Dataset and method

- Dataset: `{metadata['dataset_path']}`
- Scale: {metadata['campaign_npoints']:,} spatial points × {metadata['ntime']:,} hourly samples
- Processing: complete time series per point; spatial partitioning only; `lazy=True`
- GPU: {metadata['gpu']['name']} ({metadata['gpu']['total_bytes'] / 2**30:.2f} GiB)
- Baseline / percentile / smoothing: {metadata['baseline']} / {metadata['percentile']:.0%} / {metadata['smooth_window_hours']} h

## Largest measured performance configuration

- Points: {largest.get('npoints', 0):,}
- GPU pipeline wall time: {_number(largest.get('pipeline_wall_seconds_mean')):.3f} s
- GPU end-to-end time: {_number(largest.get('gpu_end_to_end_seconds_mean')):.3f} s
- Pipeline speedup: {_number(largest.get('pipeline_speedup_mean')):.2f}×

## Numerical consistency

{chr(10).join(f"- GPU {row['gpu_dtype']}: {row['agreement_percent']:.8f}% exceedance-mask agreement; {row['disagreements']:,} disagreements." for row in mask_rows)}

## Lazy versus eager isolated processes

{io_lines}

`peak_private_working_gib` excludes file-backed mappings and is the primary
host-working-memory measure. `peak_rss_gib` is retained separately because
some NetCDF backends memory-map source files, which can inflate process RSS
without allocating equivalent private RAM.

## Dask chunk sensitivity and repeatability

{chunk_lines}

{stability_line}

## Artifacts

- Metadata: `metadata/benchmark_metadata.json`
- Raw records: `raw/`
- Tables: `tables/`
- Presentation figures (PNG at {PRESENTATION_DPI} dpi, plus PDF/SVG): `figures/`
- Additional raw Dask chunk runs: `raw/dask_chunk_runs.csv`
- Repeatability table: `tables/repeatability.csv`
"""
    (layout["root"] / f"{BENCHMARK_ID}_report.md").write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _scaling_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if row["scenario"] == "performance_scaling"
            and row["dtype"] == "float32"
            and row["requested_batch_size"] is None
        ),
        key=lambda row: row["npoints"],
    )


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [_number(row.get(key), default=np.nan) for row in rows]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _min(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [_number(row.get(key), default=np.nan) for row in rows]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.min(finite)) if finite else None


def _max(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [_number(row.get(key), default=np.nan) for row in rows]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.max(finite)) if finite else None


def _number(value: Any, *, default: float = 0.0) -> float:
    return default if value is None else float(value)


if __name__ == "__main__":
    main()
