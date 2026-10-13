"""GPU-only, timestamped CUDAHeat numerical-pipeline benchmark.

This campaign measures GPU execution and its required pipeline overhead. It
does not time a CPU reference implementation, compare lazy against eager input,
or sweep Dask chunks. CPU is used only as an untimed correctness reference.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping

import numpy as np

from cudaheat import CUDAHeat, __version__
from cudaheat.benchmark import CUDAHeatBenchmark, records_to_rows


BENCHMARK_NAME = "CUDAHeat GPU Pipeline Benchmark"
BENCHMARK_ID = "gpu_pipeline"
DEFAULT_DATASET = Path("examples/sst_cudaheat.nc")
DEFAULT_OUTPUT_ROOT = Path("benchmark_results") / BENCHMARK_ID
DEFAULT_BASELINE = (2019, 2025)
DEFAULT_VRAM_SAFETY_VALUES = (0.0, 0.15, 0.30, 0.45)

_CPU_COLUMNS = frozenset(
    {
        "cpu_compute_seconds",
        "cpu_dtype",
        "compute_speedup",
        "end_to_end_speedup",
        "pipeline_speedup",
        "speedup",
        "cpu_values_per_second",
    }
)
_TIMINGS = (
    "gpu_compute_seconds",
    "gpu_end_to_end_seconds",
    "pipeline_wall_seconds",
    "host_read_seconds",
    "host_prepare_seconds",
    "gpu_allocation_seconds",
    "gpu_metadata_transfer_seconds",
    "transfer_to_device_seconds",
    "transfer_from_device_seconds",
    "result_assembly_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    parser.add_argument("--variable", default="analysed_sst")
    parser.add_argument(
        "--baseline", type=int, nargs=2, metavar=("START", "END"), default=DEFAULT_BASELINE
    )
    parser.add_argument("--percentile", type=float, default=0.90)
    parser.add_argument("--smooth-window-hours", type=int, default=120)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--validation-points", type=int, default=512)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--vram-safety", type=float, default=0.15)
    parser.add_argument(
        "--vram-safety-values", type=float, nargs="+", default=DEFAULT_VRAM_SAFETY_VALUES
    )
    parser.add_argument("--quick", action="store_true", help="Use a compact smoke-test matrix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    root = _timestamped_run_directory(args.output_root)
    _run_campaign(args, root)
    print(f"Benchmark artifacts: {root}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup cannot be negative")
    if args.validation_points < 1 or args.max_points is not None and args.max_points < 1:
        raise ValueError("validation-points and max-points must be positive")
    if not 0 <= args.vram_safety < 1 or any(
        not 0 <= value < 1 for value in args.vram_safety_values
    ):
        raise ValueError("VRAM safety values must be in [0, 1)")
    if args.batch_sizes and any(value < 1 for value in args.batch_sizes):
        raise ValueError("batch sizes must be positive")


def _timestamped_run_directory(root: Path) -> Path:
    """Return a never-reused UTC timestamped sibling of ``root``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = root.with_name(f"{root.name}_{stamp}")
    suffix = 2
    while candidate.exists():
        candidate = root.with_name(f"{root.name}_{stamp}_{suffix:02d}")
        suffix += 1
    return candidate


class _Progress:
    def __init__(self, groups: int) -> None:
        self.groups = groups
        self.group = 0

    def start(self, name: str, configurations: int) -> None:
        self.group += 1
        self.name, self.configurations, self.current = name, configurations, 0
        print(f"[{self.group}/{self.groups}] {name}: {configurations} configurations", flush=True)

    def configuration(self, label: str, repeats: int, warmup: int) -> None:
        self.current += 1
        print(
            f"[{self.group}/{self.groups}] {self.name} — configuration "
            f"{self.current}/{self.configurations}: {label} "
            f"({repeats} measured repeats, {warmup} warm-up)",
            flush=True,
        )


def _run_campaign(args: argparse.Namespace, root: Path) -> None:
    layout = _create_layout(root)
    repeats, warmup = (1, 0) if args.quick else (args.repeats, args.warmup)
    rows: list[dict[str, Any]] = []
    progress = _Progress(groups=5)
    common = {
        "baseline": tuple(args.baseline),
        "percentile": args.percentile,
        "smooth_window_hours": args.smooth_window_hours,
    }

    with CUDAHeat(
        args.dataset,
        variable=args.variable,
        dtype="float32",
        lazy=True,
        vram_safety=args.vram_safety,
    ) as engine:
        campaign_points = min(engine.npoints, args.max_points or engine.npoints)
        points = _point_schedule(campaign_points, quick=args.quick)
        batches = _batch_schedule(
            campaign_points, engine.batch_size, args.batch_sizes, quick=args.quick
        )

        progress.start("Scale test", len(points))
        for count in points:
            rows.extend(
                _measure(
                    engine,
                    "scale",
                    count,
                    "float32",
                    None,
                    repeats,
                    warmup,
                    common,
                    progress,
                    f"{count:,} points; automatic GPU batch",
                )
            )

        values: tuple[int | None, ...] = (None, *batches)
        progress.start("Batch-size test", len(values))
        for batch in values:
            label = "automatic GPU batch" if batch is None else f"fixed GPU batch {batch:,}"
            rows.extend(
                _measure(
                    engine,
                    "batch_size",
                    campaign_points,
                    "float32",
                    batch,
                    repeats,
                    warmup,
                    common,
                    progress,
                    label,
                )
            )

        progress.start("Float32 versus float64", 2)
        for dtype in ("float32", "float64"):
            rows.extend(
                _measure(
                    engine,
                    "precision",
                    campaign_points,
                    dtype,
                    None,
                    repeats,
                    warmup,
                    common,
                    progress,
                    f"GPU {dtype}; automatic GPU batch",
                )
            )

        progress.start("VRAM safety-margin test", len(args.vram_safety_values))
        for safety in args.vram_safety_values:
            progress.configuration(f"{safety:.0%} reserved runtime-free VRAM", repeats, warmup)
            with CUDAHeat(
                args.dataset, variable=args.variable, dtype="float32", lazy=True, vram_safety=safety
            ) as safety_engine:
                rows.extend(
                    _measure(
                        safety_engine,
                        "vram_safety",
                        campaign_points,
                        "float32",
                        None,
                        repeats,
                        warmup,
                        common,
                        None,
                        "",
                    )
                )

        progress.start("Numerical-consistency test", 2)
        validation: dict[str, Any] = {}
        for dtype in ("float32", "float64"):
            validation_points = min(args.validation_points, campaign_points)
            progress.configuration(f"GPU {dtype}; {validation_points:,} complete time series", 1, 0)
            validation[dtype] = engine.validate(
                points=validation_points,
                random_seed=17,
                gpu_dtype=dtype,
                baseline=tuple(args.baseline),
                percentile=args.percentile,
                smooth_window_hours=args.smooth_window_hours,
                validation_batch_size=64 if args.quick else 256,
            ).to_dict()
        metadata = _metadata(engine, args, campaign_points, points, batches, repeats, warmup)

    _write_artifacts(layout, metadata, rows, validation)
    _render_figures(layout["figures"], rows, validation)
    _write_readme(root, metadata)


def _measure(
    engine: CUDAHeat,
    scenario: str,
    point_count: int,
    dtype: str,
    batch_size: int | None,
    repeats: int,
    warmup: int,
    common: Mapping[str, Any],
    progress: _Progress | None,
    label: str,
) -> list[dict[str, Any]]:
    if progress:
        progress.configuration(label, repeats, warmup)
    benchmark = CUDAHeatBenchmark(engine)
    records = benchmark.run(
        repeats=repeats,
        warmup=warmup,
        point_counts=(point_count,),
        batch_sizes=(batch_size,),
        gpu_dtypes=(dtype,),
        measure_cpu=False,
        **common,
    )
    return [{"scenario": scenario, **_gpu_row(row)} for row in records_to_rows(records)]


def _gpu_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in _CPU_COLUMNS}


def _point_schedule(limit: int, *, quick: bool) -> tuple[int, ...]:
    seeds = (32, 512, limit) if quick else (32, 128, 512, 2_000, 8_000, limit)
    return tuple(sorted({item for item in seeds if item <= limit}))


def _batch_schedule(
    points: int, automatic: int, requested: list[int] | None, *, quick: bool
) -> tuple[int, ...]:
    values = requested or (
        (64, min(512, points), automatic) if quick else (64, 256, 512, 1_024, automatic)
    )
    return tuple(sorted({min(int(value), points) for value in values if value > 0}))


def _create_layout(root: Path) -> dict[str, Path]:
    """Keep every artifact directly in one timestamped run directory."""
    root.mkdir(parents=True, exist_ok=True)
    # Named aliases retain the writer's intent without creating nested folders.
    return {"root": root, "metadata": root, "raw": root, "tables": root, "figures": root}


def _metadata(
    engine: CUDAHeat,
    args: argparse.Namespace,
    points: int,
    schedule: Iterable[int],
    batches: Iterable[int],
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    import cupy as cp

    return {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_id": BENCHMARK_ID,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "description": "GPU-only benchmark of CUDAHeat's numerical preprocessing pipeline.",
        "purpose": "Measure GPU execution, required pipeline overhead, VRAM use, precision, and numerical consistency.",
        "methodology": "CPU timing is intentionally absent. CPU float64 is only an untimed numerical-validation reference.",
        "scope": "Numerical preprocessing only; marine-heatwave taxonomy and event classification are excluded.",
        "dataset_path": str(args.dataset.resolve()),
        "variable": args.variable,
        "ntime": engine.ntime,
        "dataset_npoints": engine.npoints,
        "campaign_npoints": points,
        "baseline": list(args.baseline),
        "percentile": args.percentile,
        "smooth_window_hours": args.smooth_window_hours,
        "input_strategy": "Lazy Dask-backed input for normal out-of-core operation; no lazy-versus-eager comparison is run.",
        "point_schedule": list(schedule),
        "batch_schedule": list(batches),
        "repeats": repeats,
        "warmup_runs": warmup,
        "vram_safety_default": args.vram_safety,
        "vram_safety_values": list(args.vram_safety_values),
        "gpu": engine.gpu_info.as_dict(),
        "cudaheat_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "units": _units(),
    }


def _units() -> dict[str, str]:
    return {
        "*_seconds": "seconds",
        "gpu_*_values_per_second": "values per second",
        "*_vram_bytes": "bytes",
        "safety_fraction": "fraction of runtime-free VRAM reserved",
        "npoints": "spatial points",
        "ntime": "hourly samples",
        "percentile": "fraction",
    }


def _write_artifacts(
    layout: Mapping[str, Path],
    metadata: Mapping[str, Any],
    rows: list[dict[str, Any]],
    validation: Mapping[str, Any],
) -> None:
    _write_json(
        layout["metadata"] / "run_manifest.json",
        metadata,
        "Run configuration and reproducibility context.",
        "Describe this benchmark run.",
        "Collected once before artifact rendering.",
    )
    _write_json(
        layout["raw"] / "gpu_measurements.json",
        {"records": rows},
        "Every retained GPU measurement.",
        "Provide repeat-level data without CPU performance timing.",
        "Warm-ups are excluded; each record is one complete GPU pipeline run.",
    )
    _write_csv(layout["raw"] / "gpu_measurements.csv", rows)
    summary = _summarize(
        rows, ("scenario", "npoints", "dtype", "requested_batch_size", "safety_fraction")
    )
    _write_table(
        layout,
        "performance_summary",
        summary,
        "Grouped GPU timing, throughput, and VRAM measurements.",
        "Compare benchmark configurations.",
        "Arithmetic summaries of successful repeat-level GPU runs.",
    )
    _write_table(
        layout,
        "pipeline_anatomy",
        _pipeline_anatomy(rows),
        "Full-domain pipeline stage timings.",
        "Explain where full-pipeline seconds are spent.",
        "Mean of automatic-batch float32 scale-test records.",
    )
    _write_table(
        layout,
        "vram_safety_summary",
        [row for row in summary if row["scenario"] == "vram_safety"],
        "VRAM safety-margin results.",
        "Show automatic batching's safety/performance trade-off.",
        "Grouped successful GPU-only runs.",
    )
    _write_table(
        layout,
        "precision_summary",
        [row for row in summary if row["scenario"] == "precision"],
        "Float32 and float64 GPU comparison.",
        "Compare GPU precision cost without CPU timing.",
        "Grouped successful full-domain GPU runs.",
    )
    _write_table(
        layout,
        "repeatability",
        _repeatability(rows),
        "Repeat-level GPU pipeline variability.",
        "Quantify measurement stability.",
        "Coefficient of variation across successful repeated configurations.",
    )
    _write_table(
        layout,
        "numerical_consistency",
        _validation_rows(validation),
        "GPU versus untimed CPU-reference numerical agreement.",
        "Verify scientific correctness, not CPU performance.",
        "Fixed seed 17; complete time series are never split.",
    )


def _write_table(
    layout: Mapping[str, Path],
    name: str,
    rows: list[dict[str, Any]],
    description: str,
    purpose: str,
    methodology: str,
) -> None:
    _write_json(
        layout["tables"] / f"{name}.json", {"records": rows}, description, purpose, methodology
    )
    _write_csv(layout["tables"] / f"{name}.csv", rows)


def _write_json(
    path: Path, body: Mapping[str, Any], description: str, purpose: str, methodology: str
) -> None:
    payload = {
        "description": description,
        "purpose": purpose,
        "methodology": methodology,
        "units": _units(),
        **body,
    }
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: Iterable[Mapping[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "complete":
            groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    output = []
    metrics = (
        *_TIMINGS,
        "peak_used_vram_bytes",
        "batch_size",
        "oom_retries",
        "gpu_compute_values_per_second",
        "gpu_end_to_end_values_per_second",
    )
    for key, items in sorted(
        groups.items(), key=lambda pair: tuple(str(value) for value in pair[0])
    ):
        summary = dict(zip(keys, key, strict=True))
        summary["samples"] = len(items)
        for metric in metrics:
            values = [float(row[metric]) for row in items if row.get(metric) is not None]
            if values:
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_min"] = min(values)
                summary[f"{metric}_max"] = max(values)
                summary[f"{metric}_standard_deviation"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
        output.append(summary)
    return output


def _pipeline_anatomy(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("scenario") == "scale"
        and row.get("dtype") == "float32"
        and row.get("requested_batch_size") is None
        and row.get("status") == "complete"
    ]
    if not candidates:
        return []
    points = max(row["npoints"] for row in candidates)
    chosen = [row for row in candidates if row["npoints"] == points]
    components = (
        ("Host read", "host_read_seconds"),
        ("Host preparation", "host_prepare_seconds"),
        ("GPU allocation", "gpu_allocation_seconds"),
        ("GPU metadata transfer", "gpu_metadata_transfer_seconds"),
        ("Host to GPU transfer", "transfer_to_device_seconds"),
        ("GPU compute", "gpu_compute_seconds"),
        ("GPU to host transfer", "transfer_from_device_seconds"),
        ("Result assembly", "result_assembly_seconds"),
    )
    values = [
        (name, float(np.mean([row.get(field, 0.0) or 0.0 for row in chosen])))
        for name, field in components
    ]
    pipeline = float(np.mean([row["pipeline_wall_seconds"] for row in chosen]))
    values.append(("Unattributed pipeline overhead", pipeline - sum(value for _, value in values)))
    return [
        {"npoints": points, "component": name, "seconds": value, "pipeline_wall_seconds": pipeline}
        for name, value in values
    ]


def _repeatability(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        if row.get("status") == "complete":
            key = (
                row["scenario"],
                row["npoints"],
                row["dtype"],
                row["requested_batch_size"],
                row["safety_fraction"],
            )
            grouped.setdefault(key, []).append(float(row["pipeline_wall_seconds"]))
    output = []
    for key, values in grouped.items():
        average = float(np.mean(values))
        variation = (
            float(np.std(values, ddof=1) / average * 100) if len(values) > 1 and average else 0.0
        )
        output.append(
            {
                "scenario": key[0],
                "npoints": key[1],
                "dtype": key[2],
                "requested_batch_size": key[3],
                "safety_fraction": key[4],
                "samples": len(values),
                "pipeline_wall_seconds_mean": average,
                "pipeline_wall_seconds_coefficient_of_variation_percent": variation,
            }
        )
    return output


def _validation_rows(validation: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dtype, result in validation.items():
        common = {
            "gpu_dtype": dtype,
            "reference_dtype": result["cpu_dtype"],
            "validation_points": len(result["point_indices"]),
            "ntime": result["ntime"],
            "baseline": result["baseline"],
            "percentile": result["percentile"],
            "smooth_window_hours": result["smooth_window_hours"],
        }
        for field, metrics in result["float_fields"].items():
            rows.append({**common, "record_type": "float_field", "field": field, **metrics})
        rows.append({**common, "record_type": "mask", "field": "exceedance", **result["mask"]})
        rows.append(
            {
                **common,
                "record_type": "count",
                "field": "exceedance_count",
                **result["exceedance_count"],
            }
        )
    return rows


def _render_figures(
    output: Path, rows: list[dict[str, Any]], validation: Mapping[str, Any]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    summary = _summarize(
        rows, ("scenario", "npoints", "dtype", "requested_batch_size", "safety_fraction")
    )
    scale = sorted(
        (row for row in summary if row["scenario"] == "scale"), key=lambda row: row["npoints"]
    )
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for field, label in (
        ("gpu_compute_seconds", "GPU compute"),
        ("gpu_end_to_end_seconds", "GPU device end-to-end"),
        ("pipeline_wall_seconds", "Pipeline wall"),
    ):
        ax.plot(
            [row["npoints"] for row in scale],
            [row[f"{field}_mean"] for row in scale],
            "o-",
            label=label,
        )
    ax.set(xlabel="Spatial points", ylabel="Seconds", title="GPU pipeline scaling")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, output, "01_gpu_pipeline_scaling")
    batch = sorted(
        (row for row in summary if row["scenario"] == "batch_size"),
        key=lambda row: row["batch_size_mean"],
    )
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    twin = ax.twinx()
    x = [row["batch_size_mean"] for row in batch]
    ax.plot(
        x,
        [row["pipeline_wall_seconds_mean"] for row in batch],
        "o-",
        color="#007f82",
        label="Pipeline wall",
    )
    twin.plot(
        x,
        [row["peak_used_vram_bytes_mean"] / 2**30 for row in batch],
        "s-",
        color="#7851a9",
        label="Peak VRAM",
    )
    ax.set(
        xlabel="Actual GPU batch size [points]",
        ylabel="Pipeline seconds",
        title="Batch-size trade-off",
    )
    twin.set_ylabel("Peak VRAM [GiB]")
    ax.grid(alpha=0.25)
    _save(fig, output, "02_batch_size_tradeoff")
    anatomy = _pipeline_anatomy(rows)
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    bottom = 0.0
    for row in anatomy:
        ax.bar(["Full domain"], [row["seconds"]], bottom=bottom, label=row["component"])
        bottom += row["seconds"]
    ax.set(ylabel="Seconds", title="Anatomy of the complete processing pipeline")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    _save(fig, output, "03_pipeline_anatomy")
    safety = sorted(
        (row for row in summary if row["scenario"] == "vram_safety"),
        key=lambda row: row["safety_fraction"],
    )
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    twin = ax.twinx()
    x = [100 * row["safety_fraction"] for row in safety]
    ax.plot(x, [row["pipeline_wall_seconds_mean"] for row in safety], "o-")
    twin.plot(x, [row["batch_size_mean"] for row in safety], "s-", color="#d05a33")
    ax.set(
        xlabel="Reserved runtime-free VRAM [%]",
        ylabel="Pipeline seconds",
        title="VRAM safety-margin behaviour",
    )
    twin.set_ylabel("Actual batch size [points]")
    ax.grid(alpha=0.25)
    _save(fig, output, "04_vram_safety_margin")
    precision = [row for row in summary if row["scenario"] == "precision"]
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    labels = [row["dtype"] for row in precision]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(
        x - width / 2,
        [row["gpu_compute_seconds_mean"] for row in precision],
        width,
        label="GPU compute",
    )
    ax.bar(
        x + width / 2,
        [row["pipeline_wall_seconds_mean"] for row in precision],
        width,
        label="Pipeline wall",
    )
    ax.set(xticks=x, xticklabels=labels, ylabel="Seconds", title="Float32 versus float64 GPU cost")
    ax.legend()
    _save(fig, output, "05_precision_performance")
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for dtype, result in validation.items():
        fields = result["float_fields"]
        ax.plot(
            list(fields),
            [max(metrics["rmse"] or 0.0, np.finfo(float).tiny) for metrics in fields.values()],
            "o-",
            label=f"GPU {dtype}",
        )
    ax.set_yscale("log")
    ax.set(
        xlabel="Field",
        ylabel="RMSE [temperature units]",
        title="Numerical consistency with float64 reference",
    )
    ax.legend()
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.25)
    _save(fig, output, "06_numerical_consistency")


def _save(fig: Any, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    fig.clear()


def _write_readme(root: Path, metadata: Mapping[str, Any]) -> None:
    (root / "README.md").write_text(
        f"""# CUDAHeat GPU Pipeline Benchmark

This timestamped campaign contains GPU-only performance measurements, GPU
precision comparisons, VRAM/batch behaviour, a complete pipeline anatomy, and
untimed numerical validation. CPU performance timing, Dask chunk sensitivity,
and lazy-versus-eager comparisons are deliberately absent.

- Dataset: `{metadata["dataset_path"]}`
- Scope: {metadata["scope"]}
- Raw repeat measurements: `gpu_measurements.csv` and `.json`
- Derived tables: `*_summary.csv/json`, `pipeline_anatomy.csv/json`, and `repeatability.csv/json`
- Figures: `*.png` and `*.svg`

Every JSON file carries a concise description, purpose, methodology, and units.
""",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    main()
