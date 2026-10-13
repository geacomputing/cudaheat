#!/usr/bin/env python3
"""Presentation plots for a GPU-only CUDAHeat benchmark run.

The script reads the flat JSON artifacts made by
``benchmarks/benchmark_full_pipeline_presentation.py``.  It deliberately has
no CPU-performance, Dask-sensitivity, or lazy-versus-eager plots.

Run ``python quickAndDirty/refined_plots.py`` to plot the newest valid run, or
pass ``--results-dir benchmark_results/gpu_pipeline_YYYYMMDDTHHMMSSZ``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DPI = 400
FIG_W, FIG_H = 13.32, 7.4925  # 5328 × 2997 px; exactly 16:9
BG, WHITE, INK, MUTED = "#F7F9FB", "#FFFFFF", "#17283B", "#617287"
BLUE, TEAL, ORANGE, PURPLE = "#2475B6", "#087F8C", "#E36B45", "#7567A4"
BORDER, LIGHT = "#DDE6ED", "#B3C2D0"
IO_COLOR, H2D_COLOR, COMPUTE_COLOR, D2H_COLOR = ORANGE, PURPLE, BLUE, TEAL

# Manual full-detection timings supplied for the 22,083-point SST workload.
# They deliberately remain separate from the automated benchmark JSON because
# this is a targeted one-point batch-width sweep around the storage-tile edge.
MANUAL_STORAGE_ALIGNMENT_TIMINGS = (
    (502, 34.970),
    (503, 34.536),
    (504, 34.541),
    (505, 34.619),
    (506, 34.492),
    (507, 34.467),
    (508, 34.490),
    (509, 34.594),
    (510, 34.649),
    (511, 35.228),
    (512, 20.192),
    (513, 34.717),
    (514, 34.216),
    (515, 34.014),
    (516, 34.167),
    (517, 34.156),
    (518, 34.185),
    (519, 34.172),
    (520, 34.178),
    (521, 34.316),
    (522, 34.132),
)

RESULTS_DIR: Path | None = None
OUTPUT_DIR: Path | None = None
MANIFEST: dict[str, Any] = {}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.unicode_minus": False})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Flat gpu_pipeline timestamped result directory. Defaults to the newest valid run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where refined images are written. Defaults to the result directory.",
    )
    return parser.parse_args()


def _newest_results_dir() -> Path:
    root = REPOSITORY_ROOT / "benchmark_results"
    candidates = sorted(root.glob("gpu_pipeline_*"), reverse=True)
    for candidate in candidates:
        if (candidate / "run_manifest.json").is_file() and (
            candidate / "performance_summary.json"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "No GPU-only benchmark run found. Run "
        "benchmarks/benchmark_full_pipeline_presentation.py first, or pass --results-dir."
    )


def _path(filename: str) -> Path:
    if RESULTS_DIR is None:
        raise RuntimeError("results directory has not been configured")
    return RESULTS_DIR / filename


def records(filename: str) -> list[dict[str, Any]]:
    """Return records from a self-describing GPU-benchmark JSON artifact."""
    path = _path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(
            f"{path} is not a new GPU-only benchmark JSON file; expected a 'records' array."
        )
    if not payload["records"]:
        raise ValueError(
            f"No records in {path}; the benchmark run may have failed before measuring."
        )
    return payload["records"]


def frame(title: str, subtitle: str, number: int) -> plt.Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)
    fig.text(
        0.067,
        0.915,
        "CUDAHeat  /  GPU PIPELINE BENCHMARK",
        fontsize=11.2,
        color=TEAL,
        weight="bold",
    )
    fig.text(0.067, 0.834, title, fontsize=28, color=INK, weight="bold")
    fig.text(0.068, 0.775, subtitle, fontsize=12.2, color=MUTED)
    panel = FancyBboxPatch(
        (0.056, 0.163),
        0.888,
        0.548,
        boxstyle="round,pad=0.007,rounding_size=0.016",
        transform=fig.transFigure,
        facecolor=WHITE,
        edgecolor=BORDER,
        lw=0.85,
        zorder=0,
    )
    fig.add_artist(panel)
    # fig.text(0.932, 0.053, f"{number:02d}  /  CUDAHeat", color=MUTED, fontsize=9.5, ha="right")
    return fig


def axis(
    fig: plt.Figure,
    bounds: tuple[float, float, float, float] = (0.125, 0.267, 0.735, 0.35),
    *,
    logx: bool = False,
) -> plt.Axes:
    ax = fig.add_axes(bounds, facecolor=WHITE, zorder=2)
    if logx:
        ax.set_xscale("log")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=BORDER, lw=0.8, alpha=0.85)
    ax.tick_params(axis="both", length=0, pad=10, labelcolor=MUTED, labelsize=10)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def caption(fig: plt.Figure, takeaway: str, source: str) -> None:
    fig.text(0.068, 0.105, "THE TAKEAWAY", color=TEAL, fontsize=10.2, weight="bold")
    fig.text(0.195, 0.105, takeaway, color=INK, fontsize=11.2)
    fig.text(
        0.068,
        0.053,
        f"Source: {source}  ·  measured GPU benchmark results",
        color=MUTED,
        fontsize=9.2,
    )


def save(fig: plt.Figure, filename: str) -> None:
    if OUTPUT_DIR is None:
        raise RuntimeError("output directory has not been configured")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / filename
    fig.savefig(png, dpi=DPI, bbox_inches=None, pad_inches=0)
    # fig.savefig(png.with_suffix(".svg"), bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"Saved: {png}")


def _scenario(name: str) -> list[dict[str, Any]]:
    return [row for row in records("performance_summary.json") if row["scenario"] == name]


def _full_domain_points() -> int:
    return int(MANIFEST["campaign_npoints"])


def _time_axis_note() -> str:
    return f"{int(MANIFEST['ntime']):,} hourly values per point"


def plot_gpu_pipeline_scaling() -> None:
    data = sorted(_scenario("scale"), key=lambda row: row["npoints"])
    x = np.asarray([row["npoints"] for row in data], dtype=float)
    fig = frame(
        "GPU pipeline scaling",
        f"Compute, device, and complete-pipeline time  ·  {_time_axis_note()}",
        1,
    )
    ax = axis(fig, logx=True)
    for key, label, color, marker in (
        ("gpu_compute_seconds_mean", "GPU compute", BLUE, "o"),
        ("gpu_end_to_end_seconds_mean", "GPU device end-to-end", TEAL, "s"),
        ("pipeline_wall_seconds_mean", "Complete pipeline", ORANGE, "D"),
    ):
        ax.plot(
            x,
            [row[key] for row in data],
            color=color,
            lw=3,
            marker=marker,
            ms=7,
            mfc=WHITE,
            mew=2,
            label=label,
        )
    ax.set(
        xlim=(x[0] * 0.72, x[-1] * 1.8),
        xlabel="Spatial points processed",
        ylabel="Execution time (s)",
    )
    ax.set_xticks(x, [f"{int(value):,}" for value in x])
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.18), ncol=3, frameon=False, fontsize=10)
    fig.text(0.075, 0.672, "GPU-ONLY MEASUREMENTS", color=MUTED, fontsize=10, weight="bold")
    caption(
        fig,
        "Host I/O and transfers remain visible in the complete pipeline, beside GPU kernel time.",
        "performance_summary.json",
    )
    save(fig, "benchmark_01_gpu_pipeline_scaling.png")


def plot_adaptive_batching() -> None:
    data = sorted(_scenario("scale"), key=lambda row: row["npoints"])
    x = np.asarray([row["npoints"] for row in data], dtype=float)
    batches = np.asarray([row["batch_size_mean"] for row in data])
    count = np.ceil(x / batches).astype(int)
    fig = frame(
        "Adaptive spatial batching",
        "Full temporal extent is preserved as the spatial workload increases",
        2,
    )
    ax = axis(fig, bounds=(0.128, 0.270, 0.73, 0.340), logx=True)
    ax.plot(x, batches, color=BLUE, lw=3, marker="o", ms=8, mfc=WHITE, mew=2.2, zorder=5)
    ax.fill_between(x, 0, batches, color=BLUE, alpha=0.07)
    ax.set(
        xlim=(x[0] * 0.70, x[-1] * 1.55),
        ylim=(0, max(batches) * 1.38),
        xlabel="Total spatial points in the workload",
        ylabel="Points per GPU batch",
    )
    ax.set_xticks(x, [f"{int(value):,}" for value in x])
    for index, (points, batch, n_batches) in enumerate(zip(x, batches, count, strict=True)):
        ax.annotate(
            f"{int(batch):,} pts",
            (points, batch),
            xytext=(0, 15 if index % 2 == 0 else 28),
            textcoords="offset points",
            ha="center",
            fontsize=9.6,
            color=INK,
            weight="bold",
        )
        ax.annotate(
            f"{n_batches} batch" + ("es" if n_batches != 1 else ""),
            (points, batch),
            xytext=(0, -23),
            textcoords="offset points",
            ha="center",
            fontsize=9.1,
            color=MUTED,
        )
    fig.text(0.075, 0.671, "AUTOMATIC GPU BATCH SIZE", color=MUTED, fontsize=10, weight="bold")
    fig.text(
        0.752,
        0.671,
        f"TIME AXIS: {int(MANIFEST['ntime']):,} h",
        color=TEAL,
        fontsize=9.8,
        weight="bold",
        ha="right",
    )
    caption(
        fig,
        "Automatic batches adapt to VRAM while every selected point retains the complete hourly record.",
        "performance_summary.json",
    )
    save(fig, "benchmark_02_adaptive_batching.png")


def plot_pipeline_anatomy() -> None:
    data = records("pipeline_anatomy.json")
    fig = frame(
        "Anatomy of the processing pipeline",
        "Measured components of the full-domain automatic GPU run",
        3,
    )
    ax = axis(fig, bounds=(0.13, 0.275, 0.72, 0.335))
    bottom = 0.0
    colors = [
        IO_COLOR,
        LIGHT,
        "#B3C2D0",
        PURPLE,
        H2D_COLOR,
        COMPUTE_COLOR,
        D2H_COLOR,
        TEAL,
        "#AAB7C4",
    ]
    for row, color in zip(data, colors, strict=False):
        ax.bar(
            ["Full domain"],
            [row["seconds"]],
            bottom=bottom,
            width=0.45,
            color=color,
            label=row["component"],
            zorder=3,
        )
        bottom += row["seconds"]
    ax.set(ylabel="Execution time (s)", xlabel=f"{data[0]['npoints']:,} spatial points")
    ax.set_ylim(0, bottom * 1.16)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, 1.25),
        ncol=3,
        frameon=False,
        fontsize=8.6,
        columnspacing=1.2,
    )
    fig.text(0.075, 0.671, "COMPLETE PIPELINE WALL TIME", color=MUTED, fontsize=10, weight="bold")
    caption(
        fig,
        "The stack accounts for the observed pipeline wall time, including an explicit residual when stages do not explain every second.",
        "pipeline_anatomy.json",
    )
    save(fig, "benchmark_03_pipeline_anatomy.png")


def plot_pipeline_anatomy_by_input_size() -> None:
    """Show how major pipeline stages change as the spatial input grows."""
    raw = [
        row
        for row in records("gpu_measurements.json")
        if row["scenario"] == "scale"
        and row["status"] == "complete"
        and row["dtype"] == "float32"
        and row["requested_batch_size"] is None
    ]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in raw:
        grouped.setdefault(int(row["npoints"]), []).append(row)
    if not grouped:
        raise ValueError("No successful automatic float32 scale records were retained")

    points = sorted(grouped)
    components = (
        ("Host I/O", "host_read_seconds", IO_COLOR),
        ("Host to GPU", "transfer_to_device_seconds", H2D_COLOR),
        ("GPU compute", "gpu_compute_seconds", COMPUTE_COLOR),
        ("GPU to host", "transfer_from_device_seconds", D2H_COLOR),
    )
    values: dict[str, np.ndarray] = {}
    for label, field, _ in components:
        values[label] = np.asarray(
            [np.mean([float(row[field]) for row in grouped[point]]) for point in points]
        )
    pipeline = np.asarray(
        [
            np.mean([float(row["pipeline_wall_seconds"]) for row in grouped[point]])
            for point in points
        ]
    )
    values["Other pipeline overhead"] = np.maximum(
        pipeline - sum(values[label] for label, _, _ in components), 0.0
    )

    fig = frame(
        "Pipeline anatomy across input scale",
        "Where complete-pipeline seconds are spent as the spatial workload increases",
        9,
    )
    ax = axis(fig, bounds=(0.125, 0.275, 0.735, 0.335))
    x = np.arange(len(points))
    bottom = np.zeros(len(points))
    plotted = (*components, ("Other pipeline overhead", "", "#AAB7C4"))
    for label, _, color in plotted:
        ax.bar(x, values[label], bottom=bottom, width=0.66, color=color, label=label, zorder=3)
        bottom += values[label]
    ax.set(
        xticks=x,
        xticklabels=[f"{point:,}" for point in points],
        xlabel="Spatial points processed",
        ylabel="Complete-pipeline time (s)",
        ylim=(0, max(pipeline) * 1.18),
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.18), ncol=5, frameon=False, fontsize=9.2)
    fig.text(
        0.075, 0.671, "MAJOR STAGES, STACKED IN SECONDS", color=MUTED, fontsize=10, weight="bold"
    )
    caption(
        fig,
        "The composition makes I/O, transfers, and GPU compute directly comparable at every input size.",
        "gpu_measurements.json",
    )
    save(fig, "benchmark_09_pipeline_anatomy_by_input_size.png")


def plot_batch_tradeoff() -> None:
    data = sorted(_scenario("batch_size"), key=lambda row: row["batch_size_mean"])
    x = np.asarray([row["batch_size_mean"] for row in data])
    fig = frame(
        "Batch size: speed versus VRAM",
        "Full-domain GPU runs with fixed and automatic spatial batches",
        4,
    )
    ax = axis(fig, bounds=(0.13, 0.275, 0.63, 0.335))
    right = ax.twinx()
    ax.plot(
        x,
        [row["pipeline_wall_seconds_mean"] for row in data],
        color=ORANGE,
        marker="o",
        lw=3,
        label="Pipeline wall",
    )
    ax.plot(
        x,
        [row["gpu_compute_seconds_mean"] for row in data],
        color=BLUE,
        marker="s",
        lw=2.5,
        label="GPU compute",
    )
    right.plot(
        x,
        [row["peak_used_vram_bytes_mean"] / 2**30 for row in data],
        color=PURPLE,
        marker="D",
        lw=2.5,
        label="Peak VRAM",
    )
    ax.set(xlabel="Actual GPU batch size [points]", ylabel="Time (s)")
    right.set_ylabel("Peak VRAM (GiB)", color=PURPLE, labelpad=12)
    right.tick_params(axis="y", length=0, pad=10, labelcolor=PURPLE)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.17), ncol=3, frameon=False, fontsize=9.5)
    fig.text(0.075, 0.671, "FULL-DOMAIN OPERATING POINTS", color=MUTED, fontsize=10, weight="bold")
    caption(
        fig,
        "Batch width changes memory pressure and overhead; the full pipeline reveals the practical operating point.",
        "performance_summary.json",
    )
    save(fig, "benchmark_04_batch_size_tradeoff.png")


def plot_vram_safety() -> None:
    data = sorted(records("vram_safety_summary.json"), key=lambda row: row["safety_fraction"])
    x = np.asarray([100 * row["safety_fraction"] for row in data])
    fig = frame(
        "VRAM safety margin", "How reserved runtime-free VRAM changes automatic batching", 5
    )
    ax = axis(fig, bounds=(0.13, 0.275, 0.63, 0.335))
    right = ax.twinx()
    ax.plot(
        x,
        [row["pipeline_wall_seconds_mean"] for row in data],
        color=ORANGE,
        marker="o",
        lw=3,
        label="Pipeline wall",
    )
    right.plot(
        x,
        [row["batch_size_mean"] for row in data],
        color=BLUE,
        marker="s",
        lw=3,
        label="Automatic batch",
    )
    ax.set(xticks=x, xlabel="Reserved runtime-free VRAM (%)", ylabel="Pipeline time (s)")
    right.set_ylabel("Points per GPU batch", color=BLUE, labelpad=12)
    right.tick_params(axis="y", length=0, pad=10, labelcolor=BLUE)
    for safety, batch in zip(x, [row["batch_size_mean"] for row in data], strict=True):
        right.annotate(
            f"{batch:,.0f}",
            (safety, batch),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            color=BLUE,
            weight="bold",
            fontsize=9.5,
        )
    fig.text(0.075, 0.671, "AUTOMATIC BATCHING POLICY", color=MUTED, fontsize=10, weight="bold")
    caption(
        fig,
        "Reserving more free VRAM lowers the automatic batch size and exposes its complete-pipeline cost.",
        "vram_safety_summary.json",
    )
    save(fig, "benchmark_05_vram_safety.png")


def _validation(dtype: str, record_type: str, field: str) -> dict[str, Any]:
    for row in records("numerical_consistency.json"):
        if (
            row["gpu_dtype"] == dtype
            and row["record_type"] == record_type
            and row["field"] == field
        ):
            return row
    raise ValueError(f"Missing {record_type} validation data for GPU {dtype}: {field}")


def plot_precision_and_accuracy() -> None:
    data = {row["dtype"]: row for row in records("precision_summary.json")}
    if set(data) != {"float32", "float64"}:
        raise ValueError("precision_summary.json must contain float32 and float64 records")
    fig = frame(
        "Precision versus performance",
        "GPU float32 and float64 cost, paired with numerical consistency",
        6,
    )
    ax = axis(fig, bounds=(0.133, 0.270, 0.38, 0.34))
    labels, x = ("float32", "float64"), np.arange(2)
    times = [data[dtype]["pipeline_wall_seconds_mean"] for dtype in labels]
    bars = ax.bar(x, times, width=0.47, color=[BLUE, PURPLE], zorder=3)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, max(times) * 1.38), ylabel="Pipeline time (s)")
    for bar, value in zip(bars, times, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(times) * 0.04,
            f"{value:.1f} s",
            ha="center",
            color=INK,
            weight="bold",
            fontsize=12,
        )
    ax2 = axis(fig, bounds=(0.608, 0.270, 0.255, 0.34))
    errors = [
        _validation(dtype, "float_field", "threshold")["max_absolute_error"] for dtype in labels
    ]
    ax2.bar(
        x,
        [max(value, np.finfo(float).tiny) for value in errors],
        width=0.47,
        color=[BLUE, PURPLE],
        zorder=3,
    )
    ax2.set_yscale("log")
    ax2.set(xticks=x, xticklabels=labels, ylabel="Max |threshold error| (K)")
    for point, value in zip(x, errors, strict=True):
        ax2.text(
            point,
            max(value, np.finfo(float).tiny) * 2,
            f"{value:.1e}",
            ha="center",
            fontsize=10,
            color=INK,
            weight="bold",
        )
    fig.text(0.139, 0.669, "FULL PIPELINE", color=MUTED, fontsize=10, weight="bold")
    fig.text(0.615, 0.669, "THRESHOLD NUMERICAL ERROR", color=MUTED, fontsize=10, weight="bold")
    caption(
        fig,
        "Float32 reduces GPU pipeline cost; validation makes its numerical deviation explicit.",
        "precision_summary.json + numerical_consistency.json",
    )
    save(fig, "benchmark_06_precision_and_accuracy.png")


def plot_numerical_consistency() -> None:
    validation = records("numerical_consistency.json")
    fields = [
        "climatology_raw",
        "climatology",
        "threshold_raw",
        "threshold_cycle",
        "threshold",
        "anomaly",
        "excess",
    ]
    fig = frame(
        "Numerical consistency across fields",
        "GPU results compared with the untimed float64 reference over complete time series",
        7,
    )
    ax = axis(fig, bounds=(0.13, 0.275, 0.47, 0.335))
    x = np.arange(len(fields))
    width = 0.36
    for shift, dtype, color in ((-width / 2, "float32", BLUE), (width / 2, "float64", PURPLE)):
        values = [
            next(
                row["rmse"]
                for row in validation
                if row["gpu_dtype"] == dtype
                and row["record_type"] == "float_field"
                and row["field"] == field
            )
            for field in fields
        ]
        ax.bar(
            x + shift,
            [max(value or 0.0, np.finfo(float).tiny) for value in values],
            width,
            color=color,
            label=dtype,
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set(xticks=x, xticklabels=[name.replace("_", "\n") for name in fields], ylabel="RMSE (K)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.16), ncol=2)
    mask32, mask64 = (
        _validation("float32", "mask", "exceedance"),
        _validation("float64", "mask", "exceedance"),
    )
    count32, count64 = (
        _validation("float32", "count", "exceedance_count"),
        _validation("float64", "count", "exceedance_count"),
    )
    info = axis(fig, bounds=(0.685, 0.275, 0.18, 0.335))
    info.axis("off")
    info.text(0.02, 0.9, "MASK AGREEMENT", color=TEAL, weight="bold", fontsize=10)
    info.text(
        0.02,
        0.72,
        f"float32\n{mask32['agreement_percent']:.8f}%",
        color=INK,
        fontsize=13,
        weight="bold",
    )
    info.text(
        0.02,
        0.49,
        f"float64\n{mask64['agreement_percent']:.8f}%",
        color=INK,
        fontsize=13,
        weight="bold",
    )
    info.text(
        0.02,
        0.23,
        f"Count matches\n{count32['exact_matches']}/{count32['total_points']}  ·  {count64['exact_matches']}/{count64['total_points']}",
        color=MUTED,
        fontsize=9.5,
    )
    fig.text(0.075, 0.671, "FIELD-LEVEL ERROR", color=MUTED, fontsize=10, weight="bold")
    caption(
        fig,
        "Both GPU precisions preserve scientific outputs; float64 is near the reference numerical precision.",
        "numerical_consistency.json",
    )
    save(fig, "benchmark_07_numerical_consistency.png")


def plot_repeatability() -> None:
    data = sorted(
        records("repeatability.json"),
        key=lambda row: (row["scenario"], row["npoints"], row["dtype"]),
    )
    labels = [
        f"{row['scenario'].replace('_', ' ')}\n{row['npoints']:,} · {row['dtype']}" for row in data
    ]
    values = [row["pipeline_wall_seconds_coefficient_of_variation_percent"] for row in data]
    colors = [BLUE if row["dtype"] == "float32" else PURPLE for row in data]
    fig = frame(
        "Repeatability of GPU measurements",
        "Coefficient of variation across retained repeated configurations",
        8,
    )
    ax = axis(fig, bounds=(0.13, 0.275, 0.72, 0.335))
    x = np.arange(len(data))
    ax.bar(x, values, color=colors, zorder=3)
    ax.set(xticks=x, xticklabels=labels, ylabel="Pipeline-time coefficient of variation (%)")
    ax.tick_params(axis="x", labelsize=7.5, labelrotation=45)
    ax.set_ylim(0, max(values, default=0.1) * 1.35)
    fig.text(0.075, 0.671, "REPEATED GPU RUNS", color=MUTED, fontsize=10, weight="bold")
    caption(
        fig,
        "Reported timing distributions retain every repeat; this view makes run-to-run variation visible.",
        "repeatability.json",
    )
    save(fig, "benchmark_08_repeatability.png")


def plot_storage_tile_alignment() -> None:
    """Show the manually observed performance discontinuity at batch 512."""
    batches = np.asarray([batch for batch, _ in MANUAL_STORAGE_ALIGNMENT_TIMINGS])
    times = np.asarray([seconds for _, seconds in MANUAL_STORAGE_ALIGNMENT_TIMINGS])
    fastest_index = int(np.argmin(times))
    fastest_batch = int(batches[fastest_index])
    fastest_time = float(times[fastest_index])
    nearby_times = np.delete(times, fastest_index)
    nearby_median = float(np.median(nearby_times))
    speedup_percent = 100 * (nearby_median - fastest_time) / nearby_median

    fig = frame(
        "The 512-point storage-alignment sweet spot",
        "Manual full-detection timing sweep over consecutive, discrete batch sizes",
        11,
    )
    ax = axis(fig, bounds=(0.115, 0.270, 0.770, 0.345))
    ax.plot(batches, times, color=LIGHT, lw=2.0, zorder=2)
    ax.scatter(batches, times, s=42, color=BLUE, edgecolor=WHITE, linewidth=1.0, zorder=3)
    ax.scatter(
        [fastest_batch],
        [fastest_time],
        s=160,
        color=TEAL,
        edgecolor=WHITE,
        linewidth=2.4,
        zorder=5,
        label="Fastest measured run",
    )
    ax.axvline(fastest_batch, color=TEAL, lw=1.4, ls=(0, (3, 3)), alpha=0.8, zorder=1)
    ax.axhline(nearby_median, color=MUTED, lw=1.1, ls=(0, (4, 4)), alpha=0.8, zorder=1)
    ax.set(
        xlim=(batches[0] - 0.65, batches[-1] + 0.65),
        ylim=(fastest_time * 0.82, max(times) * 1.08),
        xticks=batches,
        xlabel="Requested GPU batch size (spatial points; every value shown)",
        ylabel="Complete detection time (s)",
    )
    ax.tick_params(axis="x", labelsize=7.5, pad=7)
    for label, batch in zip(ax.get_xticklabels(), batches, strict=True):
        if batch == fastest_batch:
            label.set(color=TEAL, weight="bold")

    ax.annotate(
        "Exact storage-tile match\n512 points · 20.2 s",
        xy=(fastest_batch, fastest_time),
        xytext=(515.2, 24.2),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.6},
        bbox={"boxstyle": "round,pad=0.42", "fc": WHITE, "ec": TEAL, "lw": 1.1},
        color=INK,
        fontsize=10,
        weight="bold",
        va="center",
    )
    ax.text(
        0.985,
        0.955,
        f"Median of surrounding runs: {nearby_median:.1f} s",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=MUTED,
        fontsize=9.4,
    )
    ax.legend(loc="lower left", frameon=False, fontsize=9.2)
    fig.text(
        0.075,
        0.671,
        "MANUAL DISCRETE BATCH-SIZE SWEEP",
        color=MUTED,
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.922,
        0.671,
        f"{speedup_percent:.0f}% FASTER AT 512",
        color=TEAL,
        fontsize=10,
        weight="bold",
        ha="right",
    )
    caption(
        fig,
        "At 512 points, the GPU batch matches the NetCDF file's storage tile; adjacent sizes are much slower.",
        "manual full-detection timing sweep, batch sizes 502–522",
    )
    save(fig, "benchmark_11_storage_tile_alignment.png")


def _dashboard_metric(fig: plt.Figure, x: float, value: str, label: str) -> None:
    box = FancyBboxPatch(
        (x, 0.618),
        0.150,
        0.074,
        boxstyle="round,pad=0.007,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor=BG,
        edgecolor=BORDER,
        lw=0.8,
        zorder=2,
    )
    fig.add_artist(box)
    fig.text(x + 0.013, 0.659, value, fontsize=16, color=TEAL, weight="bold", va="center")
    fig.text(x + 0.013, 0.633, label, fontsize=8.6, color=MUTED, va="center")


def plot_cudaheat_dashboard() -> None:
    """Summarise full-domain performance, bottlenecks, and numerical confidence."""
    scale = _scenario("scale")
    full = max(scale, key=lambda row: row["npoints"])
    anatomy = records("pipeline_anatomy.json")
    precision = {row["dtype"]: row for row in records("precision_summary.json")}
    mask = _validation("float32", "mask", "exceedance")
    io_seconds = next(row["seconds"] for row in anatomy if row["component"] == "Host read")
    pipeline_seconds = full["pipeline_wall_seconds_mean"]
    io_share = 100 * io_seconds / pipeline_seconds

    fig = frame(
        "CUDAHeat benchmark dashboard",
        "Full-domain GPU execution, bottlenecks, and numerical confidence",
        10,
    )
    metric_x = (0.075, 0.250, 0.425, 0.600, 0.775)
    metrics = (
        (f"{full['npoints']:,}", "SPATIAL POINTS"),
        (f"{pipeline_seconds:.1f} s", "FULL PIPELINE"),
        (f"{full['gpu_compute_seconds_mean']:.1f} s", "GPU COMPUTE"),
        (f"{io_share:.0f}%", "PIPELINE IN HOST I/O"),
        (f"{mask['agreement_percent']:.5f}%", "FLOAT32 MASK AGREEMENT"),
    )
    for x, (value, label) in zip(metric_x, metrics, strict=True):
        _dashboard_metric(fig, x, value, label)

    anatomy_seconds = {row["component"]: float(row["seconds"]) for row in anatomy}
    time_shares = (
        ("Host I/O", anatomy_seconds["Host read"], IO_COLOR),
        ("GPU compute", anatomy_seconds["GPU compute"], COMPUTE_COLOR),
        (
            "Transfers & overhead",
            max(
                pipeline_seconds - anatomy_seconds["Host read"] - anatomy_seconds["GPU compute"], 0
            ),
            LIGHT,
        ),
    )
    fig.text(0.105, 0.565, "WHERE FULL-PIPELINE TIME GOES", color=MUTED, fontsize=10, weight="bold")
    donut = fig.add_axes((0.075, 0.225, 0.250, 0.320), facecolor=WHITE, zorder=2)
    donut.pie(
        [seconds for _, seconds, _ in time_shares],
        colors=[color for _, _, color in time_shares],
        startangle=90,
        counterclock=False,
        radius=1.10,
        wedgeprops={"width": 0.26, "edgecolor": WHITE, "linewidth": 2},
    )
    donut.text(
        0,
        0.08,
        f"{pipeline_seconds:.1f} s",
        ha="center",
        va="center",
        fontsize=14,
        color=INK,
        weight="bold",
    )
    donut.text(
        0,
        -0.15,
        "FULL PIPELINE",
        ha="center",
        va="center",
        fontsize=7.4,
        color=MUTED,
        weight="bold",
    )
    donut.set(aspect="equal")
    for spine in donut.spines.values():
        spine.set_visible(False)

    for index, (label, seconds, color) in enumerate(time_shares):
        y = 0.468 - index * 0.066
        share = 100 * seconds / pipeline_seconds
        fig.text(0.325, y, "■", color=color, fontsize=13, va="center")
        fig.text(0.346, y, label, color=INK, fontsize=9.5, va="center")
        fig.text(
            0.535,
            y,
            f"{seconds:.1f} s  ·  {share:.0f}%",
            color=MUTED,
            fontsize=9.2,
            ha="right",
            va="center",
        )

    fig.text(0.625, 0.565, "PRECISION OPERATING POINT", color=MUTED, fontsize=10, weight="bold")
    right = axis(fig, bounds=(0.645, 0.365, 0.250, 0.135))
    labels = ("float32", "float64")
    times = [precision[dtype]["pipeline_wall_seconds_mean"] for dtype in labels]
    bar_y = [0.66, 0.34]
    bars = right.barh(bar_y, times, color=[BLUE, PURPLE], height=0.16, zorder=3)
    right.set(
        yticks=bar_y,
        yticklabels=labels,
        xlabel="Full-pipeline time (s)",
        xlim=(0, max(times) * 1.40),
        ylim=(0.16, 0.84),
    )
    right.xaxis.labelpad = 8
    right.tick_params(axis="y", labelsize=10)
    right.tick_params(axis="x", labelsize=12)
    right.bar_label(
        bars,
        labels=[f"{value:.1f} s" for value in times],
        padding=5,
        color=INK,
        fontsize=9.5,
        weight="bold",
    )

    seconds_saved = times[1] - times[0]
    speed_gain = 100 * seconds_saved / times[1]
    fig.text(0.645, 0.224, f"{seconds_saved:.1f} s saved", color=INK, fontsize=13, weight="bold")
    fig.text(
        0.645,
        0.198,
        f"{speed_gain:.0f}% lower pipeline time with float32",
        color=MUTED,
        fontsize=9.5,
    )

    caption(
        fig,
        "The full-domain result is I/O-bound; float32 offers the lower-cost operating point while retaining near-perfect mask agreement.",
        "performance_summary.json + pipeline_anatomy.json + numerical_consistency.json",
    )
    save(fig, "benchmark_10_cudaheat_dashboard.png")


PLOT_REGISTRY = {
    "gpu_pipeline_scaling": plot_gpu_pipeline_scaling,
    "adaptive_batching": plot_adaptive_batching,
    "pipeline_anatomy": plot_pipeline_anatomy,
    "pipeline_anatomy_by_input_size": plot_pipeline_anatomy_by_input_size,
    "batch_tradeoff": plot_batch_tradeoff,
    "vram_safety": plot_vram_safety,
    "precision_and_accuracy": plot_precision_and_accuracy,
    "numerical_consistency": plot_numerical_consistency,
    "repeatability": plot_repeatability,
    "cudaheat_dashboard": plot_cudaheat_dashboard,
    "storage_tile_alignment": plot_storage_tile_alignment,
}


def main() -> None:
    global MANIFEST, OUTPUT_DIR, RESULTS_DIR
    args = parse_args()
    RESULTS_DIR = (args.results_dir or _newest_results_dir()).resolve()
    OUTPUT_DIR = (args.output_dir or RESULTS_DIR).resolve()
    manifest_path = _path("run_manifest.json")
    MANIFEST = json.loads(manifest_path.read_text(encoding="utf-8"))
    if MANIFEST.get("benchmark_id") != "gpu_pipeline":
        raise ValueError(f"{RESULTS_DIR} is not a GPU-only benchmark directory")
    print(f"Reading: {RESULTS_DIR}")
    for plot in PLOT_REGISTRY.values():
        plot()


if __name__ == "__main__":
    main()
