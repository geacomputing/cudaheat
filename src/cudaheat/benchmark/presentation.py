"""High-resolution, presentation-quality plots from measured CUDAHeat records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import BenchmarkRecord


_COLORS = {
    "cpu": "#3b4a6b",
    "compute": "#007f82",
    "end_to_end": "#d05a33",
    "h2d": "#4c78a8",
    "d2h": "#9ecae9",
    "float64": "#7851a9",
    "float32": "#007f82",
}


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _figure(*, panels: int = 1, width: float = 12, height: float = 7):
    plt = _plt()
    figure, axes = plt.subplots(1, panels, figsize=(width, height), dpi=160, layout="constrained")
    return figure, axes


def _style(axis, *, xlabel: str, ylabel: str, title: str) -> None:
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.tick_params(labelsize=12)
    axis.xaxis.label.set_size(15)
    axis.yaxis.label.set_size(15)
    axis.title.set_size(19)
    axis.grid(True, alpha=0.22, linewidth=0.7)


def _actual(records: Iterable[BenchmarkRecord]) -> list[BenchmarkRecord]:
    return [record for record in records if record.status == "complete"]


def _automatic_float32(records: Iterable[BenchmarkRecord]) -> list[BenchmarkRecord]:
    return [
        record
        for record in _actual(records)
        if record.dtype == "float32" and record.requested_batch_size is None
    ]


def _mean_rows(records: Iterable[BenchmarkRecord], key: str) -> list[dict[str, float]]:
    groups: dict[int, list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        groups[record.npoints].append(record)
    rows: list[dict[str, float]] = []
    for points, grouped in sorted(groups.items()):
        rows.append(
            {
                "points": float(points),
                "cpu": float(np.mean([record.cpu_compute_seconds for record in grouped if record.cpu_compute_seconds is not None])),
                "compute": float(np.mean([record.gpu_compute_seconds for record in grouped])),
                "end_to_end": float(np.mean([record.gpu_end_to_end_seconds for record in grouped])),
                "pipeline": float(
                    np.mean([record.pipeline_wall_seconds for record in grouped if record.pipeline_wall_seconds])
                ),
                "h2d": float(np.mean([record.transfer_to_device_seconds for record in grouped])),
                "d2h": float(np.mean([record.transfer_from_device_seconds for record in grouped])),
                "throughput_compute": float(
                    np.mean([record.gpu_compute_values_per_second for record in grouped]) / 1e6
                ),
                "throughput_end_to_end": float(
                    np.mean([record.gpu_end_to_end_values_per_second for record in grouped]) / 1e6
                ),
            }
        )
    return rows


def _metadata_box(axis, metadata: Mapping[str, Any] | None) -> None:
    if not metadata:
        return
    gpu_name = metadata.get("gpu_name") or metadata.get("gpu", {}).get("name")
    text = "\n".join(
        part
        for part in (
            gpu_name,
            f"{metadata.get('ntime', '—'):,} hourly samples / point" if metadata.get("ntime") else None,
            f"CPU {metadata.get('cpu_dtype', 'float64')} | GPU float32",
            f"{metadata.get('percentile', 0.90):.0%} percentile" if metadata.get("percentile") else None,
            f"baseline {metadata.get('baseline')}" if metadata.get("baseline") else None,
            f"{metadata.get('smooth_window_hours')} h smoothing" if metadata.get("smooth_window_hours") else None,
        )
        if part
    )
    axis.text(
        0.02,
        0.98,
        text,
        transform=axis.transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.9, "pad": 6},
    )


def plot_execution_time(records: Iterable[BenchmarkRecord], metadata: Mapping[str, Any] | None = None):
    """CPU, GPU compute, device E2E, and complete pipeline time by data scale."""
    rows = _mean_rows(_automatic_float32(records), "points")
    figure, axis = _figure()
    points = [row["points"] for row in rows]
    for name, label, color, marker in (
        ("cpu", "CPU float64 compute", _COLORS["cpu"], "o"),
        ("compute", "GPU float32 compute", _COLORS["compute"], "s"),
        ("end_to_end", "GPU device end-to-end", _COLORS["end_to_end"], "^") ,
        ("pipeline", "GPU pipeline wall", "#b24a70", "D"),
    ):
        axis.plot(points, [row[name] for row in rows], marker=marker, linewidth=2.2, color=color, label=label)
    _style(axis, xlabel="Dataset size [spatial points]", ylabel="Execution time [s]", title="CPU and GPU execution time")
    axis.legend(fontsize=12, frameon=True)
    _metadata_box(axis, metadata)
    return figure, axis


def plot_speedup(records: Iterable[BenchmarkRecord], metadata: Mapping[str, Any] | None = None):
    """Compute-only, device E2E, and pipeline speedups from actual measurements."""
    rows = _mean_rows(_automatic_float32(records), "points")
    figure, axis = _figure()
    points = [row["points"] for row in rows]
    compute = [row["cpu"] / row["compute"] for row in rows]
    end_to_end = [row["cpu"] / row["end_to_end"] for row in rows]
    pipeline = [row["cpu"] / row["pipeline"] for row in rows]
    axis.plot(points, compute, "o-", color=_COLORS["compute"], linewidth=2.2, label="compute-only")
    axis.plot(points, end_to_end, "s-", color=_COLORS["end_to_end"], linewidth=2.2, label="device end-to-end")
    axis.plot(points, pipeline, "D-", color="#b24a70", linewidth=2.2, label="pipeline wall")
    axis.axhline(1, color="black", linewidth=1, linestyle="--", label="1× parity")
    if compute:
        index = int(np.argmax(compute))
        axis.annotate(
            f"max {compute[index]:.1f}×",
            xy=(points[index], compute[index]),
            xytext=(8, 14),
            textcoords="offset points",
            fontsize=12,
            arrowprops={"arrowstyle": "->", "color": _COLORS["compute"]},
        )
    _style(axis, xlabel="Dataset size [spatial points]", ylabel="CPU / GPU speedup [×]", title="Measured speedup")
    axis.legend(fontsize=12)
    _metadata_box(axis, metadata)
    return figure, axis


def plot_time_breakdown(records: Iterable[BenchmarkRecord]):
    """Stacked H2D, compute, and D2H times by spatial dataset size."""
    rows = _mean_rows(_automatic_float32(records), "points")
    figure, axis = _figure()
    points = np.asarray([row["points"] for row in rows])
    h2d = np.asarray([row["h2d"] for row in rows])
    compute = np.asarray([row["compute"] for row in rows])
    d2h = np.asarray([row["d2h"] for row in rows])
    axis.bar(points, h2d, color=_COLORS["h2d"], label="CPU → GPU")
    axis.bar(points, compute, bottom=h2d, color=_COLORS["compute"], label="GPU compute")
    axis.bar(points, d2h, bottom=h2d + compute, color=_COLORS["d2h"], label="GPU → CPU")
    _style(axis, xlabel="Dataset size [spatial points]", ylabel="Time [s]", title="GPU device time breakdown")
    axis.legend(fontsize=12)
    return figure, axis


def plot_throughput(records: Iterable[BenchmarkRecord]):
    """Compute and device E2E throughput in million values per second."""
    rows = _mean_rows(_automatic_float32(records), "points")
    figure, axis = _figure()
    points = [row["points"] for row in rows]
    axis.plot(points, [row["throughput_compute"] for row in rows], "o-", linewidth=2.2, color=_COLORS["compute"], label="GPU compute")
    axis.plot(points, [row["throughput_end_to_end"] for row in rows], "s-", linewidth=2.2, color=_COLORS["end_to_end"], label="GPU device end-to-end")
    _style(axis, xlabel="Dataset size [spatial points]", ylabel="Throughput [million values s⁻¹]", title="GPU throughput")
    axis.legend(fontsize=12)
    return figure, axis


def _full_batch_rows(records: Iterable[BenchmarkRecord]) -> list[dict[str, float]]:
    valid = _actual(records)
    if not valid:
        return []
    full_points = max(record.npoints for record in valid)
    grouped: dict[int, list[BenchmarkRecord]] = defaultdict(list)
    for record in valid:
        if record.npoints == full_points and record.dtype == "float32":
            grouped[record.batch_size].append(record)
    rows = []
    for batch, items in sorted(grouped.items()):
        rows.append(
            {
                "batch": float(batch),
                "speedup": float(np.mean([record.end_to_end_speedup for record in items if record.end_to_end_speedup])),
                "vram": float(np.mean([record.peak_used_vram_bytes for record in items]) / 2**30),
                "h2d": float(np.mean([record.transfer_to_device_seconds for record in items])),
                "compute": float(np.mean([record.gpu_compute_seconds for record in items])),
                "d2h": float(np.mean([record.transfer_from_device_seconds for record in items])),
                "automatic": any(record.requested_batch_size is None for record in items),
            }
        )
    # Automatic sizing is measured in multiple campaign phases.  Small changes
    # in free VRAM can yield adjacent observed sizes (for example, 2275/2276),
    # which should be shown as one adaptive configuration rather than as
    # overlapping annotations.
    automatic = [row for row in rows if row["automatic"]]
    fixed = [row for row in rows if not row["automatic"]]
    if automatic:
        fixed.append(
            {
                key: float(np.mean([row[key] for row in automatic]))
                for key in ("batch", "speedup", "vram", "h2d", "compute", "d2h")
            }
            | {"automatic": True}
        )
    return sorted(fixed, key=lambda row: row["batch"])


def plot_speedup_batch_size(records: Iterable[BenchmarkRecord]):
    """Device end-to-end speedup by actual full-dataset batch size."""
    rows = _full_batch_rows(records)
    figure, axis = _figure()
    axis.plot([row["batch"] for row in rows], [row["speedup"] for row in rows], "o-", linewidth=2.2, color=_COLORS["end_to_end"])
    for row in rows:
        if row["automatic"]:
            axis.annotate("automatic", (row["batch"], row["speedup"]), xytext=(8, 10), textcoords="offset points", fontsize=11)
    _style(axis, xlabel="Batch size [spatial points]", ylabel="Device end-to-end speedup [×]", title="Batch-size scaling")
    return figure, axis


def plot_vram_batch_size(records: Iterable[BenchmarkRecord]):
    """Observed peak VRAM against full-dataset batch size."""
    rows = _full_batch_rows(records)
    figure, axis = _figure()
    axis.plot([row["batch"] for row in rows], [row["vram"] for row in rows], "o-", linewidth=2.2, color="#7851a9")
    _style(axis, xlabel="Batch size [spatial points]", ylabel="Peak GPU memory [GiB]", title="Observed VRAM usage")
    return figure, axis


def plot_transfer_compute_batch_size(records: Iterable[BenchmarkRecord]):
    """H2D, compute, and D2H by actual full-dataset batch size."""
    rows = _full_batch_rows(records)
    figure, axis = _figure()
    batches = [row["batch"] for row in rows]
    axis.plot(batches, [row["h2d"] for row in rows], "o-", color=_COLORS["h2d"], linewidth=2.2, label="CPU → GPU")
    axis.plot(batches, [row["compute"] for row in rows], "s-", color=_COLORS["compute"], linewidth=2.2, label="GPU compute")
    axis.plot(batches, [row["d2h"] for row in rows], "^-", color=_COLORS["d2h"], linewidth=2.2, label="GPU → CPU")
    _style(axis, xlabel="Batch size [spatial points]", ylabel="Time [s]", title="Transfer and compute by batch size")
    axis.legend(fontsize=12)
    return figure, axis


def plot_dtype_performance(records: Iterable[BenchmarkRecord]):
    """Float32 versus float64 GPU compute and device E2E measurements."""
    valid = [record for record in _actual(records) if record.requested_batch_size is None]
    grouped: dict[tuple[int, str], list[BenchmarkRecord]] = defaultdict(list)
    for record in valid:
        grouped[(record.npoints, record.dtype)].append(record)
    points = sorted({key[0] for key in grouped})
    figure, axes = _figure(panels=2, width=15, height=7)
    for dtype, color in (("float32", _COLORS["float32"]), ("float64", _COLORS["float64"])):
        available = [point for point in points if (point, dtype) in grouped]
        axes[0].plot(available, [np.mean([r.gpu_compute_seconds for r in grouped[(p, dtype)]]) for p in available], "o-", color=color, linewidth=2.2, label=dtype)
        axes[1].plot(available, [np.mean([r.gpu_end_to_end_seconds for r in grouped[(p, dtype)]]) for p in available], "o-", color=color, linewidth=2.2, label=dtype)
    _style(axes[0], xlabel="Dataset size [spatial points]", ylabel="GPU compute [s]", title="GPU compute precision")
    _style(axes[1], xlabel="Dataset size [spatial points]", ylabel="GPU device E2E [s]", title="GPU end-to-end precision")
    axes[0].legend(fontsize=12)
    axes[1].legend(fontsize=12)
    return figure, axes


def plot_repeated_variability(records: Iterable[BenchmarkRecord]):
    """Coefficient of variation of repeated device end-to-end measurements."""
    valid = _actual(records)
    groups: dict[tuple[int, str, int | None, int], list[float]] = defaultdict(list)
    for record in valid:
        groups[(record.npoints, record.dtype, record.requested_batch_size, record.batch_size)].append(
            record.gpu_end_to_end_seconds
        )
    figure, axis = _figure()
    for dtype, color, marker, label in (
        ("float32", _COLORS["float32"], "o", "float32"),
        ("float64", _COLORS["float64"], "s", "float64"),
    ):
        points = []
        variability = []
        for (_, group_dtype, _, _), values in groups.items():
            if group_dtype != dtype or len(values) < 2:
                continue
            average = float(np.mean(values))
            points.append(average)
            variability.append(float(np.std(values, ddof=1) / average * 100))
        if points:
            axis.scatter(points, variability, s=46, color=color, marker=marker, alpha=0.85, label=label)
    axis.set_xscale("log")
    _style(
        axis,
        xlabel="Mean GPU device end-to-end time [s] (log scale)",
        ylabel="Repeat coefficient of variation [%]",
        title="Repeated-run variability by configuration",
    )
    axis.legend(title="GPU precision", fontsize=12, title_fontsize=12)
    return figure, axis


def plot_speedup_heatmap(records: Iterable[BenchmarkRecord]):
    """Observed mean speedup grid; missing configurations are deliberately blank."""
    valid = [record for record in _actual(records) if record.end_to_end_speedup is not None]
    points = sorted({record.npoints for record in valid})
    batches = sorted({record.batch_size for record in valid})
    matrix = np.full((len(points), len(batches)), np.nan)
    for row, point in enumerate(points):
        for column, batch in enumerate(batches):
            values = [r.end_to_end_speedup for r in valid if r.npoints == point and r.batch_size == batch]
            if values:
                matrix[row, column] = np.mean(values)
    figure, axis = _figure(width=14, height=8)
    image = axis.imshow(matrix, aspect="auto", interpolation="none")
    # A dense campaign can have more measured sizes than a presentation axis
    # can label.  Keep representative labels (including both endpoints).
    x_ticks = np.unique(np.linspace(0, len(batches) - 1, min(10, len(batches)), dtype=int))
    y_ticks = np.unique(np.linspace(0, len(points) - 1, min(12, len(points)), dtype=int))
    axis.set_xticks(x_ticks, [f"{batches[index]:,}" for index in x_ticks])
    axis.set_yticks(y_ticks, [f"{points[index]:,}" for index in y_ticks])
    axis.tick_params(axis="x", rotation=0)
    _style(axis, xlabel="Batch size [spatial points]", ylabel="Dataset size [spatial points]", title="Observed device end-to-end speedup")
    figure.colorbar(image, ax=axis, label="CPU / GPU speedup [×]")
    return figure, axis


def plot_full_dataset_summary(records: Iterable[BenchmarkRecord], metadata: Mapping[str, Any] | None = None):
    """One compact figure explaining the measured full-dataset result."""
    valid = _automatic_float32(records)
    full_points = max(record.npoints for record in valid)
    full = [record for record in valid if record.npoints == full_points]
    cpu = float(np.mean([record.cpu_compute_seconds for record in full if record.cpu_compute_seconds]))
    compute = float(np.mean([record.gpu_compute_seconds for record in full]))
    end_to_end = float(np.mean([record.gpu_end_to_end_seconds for record in full]))
    pipeline = float(np.mean([record.pipeline_wall_seconds for record in full if record.pipeline_wall_seconds]))
    figure, axes = _figure(panels=2, width=15, height=7)
    labels = ["CPU float64\ncompute", "GPU float32\ncompute", "GPU device\nend-to-end", "GPU pipeline\nwall"]
    values = [cpu, compute, end_to_end, pipeline]
    axes[0].bar(labels, values, color=[_COLORS["cpu"], _COLORS["compute"], _COLORS["end_to_end"], "#b24a70"])
    axes[0].bar_label(axes[0].containers[0], labels=[f"{value:.2f} s" for value in values], fontsize=12, padding=4)
    _style(axes[0], xlabel="", ylabel="Time [s]", title=f"Full dataset: {full_points:,} points × {full[0].ntime:,} hours")
    axes[1].axis("off")
    text = (
        f"{full_points * full[0].ntime / 1e9:.3f} billion SST values\n\n"
        f"Compute-only speedup: {cpu / compute:.1f}×\n"
        f"Device end-to-end speedup: {cpu / end_to_end:.1f}×\n"
        f"Pipeline-wall speedup: {cpu / pipeline:.1f}×\n\n"
        f"H2D: {np.mean([r.transfer_to_device_seconds for r in full]):.2f} s\n"
        f"GPU compute: {compute:.2f} s\n"
        f"Peak VRAM: {np.mean([r.peak_used_vram_bytes for r in full]) / 2**30:.2f} GiB"
    )
    axes[1].text(0.06, 0.88, text, va="top", fontsize=17, linespacing=1.45, bbox={"facecolor": "#f6f6f6", "edgecolor": "#999999", "pad": 12})
    return figure, axes


def plot_validation_summary(validation: Mapping[str, Any]):
    """Scientific validation overview from a ``ValidationResult.to_dict`` payload."""
    mask = validation["mask"]
    fields = validation["float_fields"]
    names = [name for name in ("climatology", "threshold", "anomaly", "excess") if name in fields]
    figure, axes = _figure(panels=2, width=15, height=7)
    axes[0].bar(names, [fields[name]["rmse"] for name in names], color=_COLORS["compute"])
    axes[0].set_yscale("log")
    _style(axes[0], xlabel="Field", ylabel="RMSE [°C]", title="CPU float64 vs GPU float32")
    text = (
        f"Classifications: {mask['total_elements']:,}\n"
        f"Agreement: {mask['agreement_percent']:.8f}%\n"
        f"Disagreements: {mask['disagreements']:,}\n"
        f"False positives / negatives: {mask['false_positives']} / {mask['false_negatives']}\n"
        f"Threshold distance mean: {mask['threshold_distance_mean']:.3g} °C\n"
        f"NaN-location disagreements: {max(field['nan_location_disagreement'] for field in fields.values()):,}"
    )
    axes[1].axis("off")
    axes[1].text(0.06, 0.9, text, va="top", fontsize=17, linespacing=1.5, bbox={"facecolor": "#f6f6f6", "edgecolor": "#999999", "pad": 12})
    return figure, axes


def save_figure_set(figure, output_directory: str | Path, name: str) -> dict[str, Path]:
    """Save crisp PNG, PDF, and SVG variants without overwriting a run directory."""
    base = Path(output_directory) / "figures"
    outputs = {
        "png": base / "png" / f"{name}.png",
        "pdf": base / "pdf" / f"{name}.pdf",
        "svg": base / "svg" / f"{name}.svg",
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(outputs["png"], dpi=300, bbox_inches="tight")
    figure.savefig(outputs["pdf"], bbox_inches="tight")
    figure.savefig(outputs["svg"], bbox_inches="tight")
    _plt().close(figure)
    return outputs
