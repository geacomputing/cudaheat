"""Matplotlib plots built exclusively from actual benchmark records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from .metrics import BenchmarkRecord


def _completed(records: Sequence[BenchmarkRecord]) -> list[BenchmarkRecord]:
    return [record for record in records if record.status == "complete"]


def plot_cpu_vs_gpu_time(records: Sequence[BenchmarkRecord]):
    """Plot CPU, GPU compute-only, and GPU end-to-end times by point count."""
    import matplotlib.pyplot as plt

    valid = _completed(records)
    figure, axis = plt.subplots()
    axis.plot([r.npoints for r in valid], [r.gpu_compute_seconds for r in valid], "o", label="GPU compute")
    axis.plot([r.npoints for r in valid], [r.gpu_end_to_end_seconds for r in valid], "o", label="GPU end-to-end")
    cpu = [record for record in valid if record.cpu_compute_seconds is not None]
    if cpu:
        axis.plot([r.npoints for r in cpu], [r.cpu_compute_seconds for r in cpu], "o", label="CPU compute")
    axis.set(xlabel="Spatial points (complete time series)", ylabel="Seconds", title="Measured execution time")
    axis.legend()
    return figure, axis


def plot_speedup_vs_dataset_size(records: Sequence[BenchmarkRecord]):
    """Plot compute-only and end-to-end speedups by spatial scale."""
    import matplotlib.pyplot as plt

    valid = [record for record in _completed(records) if record.compute_speedup is not None]
    figure, axis = plt.subplots()
    axis.plot([r.npoints for r in valid], [r.compute_speedup for r in valid], "o", label="compute-only")
    axis.plot([r.npoints for r in valid], [r.end_to_end_speedup for r in valid], "o", label="end-to-end")
    axis.axhline(1, color="black", linewidth=0.8)
    axis.set(xlabel="Spatial points (complete time series)", ylabel="CPU / GPU speedup", title="Measured speedup")
    axis.legend()
    return figure, axis


def plot_speedup_vs_batch_size(records: Sequence[BenchmarkRecord]):
    """Plot actual end-to-end speedup for controlled batch-size runs."""
    import matplotlib.pyplot as plt

    valid = [record for record in _completed(records) if record.end_to_end_speedup is not None]
    figure, axis = plt.subplots()
    axis.scatter([r.batch_size for r in valid], [r.end_to_end_speedup for r in valid])
    axis.set(xlabel="Actual spatial batch size", ylabel="End-to-end speedup", title="Measured batch scaling")
    return figure, axis


def plot_vram_vs_batch_size(records: Sequence[BenchmarkRecord]):
    """Plot measured peak VRAM use versus actual spatial batch size."""
    import matplotlib.pyplot as plt

    valid = _completed(records)
    figure, axis = plt.subplots()
    axis.scatter([r.batch_size for r in valid], [r.peak_used_vram_bytes / 2**30 for r in valid])
    axis.set(xlabel="Spatial batch size", ylabel="Peak VRAM used (GiB)", title="Measured VRAM scaling")
    return figure, axis


def plot_transfer_vs_compute(records: Sequence[BenchmarkRecord]):
    """Compare actual host-to-device transfer and GPU compute time."""
    import matplotlib.pyplot as plt

    valid = _completed(records)
    figure, axis = plt.subplots()
    axis.scatter(
        [r.transfer_to_device_seconds for r in valid],
        [r.gpu_compute_seconds for r in valid],
    )
    axis.set(xlabel="CPU → GPU transfer (s)", ylabel="GPU compute (s)", title="Transfer versus compute")
    return figure, axis


def plot_throughput_vs_dataset_size(records: Sequence[BenchmarkRecord]):
    """Plot values/s for GPU compute and end-to-end measurements."""
    import matplotlib.pyplot as plt

    valid = _completed(records)
    figure, axis = plt.subplots()
    axis.plot(
        [r.npoints for r in valid],
        [r.gpu_compute_values_per_second / 1e6 for r in valid],
        "o",
        label="GPU compute",
    )
    axis.plot(
        [r.npoints for r in valid],
        [r.gpu_end_to_end_values_per_second / 1e6 for r in valid],
        "o",
        label="GPU end-to-end",
    )
    axis.set(xlabel="Spatial points (complete time series)", ylabel="Million values/s", title="Measured throughput")
    axis.legend()
    return figure, axis


def plot_dtype_comparison(records: Sequence[BenchmarkRecord]):
    """Plot actual end-to-end times grouped by measured GPU dtype."""
    import matplotlib.pyplot as plt

    grouped: dict[str, list[float]] = defaultdict(list)
    for record in _completed(records):
        grouped[record.dtype].append(record.gpu_end_to_end_seconds)
    figure, axis = plt.subplots()
    axis.boxplot(grouped.values(), tick_labels=grouped.keys())
    axis.set(xlabel="GPU dtype", ylabel="GPU end-to-end seconds", title="Measured dtype performance")
    return figure, axis


def plot_repeated_run_boxplot(records: Sequence[BenchmarkRecord]):
    """Show end-to-end timing distributions for each measured configuration."""
    import matplotlib.pyplot as plt

    grouped: dict[str, list[float]] = defaultdict(list)
    for record in _completed(records):
        label = f"{record.npoints} pts / {record.dtype} / {record.batch_size} batch"
        grouped[label].append(record.gpu_end_to_end_seconds)
    figure, axis = plt.subplots()
    axis.boxplot(grouped.values(), tick_labels=grouped.keys())
    axis.tick_params(axis="x", rotation=30)
    axis.set(ylabel="GPU end-to-end seconds", title="Repeated actual runs")
    return figure, axis


def plot_speedup_heatmap(records: Sequence[BenchmarkRecord]):
    """Plot observed end-to-end speedup for point-count × batch-size pairs.

    Missing configurations remain blank; no interpolation or fabricated values
    are used.
    """
    import matplotlib.pyplot as plt

    valid = [record for record in _completed(records) if record.end_to_end_speedup is not None]
    points = sorted({record.npoints for record in valid})
    batches = sorted({record.batch_size for record in valid})
    matrix = np.full((len(points), len(batches)), np.nan)
    for row, point_count in enumerate(points):
        for column, batch_size in enumerate(batches):
            values = [
                record.end_to_end_speedup
                for record in valid
                if record.npoints == point_count and record.batch_size == batch_size
            ]
            if values:
                matrix[row, column] = np.mean(values)
    figure, axis = plt.subplots()
    image = axis.imshow(matrix, aspect="auto")
    axis.set_xticks(range(len(batches)), batches)
    axis.set_yticks(range(len(points)), points)
    axis.set(xlabel="Spatial batch size", ylabel="Spatial points", title="Observed end-to-end speedup")
    figure.colorbar(image, ax=axis, label="CPU / GPU speedup")
    return figure, axis
