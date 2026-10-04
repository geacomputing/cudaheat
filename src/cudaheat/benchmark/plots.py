"""Matplotlib visualisations of actual :mod:`cudaheat.benchmark` records."""

from __future__ import annotations

from collections.abc import Sequence

from .metrics import BenchmarkRecord


def plot_cpu_vs_gpu_time(records: Sequence[BenchmarkRecord]):
    """Plot measured end-to-end GPU and available CPU times against point count."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([r.npoints for r in records], [r.gpu_end_to_end_seconds for r in records], "o-", label="GPU end-to-end")
    cpu_records = [record for record in records if record.cpu_compute_seconds is not None]
    if cpu_records:
        axis.plot([r.npoints for r in cpu_records], [r.cpu_compute_seconds for r in cpu_records], "o-", label="CPU")
    axis.set(xlabel="Spatial points", ylabel="Seconds", title="Measured CPU and GPU time")
    axis.legend()
    return figure, axis


def plot_vram_vs_batch_size(records: Sequence[BenchmarkRecord]):
    """Plot measured peak VRAM use versus actual batch size."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.scatter(
        [r.batch_size for r in records],
        [r.peak_used_vram_bytes / 2**30 for r in records],
    )
    axis.set(xlabel="Spatial batch size", ylabel="Peak VRAM used (GiB)", title="Measured VRAM scaling")
    return figure, axis
