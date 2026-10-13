"""Machine-readable benchmark records, derived metrics, and summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median, stdev
from typing import Any, Iterable


@dataclass(frozen=True)
class BenchmarkRecord:
    """One actual measurement. No field represents an estimated performance value."""

    repeat: int
    ntime: int
    npoints: int
    dtype: str
    device_id: int
    cpu_compute_seconds: float | None
    gpu_compute_seconds: float
    transfer_to_device_seconds: float
    transfer_from_device_seconds: float
    gpu_end_to_end_seconds: float
    batch_size: int
    free_vram_bytes: int
    peak_used_vram_bytes: int
    safety_fraction: float
    timestamp: str | None = None
    status: str = "complete"
    error: str | None = None
    cpu_dtype: str | None = "float64"
    gpu_name: str | None = None
    total_vram_bytes: int | None = None
    used_vram_before_bytes: int | None = None
    pool_used_bytes: int | None = None
    bytes_per_point: int | None = None
    raw_data_bytes_per_point: int | None = None
    oom_retries: int = 0
    requested_batch_size: int | None = None
    baseline: tuple[int, int] | None = None
    percentile: float | None = None
    smooth_window_hours: int | None = None
    host_read_seconds: float = 0.0
    host_prepare_seconds: float = 0.0
    gpu_allocation_seconds: float = 0.0
    gpu_metadata_transfer_seconds: float = 0.0
    result_assembly_seconds: float = 0.0
    pipeline_wall_seconds: float | None = None

    @property
    def compute_speedup(self) -> float | None:
        if self.cpu_compute_seconds is None or self.gpu_compute_seconds <= 0:
            return None
        return self.cpu_compute_seconds / self.gpu_compute_seconds

    @property
    def end_to_end_speedup(self) -> float | None:
        if self.cpu_compute_seconds is None or self.gpu_end_to_end_seconds <= 0:
            return None
        return self.cpu_compute_seconds / self.gpu_end_to_end_seconds

    @property
    def pipeline_speedup(self) -> float | None:
        """CPU compute divided by complete measured GPU pipeline wall time."""
        if self.cpu_compute_seconds is None or not self.pipeline_wall_seconds:
            return None
        return self.cpu_compute_seconds / self.pipeline_wall_seconds

    @property
    def speedup(self) -> float | None:
        """Backward-compatible alias for end-to-end speedup."""
        return self.end_to_end_speedup

    @property
    def values_processed(self) -> int:
        return self.ntime * self.npoints

    @property
    def cpu_values_per_second(self) -> float | None:
        if self.cpu_compute_seconds is None or self.cpu_compute_seconds <= 0:
            return None
        return self.values_processed / self.cpu_compute_seconds

    @property
    def gpu_compute_values_per_second(self) -> float | None:
        if self.gpu_compute_seconds <= 0:
            return None
        return self.values_processed / self.gpu_compute_seconds

    @property
    def gpu_end_to_end_values_per_second(self) -> float | None:
        if self.gpu_end_to_end_seconds <= 0:
            return None
        return self.values_processed / self.gpu_end_to_end_seconds

    @property
    def throughput_points_per_second(self) -> float:
        """Backward-compatible end-to-end spatial-series throughput."""
        if self.gpu_end_to_end_seconds <= 0:
            return float("inf")
        return self.npoints / self.gpu_end_to_end_seconds

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row.update(
            {
                "compute_speedup": self.compute_speedup,
                "end_to_end_speedup": self.end_to_end_speedup,
                "pipeline_speedup": self.pipeline_speedup,
                "speedup": self.speedup,
                "values_processed": self.values_processed,
                "cpu_values_per_second": self.cpu_values_per_second,
                "gpu_compute_values_per_second": self.gpu_compute_values_per_second,
                "gpu_end_to_end_values_per_second": self.gpu_end_to_end_values_per_second,
                "throughput_points_per_second": self.throughput_points_per_second,
            }
        )
        return row


@dataclass(frozen=True)
class MetricSummary:
    """Distribution statistics over actual repeated measurements."""

    count: int
    mean: float
    median: float
    standard_deviation: float
    minimum: float
    maximum: float


def records_to_rows(records: Iterable[BenchmarkRecord]) -> list[dict[str, Any]]:
    """Convert individual runs, retaining all repeats, into exportable rows."""
    return [record.as_dict() for record in records]


def summarize_values(values: Iterable[float | None]) -> MetricSummary | None:
    """Return distribution statistics for defined actual measurements."""
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return MetricSummary(
        count=len(cleaned),
        mean=mean(cleaned),
        median=median(cleaned),
        standard_deviation=stdev(cleaned) if len(cleaned) > 1 else 0.0,
        minimum=min(cleaned),
        maximum=max(cleaned),
    )


def summarize_records(
    records: Iterable[BenchmarkRecord],
) -> dict[str, dict[str, MetricSummary | None]]:
    """Summarize timings, speedups, and throughput by measured configuration."""
    grouped: dict[str, list[BenchmarkRecord]] = {}
    for record in records:
        key = (
            f"points={record.npoints};dtype={record.dtype};"
            f"batch={record.requested_batch_size or 'automatic'}"
        )
        grouped.setdefault(key, []).append(record)
    result: dict[str, dict[str, MetricSummary | None]] = {}
    for key, group in grouped.items():
        result[key] = {
            "cpu_compute_seconds": summarize_values(record.cpu_compute_seconds for record in group),
            "gpu_compute_seconds": summarize_values(record.gpu_compute_seconds for record in group),
            "gpu_end_to_end_seconds": summarize_values(
                record.gpu_end_to_end_seconds for record in group
            ),
            "compute_speedup": summarize_values(record.compute_speedup for record in group),
            "end_to_end_speedup": summarize_values(record.end_to_end_speedup for record in group),
            "pipeline_speedup": summarize_values(record.pipeline_speedup for record in group),
            "gpu_compute_values_per_second": summarize_values(
                record.gpu_compute_values_per_second for record in group
            ),
            "gpu_end_to_end_values_per_second": summarize_values(
                record.gpu_end_to_end_values_per_second for record in group
            ),
        }
    return result
