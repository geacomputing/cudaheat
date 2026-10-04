"""Machine-readable benchmark record types and summary calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True)
class BenchmarkRecord:
    """One measured benchmark repetition; values are never estimates."""

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

    @property
    def speedup(self) -> float | None:
        if not self.cpu_compute_seconds or self.gpu_end_to_end_seconds == 0:
            return None
        return self.cpu_compute_seconds / self.gpu_end_to_end_seconds

    @property
    def throughput_points_per_second(self) -> float:
        if self.gpu_end_to_end_seconds == 0:
            return float("inf")
        return self.npoints / self.gpu_end_to_end_seconds

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["speedup"] = self.speedup
        row["throughput_points_per_second"] = self.throughput_points_per_second
        return row


def records_to_rows(records: Iterable[BenchmarkRecord]) -> list[dict[str, Any]]:
    """Convert individual runs, retaining all repeats, into exportable rows."""
    return [record.as_dict() for record in records]


def mean_speedup(records: Iterable[BenchmarkRecord]) -> float | None:
    """Mean of defined measured speedups, useful for concise reports."""
    values = [record.speedup for record in records if record.speedup is not None]
    return mean(values) if values else None
