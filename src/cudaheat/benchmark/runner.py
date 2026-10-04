"""Run and export CUDAHeat measurements without coupling benchmarks to the engine."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from ..core import CUDAHeat, PipelineResult
from .metrics import BenchmarkRecord, records_to_rows


class CUDAHeatBenchmark:
    """Collect repeatable GPU timings and optional CPU-reference timings."""

    def __init__(self, engine: CUDAHeat) -> None:
        self.engine = engine
        self.records: list[BenchmarkRecord] = []

    def run(
        self,
        *,
        repeats: int = 3,
        cpu_reference: Callable[[], None] | None = None,
        **pipeline_kwargs: Any,
    ) -> list[BenchmarkRecord]:
        """Run real pipeline repetitions and retain every individual observation.

        ``cpu_reference`` should execute an equivalent CPU operation. It is
        optional because benchmark datasets and CPU methodologies vary.
        """
        if repeats < 1:
            raise ValueError("repeats must be at least one")
        new_records: list[BenchmarkRecord] = []
        for repeat in range(repeats):
            cpu_seconds = None
            if cpu_reference is not None:
                start = perf_counter()
                cpu_reference()
                cpu_seconds = perf_counter() - start
            result = self.engine.run(**pipeline_kwargs)
            record = self._record_from_result(repeat, result, cpu_seconds)
            self.records.append(record)
            new_records.append(record)
        return new_records

    def _record_from_result(
        self, repeat: int, result: PipelineResult, cpu_seconds: float | None
    ) -> BenchmarkRecord:
        telemetry = result.telemetry
        if not telemetry:
            raise RuntimeError("CUDAHeat returned no batch telemetry")
        return BenchmarkRecord(
            repeat=repeat,
            ntime=self.engine.ntime,
            npoints=self.engine.npoints,
            dtype=self.engine.dtype.name,
            device_id=self.engine.device_id,
            cpu_compute_seconds=cpu_seconds,
            gpu_compute_seconds=sum(item.gpu_compute_seconds for item in telemetry),
            transfer_to_device_seconds=sum(item.transfer_to_device_seconds for item in telemetry),
            transfer_from_device_seconds=sum(item.transfer_from_device_seconds for item in telemetry),
            gpu_end_to_end_seconds=sum(
                item.transfer_to_device_seconds
                + item.gpu_compute_seconds
                + item.transfer_from_device_seconds
                for item in telemetry
            ),
            batch_size=max(item.batch_size for item in telemetry),
            free_vram_bytes=telemetry[0].free_before_bytes,
            peak_used_vram_bytes=max(item.peak_used_bytes for item in telemetry),
            safety_fraction=self.engine.vram_safety,
        )

    def export_json(self, path: str | Path) -> Path:
        """Export all individual benchmark runs as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records_to_rows(self.records), indent=2), encoding="utf-8")
        return target

    def export_csv(self, path: str | Path) -> Path:
        """Export all individual benchmark runs as CSV."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = records_to_rows(self.records)
        if not rows:
            raise ValueError("cannot export an empty benchmark")
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return target
