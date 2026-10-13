"""Reusable CPU/GPU benchmarking separate from CUDAHeat's scientific core."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np

from ..climatology import baseline_rows, cpu_pre_event_fields
from ..core import CUDAHeat, PipelineResult
from ..exceptions import CUDAHeatError
from ..memory import cupy
from .metrics import BenchmarkRecord, records_to_rows, summarize_records


class CUDAHeatBenchmark:
    """Measure actual complete-time CPU and GPU pipeline runs."""

    def __init__(self, engine: CUDAHeat) -> None:
        self.engine = engine
        self.records: list[BenchmarkRecord] = []
        self.metadata: dict[str, Any] = {}

    def run(
        self,
        *,
        repeats: int = 3,
        warmup: int = 1,
        point_counts: Iterable[int] | None = None,
        batch_sizes: Iterable[int | None] | None = None,
        gpu_dtypes: Iterable[str] | None = None,
        cpu_dtype: str = "float64",
        cpu_reference: Callable[[], None] | None = None,
        reuse_cpu_reference: bool = True,
        measure_cpu: bool = True,
        **pipeline_kwargs: Any,
    ) -> list[BenchmarkRecord]:
        """Run a matrix of actual measurements and retain every repeat.

        The default CPU reference is ``cpu_pre_event_fields``. Set
        ``measure_cpu=False`` for GPU-only campaigns. Warm-up runs are
        intentionally excluded from the returned and exported measurements.
        """
        if repeats < 1 or warmup < 0:
            raise ValueError("repeats must be positive and warmup cannot be negative")
        cpu_dtype_value = np.dtype(cpu_dtype)
        if cpu_dtype_value not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("cpu_dtype must be float32 or float64")
        point_counts = tuple(dict.fromkeys(point_counts or (self.engine.npoints,)))
        batch_sizes = tuple(batch_sizes or (None,))
        gpu_dtypes = tuple(gpu_dtypes or (self.engine.dtype.name,))
        if not gpu_dtypes:
            raise ValueError("gpu_dtypes must contain at least one precision")
        if any(count < 1 or count > self.engine.npoints for count in point_counts):
            raise ValueError("point_counts must be in [1, engine.npoints]")
        if any(size is not None and size < 1 for size in batch_sizes):
            raise ValueError("batch_sizes must contain positive integers or None")
        self.metadata = self._metadata(
            pipeline_kwargs, cpu_dtype_value.name if measure_cpu else None
        )
        new_records: list[BenchmarkRecord] = []
        for point_count in point_counts:
            cpu_measurements: list[float] | None = None
            if measure_cpu and reuse_cpu_reference:
                with self._engine_for_configuration(point_count, gpu_dtypes[0]) as cpu_engine:
                    for _ in range(warmup):
                        self._run_cpu(cpu_engine, cpu_dtype_value, cpu_reference, pipeline_kwargs)
                    cpu_measurements = [
                        self._run_cpu(cpu_engine, cpu_dtype_value, cpu_reference, pipeline_kwargs)
                        for _ in range(repeats)
                    ]
            for gpu_dtype in gpu_dtypes:
                for requested_batch_size in batch_sizes:
                    with self._engine_for_configuration(point_count, gpu_dtype) as configured:
                        for _ in range(warmup):
                            if measure_cpu and not reuse_cpu_reference:
                                self._run_cpu(
                                    configured, cpu_dtype_value, cpu_reference, pipeline_kwargs
                                )
                            configured.run(batch_size=requested_batch_size, **pipeline_kwargs)
                        for repeat in range(repeats):
                            cpu_seconds = None
                            if measure_cpu:
                                cpu_seconds = (
                                    cpu_measurements[repeat]
                                    if cpu_measurements is not None
                                    else self._run_cpu(
                                        configured, cpu_dtype_value, cpu_reference, pipeline_kwargs
                                    )
                                )
                            try:
                                pipeline_start = perf_counter()
                                result = configured.run(
                                    batch_size=requested_batch_size, **pipeline_kwargs
                                )
                                pipeline_wall_seconds = perf_counter() - pipeline_start
                            except CUDAHeatError as exc:
                                record = self._failed_record(
                                    configured,
                                    repeat,
                                    cpu_seconds,
                                    requested_batch_size,
                                    cpu_dtype_value.name if measure_cpu else None,
                                    pipeline_kwargs,
                                    exc,
                                )
                                self.records.append(record)
                                new_records.append(record)
                                break
                            record = self._record_from_result(
                                repeat,
                                result,
                                cpu_seconds,
                                requested_batch_size,
                                cpu_dtype_value.name if measure_cpu else None,
                                pipeline_wall_seconds,
                            )
                            self.records.append(record)
                            new_records.append(record)
        return new_records

    def _engine_for_configuration(self, point_count: int, gpu_dtype: str) -> CUDAHeat:
        if self.engine._dataset is None:
            raise RuntimeError("the benchmark source engine is closed")
        indices = self.engine.point_indices[:point_count]
        if indices.size and np.all(np.diff(indices) == 1):
            point_selection: Iterable[int] | slice = slice(int(indices[0]), int(indices[-1]) + 1)
        else:
            point_selection = indices
        return CUDAHeat(
            self.engine._dataset,
            variable=self.engine.variable,
            dtype=gpu_dtype,
            vram_safety=self.engine.vram_safety,
            devices=self.engine.devices,
            time_dim=self.engine.time_dim,
            point_indices=point_selection,
            calendar_mode=self.engine.calendar_mode,
        )

    @staticmethod
    def _run_cpu(
        engine: CUDAHeat,
        dtype: np.dtype[Any],
        cpu_reference: Callable[[], None] | None,
        pipeline_kwargs: dict[str, Any],
    ) -> float:
        if cpu_reference is not None:
            start = perf_counter()
            cpu_reference()
            return perf_counter() - start
        rows = baseline_rows(engine._time, pipeline_kwargs.get("baseline"))
        percentile = pipeline_kwargs.get("percentile", 0.90)
        smooth_window_hours = engine._resolve_smooth_window_hours(
            pipeline_kwargs.get("smooth_window_hours")
        )
        smooth_min_periods = engine._resolve_smooth_min_periods(
            smooth_window_hours, pipeline_kwargs.get("smooth_min_periods")
        )
        climatology_nbins = engine._climatology_nbins(rows)
        # This cap limits host RAM, but every calculation still receives all time.
        cpu_batch_size = min(engine.npoints, 256)
        seconds = 0.0
        for point_start in range(0, engine.npoints, cpu_batch_size):
            point_stop = min(point_start + cpu_batch_size, engine.npoints)
            values = engine._load_spatial_batch(point_start, point_stop, dtype)
            start = perf_counter()
            fields = cpu_pre_event_fields(
                values,
                engine._bins,
                rows,
                percentile,
                smooth_window_hours,
                climatology_nbins,
                smooth_min_periods,
                engine.calendar_mode,
            )
            seconds += perf_counter() - start
            del fields
        return seconds

    def _record_from_result(
        self,
        repeat: int,
        result: PipelineResult,
        cpu_seconds: float | None,
        requested_batch_size: int | None,
        cpu_dtype: str | None,
        pipeline_wall_seconds: float,
    ) -> BenchmarkRecord:
        telemetry = result.telemetry
        if not telemetry:
            raise RuntimeError("CUDAHeat returned no batch telemetry")
        first = telemetry[0]
        return BenchmarkRecord(
            repeat=repeat,
            ntime=self.engine.ntime,
            npoints=int(result.metadata["npoints"]),
            dtype=result.metadata["dtype"],
            device_id=first.device_id,
            cpu_compute_seconds=cpu_seconds,
            gpu_compute_seconds=sum(item.gpu_compute_seconds for item in telemetry),
            transfer_to_device_seconds=sum(item.transfer_to_device_seconds for item in telemetry),
            transfer_from_device_seconds=sum(
                item.transfer_from_device_seconds for item in telemetry
            ),
            gpu_end_to_end_seconds=sum(
                item.transfer_to_device_seconds
                + item.gpu_compute_seconds
                + item.transfer_from_device_seconds
                for item in telemetry
            ),
            batch_size=max(item.batch_size for item in telemetry),
            free_vram_bytes=first.free_before_bytes,
            peak_used_vram_bytes=max(item.peak_used_bytes for item in telemetry),
            safety_fraction=first.safety_fraction,
            timestamp=datetime.now(timezone.utc).isoformat(),
            cpu_dtype=cpu_dtype,
            gpu_name=self.engine.gpu_info.name,
            total_vram_bytes=self.engine.gpu_info.total_bytes,
            used_vram_before_bytes=first.used_before_bytes,
            pool_used_bytes=max(item.pool_used_bytes for item in telemetry),
            bytes_per_point=first.bytes_per_point,
            raw_data_bytes_per_point=self.engine.ntime
            * np.dtype(result.metadata["dtype"]).itemsize,
            oom_retries=sum(item.oom_retries for item in telemetry),
            requested_batch_size=requested_batch_size,
            baseline=result.metadata["baseline"],
            percentile=result.metadata["percentile"],
            smooth_window_hours=result.metadata["smooth_window_hours"],
            host_read_seconds=sum(item.host_read_seconds for item in telemetry),
            host_prepare_seconds=sum(item.host_prepare_seconds for item in telemetry),
            gpu_allocation_seconds=sum(item.gpu_allocation_seconds for item in telemetry),
            gpu_metadata_transfer_seconds=sum(
                item.gpu_metadata_transfer_seconds for item in telemetry
            ),
            result_assembly_seconds=sum(item.result_assembly_seconds for item in telemetry)
            + result.metadata.get("result_dataset_assembly_seconds", 0.0),
            pipeline_wall_seconds=pipeline_wall_seconds,
        )

    def _failed_record(
        self,
        engine: CUDAHeat,
        repeat: int,
        cpu_seconds: float | None,
        requested_batch_size: int | None,
        cpu_dtype: str | None,
        pipeline_kwargs: dict[str, Any],
        error: Exception,
    ) -> BenchmarkRecord:
        return BenchmarkRecord(
            repeat=repeat,
            ntime=engine.ntime,
            npoints=engine.npoints,
            dtype=engine.dtype.name,
            device_id=engine.device_id,
            cpu_compute_seconds=cpu_seconds,
            gpu_compute_seconds=0.0,
            transfer_to_device_seconds=0.0,
            transfer_from_device_seconds=0.0,
            gpu_end_to_end_seconds=0.0,
            batch_size=0,
            free_vram_bytes=engine.gpu_info.free_bytes,
            peak_used_vram_bytes=engine.gpu_info.used_bytes,
            safety_fraction=engine.vram_safety,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="invalid",
            error=str(error),
            cpu_dtype=cpu_dtype,
            gpu_name=engine.gpu_info.name,
            total_vram_bytes=engine.gpu_info.total_bytes,
            used_vram_before_bytes=engine.gpu_info.used_bytes,
            bytes_per_point=engine.batch_estimate.bytes_per_point,
            raw_data_bytes_per_point=engine.ntime * engine.dtype.itemsize,
            requested_batch_size=requested_batch_size,
            baseline=pipeline_kwargs.get("baseline"),
            percentile=pipeline_kwargs.get("percentile", 0.90),
            smooth_window_hours=engine._resolve_smooth_window_hours(
                pipeline_kwargs.get("smooth_window_hours")
            ),
        )

    def summary(self) -> dict[str, dict[str, Any]]:
        """Distribution summaries grouped by measured configuration."""
        return summarize_records(self.records)

    def export_json(self, path: str | Path) -> Path:
        """Export metadata, individual measurements, and repeat summaries as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "records": records_to_rows(self.records),
            "summary": _json_ready(self.summary()),
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def export_csv(self, path: str | Path) -> Path:
        """Export individual repeat measurements as flat CSV rows."""
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

    def _metadata(self, pipeline_kwargs: dict[str, Any], cpu_dtype: str | None) -> dict[str, Any]:
        cp = cupy()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cudaheat_version": __import__("cudaheat").__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "cupy_version": cp.__version__,
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "gpu_count": int(cp.cuda.runtime.getDeviceCount()),
            "gpu": self.engine.gpu_info.as_dict(),
            "dataset_path": self.engine.source_path,
            "variable": self.engine.variable,
            "ntime": self.engine.ntime,
            "npoints": self.engine.npoints,
            "cpu_dtype": cpu_dtype,
            "baseline": pipeline_kwargs.get("baseline"),
            "percentile": pipeline_kwargs.get("percentile", 0.90),
            "smooth_window_hours": self.engine._resolve_smooth_window_hours(
                pipeline_kwargs.get("smooth_window_hours")
            ),
        }


def _json_ready(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {name: _json_ready(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value
