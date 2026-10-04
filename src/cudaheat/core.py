"""Out-of-core CUDAHeat engine: complete time series, spatial batches only."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import xarray as xr

from .climatology import (
    HOURS_PER_LEAP_YEAR,
    PreEventFields,
    baseline_rows,
    gpu_pre_event_fields,
    hourly_climatology_index,
)
from .exceptions import InsufficientVRAMError
from .memory import (
    BatchEstimate,
    BatchTelemetry,
    GPUInfo,
    VRAMEstimator,
    cupy,
    get_gpu_info,
    reduced_batch_size,
    release_gpu_memory,
)

logger = logging.getLogger(__name__)


@dataclass
class BatchOutput:
    """CPU results deliberately transferred for one completed spatial batch."""

    point_start: int
    point_stop: int
    fields: dict[str, np.ndarray]
    telemetry: BatchTelemetry


@dataclass
class PipelineResult:
    """Compact result data, measured telemetry, and reproducibility metadata."""

    data: xr.Dataset
    telemetry: tuple[BatchTelemetry, ...]
    metadata: dict[str, Any]

    @property
    def exceedance_count(self) -> xr.DataArray:
        """Number of threshold exceedances per spatial point."""
        return self.data["exceedance_count"]


class CUDAHeat:
    """GPU-native preprocessing for hourly gridded SST.

    Every GPU input has shape ``(complete_time, spatial_batch)``.  The engine
    will only reduce ``spatial_batch`` after an OOM; it has no code path that
    partitions a time series.

    Parameters
    ----------
    source:
        A NetCDF path or already-open :class:`xarray.Dataset`.
    variable:
        Name of the SST variable.
    dtype:
        Explicit GPU precision, either ``"float32"`` (default) or
        ``"float64"``. Source precision is never silently retained instead.
    vram_safety:
        Fraction of currently free VRAM held in reserve (default 0.30).
    devices:
        Selected CUDA device IDs. The first milestone runs batches sequentially
        on the first selected device; the executor is device-explicit so future
        schedulers can distribute independent spatial batches over this list.
    """

    _RETURNABLE_FIELDS = frozenset(
        {"climatology", "threshold_cycle", "threshold", "anomaly", "excess", "mask"}
    )

    def __init__(
        self,
        source: str | Path | xr.Dataset,
        *,
        variable: str,
        dtype: str | np.dtype[Any] = "float32",
        vram_safety: float = 0.30,
        devices: Iterable[int] = (0,),
        time_dim: str | None = "time",
    ) -> None:
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")
        self.vram_safety = float(vram_safety)
        self.devices = tuple(int(device) for device in devices)
        if not self.devices:
            raise ValueError("at least one CUDA device must be selected")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("devices must not contain duplicates")
        self.device_id = self.devices[0]
        self._owns_dataset = not isinstance(source, xr.Dataset)
        self._dataset = xr.open_dataset(source) if self._owns_dataset else source
        if variable not in self._dataset:
            self.close()
            raise KeyError(f"variable {variable!r} is not present in the dataset")
        self.variable = variable
        self._source = self._dataset[variable]
        self.time_dim = self._resolve_time_dim(time_dim)
        self.spatial_dims = tuple(dim for dim in self._source.dims if dim != self.time_dim)
        if not self.spatial_dims:
            self.close()
            raise ValueError("SST variable needs at least one spatial dimension besides time")
        self._stacked = (
            self._source.transpose(self.time_dim, *self.spatial_dims)
            .stack(point=self.spatial_dims)
            .transpose(self.time_dim, "point")
        )
        self.ntime = int(self._stacked.sizes[self.time_dim])
        self.npoints = int(self._stacked.sizes["point"])
        if self.ntime < 1 or self.npoints < 1:
            self.close()
            raise ValueError("SST variable must have non-empty time and spatial dimensions")
        self._time = self._stacked[self.time_dim]
        self._bins = hourly_climatology_index(self._time)
        self._estimator = VRAMEstimator(
            self.ntime, self.dtype.itemsize, safety_fraction=self.vram_safety
        )
        self.gpu_info = get_gpu_info(self.device_id)
        self.gpu_infos = tuple(get_gpu_info(device) for device in self.devices)
        self.batch_estimate = self._estimator.estimate(self.gpu_info.free_bytes, self.npoints)
        if self.batch_estimate.batch_size < 1:
            self.close()
            raise InsufficientVRAMError(self._single_series_message(self.batch_estimate))
        self.batch_size = self.batch_estimate.batch_size
        self.last_result: PipelineResult | None = None
        logger.info(
            "CUDAHeat selected GPU %s (%s); free %.2f GiB / total %.2f GiB; "
            "dtype=%s; initial spatial batch=%d; %.2f MiB/point estimate",
            self.device_id,
            self.gpu_info.name,
            self.gpu_info.free_bytes / 2**30,
            self.gpu_info.total_bytes / 2**30,
            self.dtype.name,
            self.batch_size,
            self.batch_estimate.bytes_per_point / 2**20,
        )
        if len(self.devices) > 1:
            logger.info("Selected devices %s; batches currently execute sequentially on device %d", self.devices, self.device_id)

    def __enter__(self) -> "CUDAHeat":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close a NetCDF dataset opened by this instance."""
        if getattr(self, "_owns_dataset", False) and getattr(self, "_dataset", None) is not None:
            self._dataset.close()
            self._dataset = None

    def _resolve_time_dim(self, requested: str | None) -> str:
        if requested is not None and requested in self._source.dims:
            return requested
        named = [dim for dim in self._source.dims if dim.lower() == "time"]
        if len(named) == 1:
            return named[0]
        datetime_dims = [
            dim
            for dim in self._source.dims
            if dim in self._source.coords
            and np.issubdtype(self._source[dim].dtype, np.datetime64)
        ]
        if len(datetime_dims) == 1:
            return datetime_dims[0]
        raise ValueError("could not identify one time dimension; pass time_dim explicitly")

    def _single_series_message(self, estimate: BatchEstimate) -> str:
        return (
            "A single complete SST time series cannot fit in the configured GPU budget. "
            f"ntime={self.ntime}, estimated={estimate.bytes_per_point:,} bytes/point, "
            f"budget={estimate.budget_bytes:,} bytes after vram_safety={self.vram_safety:.2f}. "
            "CUDAHeat will not split time; use float32, choose a larger GPU, or reduce "
            "the scientific memory model only after measurement."
        )

    def _load_spatial_batch(self, point_start: int, point_stop: int) -> np.ndarray:
        """Read one full-time spatial slice from Xarray into CPU RAM.

        This is intentionally the only disk-to-RAM selection used by the engine:
        it slices ``point`` and never ``time``.
        """
        selected = self._stacked.isel(point=slice(point_start, point_stop)).load()
        values = np.asarray(selected.data, dtype=self.dtype)
        if values.shape != (self.ntime, point_stop - point_start):
            raise RuntimeError("spatial batch lost part of its time series")
        return np.ascontiguousarray(values)

    def run(
        self,
        *,
        baseline: tuple[int, int] | None = None,
        percentile: float = 0.90,
        smooth_window_hours: int = 120,
        return_fields: Iterable[str] = (),
        on_batch: Callable[[BatchOutput], None] | None = None,
    ) -> PipelineResult:
        """Run the first-milestone GPU pipeline over all spatial batches.

        By default only compact ``exceedance_count`` is transferred to CPU. Use
        ``return_fields={"mask"}`` (or another explicit field) for small inputs;
        for production-size arrays use ``on_batch`` to persist selected batch
        outputs without retaining the whole result in host memory.
        """
        requested = frozenset(return_fields)
        unknown = requested - self._RETURNABLE_FIELDS
        if unknown:
            raise ValueError(f"unknown return field(s): {sorted(unknown)!r}")
        rows = baseline_rows(self._time, baseline)
        buffers = self._allocate_output_buffers(requested)
        exceedance_count = np.zeros(self.npoints, dtype=np.uint64)
        telemetry: list[BatchTelemetry] = []
        point_start = 0
        active_batch_size = self.batch_size

        while point_start < self.npoints:
            current_size = min(active_batch_size, self.npoints - point_start)
            retries = 0
            while True:
                point_stop = point_start + current_size
                cpu_values = self._load_spatial_batch(point_start, point_stop)
                try:
                    output, batch_telemetry = self._execute_gpu_batch(
                        cpu_values, point_start, point_stop, rows, percentile, smooth_window_hours, retries
                    )
                except Exception as exc:
                    if not self._is_gpu_oom(exc):
                        raise
                    release_gpu_memory(self.device_id)
                    if current_size == 1:
                        raise InsufficientVRAMError(self._single_series_message(self.batch_estimate)) from exc
                    retries += 1
                    current_size = reduced_batch_size(current_size)
                    active_batch_size = current_size
                    logger.warning(
                        "GPU OOM for points %d:%d; retrying identical complete time series "
                        "with %d spatial points (retry %d)",
                        point_start,
                        point_stop,
                        current_size,
                        retries,
                    )
                    continue
                self._store_batch(output.fields, buffers, point_start, point_stop)
                exceedance_count[point_start:point_stop] = output.fields.pop("exceedance_count")
                completed = BatchOutput(point_start, point_stop, output.fields, batch_telemetry)
                if on_batch is not None:
                    on_batch(completed)
                telemetry.append(batch_telemetry)
                point_start = point_stop
                break

        self.batch_size = active_batch_size
        result = PipelineResult(
            data=self._to_dataset(exceedance_count, buffers),
            telemetry=tuple(telemetry),
            metadata={
                "variable": self.variable,
                "dtype": self.dtype.name,
                "baseline": baseline,
                "percentile": percentile,
                "smooth_window_hours": smooth_window_hours,
                "time_dimension": self.time_dim,
                "spatial_dimensions": self.spatial_dims,
                "time_was_split": False,
                "vram_safety": self.vram_safety,
                "initial_batch_estimate": self.batch_estimate.as_dict(),
                "devices": self.devices,
            },
        )
        self.last_result = result
        logger.info(
            "CUDAHeat complete: %d spatial points in %d batch(es); time was never split",
            self.npoints,
            len(telemetry),
        )
        return result

    def climatology(self, **kwargs: Any) -> xr.DataArray:
        """Return the circularly smoothed hourly climatological cycle."""
        return self.run(return_fields={"climatology"}, **kwargs).data["climatology"]

    def thresholds(self, **kwargs: Any) -> xr.DataArray:
        """Return the timestamp-aligned, smoothed percentile threshold."""
        return self.run(return_fields={"threshold"}, **kwargs).data["threshold"]

    def mask(self, **kwargs: Any) -> xr.DataArray:
        """Return an in-memory exceedance mask; use only for manageable output sizes."""
        return self.run(return_fields={"mask"}, **kwargs).data["mask"]

    def detect(self, *_: Any, **__: Any) -> None:
        """Reserve the public extension point for future GPU MHW event detection."""
        raise NotImplementedError(
            "Event detection is intentionally outside CUDAHeat's first milestone; "
            "the GPU exceedance mask is the current pipeline endpoint."
        )

    def _execute_gpu_batch(
        self,
        cpu_values: np.ndarray,
        point_start: int,
        point_stop: int,
        rows: np.ndarray,
        percentile: float,
        smooth_window_hours: int,
        retries: int,
    ) -> tuple[BatchOutput, BatchTelemetry]:
        cp = cupy()
        with cp.cuda.Device(self.device_id):
            free_before, total_bytes = cp.cuda.runtime.memGetInfo()
            transfer_start = perf_counter()
            gpu_values = cp.asarray(cpu_values)
            cp.cuda.get_current_stream().synchronize()
            transfer_to_device = perf_counter() - transfer_start
            gpu_bins = cp.asarray(self._bins)
            gpu_rows = cp.asarray(rows)
            event_start = cp.cuda.Event()
            event_end = cp.cuda.Event()
            event_start.record()
            gpu_fields = gpu_pre_event_fields(
                gpu_values,
                gpu_bins,
                gpu_rows,
                percentile=percentile,
                smooth_window_hours=smooth_window_hours,
                nbins=HOURS_PER_LEAP_YEAR,
            )
            event_end.record()
            event_end.synchronize()
            gpu_compute = cp.cuda.get_elapsed_time(event_start, event_end) / 1000
            free_after_compute, _ = cp.cuda.runtime.memGetInfo()
            pool_used = cp.get_default_memory_pool().used_bytes()
            transfer_start = perf_counter()
            fields = self._transfer_requested_fields(gpu_fields)
            fields["exceedance_count"] = cp.asnumpy(
                cp.sum(gpu_fields.exceedance, axis=0, dtype=cp.uint64)
            )
            cp.cuda.get_current_stream().synchronize()
            transfer_from_device = perf_counter() - transfer_start
            free_after, _ = cp.cuda.runtime.memGetInfo()
            telemetry = BatchTelemetry(
                device_id=self.device_id,
                point_start=point_start,
                point_stop=point_stop,
                batch_size=point_stop - point_start,
                dtype=self.dtype.name,
                bytes_per_point=self._estimator.bytes_per_point,
                safety_fraction=self.vram_safety,
                oom_retries=retries,
                free_before_bytes=int(free_before),
                used_before_bytes=int(total_bytes - free_before),
                free_after_bytes=int(free_after),
                used_after_bytes=int(total_bytes - free_after),
                peak_used_bytes=int(total_bytes - min(free_before, free_after_compute, free_after)),
                pool_used_bytes=int(pool_used),
                transfer_to_device_seconds=transfer_to_device,
                gpu_compute_seconds=float(gpu_compute),
                transfer_from_device_seconds=transfer_from_device,
            )
            # CuPy pools are intentionally released after telemetry has been sampled.
            del gpu_fields, gpu_values, gpu_bins, gpu_rows
            release_gpu_memory(self.device_id)
            return BatchOutput(point_start, point_stop, fields, telemetry), telemetry

    def _transfer_requested_fields(self, fields: PreEventFields) -> dict[str, np.ndarray]:
        requested = getattr(self, "_active_return_fields", frozenset())
        mapping = {
            "climatology": fields.climatology,
            "threshold_cycle": fields.threshold_cycle,
            "threshold": fields.threshold,
            "anomaly": fields.anomaly,
            "excess": fields.excess,
            "mask": fields.exceedance,
        }
        cp = cupy()
        return {name: cp.asnumpy(mapping[name]) for name in requested}

    def _allocate_output_buffers(self, requested: frozenset[str]) -> dict[str, np.ndarray]:
        self._active_return_fields = requested
        specs: dict[str, tuple[tuple[int, int], np.dtype[Any]]] = {
            "climatology": ((HOURS_PER_LEAP_YEAR, self.npoints), self.dtype),
            "threshold_cycle": ((HOURS_PER_LEAP_YEAR, self.npoints), self.dtype),
            "threshold": ((self.ntime, self.npoints), self.dtype),
            "anomaly": ((self.ntime, self.npoints), self.dtype),
            "excess": ((self.ntime, self.npoints), self.dtype),
            "mask": ((self.ntime, self.npoints), np.dtype(bool)),
        }
        return {name: np.empty(*specs[name]) for name in requested}

    @staticmethod
    def _store_batch(
        fields: dict[str, np.ndarray], buffers: dict[str, np.ndarray], start: int, stop: int
    ) -> None:
        for name, values in fields.items():
            if name != "exceedance_count":
                buffers[name][..., start:stop] = values

    def _to_dataset(self, exceedance_count: np.ndarray, buffers: dict[str, np.ndarray]) -> xr.Dataset:
        data = {"exceedance_count": self._to_dataarray(exceedance_count, "exceedance_count")}
        for name, values in buffers.items():
            leading_dim = "climatology_hour" if name in {"climatology", "threshold_cycle"} else self.time_dim
            data[name] = self._to_dataarray(values, name, leading_dim)
        return xr.Dataset(data)

    def _to_dataarray(
        self, values: np.ndarray, name: str, leading_dim: str | None = None
    ) -> xr.DataArray:
        if leading_dim is None:
            array = xr.DataArray(values, dims=("point",), coords={"point": self._stacked["point"]}, name=name)
        elif leading_dim == self.time_dim:
            array = xr.DataArray(
                values,
                dims=(self.time_dim, "point"),
                coords={self.time_dim: self._time, "point": self._stacked["point"]},
                name=name,
            )
        else:
            array = xr.DataArray(
                values,
                dims=(leading_dim, "point"),
                coords={leading_dim: np.arange(HOURS_PER_LEAP_YEAR), "point": self._stacked["point"]},
                name=name,
            )
        return array.unstack("point")

    @staticmethod
    def _is_gpu_oom(exc: Exception) -> bool:
        cp = cupy()
        if isinstance(exc, cp.cuda.memory.OutOfMemoryError):
            return True
        return isinstance(exc, cp.cuda.runtime.CUDARuntimeError) and getattr(exc, "status", None) == 2
