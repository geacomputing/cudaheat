"""Out-of-core CUDAHeat engine: complete time series, spatial batches only."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import xarray as xr

from .climatology import (
    CALENDAR_MODE_DAY_OF_YEAR,
    CALENDAR_MODE_MONTH_DAY_HOUR,
    HOURS_PER_LEAP_YEAR,
    PreEventFields,
    baseline_rows,
    climatology_bin_count,
    gpu_pre_event_fields,
    hourly_climatology_index,
    validate_contiguous_hourly_time,
)
from .detection import EventRuns, detect_events
from .exceptions import InsufficientVRAMError
from .io import NetCDFReader, is_optimized_point_layout
from .memory import (
    BatchEstimate,
    BatchTelemetry,
    VRAMEstimator,
    cupy,
    guess_batch_size,
    get_gpu_info,
    reduced_batch_size,
    release_gpu_memory,
)
from .preprocessing import PreparedNetCDF, prepare_netcdf_input
from .result import EventDetectionResult, EventGrid
from .timing import log_elapsed_time

logger = logging.getLogger(__name__)
_CUDAHEAT_SMOOTH_WINDOW_HOURS = 31 * 24
_HOBDAY_SAMPLING_WINDOW_HALF_WIDTH_DAYS = 5


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


@dataclass
class EventBatchOutput:
    """Event runs and explicitly retained cycles for one spatial GPU batch."""

    point_start: int
    point_stop: int
    events: EventRuns
    fields: dict[str, np.ndarray]
    telemetry: BatchTelemetry


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
        Fraction of currently free VRAM held in reserve (default 0.15).
    devices:
        Selected CUDA device IDs. The first milestone runs batches sequentially
        on the first selected device; the executor is device-explicit so future
        schedulers can distribute independent spatial batches over this list.
    lazy:
        Open a NetCDF path with Dask-backed Xarray arrays. Each CPU batch still
        contains every time step for its selected spatial points.
    lazy_spatial_chunk:
        Maximum source chunk width along the final spatial dimension for lazy
        reads. It affects disk-to-CPU I/O only, not CUDAHeat's GPU batch size.
    calendar_mode:
        ``"month_day_hour"`` (default) groups equivalent calendar hours in a
        fixed 366-day cycle. ``"dayofyear"`` groups by fractional day of year
        (day-of-year plus clock hour); both modes use the same hourly Hobday
        sampling and smoothing defaults.
    timing_logging:
        Print per-batch disk-I/O and host-to-GPU transfer timing records.
        Defaults to ``True`` for backwards compatibility. Disable it for
        quiet production runs; telemetry is still collected in the result.
    batch_size:
        An explicit positive spatial batch size, or ``"guess"`` (default).
        The guess selects the closest of 512, 1024, 1536, or 2048 to one
        quarter of the safe VRAM ceiling. It is immediate: no trial CUDA
        work, calibration, or performance-cache file is created.
    input_optimization:
        For NetCDF paths, ``"auto"`` (default) uses an existing CUDAHeat
        ``time × point`` file or creates a validated sibling cache for a grid.
        ``"skip"`` retains the historical direct-input behaviour and
        ``"require"`` rejects an unoptimized grid. Dataset inputs are never
        written or replaced.
    """

    _RETURNABLE_FIELDS = frozenset(
        {
            "climatology_raw",
            "climatology",
            "threshold_raw",
            "threshold_cycle",
            "threshold",
            "anomaly",
            "excess",
            "mask",
        }
    )
    _RETAINABLE_DETECTION_FIELDS = frozenset({"climatology", "threshold_cycle"})

    def __init__(
        self,
        source: str | Path | xr.Dataset,
        *,
        variable: str,
        dtype: str | np.dtype[Any] = "float32",
        vram_safety: float = 0.15,
        devices: Iterable[int] = (0,),
        time_dim: str | None = "time",
        point_indices: Iterable[int] | slice | None = None,
        lazy: bool = False,
        lazy_spatial_chunk: int = 256,
        calendar_mode: str = CALENDAR_MODE_MONTH_DAY_HOUR,
        timing_logging: bool = True,
        batch_size: int | Literal["guess"] = "guess",
        input_optimization: Literal["auto", "skip", "require"] = "auto",
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
        self.lazy = bool(lazy)
        self.lazy_spatial_chunk = int(lazy_spatial_chunk)
        self.calendar_mode = str(calendar_mode)
        self.timing_logging = bool(timing_logging)
        if batch_size != "guess" and (
            not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer or 'guess'")
        if input_optimization not in {"auto", "skip", "require"}:
            raise ValueError("input_optimization must be 'auto', 'skip', or 'require'")
        self.requested_batch_size = batch_size
        self.input_optimization = input_optimization
        self.original_source_path = str(source) if not isinstance(source, xr.Dataset) else None
        self.input_preparation: PreparedNetCDF | None = None
        self.source_path = self.original_source_path
        self._owns_dataset = not isinstance(source, xr.Dataset)
        self._reader: NetCDFReader | None = None
        if self._owns_dataset:
            # Preparation occurs before Xarray opens the active input, so an
            # ordinary grid can safely become a separate compact cache file.
            self.input_preparation = prepare_netcdf_input(
                source,
                variable=variable,
                dtype=self.dtype,
                strategy=self.input_optimization,
                time_dim=time_dim,
            )
            self.source_path = str(self.input_preparation.active_path)
            self._reader = NetCDFReader(
                self.input_preparation.active_path,
                variable=variable,
                lazy=self.lazy,
                time_dim=time_dim,
                spatial_chunk=self.lazy_spatial_chunk,
            )
            self._dataset = self._reader.dataset
        else:
            if self.lazy:
                raise ValueError("lazy=True requires a NetCDF path, not an already-open Dataset")
            self._dataset = source
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
        self._optimized_point_layout = is_optimized_point_layout(self._source, self.time_dim)
        if self._optimized_point_layout:
            # The one-time preprocessor already performed spatial stacking and
            # permanent-land removal.  Keeping this direct prevents a second
            # stack operation and preserves compact point-index semantics.
            self._stacked = self._source.transpose(self.time_dim, "point")
        else:
            self._stacked = (
                self._source.transpose(self.time_dim, *self.spatial_dims)
                .stack(point=self.spatial_dims)
                .transpose(self.time_dim, "point")
            )
        if self._reader is not None:
            self._stacked = self._reader.rechunk_stacked(self._stacked)
        all_point_count = int(self._stacked.sizes["point"])
        self.point_indices = self._normalise_point_indices(point_indices, all_point_count)
        if point_indices is not None:
            # Preserve a contiguous slice for Dask-backed input.  Replacing it
            # with its equivalent integer array turns this into fancy indexing
            # and can make a small spatial subset depend on every source chunk.
            selection = point_indices if isinstance(point_indices, slice) else self.point_indices
            self._stacked = self._stacked.isel(point=selection)
        self._point_coordinates = (
            {
                name: np.array(coordinate.values, copy=True)
                for name, coordinate in self._stacked.coords.items()
                if name != "point" and coordinate.dims == ("point",)
            }
            if self._optimized_point_layout
            else {}
        )
        original_flat = self._point_coordinates.get("original_flat_index")
        self._original_flat_index = (
            np.asarray(original_flat, dtype=np.int64) if original_flat is not None else None
        )
        self.ntime = int(self._stacked.sizes[self.time_dim])
        self.npoints = int(self._stacked.sizes["point"])
        if self.ntime < 1 or self.npoints < 1:
            self.close()
            raise ValueError("SST variable must have non-empty time and spatial dimensions")
        self._time = self._stacked[self.time_dim]
        validate_contiguous_hourly_time(self._time)
        self._bins = hourly_climatology_index(self._time, calendar_mode=self.calendar_mode)
        self._estimator = VRAMEstimator(
            self.ntime, self.dtype.itemsize, safety_fraction=self.vram_safety
        )
        self.gpu_info = get_gpu_info(self.device_id)
        self.gpu_infos = tuple(get_gpu_info(device) for device in self.devices)
        self.batch_estimate = self._estimator.estimate(self.gpu_info.free_bytes, self.npoints)
        if self.batch_estimate.batch_size < 1:
            self.close()
            raise InsufficientVRAMError(self._single_series_message(self.batch_estimate))
        self.batch_size = self._resolve_configured_batch_size()
        self.last_result: PipelineResult | None = None
        self.last_detection_result: EventDetectionResult | None = None
        self._print_input_preparation()
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
        self._print_batch_choice()
        if len(self.devices) > 1:
            logger.info(
                "Selected devices %s; batches currently execute sequentially on device %d",
                self.devices,
                self.device_id,
            )

    def __enter__(self) -> "CUDAHeat":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close a NetCDF dataset opened by this instance."""
        if getattr(self, "_owns_dataset", False) and getattr(self, "_dataset", None) is not None:
            if self._reader is not None:
                self._reader.close()
            else:
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
            if dim in self._source.coords and np.issubdtype(self._source[dim].dtype, np.datetime64)
        ]
        if len(datetime_dims) == 1:
            return datetime_dims[0]
        raise ValueError("could not identify one time dimension; pass time_dim explicitly")

    @staticmethod
    def _normalise_point_indices(
        point_indices: Iterable[int] | slice | None, all_point_count: int
    ) -> np.ndarray:
        if point_indices is None:
            return np.arange(all_point_count, dtype=np.intp)
        if isinstance(point_indices, slice):
            indices = np.arange(all_point_count, dtype=np.intp)[point_indices]
        else:
            indices = np.asarray(tuple(point_indices), dtype=np.intp)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("point_indices must select at least one spatial point")
        if np.any(indices < 0) or np.any(indices >= all_point_count):
            raise IndexError("point_indices contains an out-of-range spatial point")
        if np.unique(indices).size != indices.size:
            raise ValueError("point_indices must not contain duplicates")
        return indices

    def _single_series_message(self, estimate: BatchEstimate) -> str:
        return (
            "A single complete SST time series cannot fit in the configured GPU budget. "
            f"ntime={self.ntime}, estimated={estimate.bytes_per_point:,} bytes/point, "
            f"budget={estimate.budget_bytes:,} bytes after vram_safety={self.vram_safety:.2f}. "
            "CUDAHeat will not split time; use float32, choose a larger GPU, or reduce "
            "the scientific memory model only after measurement."
        )

    def _resolve_configured_batch_size(self) -> int:
        """Resolve the engine-level explicit size or immediate structural guess."""
        if self.requested_batch_size == "guess":
            return guess_batch_size(self.batch_estimate, npoints=self.npoints)
        return min(int(self.requested_batch_size), self.npoints)

    def _print_batch_choice(self) -> None:
        """Report the immediate batch decision without running calibration work."""
        if not self.timing_logging:
            return
        print(
            "[CUDAHeat batch] "
            f"mode={self.requested_batch_size} safe_ceiling={self.batch_estimate.batch_size} "
            f"selected={self.batch_size}",
            flush=True,
        )

    def _print_input_preparation(self) -> None:
        """Report the non-destructive input decision before CUDA execution begins."""
        if not self.timing_logging or self.input_preparation is None:
            return
        prepared = self.input_preparation
        if prepared.was_already_optimized:
            finding = "already_optimized"
        elif prepared.reused_cached_optimization:
            finding = "reused_optimized_cache"
        else:
            finding = "optimized_new_input"
        report = prepared.report
        details = ""
        if report is not None:
            details = (
                f" chunks={report.output_chunks} retained_points={report.retained_points} "
                f"removed_points={report.removed_points} elapsed={report.elapsed_seconds:.2f}s"
            )
        print(
            "[CUDAHeat input] "
            f"{finding} | source={prepared.source_path} active={prepared.active_path}{details}",
            flush=True,
        )

    def _load_spatial_batch(
        self, point_start: int, point_stop: int, dtype: str | np.dtype[Any] | None = None
    ) -> np.ndarray:
        """Read one full-time spatial slice from Xarray into CPU RAM.

        This is intentionally the only disk-to-RAM selection used by the engine:
        it slices ``point`` and never ``time``.
        """
        values, _, _ = self._load_spatial_batch_profiled(point_start, point_stop, dtype)
        return values

    def _load_spatial_batch_profiled(
        self, point_start: int, point_stop: int, dtype: str | np.dtype[Any] | None = None
    ) -> tuple[np.ndarray, float, float]:
        """Load one full-time spatial batch and separate I/O from preparation."""
        read_start = perf_counter()
        target_dtype = self.dtype if dtype is None else np.dtype(dtype)
        if target_dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("batch dtype must be float32 or float64")
        if self._reader is not None:
            values = self._reader.materialize_spatial_batch(
                self._stacked,
                point_start,
                point_stop,
                dtype=target_dtype,
            )
        else:
            selected = self._stacked.isel(point=slice(point_start, point_stop)).load()
            try:
                values = np.ascontiguousarray(selected.data, dtype=target_dtype)
            finally:
                del selected
        host_read_seconds = log_elapsed_time(
            "disk-to-CPU batch I/O",
            read_start,
            enabled=self.timing_logging,
            points=f"{point_start}:{point_stop}",
            time_steps=self.ntime,
            host_bytes=values.nbytes,
        )
        prepare_start = perf_counter()
        if values.shape != (self.ntime, point_stop - point_start):
            raise RuntimeError("spatial batch lost part of its time series")
        return values, host_read_seconds, perf_counter() - prepare_start

    def run(
        self,
        *,
        baseline: tuple[int, int] | None = None,
        percentile: float = 0.90,
        smooth_window_hours: int | None = None,
        smooth_min_periods: int | None = None,
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
        smooth_window_hours = self._resolve_smooth_window_hours(smooth_window_hours)
        rows = baseline_rows(self._time, baseline)
        climatology_nbins = self._climatology_nbins(rows)
        resolved_smooth_min_periods = self._resolve_smooth_min_periods(
            smooth_window_hours, smooth_min_periods
        )
        buffers = self._allocate_output_buffers(requested, climatology_nbins)
        exceedance_count = np.zeros(self.npoints, dtype=np.uint64)
        telemetry: list[BatchTelemetry] = []
        point_start = 0
        active_batch_size = self.batch_size

        while point_start < self.npoints:
            current_size = min(active_batch_size, self.npoints - point_start)
            retries = 0
            while True:
                point_stop = point_start + current_size
                logger.info(
                    "Processing spatial points %d:%d with all %d time steps on GPU %d",
                    point_start,
                    point_stop,
                    self.ntime,
                    self.device_id,
                )
                cpu_values, host_read_seconds, host_prepare_seconds = (
                    self._load_spatial_batch_profiled(point_start, point_stop)
                )
                try:
                    output, batch_telemetry = self._execute_gpu_batch(
                        cpu_values,
                        point_start,
                        point_stop,
                        rows,
                        percentile,
                        smooth_window_hours,
                        resolved_smooth_min_periods,
                        climatology_nbins,
                        retries,
                        host_read_seconds,
                        host_prepare_seconds,
                    )
                except Exception as exc:
                    del cpu_values
                    if not self._is_gpu_oom(exc):
                        raise
                    release_gpu_memory(self.device_id)
                    if current_size == 1:
                        raise InsufficientVRAMError(
                            self._single_series_message(self.batch_estimate)
                        ) from exc
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
                del cpu_values
                assembly_start = perf_counter()
                self._store_batch(output.fields, buffers, point_start, point_stop)
                exceedance_count[point_start:point_stop] = output.fields.pop("exceedance_count")
                batch_telemetry = replace(
                    batch_telemetry,
                    result_assembly_seconds=perf_counter() - assembly_start,
                )
                completed = BatchOutput(point_start, point_stop, output.fields, batch_telemetry)
                if on_batch is not None:
                    on_batch(completed)
                telemetry.append(batch_telemetry)
                logger.info(
                    "Completed points %d:%d: transfer %.3fs + GPU %.3fs + return %.3fs; "
                    "sampled peak %.2f GiB",
                    point_start,
                    point_stop,
                    batch_telemetry.transfer_to_device_seconds,
                    batch_telemetry.gpu_compute_seconds,
                    batch_telemetry.transfer_from_device_seconds,
                    batch_telemetry.peak_used_bytes / 2**30,
                )
                point_start = point_stop
                break

        self.batch_size = active_batch_size
        final_assembly_start = perf_counter()
        data = self._to_dataset(exceedance_count, buffers, climatology_nbins)
        final_assembly_seconds = perf_counter() - final_assembly_start
        result = PipelineResult(
            data=data,
            telemetry=tuple(telemetry),
            metadata={
                "variable": self.variable,
                "dtype": self.dtype.name,
                "npoints": self.npoints,
                "baseline": baseline,
                "percentile": percentile,
                "smooth_window_hours": smooth_window_hours,
                "smooth_min_periods": resolved_smooth_min_periods,
                "sampling_window_half_width_days": _HOBDAY_SAMPLING_WINDOW_HALF_WIDTH_DAYS,
                "same_hour_smoothing": True,
                "calendar_mode": self.calendar_mode,
                "time_dimension": self.time_dim,
                "spatial_dimensions": self.spatial_dims,
                "optimized_point_layout": self._optimized_point_layout,
                "time_was_split": False,
                "vram_safety": self.vram_safety,
                "initial_batch_estimate": self.batch_estimate.as_dict(),
                "devices": self.devices,
                "requested_batch_size": self.requested_batch_size,
                "batch_size": self.batch_size,
                "batch_selection": "guess" if self.requested_batch_size == "guess" else "explicit",
                "source_path": self.source_path,
                "original_source_path": self.original_source_path,
                "input_preparation": (
                    None if self.input_preparation is None else self.input_preparation.as_dict()
                ),
                "result_dataset_assembly_seconds": final_assembly_seconds,
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

    def validate(self, **kwargs: Any) -> Any:
        """Compare float64 CPU fields with GPU fields without splitting time.

        The returned :class:`cudaheat.validation.ValidationResult` contains
        numerical, NaN, mask-boundary, and per-point count diagnostics.
        """
        from .validation import validate_cpu_gpu

        return validate_cpu_gpu(self, **kwargs)

    def detect(
        self,
        *,
        baseline: tuple[int, int] | None = None,
        percentile: float = 0.90,
        smooth_window_hours: int | None = None,
        smooth_min_periods: int | None = None,
        min_duration_hours: int = 120,
        max_internal_gap_hours: int = 6,
        join_gap_hours: int = 48,
        retain_fields: Iterable[str] = (),
    ) -> EventDetectionResult:
        """Detect vectorized threshold-exceedance events on the GPU.

        The pipeline preserves raw exceedance, fills short internal gaps,
        filters contiguous runs by duration, then joins nearby candidates.
        By default only the compact event catalogue is retained on the host.
        ``retain_fields={"climatology", "threshold_cycle"}`` additionally
        preserves the smoothed annual cycles needed for CPU-only event plots.
        These fields have shape ``(climatology_hour, point)``; a full
        timestamp-aligned threshold is deliberately not retained.
        """
        if not isinstance(min_duration_hours, int) or isinstance(min_duration_hours, bool):
            raise TypeError("min_duration_hours must be an integer")
        if min_duration_hours < 1:
            raise ValueError("min_duration_hours must be positive")
        if not isinstance(max_internal_gap_hours, int) or isinstance(max_internal_gap_hours, bool):
            raise TypeError("max_internal_gap_hours must be an integer")
        if max_internal_gap_hours < 0:
            raise ValueError("max_internal_gap_hours must be non-negative")
        if not isinstance(join_gap_hours, int) or isinstance(join_gap_hours, bool):
            raise TypeError("join_gap_hours must be an integer")
        if join_gap_hours < 0:
            raise ValueError("join_gap_hours must be non-negative")
        retained = frozenset(retain_fields)
        unknown_retained = retained - self._RETAINABLE_DETECTION_FIELDS
        if unknown_retained:
            raise ValueError(
                "unknown retained detection field(s): " f"{sorted(unknown_retained)!r}"
            )
        smooth_window_hours = self._resolve_smooth_window_hours(smooth_window_hours)
        rows = baseline_rows(self._time, baseline)
        climatology_nbins = self._climatology_nbins(rows)
        resolved_smooth_min_periods = self._resolve_smooth_min_periods(
            smooth_window_hours, smooth_min_periods
        )
        retained_buffers = {
            name: np.empty((climatology_nbins, self.npoints), dtype=self.dtype)
            for name in retained
        }
        event_parts: list[EventRuns] = []
        telemetry: list[BatchTelemetry] = []
        point_start = 0
        active_batch_size = self.batch_size

        while point_start < self.npoints:
            current_size = min(active_batch_size, self.npoints - point_start)
            retries = 0
            while True:
                point_stop = point_start + current_size
                cpu_values, host_read_seconds, host_prepare_seconds = (
                    self._load_spatial_batch_profiled(point_start, point_stop)
                )
                try:
                    output, batch_telemetry = self._execute_gpu_detection_batch(
                        cpu_values,
                        point_start,
                        point_stop,
                        rows,
                        percentile,
                        smooth_window_hours,
                        resolved_smooth_min_periods,
                        climatology_nbins,
                        min_duration_hours,
                        max_internal_gap_hours,
                        join_gap_hours,
                        retries,
                        host_read_seconds,
                        host_prepare_seconds,
                        retained,
                    )
                except Exception as exc:
                    del cpu_values
                    if not self._is_gpu_oom(exc):
                        raise
                    release_gpu_memory(self.device_id)
                    if current_size == 1:
                        raise InsufficientVRAMError(
                            self._single_series_message(self.batch_estimate)
                        ) from exc
                    retries += 1
                    current_size = reduced_batch_size(current_size)
                    active_batch_size = current_size
                    logger.warning(
                        "GPU OOM during detection for points %d:%d; retrying with %d points (retry %d)",
                        point_start,
                        point_stop,
                        current_size,
                        retries,
                    )
                    continue
                assembly_start = perf_counter()
                event_parts.append(
                    EventRuns(
                        point=np.asarray(output.events.point, dtype=np.intp) + point_start,
                        start=np.asarray(output.events.start, dtype=np.intp),
                        stop=np.asarray(output.events.stop, dtype=np.intp),
                        duration=np.asarray(output.events.duration, dtype=np.intp),
                    )
                )
                self._store_batch(output.fields, retained_buffers, point_start, point_stop)
                del cpu_values
                telemetry.append(
                    replace(
                        batch_telemetry,
                        result_assembly_seconds=perf_counter() - assembly_start,
                    )
                )
                del output
                point_start = point_stop
                break

        self.batch_size = active_batch_size
        final_assembly_start = perf_counter()
        data = self._event_dataset(event_parts)
        retained_field_data = self._retained_cycle_dataset(retained_buffers, climatology_nbins)
        grid = EventGrid(
            time=np.array(self._time.values, copy=True),
            point_index=self._stacked.indexes["point"].copy(),
            point_indices=self.point_indices.copy(),
            time_dim=self.time_dim,
            point_coordinates=self._point_coordinates.copy(),
        )
        result = EventDetectionResult(
            data=data,
            telemetry=tuple(telemetry),
            metadata={
                "variable": self.variable,
                "dtype": self.dtype.name,
                "time_dimension": self.time_dim,
                "spatial_dimensions": self.spatial_dims,
                "optimized_point_layout": self._optimized_point_layout,
                "baseline": baseline,
                "percentile": percentile,
                "smooth_window_hours": smooth_window_hours,
                "smooth_min_periods": resolved_smooth_min_periods,
                "sampling_window_half_width_days": _HOBDAY_SAMPLING_WINDOW_HALF_WIDTH_DAYS,
                "same_hour_smoothing": True,
                "calendar_mode": self.calendar_mode,
                "min_duration_hours": min_duration_hours,
                "max_internal_gap_hours": max_internal_gap_hours,
                "join_gap_hours": join_gap_hours,
                "retained_fields": tuple(sorted(retained)),
                "time_was_split": False,
                "detection_stages": (
                    "exceedance",
                    "internal_gap_fill",
                    "contiguous_runs",
                    "minimum_duration",
                    "join_candidate_runs",
                ),
                "requested_batch_size": self.requested_batch_size,
                "batch_size": self.batch_size,
                "batch_selection": "guess" if self.requested_batch_size == "guess" else "explicit",
                "source_path": self.source_path,
                "original_source_path": self.original_source_path,
                "input_preparation": (
                    None if self.input_preparation is None else self.input_preparation.as_dict()
                ),
                "result_dataset_assembly_seconds": perf_counter() - final_assembly_start,
            },
            _grid=grid,
            retained_fields=retained_field_data,
        )
        self.last_detection_result = result
        return result

    def _execute_gpu_batch(
        self,
        cpu_values: np.ndarray,
        point_start: int,
        point_stop: int,
        rows: np.ndarray,
        percentile: float,
        smooth_window_hours: int,
        smooth_min_periods: int,
        climatology_nbins: int,
        retries: int,
        host_read_seconds: float,
        host_prepare_seconds: float,
    ) -> tuple[BatchOutput, BatchTelemetry]:
        cp = cupy()
        with cp.cuda.Device(self.device_id):
            free_before, total_bytes = cp.cuda.runtime.memGetInfo()
            allocation_start = perf_counter()
            gpu_values = cp.empty(cpu_values.shape, dtype=cpu_values.dtype)
            cp.cuda.get_current_stream().synchronize()
            gpu_allocation_seconds = perf_counter() - allocation_start
            transfer_start = perf_counter()
            gpu_values.set(cpu_values)
            cp.cuda.get_current_stream().synchronize()
            transfer_to_device = log_elapsed_time(
                "CPU-to-GPU batch transfer",
                transfer_start,
                enabled=self.timing_logging,
                points=f"{point_start}:{point_stop}",
                time_steps=cpu_values.shape[0],
                host_bytes=cpu_values.nbytes,
                device_id=self.device_id,
            )
            metadata_start = perf_counter()
            gpu_bins = cp.asarray(self._bins)
            gpu_rows = cp.asarray(rows)
            cp.cuda.get_current_stream().synchronize()
            gpu_metadata_transfer_seconds = perf_counter() - metadata_start
            event_start = cp.cuda.Event()
            event_end = cp.cuda.Event()
            event_start.record()
            gpu_fields = gpu_pre_event_fields(
                gpu_values,
                gpu_bins,
                gpu_rows,
                percentile=percentile,
                smooth_window_hours=smooth_window_hours,
                smooth_min_periods=smooth_min_periods,
                nbins=climatology_nbins,
                calendar_mode=self.calendar_mode,
                sampling_window_half_width_days=_HOBDAY_SAMPLING_WINDOW_HALF_WIDTH_DAYS,
                smooth_same_hour=True,
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
                host_read_seconds=host_read_seconds,
                host_prepare_seconds=host_prepare_seconds,
                gpu_allocation_seconds=gpu_allocation_seconds,
                gpu_metadata_transfer_seconds=gpu_metadata_transfer_seconds,
            )
            # CuPy pools are intentionally released after telemetry has been sampled.
            del gpu_fields, gpu_values, gpu_bins, gpu_rows
            release_gpu_memory(self.device_id)
            return BatchOutput(point_start, point_stop, fields, telemetry), telemetry

    def _execute_gpu_detection_batch(
        self,
        cpu_values: np.ndarray,
        point_start: int,
        point_stop: int,
        rows: np.ndarray,
        percentile: float,
        smooth_window_hours: int,
        smooth_min_periods: int,
        climatology_nbins: int,
        min_duration_hours: int,
        max_internal_gap_hours: int,
        join_gap_hours: int,
        retries: int,
        host_read_seconds: float,
        host_prepare_seconds: float,
        retained: frozenset[str],
    ) -> tuple[EventBatchOutput, BatchTelemetry]:
        """Preprocess and detect one batch, retaining requested compact cycles."""
        cp = cupy()
        with cp.cuda.Device(self.device_id):
            free_before, total_bytes = cp.cuda.runtime.memGetInfo()
            allocation_start = perf_counter()
            gpu_values = cp.empty(cpu_values.shape, dtype=cpu_values.dtype)
            cp.cuda.get_current_stream().synchronize()
            gpu_allocation_seconds = perf_counter() - allocation_start
            transfer_start = perf_counter()
            gpu_values.set(cpu_values)
            cp.cuda.get_current_stream().synchronize()
            transfer_to_device = log_elapsed_time(
                "CPU-to-GPU batch transfer",
                transfer_start,
                enabled=self.timing_logging,
                points=f"{point_start}:{point_stop}",
                time_steps=cpu_values.shape[0],
                host_bytes=cpu_values.nbytes,
                device_id=self.device_id,
            )
            metadata_start = perf_counter()
            gpu_bins = cp.asarray(self._bins)
            gpu_rows = cp.asarray(rows)
            cp.cuda.get_current_stream().synchronize()
            gpu_metadata_transfer_seconds = perf_counter() - metadata_start
            event_start = cp.cuda.Event()
            event_end = cp.cuda.Event()
            event_start.record()
            gpu_fields = gpu_pre_event_fields(
                gpu_values,
                gpu_bins,
                gpu_rows,
                percentile=percentile,
                smooth_window_hours=smooth_window_hours,
                smooth_min_periods=smooth_min_periods,
                nbins=climatology_nbins,
                calendar_mode=self.calendar_mode,
                sampling_window_half_width_days=_HOBDAY_SAMPLING_WINDOW_HALF_WIDTH_DAYS,
                smooth_same_hour=True,
            )
            detected = detect_events(
                gpu_values,
                gpu_fields.threshold,
                min_duration_hours=min_duration_hours,
                max_internal_gap_hours=max_internal_gap_hours,
                join_gap_hours=join_gap_hours,
                xp=cp,
            )
            event_end.record()
            event_end.synchronize()
            gpu_compute = cp.cuda.get_elapsed_time(event_start, event_end) / 1000
            free_after_compute, _ = cp.cuda.runtime.memGetInfo()
            pool_used = cp.get_default_memory_pool().used_bytes()
            transfer_start = perf_counter()
            events = EventRuns(
                point=cp.asnumpy(detected.events.point),
                start=cp.asnumpy(detected.events.start),
                stop=cp.asnumpy(detected.events.stop),
                duration=cp.asnumpy(detected.events.duration),
            )
            retained_fields = {
                name: cp.asnumpy(getattr(gpu_fields, name)) for name in retained
            }
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
                host_read_seconds=host_read_seconds,
                host_prepare_seconds=host_prepare_seconds,
                gpu_allocation_seconds=gpu_allocation_seconds,
                gpu_metadata_transfer_seconds=gpu_metadata_transfer_seconds,
            )
            del detected, gpu_fields, gpu_values, gpu_bins, gpu_rows
            release_gpu_memory(self.device_id)
            return (
                EventBatchOutput(point_start, point_stop, events, retained_fields, telemetry),
                telemetry,
            )

    def _transfer_requested_fields(self, fields: PreEventFields) -> dict[str, np.ndarray]:
        requested = getattr(self, "_active_return_fields", frozenset())
        mapping = {
            "climatology_raw": fields.climatology_raw,
            "climatology": fields.climatology,
            "threshold_raw": fields.threshold_raw,
            "threshold_cycle": fields.threshold_cycle,
            "threshold": fields.threshold,
            "anomaly": fields.anomaly,
            "excess": fields.excess,
            "mask": fields.exceedance,
        }
        cp = cupy()
        return {name: cp.asnumpy(mapping[name]) for name in requested}

    def _allocate_output_buffers(
        self, requested: frozenset[str], climatology_nbins: int
    ) -> dict[str, np.ndarray]:
        self._active_return_fields = requested
        specs: dict[str, tuple[tuple[int, int], np.dtype[Any]]] = {
            "climatology_raw": ((climatology_nbins, self.npoints), self.dtype),
            "climatology": ((climatology_nbins, self.npoints), self.dtype),
            "threshold_raw": ((climatology_nbins, self.npoints), self.dtype),
            "threshold_cycle": ((climatology_nbins, self.npoints), self.dtype),
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

    def _to_dataset(
        self, exceedance_count: np.ndarray, buffers: dict[str, np.ndarray], climatology_nbins: int
    ) -> xr.Dataset:
        data = {"exceedance_count": self._to_dataarray(exceedance_count, "exceedance_count")}
        for name, values in buffers.items():
            leading_dim = (
                "climatology_hour"
                if name in {"climatology_raw", "climatology", "threshold_raw", "threshold_cycle"}
                else self.time_dim
            )
            data[name] = self._to_dataarray(values, name, leading_dim, climatology_nbins)
        return xr.Dataset(data)

    def _retained_cycle_dataset(
        self, buffers: dict[str, np.ndarray], climatology_nbins: int
    ) -> xr.Dataset:
        """Wrap retained cycle buffers without expanding them over time."""
        data = {
            name: xr.DataArray(
                values,
                dims=("climatology_hour", "point"),
                coords={
                    "climatology_hour": np.arange(climatology_nbins),
                    "point": self._stacked["point"],
                },
                name=name,
            )
            for name, values in buffers.items()
        }
        return xr.Dataset(data)

    def _event_dataset(self, event_parts: list[EventRuns]) -> xr.Dataset:
        """Assemble compact host event runs into a stable xarray event table."""
        point = np.concatenate([part.point for part in event_parts])
        start = np.concatenate([part.start for part in event_parts])
        stop = np.concatenate([part.stop for part in event_parts])
        duration = np.concatenate([part.duration for part in event_parts])
        event_ids = np.arange(1, start.size + 1, dtype=np.int64)
        time_values = np.asarray(self._time.values)
        data = xr.Dataset(
            {
                "point_index": ("event", self.point_indices[point]),
                "start_index": ("event", start),
                "stop_index": ("event", stop),
                "duration_hours": ("event", duration),
                "start_time": ("event", time_values[start]),
                "end_time": ("event", time_values[stop - 1]),
            },
            coords={"event": event_ids},
        )
        if self._original_flat_index is not None:
            data["original_flat_index"] = ("event", self._original_flat_index[point])
            data["point_index"].attrs["description"] = "CUDAHeat compact point index."
            data["original_flat_index"].attrs["description"] = (
                "C-order flattened point index in the source latitude x longitude grid."
            )
        event_count = np.bincount(point, minlength=self.npoints).astype(np.uint64, copy=False)
        # ``original_flat_index`` is event-level provenance above.  It cannot
        # also be a point-level coordinate in this same Dataset under Xarray's
        # coordinate/data-variable naming rules.
        data["event_count"] = self._to_dataarray(
            event_count, "event_count", exclude_point_coordinates={"original_flat_index"}
        )
        data["stop_index"].attrs["description"] = (
            "Exclusive time index; event interval is [start, stop)."
        )
        return data

    def _to_dataarray(
        self,
        values: np.ndarray,
        name: str,
        leading_dim: str | None = None,
        climatology_nbins: int = HOURS_PER_LEAP_YEAR,
        exclude_point_coordinates: set[str] | None = None,
    ) -> xr.DataArray:
        if leading_dim is None:
            array = xr.DataArray(
                values, dims=("point",), coords={"point": self._stacked["point"]}, name=name
            )
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
                coords={leading_dim: np.arange(climatology_nbins), "point": self._stacked["point"]},
                name=name,
            )
        if not self._optimized_point_layout:
            return array.unstack("point")
        excluded = exclude_point_coordinates or set()
        return array.assign_coords(
            {
                name: ("point", values)
                for name, values in self._point_coordinates.items()
                if name not in array.coords and name not in excluded
            }
        )

    def _climatology_nbins(self, rows: np.ndarray) -> int:
        return climatology_bin_count(self._bins, rows, calendar_mode=self.calendar_mode)

    def _resolve_smooth_min_periods(self, window: int, requested: int | None) -> int:
        if requested is None:
            requested = 1
        if not isinstance(requested, int) or isinstance(requested, bool):
            raise TypeError("smooth_min_periods must be an integer or None")
        if not 1 <= requested <= window // 24:
            raise ValueError(
                "smooth_min_periods must be between 1 and the number of smoothing days"
            )
        return requested

    def _resolve_smooth_window_hours(self, requested: int | None) -> int:
        if requested is None:
            return _CUDAHEAT_SMOOTH_WINDOW_HOURS
        if not isinstance(requested, int) or isinstance(requested, bool):
            raise TypeError("smooth_window_hours must be an integer or None")
        if requested < 24 or requested % 24:
            raise ValueError(
                "smooth_window_hours must be a positive whole number of days for hourly Hobday smoothing"
            )
        return requested

    @staticmethod
    def _is_gpu_oom(exc: Exception) -> bool:
        cp = cupy()
        if isinstance(exc, cp.cuda.memory.OutOfMemoryError):
            return True
        return (
            isinstance(exc, cp.cuda.runtime.CUDARuntimeError) and getattr(exc, "status", None) == 2
        )
