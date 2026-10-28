"""Structured scientific validation of CPU reference and GPU CUDAHeat fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from .climatology import PreEventFields, baseline_rows, cpu_pre_event_fields, gpu_pre_event_fields
from .exceptions import InsufficientVRAMError
from .memory import cupy, release_gpu_memory, reduced_batch_size


_HOBDAY_SAMPLING_WINDOW_HALF_WIDTH_DAYS = 5

if TYPE_CHECKING:
    from .core import CUDAHeat


@dataclass(frozen=True)
class FloatMetrics:
    """Numerical agreement and NaN diagnostics for one floating-point field."""

    compared_values: int
    cpu_nan_count: int
    gpu_nan_count: int
    nan_location_disagreement: int
    max_absolute_error: float | None
    mean_absolute_error: float | None
    rmse: float | None
    max_relative_error: float | None
    mean_relative_error: float | None
    cpu_min: float | None
    cpu_mean: float | None
    cpu_max: float | None
    gpu_min: float | None
    gpu_mean: float | None
    gpu_max: float | None


@dataclass(frozen=True)
class MaskMetrics:
    """Classification agreement, including threshold-boundary diagnostics."""

    total_elements: int
    disagreements: int
    disagreement_fraction: float
    agreement_percent: float
    false_positives: int
    false_negatives: int
    threshold_distance_min: float | None
    threshold_distance_mean: float | None
    threshold_distance_max: float | None
    examples: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CountMetrics:
    """Per-point exceedance-count comparison."""

    total_points: int
    exact_matches: int
    mismatched_points: int
    max_absolute_difference: int
    mean_absolute_difference: float
    examples: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ValidationResult:
    """Programmatic result of a complete-time CPU/GPU validation run."""

    level: str
    point_indices: tuple[int, ...]
    ntime: int
    cpu_dtype: str
    gpu_dtype: str
    baseline: tuple[int, int] | None
    percentile: float
    smooth_window_hours: int
    float_fields: dict[str, FloatMetrics]
    mask: MaskMetrics
    exceedance_count: CountMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe structured validation results."""
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        """Write the structured result without scientific arrays."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    def summary(self) -> str:
        """Create a short human-readable scientific-validation summary."""
        lines = [
            f"CPU {self.cpu_dtype} vs GPU {self.gpu_dtype}; "
            f"{len(self.point_indices)} complete time series × {self.ntime} hours",
            f"mask agreement: {self.mask.agreement_percent:.8f}% "
            f"({self.mask.disagreements}/{self.mask.total_elements} disagreements)",
            f"count matches: {self.exceedance_count.exact_matches}/{self.exceedance_count.total_points}",
        ]
        for name, metrics in self.float_fields.items():
            lines.append(f"{name}: max abs={metrics.max_absolute_error}, RMSE={metrics.rmse}")
        return "\n".join(lines)


class _FloatAccumulator:
    def __init__(self, relative_epsilon: float) -> None:
        self.relative_epsilon = relative_epsilon
        self.compared = self.cpu_nans = self.gpu_nans = self.nan_disagreements = 0
        self.abs_sum = self.square_sum = self.relative_sum = 0.0
        self.max_abs: float | None = None
        self.max_relative: float | None = None
        self.relative_count = 0
        self.cpu_sum = self.gpu_sum = 0.0
        self.cpu_count = self.gpu_count = 0
        self.cpu_min: float | None = None
        self.cpu_max: float | None = None
        self.gpu_min: float | None = None
        self.gpu_max: float | None = None

    def update(self, cpu_values: np.ndarray, gpu_values: np.ndarray) -> None:
        cpu_nan = np.isnan(cpu_values)
        gpu_nan = np.isnan(gpu_values)
        self.cpu_nans += int(cpu_nan.sum())
        self.gpu_nans += int(gpu_nan.sum())
        self.nan_disagreements += int(np.logical_xor(cpu_nan, gpu_nan).sum())
        valid = ~(cpu_nan | gpu_nan)
        if valid.any():
            cpu = cpu_values[valid].astype(np.float64, copy=False)
            gpu = gpu_values[valid].astype(np.float64, copy=False)
            absolute = np.abs(cpu - gpu)
            self.compared += int(absolute.size)
            self.abs_sum += float(absolute.sum())
            self.square_sum += float(np.square(absolute).sum())
            candidate = float(absolute.max())
            self.max_abs = candidate if self.max_abs is None else max(self.max_abs, candidate)
            relative_valid = np.abs(cpu) > self.relative_epsilon
            if relative_valid.any():
                relative = absolute[relative_valid] / np.abs(cpu[relative_valid])
                self.relative_count += int(relative.size)
                self.relative_sum += float(relative.sum())
                candidate = float(relative.max())
                self.max_relative = (
                    candidate if self.max_relative is None else max(self.max_relative, candidate)
                )
        self._update_summary(cpu_values, True)
        self._update_summary(gpu_values, False)

    def _update_summary(self, values: np.ndarray, is_cpu: bool) -> None:
        finite = values[np.isfinite(values)].astype(np.float64, copy=False)
        if finite.size == 0:
            return
        low, high, total = float(finite.min()), float(finite.max()), float(finite.sum())
        if is_cpu:
            self.cpu_count += int(finite.size)
            self.cpu_sum += total
            self.cpu_min = low if self.cpu_min is None else min(self.cpu_min, low)
            self.cpu_max = high if self.cpu_max is None else max(self.cpu_max, high)
        else:
            self.gpu_count += int(finite.size)
            self.gpu_sum += total
            self.gpu_min = low if self.gpu_min is None else min(self.gpu_min, low)
            self.gpu_max = high if self.gpu_max is None else max(self.gpu_max, high)

    def result(self) -> FloatMetrics:
        return FloatMetrics(
            compared_values=self.compared,
            cpu_nan_count=self.cpu_nans,
            gpu_nan_count=self.gpu_nans,
            nan_location_disagreement=self.nan_disagreements,
            max_absolute_error=self.max_abs,
            mean_absolute_error=self.abs_sum / self.compared if self.compared else None,
            rmse=(self.square_sum / self.compared) ** 0.5 if self.compared else None,
            max_relative_error=self.max_relative,
            mean_relative_error=self.relative_sum / self.relative_count if self.relative_count else None,
            cpu_min=self.cpu_min,
            cpu_mean=self.cpu_sum / self.cpu_count if self.cpu_count else None,
            cpu_max=self.cpu_max,
            gpu_min=self.gpu_min,
            gpu_mean=self.gpu_sum / self.gpu_count if self.gpu_count else None,
            gpu_max=self.gpu_max,
        )


class _MaskAccumulator:
    def __init__(self, maximum_examples: int) -> None:
        self.maximum_examples = maximum_examples
        self.total = self.disagreements = self.false_positives = self.false_negatives = 0
        self.distance_sum = 0.0
        self.distance_count = 0
        self.distance_min: float | None = None
        self.distance_max: float | None = None
        self.examples: list[dict[str, Any]] = []

    def update(
        self,
        cpu_mask: np.ndarray,
        gpu_mask: np.ndarray,
        cpu_values: np.ndarray,
        cpu_threshold: np.ndarray,
        gpu_values: np.ndarray,
        gpu_threshold: np.ndarray,
        time: np.ndarray,
        global_points: np.ndarray,
    ) -> None:
        self.total += int(cpu_mask.size)
        mismatch = cpu_mask != gpu_mask
        mismatch_count = int(mismatch.sum())
        self.disagreements += mismatch_count
        self.false_positives += int(np.logical_and(gpu_mask, ~cpu_mask).sum())
        self.false_negatives += int(np.logical_and(cpu_mask, ~gpu_mask).sum())
        if mismatch_count == 0:
            return
        distances = np.abs(cpu_values[mismatch] - cpu_threshold[mismatch]).astype(np.float64)
        finite_distances = distances[np.isfinite(distances)]
        if finite_distances.size:
            self.distance_count += int(finite_distances.size)
            self.distance_sum += float(finite_distances.sum())
            low, high = float(finite_distances.min()), float(finite_distances.max())
            self.distance_min = low if self.distance_min is None else min(self.distance_min, low)
            self.distance_max = high if self.distance_max is None else max(self.distance_max, high)
        remaining = self.maximum_examples - len(self.examples)
        if remaining <= 0:
            return
        for time_index, point_column in np.argwhere(mismatch)[:remaining]:
            self.examples.append(
                {
                    "time_index": int(time_index),
                    "time": str(time[time_index]),
                    "point_index": int(global_points[point_column]),
                    "cpu_sst": float(cpu_values[time_index, point_column]),
                    "gpu_sst": float(gpu_values[time_index, point_column]),
                    "cpu_threshold": float(cpu_threshold[time_index, point_column]),
                    "gpu_threshold": float(gpu_threshold[time_index, point_column]),
                    "abs_cpu_sst_minus_threshold": float(
                        abs(cpu_values[time_index, point_column] - cpu_threshold[time_index, point_column])
                    ),
                    "cpu_exceedance": bool(cpu_mask[time_index, point_column]),
                    "gpu_exceedance": bool(gpu_mask[time_index, point_column]),
                }
            )

    def result(self) -> MaskMetrics:
        return MaskMetrics(
            total_elements=self.total,
            disagreements=self.disagreements,
            disagreement_fraction=self.disagreements / self.total if self.total else 0.0,
            agreement_percent=100 * (self.total - self.disagreements) / self.total if self.total else 100.0,
            false_positives=self.false_positives,
            false_negatives=self.false_negatives,
            threshold_distance_min=self.distance_min,
            threshold_distance_mean=self.distance_sum / self.distance_count if self.distance_count else None,
            threshold_distance_max=self.distance_max,
            examples=tuple(self.examples),
        )


class _CountAccumulator:
    def __init__(self, maximum_examples: int) -> None:
        self.maximum_examples = maximum_examples
        self.total = self.matches = self.abs_sum = self.max_abs = 0
        self.examples: list[dict[str, Any]] = []

    def update(self, cpu_mask: np.ndarray, gpu_mask: np.ndarray, global_points: np.ndarray) -> None:
        cpu_counts = cpu_mask.sum(axis=0, dtype=np.uint64)
        gpu_counts = gpu_mask.sum(axis=0, dtype=np.uint64)
        differences = np.abs(cpu_counts.astype(np.int64) - gpu_counts.astype(np.int64))
        self.total += int(differences.size)
        self.matches += int((differences == 0).sum())
        self.abs_sum += int(differences.sum())
        self.max_abs = max(self.max_abs, int(differences.max(initial=0)))
        remaining = self.maximum_examples - len(self.examples)
        for point_column in np.flatnonzero(differences)[:remaining]:
            self.examples.append(
                {
                    "point_index": int(global_points[point_column]),
                    "cpu_count": int(cpu_counts[point_column]),
                    "gpu_count": int(gpu_counts[point_column]),
                    "absolute_difference": int(differences[point_column]),
                }
            )

    def result(self) -> CountMetrics:
        return CountMetrics(
            total_points=self.total,
            exact_matches=self.matches,
            mismatched_points=self.total - self.matches,
            max_absolute_difference=self.max_abs,
            mean_absolute_difference=self.abs_sum / self.total if self.total else 0.0,
            examples=tuple(self.examples),
        )


def select_validation_points(
    engine: "CUDAHeat",
    *,
    points: int,
    random_seed: int = 0,
    point_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Select deterministic finite SST points, loading complete series only.

    Explicit indices are honoured exactly. Random selection skips columns that
    are completely NaN, using full time series to establish validity.
    """
    if point_indices is not None:
        selected = np.asarray(tuple(point_indices), dtype=np.intp)
        if selected.ndim != 1 or selected.size == 0:
            raise ValueError("point_indices must contain at least one index")
        if np.any(selected < 0) or np.any(selected >= engine.npoints):
            raise IndexError("validation point index is out of range")
        if np.unique(selected).size != selected.size:
            raise ValueError("validation point indices must be unique")
        return selected
    if points < 1:
        raise ValueError("points must be positive")
    rng = np.random.default_rng(random_seed)
    candidates = rng.permutation(engine.npoints)
    selected: list[int] = []
    probe_size = max(256, points * 4)
    for start in range(0, candidates.size, probe_size):
        candidates_here = candidates[start : start + probe_size]
        values = _load_points(engine, candidates_here, np.dtype("float64"))
        finite = np.isfinite(values).any(axis=0)
        selected.extend(int(index) for index in candidates_here[finite])
        if len(selected) >= points:
            return np.asarray(selected[:points], dtype=np.intp)
    raise ValueError(f"only {len(selected)} spatial points contain finite SST observations")


def validate_cpu_gpu(
    engine: "CUDAHeat",
    *,
    points: int = 10,
    point_indices: Iterable[int] | None = None,
    full_dataset: bool = False,
    random_seed: int = 0,
    gpu_dtype: str = "float32",
    baseline: tuple[int, int] | None = None,
    percentile: float = 0.90,
    smooth_window_hours: int | None = None,
    smooth_min_periods: int | None = None,
    validation_batch_size: int | None = None,
    relative_epsilon: float = 1e-12,
    maximum_examples: int = 20,
) -> ValidationResult:
    """Compare a float64 CPU reference against the selected GPU precision.

    Both targeted and full modes retain every time step for every selected
    point. ``validation_batch_size`` partitions spatial points only.
    """
    gpu_dtype_value = np.dtype(gpu_dtype)
    if gpu_dtype_value not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("gpu_dtype must be float32 or float64")
    if full_dataset and point_indices is not None:
        raise ValueError("full_dataset and explicit point_indices are mutually exclusive")
    selected = (
        np.arange(engine.npoints, dtype=np.intp)
        if full_dataset
        else select_validation_points(
            engine, points=points, random_seed=random_seed, point_indices=point_indices
        )
    )
    if validation_batch_size is not None and validation_batch_size < 1:
        raise ValueError("validation_batch_size must be positive")
    active_batch_size = min(validation_batch_size or 256, selected.size)
    smooth_window_hours = engine._resolve_smooth_window_hours(smooth_window_hours)
    rows = baseline_rows(engine._time, baseline)
    climatology_nbins = engine._climatology_nbins(rows)
    resolved_smooth_min_periods = engine._resolve_smooth_min_periods(
        smooth_window_hours, smooth_min_periods
    )
    float_names = (
        "climatology_raw",
        "climatology",
        "threshold_raw",
        "threshold_cycle",
        "threshold",
        "anomaly",
        "excess",
    )
    floats = {name: _FloatAccumulator(relative_epsilon) for name in float_names}
    mask = _MaskAccumulator(maximum_examples)
    counts = _CountAccumulator(maximum_examples)
    offset = 0
    while offset < selected.size:
        current_size = min(active_batch_size, selected.size - offset)
        while True:
            local_indices = selected[offset : offset + current_size]
            cpu_values = _load_points(engine, local_indices, np.dtype("float64"))
            cpu_fields = cpu_pre_event_fields(
                cpu_values,
                engine._bins,
                rows,
                percentile,
                smooth_window_hours,
                climatology_nbins,
                resolved_smooth_min_periods,
                engine.calendar_mode,
                _HOBDAY_SAMPLING_WINDOW_HALF_WIDTH_DAYS,
                True,
            )
            try:
                gpu_fields = _gpu_fields(
                    engine,
                    cpu_values,
                    gpu_dtype_value,
                    rows,
                    percentile,
                    smooth_window_hours,
                    climatology_nbins,
                    resolved_smooth_min_periods,
                    engine.calendar_mode,
                )
            except Exception as exc:
                if not engine._is_gpu_oom(exc):
                    raise
                release_gpu_memory(engine.device_id)
                if current_size == 1:
                    raise InsufficientVRAMError(
                        "A single complete time series cannot fit during validation; "
                        "CUDAHeat will not split time."
                    ) from exc
                current_size = reduced_batch_size(current_size)
                active_batch_size = current_size
                continue
            break
        for name in float_names:
            floats[name].update(getattr(cpu_fields, name), getattr(gpu_fields, name))
        cpu_mask = cpu_fields.exceedance
        gpu_mask = gpu_fields.exceedance
        global_points = engine.point_indices[local_indices]
        mask.update(
            cpu_mask,
            gpu_mask,
            cpu_values,
            cpu_fields.threshold,
            gpu_fields_input_values(cpu_values, gpu_dtype_value),
            gpu_fields.threshold,
            np.asarray(engine._time.values),
            global_points,
        )
        counts.update(cpu_mask, gpu_mask, global_points)
        offset += current_size
    return ValidationResult(
        level="full" if full_dataset else "targeted",
        point_indices=tuple(int(index) for index in engine.point_indices[selected]),
        ntime=engine.ntime,
        cpu_dtype="float64",
        gpu_dtype=gpu_dtype_value.name,
        baseline=baseline,
        percentile=percentile,
        smooth_window_hours=smooth_window_hours,
        float_fields={name: accumulator.result() for name, accumulator in floats.items()},
        mask=mask.result(),
        exceedance_count=counts.result(),
    )


def _load_points(engine: "CUDAHeat", local_indices: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    if local_indices.size == 0:
        raise ValueError("no validation points selected")
    if np.all(np.diff(local_indices) == 1):
        return engine._load_spatial_batch(int(local_indices[0]), int(local_indices[-1]) + 1, dtype)
    selected = engine._stacked.isel(point=local_indices).load()
    values = np.asarray(selected.data, dtype=dtype)
    if values.shape != (engine.ntime, local_indices.size):
        raise RuntimeError("validation selection lost part of a time series")
    return np.ascontiguousarray(values)


def _gpu_fields(
    engine: "CUDAHeat",
    cpu_values: np.ndarray,
    gpu_dtype: np.dtype[Any],
    rows: np.ndarray,
    percentile: float,
    smooth_window_hours: int,
    climatology_nbins: int,
    smooth_min_periods: int,
    calendar_mode: str,
) -> PreEventFields:
    cp = cupy()
    with cp.cuda.Device(engine.device_id):
        values = cp.asarray(cpu_values.astype(gpu_dtype, copy=False))
        bins = cp.asarray(engine._bins)
        gpu_rows = cp.asarray(rows)
        fields = gpu_pre_event_fields(
            values,
            bins,
            gpu_rows,
            percentile,
            smooth_window_hours,
            climatology_nbins,
            smooth_min_periods,
            calendar_mode,
            _HOBDAY_SAMPLING_WINDOW_HALF_WIDTH_DAYS,
            True,
        )
        copied = PreEventFields(*(cp.asnumpy(getattr(fields, name)) for name in fields.__dataclass_fields__))
        del fields, values, bins, gpu_rows
        release_gpu_memory(engine.device_id)
        return copied


def gpu_fields_input_values(cpu_values: np.ndarray, gpu_dtype: np.dtype[Any]) -> np.ndarray:
    """Represent the explicit source-to-GPU precision conversion in diagnostics."""
    return cpu_values.astype(gpu_dtype, copy=False).astype(np.float64, copy=False)
