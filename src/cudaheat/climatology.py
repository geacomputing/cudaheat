"""CPU and CuPy implementations of pre-event hourly climatology operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


HOURS_PER_LEAP_YEAR = 366 * 24
_MONTH_OFFSETS = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])


@dataclass
class PreEventFields:
    """Arrays created before future MHW event detection."""

    climatology: Any
    threshold_cycle: Any
    threshold: Any
    anomaly: Any
    excess: Any
    exceedance: Any


def hourly_climatology_index(time: Any) -> np.ndarray:
    """Map timestamps to a leap-year ``month/day/hour`` climatology index.

    A fixed 366-day cycle preserves month/day/hour alignment across leap and
    non-leap years.  The input must be hourly; the returned index is CPU-side
    metadata and is tiny relative to SST batches.
    """
    try:
        month = np.asarray(time.dt.month.values, dtype=np.int16)
        day = np.asarray(time.dt.day.values, dtype=np.int16)
        hour = np.asarray(time.dt.hour.values, dtype=np.int16)
        minute = np.asarray(time.dt.minute.values, dtype=np.int16)
        second = np.asarray(time.dt.second.values, dtype=np.int16)
    except (AttributeError, TypeError) as exc:
        raise ValueError("time coordinate must provide datetime-like .dt accessors") from exc
    if np.any(minute != 0) or np.any(second != 0):
        raise ValueError("CUDAHeat currently requires timestamps aligned to whole hours")
    return ((_MONTH_OFFSETS[month - 1] + day - 1) * 24 + hour).astype(np.int32)


def baseline_rows(time: Any, baseline: tuple[int, int] | None) -> np.ndarray:
    """Return rows used to build climatology and percentile statistics."""
    if baseline is None:
        return np.arange(time.size, dtype=np.intp)
    start, end = baseline
    if start > end:
        raise ValueError("baseline start year must not exceed end year")
    years = np.asarray(time.dt.year.values)
    rows = np.flatnonzero((years >= start) & (years <= end))
    if rows.size == 0:
        raise ValueError(f"baseline {baseline!r} contains no timestamps in this dataset")
    return rows.astype(np.intp)


def _smooth_circular(array: Any, window: int, xp: Any) -> Any:
    """Nan-aware centered circular moving mean along climatological time."""
    if window <= 1:
        return array
    if window > array.shape[0]:
        raise ValueError("smooth_window_hours may not exceed the climatological cycle")
    left = window // 2
    right = window - left - 1
    padded = xp.concatenate((array[-left:] if left else array[:0], array, array[:right]), axis=0)
    valid = ~xp.isnan(padded)
    filled = xp.where(valid, padded, 0)
    zeros = xp.zeros((1, array.shape[1]), dtype=array.dtype)
    sums = xp.concatenate((zeros, xp.cumsum(filled, axis=0)), axis=0)
    counts = xp.concatenate((zeros, xp.cumsum(valid, axis=0, dtype=array.dtype)), axis=0)
    numerator = sums[window:] - sums[:-window]
    denominator = counts[window:] - counts[:-window]
    safe_denominator = xp.where(denominator > 0, denominator, 1)
    return xp.where(denominator > 0, numerator / safe_denominator, xp.nan)


def smooth_circular_cpu(array: np.ndarray, window: int) -> np.ndarray:
    """CPU reference for circular climatology smoothing."""
    return _smooth_circular(array, window, np)


def _nanmean_by_bin_cpu(values: np.ndarray, bins: np.ndarray, nbins: int) -> np.ndarray:
    result = np.full((nbins, values.shape[1]), np.nan, dtype=values.dtype)
    for bin_id in np.unique(bins):
        selected = values[bins == bin_id]
        valid = ~np.isnan(selected)
        counts = valid.sum(axis=0)
        sums = np.where(valid, selected, 0).sum(axis=0)
        result[bin_id] = np.where(counts > 0, sums / np.where(counts > 0, counts, 1), np.nan)
    return result


def _nanpercentile_by_bin_cpu(
    values: np.ndarray, bins: np.ndarray, nbins: int, percentile: float
) -> np.ndarray:
    result = np.full((nbins, values.shape[1]), np.nan, dtype=values.dtype)
    for bin_id in np.unique(bins):
        result[bin_id] = _nanpercentile_cpu(values[bins == bin_id], percentile)
    return result


def _nanpercentile_cpu(values: np.ndarray, percentile: float) -> np.ndarray:
    """NaN-safe linear percentile matching the GPU definition without warnings."""
    valid = ~np.isnan(values)
    counts = valid.sum(axis=0)
    sorted_values = np.sort(np.where(valid, values, np.inf), axis=0)
    rank = (counts - 1) * percentile
    lower_index = np.maximum(np.floor(rank).astype(np.intp), 0)
    upper_index = np.maximum(np.ceil(rank).astype(np.intp), 0)
    lower = np.take_along_axis(sorted_values, lower_index[None, :], axis=0)[0]
    upper = np.take_along_axis(sorted_values, upper_index[None, :], axis=0)[0]
    lower = np.where(counts > 0, lower, 0)
    upper = np.where(counts > 0, upper, 0)
    output = lower + (upper - lower) * (rank - np.floor(rank))
    return np.where(counts > 0, output, np.nan)


def cpu_pre_event_fields(
    values: np.ndarray,
    bins: np.ndarray,
    baseline_indices: np.ndarray,
    percentile: float = 0.90,
    smooth_window_hours: int = 120,
    nbins: int = HOURS_PER_LEAP_YEAR,
) -> PreEventFields:
    """NumPy reference implementation for validation and small CPU workloads."""
    _validate_inputs(values, bins, baseline_indices, percentile, smooth_window_hours, nbins)
    reference = values[baseline_indices]
    reference_bins = bins[baseline_indices]
    climatology = smooth_circular_cpu(
        _nanmean_by_bin_cpu(reference, reference_bins, nbins), smooth_window_hours
    )
    threshold_cycle = smooth_circular_cpu(
        _nanpercentile_by_bin_cpu(reference, reference_bins, nbins, percentile),
        smooth_window_hours,
    )
    climatology_by_time = climatology[bins]
    threshold = threshold_cycle[bins]
    anomaly = values - climatology_by_time
    excess = values - threshold
    exceedance = np.logical_and(values > threshold, ~np.isnan(values))
    return PreEventFields(climatology, threshold_cycle, threshold, anomaly, excess, exceedance)


def _nanmean_by_bin_gpu(values: Any, bins: Any, nbins: int, cp: Any) -> Any:
    sums = cp.zeros((nbins, values.shape[1]), dtype=values.dtype)
    counts = cp.zeros((nbins, values.shape[1]), dtype=values.dtype)
    valid = ~cp.isnan(values)
    cp.add.at(sums, bins, cp.where(valid, values, 0))
    cp.add.at(counts, bins, valid.astype(values.dtype))
    return cp.where(counts > 0, sums / counts, cp.nan)


def _nanpercentile_by_bin_gpu(
    values: Any, bins: Any, nbins: int, percentile: float, cp: Any
) -> Any:
    """NaN-aware GPU percentile without a bespoke CUDA kernel.

    CuPy 13 does not expose ``nanpercentile``.  Sorting with CuPy and applying
    NumPy's default linear percentile interpolation keeps the work in VRAM and
    gives the same definition as the CPU reference.
    """
    result = cp.full((nbins, values.shape[1]), cp.nan, dtype=values.dtype)
    # Bins are CPU-created integer metadata.  The loop schedules independent GPU
    # percentile reductions and retains the complete series for every point.
    for bin_id in cp.asnumpy(cp.unique(bins)):
        rows = bins == bin_id
        result[int(bin_id)] = _nanpercentile_gpu(values[rows], percentile, cp)
    return result


def _nanpercentile_gpu(values: Any, percentile: float, cp: Any) -> Any:
    """GPU equivalent of ``numpy.nanpercentile(..., method='linear')``."""
    valid = ~cp.isnan(values)
    counts = cp.sum(valid, axis=0)
    sorted_values = cp.sort(cp.where(valid, values, cp.inf), axis=0)
    rank = (counts - 1) * percentile
    lower_index = cp.maximum(cp.floor(rank).astype(cp.intp), 0)
    upper_index = cp.maximum(cp.ceil(rank).astype(cp.intp), 0)
    lower = cp.take_along_axis(sorted_values, lower_index[None, :], axis=0)[0]
    upper = cp.take_along_axis(sorted_values, upper_index[None, :], axis=0)[0]
    lower = cp.where(counts > 0, lower, 0)
    upper = cp.where(counts > 0, upper, 0)
    fraction = rank - cp.floor(rank)
    percentile_values = lower + (upper - lower) * fraction
    return cp.where(counts > 0, percentile_values, cp.nan)


def gpu_pre_event_fields(
    values: Any,
    bins: Any,
    baseline_indices: Any,
    percentile: float = 0.90,
    smooth_window_hours: int = 120,
    nbins: int = HOURS_PER_LEAP_YEAR,
) -> PreEventFields:
    """CuPy pre-event pipeline; inputs and outputs remain in GPU VRAM."""
    from .memory import cupy

    cp = cupy()
    _validate_inputs(values, bins, baseline_indices, percentile, smooth_window_hours, nbins)
    reference = values[baseline_indices]
    reference_bins = bins[baseline_indices]
    climatology = _smooth_circular(
        _nanmean_by_bin_gpu(reference, reference_bins, nbins, cp), smooth_window_hours, cp
    )
    threshold_cycle = _smooth_circular(
        _nanpercentile_by_bin_gpu(reference, reference_bins, nbins, percentile, cp),
        smooth_window_hours,
        cp,
    )
    climatology_by_time = climatology[bins]
    threshold = threshold_cycle[bins]
    anomaly = values - climatology_by_time
    excess = values - threshold
    exceedance = cp.logical_and(values > threshold, ~cp.isnan(values))
    return PreEventFields(climatology, threshold_cycle, threshold, anomaly, excess, exceedance)


def _validate_inputs(
    values: Any,
    bins: Any,
    baseline_indices: Any,
    percentile: float,
    smooth_window_hours: int,
    nbins: int,
) -> None:
    if values.ndim != 2:
        raise ValueError("values must have shape (time, point)")
    if bins.ndim != 1 or bins.shape[0] != values.shape[0]:
        raise ValueError("bins must have exactly one entry for every complete time step")
    if baseline_indices.ndim != 1 or baseline_indices.size == 0:
        raise ValueError("baseline_indices must be a non-empty one-dimensional array")
    if not 0 < percentile < 1:
        raise ValueError("percentile must be strictly between 0 and 1")
    if not 1 <= smooth_window_hours <= nbins:
        raise ValueError("smooth_window_hours must be between 1 and the climatological cycle length")
