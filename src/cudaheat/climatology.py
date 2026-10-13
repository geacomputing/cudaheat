"""CPU and CuPy implementations of pre-event hourly climatology operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


HOURS_PER_LEAP_YEAR = 366 * 24
_MONTH_OFFSETS = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
CALENDAR_MODE_MONTH_DAY_HOUR = "month_day_hour"
CALENDAR_MODE_DAY_OF_YEAR = "dayofyear"
_CALENDAR_MODES = frozenset((CALENDAR_MODE_MONTH_DAY_HOUR, CALENDAR_MODE_DAY_OF_YEAR))


@dataclass
class PreEventFields:
    """Arrays created before future MHW event detection."""

    climatology_raw: Any
    climatology: Any
    threshold_raw: Any
    threshold_cycle: Any
    threshold: Any
    anomaly: Any
    excess: Any
    exceedance: Any


def hourly_climatology_index(
    time: Any, *, calendar_mode: str = CALENDAR_MODE_MONTH_DAY_HOUR
) -> np.ndarray:
    """Map hourly timestamps to a climatology index.

    ``"month_day_hour"`` is the historic CUDAHeat default: a fixed 366-day
    month/day/hour cycle, so a date has the same bin in leap and non-leap
    years. ``"dayofyear"`` reproduces HourWave's grouping convention,
    ``(time.dt.dayofyear - 1) * 24 + time.dt.hour``. In the latter convention,
    dates after February shift by one day between leap and non-leap years.

    The input must be aligned to whole hours. The returned index is CPU-side
    metadata and is tiny relative to SST batches.
    """
    try:
        month = np.asarray(time.dt.month.values, dtype=np.int16)
        day = np.asarray(time.dt.day.values, dtype=np.int16)
        dayofyear = np.asarray(time.dt.dayofyear.values, dtype=np.int16)
        hour = np.asarray(time.dt.hour.values, dtype=np.int16)
        minute = np.asarray(time.dt.minute.values, dtype=np.int16)
        second = np.asarray(time.dt.second.values, dtype=np.int16)
    except (AttributeError, TypeError) as exc:
        raise ValueError("time coordinate must provide datetime-like .dt accessors") from exc
    if np.any(minute != 0) or np.any(second != 0):
        raise ValueError("CUDAHeat currently requires timestamps aligned to whole hours")
    if calendar_mode == CALENDAR_MODE_MONTH_DAY_HOUR:
        return ((_MONTH_OFFSETS[month - 1] + day - 1) * 24 + hour).astype(np.int32)
    if calendar_mode == CALENDAR_MODE_DAY_OF_YEAR:
        return ((dayofyear - 1) * 24 + hour).astype(np.int32)
    raise ValueError(
        f"calendar_mode must be one of {sorted(_CALENDAR_MODES)!r}; got {calendar_mode!r}"
    )


def climatology_bin_count(
    bins: np.ndarray,
    baseline_indices: np.ndarray,
    *,
    calendar_mode: str = CALENDAR_MODE_MONTH_DAY_HOUR,
) -> int:
    """Return the cycle length required by one baseline and calendar mode.

    The fixed month/day/hour convention always exposes all 366 days. HourWave
    groups by observed day-of-year labels, so a non-leap baseline has 365 × 24
    groups while a baseline containing a leap year has 366 × 24 groups.
    """
    if calendar_mode == CALENDAR_MODE_MONTH_DAY_HOUR:
        return HOURS_PER_LEAP_YEAR
    if calendar_mode != CALENDAR_MODE_DAY_OF_YEAR:
        raise ValueError(
            f"calendar_mode must be one of {sorted(_CALENDAR_MODES)!r}; got {calendar_mode!r}"
        )
    reference_bins = np.asarray(bins)[np.asarray(baseline_indices)]
    if reference_bins.size == 0:
        raise ValueError("baseline_indices must be non-empty")
    return int(np.max(reference_bins)) + 1


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


def _smooth_circular(array: Any, window: int, xp: Any, min_periods: int = 1) -> Any:
    """Nan-aware centered circular moving mean along climatological time."""
    if window <= 1:
        return array
    if window > array.shape[0]:
        raise ValueError("smooth_window_hours may not exceed the climatological cycle")
    if not 1 <= min_periods <= window:
        raise ValueError("smooth_min_periods must be between 1 and smooth_window_hours")
    left = window // 2
    right = window - left - 1
    padded = xp.concatenate((array[-left:] if left else array[:0], array, array[:right]), axis=0)
    valid = ~xp.isnan(padded)
    filled = xp.where(valid, padded, 0)
    # A float32 prefix sum over a 366-day cycle loses enough precision that
    # subtracting nearby large totals can shift a short rolling mean by mK.
    # xarray's rolling reduction avoids that cancellation; accumulate in
    # float64, then restore the requested output precision.
    accumulation_dtype = xp.float64 if array.dtype.itemsize < 8 else array.dtype
    zeros = xp.zeros((1, array.shape[1]), dtype=accumulation_dtype)
    sums = xp.concatenate((zeros, xp.cumsum(filled, axis=0, dtype=accumulation_dtype)), axis=0)
    counts = xp.concatenate((zeros, xp.cumsum(valid, axis=0, dtype=accumulation_dtype)), axis=0)
    numerator = sums[window:] - sums[:-window]
    denominator = counts[window:] - counts[:-window]
    safe_denominator = xp.where(denominator > 0, denominator, 1)
    output = xp.where(denominator >= min_periods, numerator / safe_denominator, xp.nan)
    return output.astype(array.dtype, copy=False)


def _smooth_hourwave_cycle(
    array: Any, reference_bins: Any, window: int, xp: Any, min_periods: int
) -> Any:
    """Smooth populated day-of-year groups exactly as HourWave's groupby does.

    Xarray's ``groupby`` omits timestamp groups absent from the baseline, then
    rolls over the remaining ordered labels.  Keep that source behavior while
    returning CUDAHeat's dense cycle representation: absent labels stay NaN.
    """
    populated_bins = xp.unique(reference_bins)
    compact = _smooth_circular(array[populated_bins], window, xp, min_periods)
    output = xp.full(array.shape, xp.nan, dtype=array.dtype)
    output[populated_bins] = compact
    return output


def _smooth_cycle(
    array: Any,
    reference_bins: Any,
    window: int,
    xp: Any,
    min_periods: int,
    calendar_mode: str,
) -> Any:
    if calendar_mode == CALENDAR_MODE_DAY_OF_YEAR:
        return _smooth_hourwave_cycle(array, reference_bins, window, xp, min_periods)
    return _smooth_circular(array, window, xp, min_periods)


def smooth_circular_cpu(array: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    """CPU reference for circular climatology smoothing."""
    return _smooth_circular(array, window, np, min_periods)


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


def _cycle_by_time(cycle: Any, bins: Any, xp: Any) -> Any:
    """Align a climatology cycle to time without indexing missing baseline bins.

    A day-of-year baseline without a leap year has no day-366 group. Returning
    NaN for that target group mirrors an xarray ``reindex`` instead of raising
    from direct label selection.
    """
    valid = (bins >= 0) & (bins < cycle.shape[0])
    safe_bins = xp.clip(bins, 0, cycle.shape[0] - 1)
    aligned = cycle[safe_bins]
    return xp.where(valid[:, None], aligned, xp.nan)


def cpu_pre_event_fields(
    values: np.ndarray,
    bins: np.ndarray,
    baseline_indices: np.ndarray,
    percentile: float = 0.90,
    smooth_window_hours: int = 120,
    nbins: int = HOURS_PER_LEAP_YEAR,
    smooth_min_periods: int = 1,
    calendar_mode: str = CALENDAR_MODE_MONTH_DAY_HOUR,
) -> PreEventFields:
    """NumPy reference implementation for validation and small CPU workloads."""
    _validate_inputs(
        values, bins, baseline_indices, percentile, smooth_window_hours, smooth_min_periods, nbins
    )
    _validate_calendar_mode(calendar_mode)
    reference = values[baseline_indices]
    reference_bins = bins[baseline_indices]
    climatology_raw = _nanmean_by_bin_cpu(reference, reference_bins, nbins)
    climatology = _smooth_cycle(
        climatology_raw,
        reference_bins,
        smooth_window_hours,
        np,
        smooth_min_periods,
        calendar_mode,
    )
    threshold_raw = _nanpercentile_by_bin_cpu(reference, reference_bins, nbins, percentile)
    threshold_cycle = _smooth_cycle(
        threshold_raw,
        reference_bins,
        smooth_window_hours,
        np,
        smooth_min_periods,
        calendar_mode,
    )
    climatology_by_time = _cycle_by_time(climatology, bins, np)
    threshold = _cycle_by_time(threshold_cycle, bins, np)
    anomaly = values - climatology_by_time
    excess = values - threshold
    exceedance = np.logical_and(values > threshold, ~np.isnan(values))
    return PreEventFields(
        climatology_raw,
        climatology,
        threshold_raw,
        threshold_cycle,
        threshold,
        anomaly,
        excess,
        exceedance,
    )


def _nanmean_by_bin_gpu(values: Any, bins: Any, nbins: int, cp: Any) -> Any:
    sums = cp.zeros((nbins, values.shape[1]), dtype=values.dtype)
    counts = cp.zeros((nbins, values.shape[1]), dtype=values.dtype)
    valid = ~cp.isnan(values)
    cp.add.at(sums, bins, cp.where(valid, values, 0))
    cp.add.at(counts, bins, valid.astype(values.dtype))
    return cp.where(counts > 0, sums / counts, cp.nan)


def _nanpercentile_by_bin_gpu_legacy(
    values: Any, bins: Any, nbins: int, percentile: float, cp: Any
) -> Any:
    """Previous tiled percentile implementation retained for equivalence checks.

    It transfers the bin vector to the host and constructs per-tile gather
    indices in Python.  Keep this private reference until the replacement has
    been validated against it.
    """
    result = cp.full((nbins, values.shape[1]), cp.nan, dtype=values.dtype)
    bins_cpu = cp.asnumpy(bins)
    populated_bins = np.unique(bins_cpu)
    tile_size = 256
    for tile_start in range(0, populated_bins.size, tile_size):
        tile_bins = populated_bins[tile_start : tile_start + tile_size]
        row_lists = [np.flatnonzero(bins_cpu == bin_id) for bin_id in tile_bins]
        max_rows = max(rows.size for rows in row_lists)
        gather_indices = np.zeros((tile_bins.size, max_rows), dtype=np.intp)
        present = np.zeros((tile_bins.size, max_rows), dtype=bool)
        for row, indices in enumerate(row_lists):
            gather_indices[row, : indices.size] = indices
            present[row, : indices.size] = True
        grouped = values[cp.asarray(gather_indices)]
        grouped = cp.where(cp.asarray(present)[..., None], grouped, cp.nan)
        result[cp.asarray(tile_bins)] = _nanpercentile_grouped_gpu(grouped, percentile, cp)
    return result


def _nanpercentile_by_bin_gpu(
    values: Any, bins: Any, nbins: int, percentile: float, cp: Any
) -> Any:
    """NaN-aware tiled percentile, retaining bin grouping on the GPU.

    CuPy has no ``nanpercentile``.  The calculation therefore still sorts each
    bounded ``(bin, baseline_sample, point)`` tile, but GPU sorting of the bin
    vector replaces the previous host copy and Python construction of one row
    list per populated bin.  One scalar maximum tile count is read per tile to
    preserve the previous bounded temporary allocation for irregular inputs.
    """
    result = cp.full((nbins, values.shape[1]), cp.nan, dtype=values.dtype)
    tile_size = 256

    # Stable ordering is unnecessary because every gathered group is sorted
    # again by _nanpercentile_grouped_gpu.  It only needs to make equal-bin rows
    # contiguous so that searchsorted can describe each ragged group.
    row_order = cp.argsort(bins)
    sorted_bins = bins[row_order]
    populated_bins = cp.unique(sorted_bins)

    for tile_start in range(0, populated_bins.size, tile_size):
        tile_bins = populated_bins[tile_start : tile_start + tile_size]
        starts = cp.searchsorted(sorted_bins, tile_bins, side="left")
        stops = cp.searchsorted(sorted_bins, tile_bins, side="right")
        counts = stops - starts

        # A scalar read keeps this ragged tile compact even for unusual input
        # calendars.  The former implementation already synchronised once per
        # tile while transferring its gather metadata, but moved the full bin
        # vector and did all group construction on the CPU.
        max_rows = int(counts.max())
        positions = starts[:, None] + cp.arange(max_rows, dtype=starts.dtype)
        present = positions < stops[:, None]
        safe_positions = cp.minimum(positions, values.shape[0] - 1)
        grouped = values[row_order[safe_positions]]
        grouped = cp.where(present[..., None], grouped, cp.nan)
        result[tile_bins] = _nanpercentile_grouped_gpu(grouped, percentile, cp)
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


def _nanpercentile_grouped_gpu(values: Any, percentile: float, cp: Any) -> Any:
    """Linear NaN percentile for ``(bin, baseline_sample, point)`` values."""
    valid = ~cp.isnan(values)
    counts = cp.sum(valid, axis=1)
    sorted_values = cp.sort(cp.where(valid, values, cp.inf), axis=1)
    rank = (counts - 1) * percentile
    lower_index = cp.maximum(cp.floor(rank).astype(cp.intp), 0)
    upper_index = cp.maximum(cp.ceil(rank).astype(cp.intp), 0)
    lower = cp.take_along_axis(sorted_values, lower_index[:, None, :], axis=1)[:, 0, :]
    upper = cp.take_along_axis(sorted_values, upper_index[:, None, :], axis=1)[:, 0, :]
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
    smooth_min_periods: int = 1,
    calendar_mode: str = CALENDAR_MODE_MONTH_DAY_HOUR,
) -> PreEventFields:
    """CuPy pre-event pipeline; inputs and outputs remain in GPU VRAM."""
    from .memory import cupy

    cp = cupy()
    _validate_inputs(
        values, bins, baseline_indices, percentile, smooth_window_hours, smooth_min_periods, nbins
    )
    _validate_calendar_mode(calendar_mode)
    reference = values[baseline_indices]
    reference_bins = bins[baseline_indices]
    climatology_raw = _nanmean_by_bin_gpu(reference, reference_bins, nbins, cp)
    climatology = _smooth_cycle(
        climatology_raw,
        reference_bins,
        smooth_window_hours,
        cp,
        smooth_min_periods,
        calendar_mode,
    )
    threshold_raw = _nanpercentile_by_bin_gpu(reference, reference_bins, nbins, percentile, cp)
    threshold_cycle = _smooth_cycle(
        threshold_raw,
        reference_bins,
        smooth_window_hours,
        cp,
        smooth_min_periods,
        calendar_mode,
    )
    climatology_by_time = _cycle_by_time(climatology, bins, cp)
    threshold = _cycle_by_time(threshold_cycle, bins, cp)
    anomaly = values - climatology_by_time
    excess = values - threshold
    exceedance = cp.logical_and(values > threshold, ~cp.isnan(values))
    return PreEventFields(
        climatology_raw,
        climatology,
        threshold_raw,
        threshold_cycle,
        threshold,
        anomaly,
        excess,
        exceedance,
    )


def _validate_inputs(
    values: Any,
    bins: Any,
    baseline_indices: Any,
    percentile: float,
    smooth_window_hours: int,
    smooth_min_periods: int,
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
    if not 1 <= smooth_min_periods <= smooth_window_hours:
        raise ValueError("smooth_min_periods must be between 1 and smooth_window_hours")


def _validate_calendar_mode(calendar_mode: str) -> None:
    if calendar_mode not in _CALENDAR_MODES:
        raise ValueError(
            f"calendar_mode must be one of {sorted(_CALENDAR_MODES)!r}; got {calendar_mode!r}"
        )
