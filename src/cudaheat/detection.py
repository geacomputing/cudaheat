"""Vectorized dense-to-sparse building blocks for GPU event detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventRuns:
    """Sparse half-open intervals, ordered by point then start time."""

    point: Any
    start: Any
    stop: Any
    duration: Any


@dataclass(frozen=True)
class DetectionFields:
    """Raw and filled detection masks plus compact final events."""

    exceedance: Any
    events: EventRuns
    labels: Any | None = None
    filled_exceedance: Any | None = None


def compute_exceedance(values: Any, threshold: Any, *, xp: Any) -> Any:
    """Return the NaN-safe ``values > threshold`` mask for every time and point.

    This intentionally matches the preprocessing pipeline: a NaN SST value
    never exceeds its threshold. A NaN threshold also produces ``False`` via
    the comparison itself.
    """
    _validate_values_and_threshold(values, threshold)
    return xp.logical_and(values > threshold, ~xp.isnan(values))


def find_contiguous_runs(exceedance: Any, *, xp: Any) -> EventRuns:
    """Find every contiguous ``True`` run without Python loops over points/time.

    Returned intervals are ``[start, stop)`` and are sorted by ``point`` then
    ``start``. That order makes future gap-joining a vectorized neighbour
    comparison.
    """
    return _find_boolean_runs(exceedance, xp=xp)


def fill_internal_gaps(exceedance: Any, max_internal_gap_hours: int, *, xp: Any) -> Any:
    """Fill short False runs bounded by True values using interval deltas.

    The input remains unchanged: the raw exceedance mask is retained by the
    caller for later metrics, while this returned mask controls detection.
    """
    _validate_nonnegative_hours(max_internal_gap_hours, "max_internal_gap_hours")
    if exceedance.ndim != 2:
        raise ValueError("exceedance must have shape (time, point)")

    ntime, npoints = exceedance.shape
    if ntime < 1 or npoints < 1:
        raise ValueError("exceedance must have at least one time step and point")
    if max_internal_gap_hours == 0:
        return exceedance

    gaps = _find_boolean_runs(~exceedance, xp=xp)
    fillable = (
        (gaps.duration <= max_internal_gap_hours)
        & (gaps.start > 0)
        & (gaps.stop < ntime)
    )
    starts = gaps.start[fillable]
    stops = gaps.stop[fillable]
    points = gaps.point[fillable]

    # A single difference array scatters every interval boundary, then one
    # scan reconstructs all fillable gaps without per-gap slicing.
    deltas = xp.zeros((ntime + 1, npoints), dtype=xp.int32)
    xp.add.at(deltas, (starts, points), 1)
    xp.add.at(deltas, (stops, points), -1)
    gap_fill = xp.cumsum(deltas[:-1], axis=0, dtype=xp.int32) > 0
    return xp.logical_or(exceedance, gap_fill)


def join_candidate_runs(runs: EventRuns, join_gap_hours: int, *, xp: Any) -> EventRuns:
    """Join neighbouring duration-qualified runs through a sparse prefix scan."""
    _validate_nonnegative_hours(join_gap_hours, "join_gap_hours")
    if runs.start.size == 0:
        return runs

    join_previous = xp.zeros(runs.start.shape, dtype=bool)
    join_previous[1:] = (
        (runs.point[1:] == runs.point[:-1])
        & ((runs.start[1:] - runs.stop[:-1]) <= join_gap_hours)
    )
    new_event = ~join_previous
    group_ids = xp.cumsum(new_event, dtype=xp.int64) - 1
    group_starts = xp.nonzero(new_event)[0]

    # Scatter reductions collapse every sparse group uniformly.  The initial
    # first interval gives each group a valid identity without a host read.
    # CuPy's atomic minimum/maximum supports int32, but not int64 data.
    start_values = runs.start.astype(xp.int32, copy=False)
    stop_values = runs.stop.astype(xp.int32, copy=False)
    start = start_values[group_starts].copy()
    stop = stop_values[group_starts].copy()
    xp.minimum.at(start, group_ids, start_values)
    xp.maximum.at(stop, group_ids, stop_values)
    point = runs.point[group_starts]
    start = start.astype(runs.start.dtype, copy=False)
    stop = stop.astype(runs.stop.dtype, copy=False)
    return EventRuns(point=point, start=start, stop=stop, duration=stop - start)


def _find_boolean_runs(mask: Any, *, xp: Any) -> EventRuns:
    """Return maximal True intervals from a two-dimensional boolean mask."""
    if mask.ndim != 2:
        raise ValueError("mask must have shape (time, point)")
    ntime, npoints = mask.shape
    if ntime < 1 or npoints < 1:
        raise ValueError("mask must have at least one time step and point")

    false_row = xp.zeros((1, npoints), dtype=bool)
    starts = xp.logical_and(mask, ~xp.concatenate((false_row, mask[:-1]), axis=0))
    ends = xp.logical_and(mask, ~xp.concatenate((mask[1:], false_row), axis=0))
    start_time, start_point = xp.nonzero(starts)
    end_time, end_point = xp.nonzero(ends)

    # ``nonzero`` is time-major. Reordering both transition lists by point
    # pairs each run start with its corresponding run end using only device
    # sorting; their per-point order remains chronological.
    start_order = xp.argsort(start_point.astype(xp.int64) * ntime + start_time)
    end_order = xp.argsort(end_point.astype(xp.int64) * ntime + end_time)
    point = start_point[start_order]
    start = start_time[start_order]
    stop = end_time[end_order] + 1
    return EventRuns(point=point, start=start, stop=stop, duration=stop - start)


def filter_min_duration(runs: EventRuns, min_duration_hours: int, *, xp: Any) -> EventRuns:
    """Keep only runs lasting at least ``min_duration_hours``."""
    _validate_min_duration(min_duration_hours)
    keep = runs.duration >= min_duration_hours
    return EventRuns(
        point=runs.point[keep],
        start=runs.start[keep],
        stop=runs.stop[keep],
        duration=runs.duration[keep],
    )


def build_event_labels(runs: EventRuns, shape: tuple[int, int], *, xp: Any) -> Any:
    """Build one-based dense labels through interval deltas and a prefix sum.

    This is optional because the compact :class:`EventRuns` table is much
    smaller for typical sparse events. It has no per-event slice assignment.
    """
    ntime, npoints = shape
    if ntime < 1 or npoints < 1:
        raise ValueError("label shape must have positive time and point dimensions")
    # CuPy scatter-add supports int32 (not signed int64); an int32 label is
    # sufficient for a dense time-by-point event map.
    deltas = xp.zeros((ntime + 1, npoints), dtype=xp.int32)
    event_ids = xp.arange(1, runs.start.size + 1, dtype=xp.int32)
    xp.add.at(deltas, (runs.start, runs.point), event_ids)
    xp.add.at(deltas, (runs.stop, runs.point), -event_ids)
    return xp.cumsum(deltas[:-1], axis=0, dtype=xp.int32)


def detect_events(
    values: Any,
    threshold: Any,
    *,
    min_duration_hours: int = 1,
    max_internal_gap_hours: int = 0,
    join_gap_hours: int = 0,
    return_labels: bool = False,
    xp: Any,
) -> DetectionFields:
    """Run the vectorized event-detection pipeline on one GPU batch.

    ``xp`` is normally CuPy. Keeping it explicit makes this algorithm easy to
    validate with NumPy while retaining exactly the same array-oriented steps.
    """
    exceedance = compute_exceedance(values, threshold, xp=xp)
    filled_exceedance = fill_internal_gaps(
        exceedance, max_internal_gap_hours, xp=xp
    )
    runs = find_contiguous_runs(filled_exceedance, xp=xp)
    candidates = filter_min_duration(runs, min_duration_hours, xp=xp)
    events = join_candidate_runs(candidates, join_gap_hours, xp=xp)
    labels = build_event_labels(events, values.shape, xp=xp) if return_labels else None
    return DetectionFields(
        exceedance=exceedance,
        events=events,
        labels=labels,
        filled_exceedance=filled_exceedance,
    )


def _validate_values_and_threshold(values: Any, threshold: Any) -> None:
    if values.ndim != 2 or threshold.ndim != 2:
        raise ValueError("values and threshold must both have shape (time, point)")
    if values.shape != threshold.shape:
        raise ValueError("values and threshold must have identical shapes")


def _validate_min_duration(min_duration_hours: int) -> None:
    if not isinstance(min_duration_hours, int) or isinstance(min_duration_hours, bool):
        raise TypeError("min_duration_hours must be an integer")
    if min_duration_hours < 1:
        raise ValueError("min_duration_hours must be positive")


def _validate_nonnegative_hours(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
