"""Independent regression checks added by the numerical audit.

These references deliberately do not call CUDAHeat's climatology or run
segmentation helpers.
"""

from __future__ import annotations

import inspect
import numpy as np
import pytest
import xarray as xr

from cudaheat.climatology import (
    HOURS_PER_LEAP_YEAR,
    cpu_pre_event_fields,
    hourly_climatology_index,
    validate_contiguous_hourly_time,
)
from cudaheat.detection import detect_events
from cudaheat.core import CUDAHeat


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Independent half-open Boolean-run reference."""
    output: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == mask.size - 1):
            output.append((start, index if not value else index + 1))
            start = None
    return output


def _reference_events(
    values: np.ndarray,
    threshold: np.ndarray,
    *,
    min_duration: int,
    internal_gap: int,
    joining_gap: int,
) -> list[tuple[int, int]]:
    """Plain-Python reference for CUDAHeat's documented stage order."""
    mask = np.isfinite(values) & (values > threshold)
    filled = mask.copy()
    for start, stop in _runs(~mask):
        if start > 0 and stop < mask.size and stop - start <= internal_gap:
            filled[start:stop] = True
    qualified = [(start, stop) for start, stop in _runs(filled) if stop - start >= min_duration]
    if not qualified:
        return []
    output = [qualified[0]]
    for start, stop in qualified[1:]:
        previous_start, previous_stop = output[-1]
        if start - previous_stop <= joining_gap:
            output[-1] = (previous_start, stop)
        else:
            output.append((start, stop))
    return output


@pytest.mark.parametrize("gap, expected", [(0, [(0, 120)]), (1, [(0, 121)]), (6, [(0, 126)]), (7, [])])
def test_audit_internal_gap_boundaries(gap: int, expected: list[tuple[int, int]]):
    values = np.zeros(260)
    values[:60] = 1
    values[60 + gap : 120 + gap] = 1
    fields = detect_events(
        values[:, None],
        np.full((values.size, 1), 0.5),
        min_duration_hours=120,
        max_internal_gap_hours=6,
        join_gap_hours=48,
        xp=np,
    )
    actual = list(zip(fields.events.start.tolist(), fields.events.stop.tolist(), strict=True))
    assert actual == expected


@pytest.mark.parametrize("gap, count", [(47, 1), (48, 1), (49, 2)])
def test_audit_joining_gap_boundaries(gap: int, count: int):
    values = np.zeros(400)
    values[:120] = 1
    values[120 + gap : 240 + gap] = 1
    fields = detect_events(
        values[:, None],
        np.full((values.size, 1), 0.5),
        min_duration_hours=120,
        max_internal_gap_hours=6,
        join_gap_hours=48,
        xp=np,
    )
    assert fields.events.start.size == count
    if count == 1:
        assert (int(fields.events.start[0]), int(fields.events.stop[0])) == (0, 240 + gap)


def test_audit_detection_matches_independent_random_reference():
    rng = np.random.default_rng(20261023)
    for _ in range(100):
        values = rng.normal(size=300)
        values[rng.choice(values.size, size=12, replace=False)] = np.nan
        threshold = rng.normal(scale=0.4, size=values.size)
        reference = _reference_events(
            values, threshold, min_duration=5, internal_gap=3, joining_gap=9
        )
        fields = detect_events(
            values[:, None],
            threshold[:, None],
            min_duration_hours=5,
            max_internal_gap_hours=3,
            join_gap_hours=9,
            xp=np,
        )
        actual = list(zip(fields.events.start.tolist(), fields.events.stop.tolist(), strict=True))
        assert actual == reference


def test_audit_missing_samples_are_not_raw_exceedances_but_follow_gap_policy():
    values = np.ones((126, 1))
    values[60:66] = np.nan
    fields = detect_events(
        values,
        np.zeros_like(values),
        min_duration_hours=120,
        max_internal_gap_hours=6,
        join_gap_hours=48,
        xp=np,
    )
    assert not fields.exceedance[60:66].any()
    assert fields.filled_exceedance[60:66].all()
    assert (int(fields.events.start[0]), int(fields.events.stop[0])) == (0, 126)


def test_audit_multiple_internal_gaps_count_toward_event_duration():
    values = np.zeros((138, 1))
    values[:30] = 1
    values[36:66] = 1
    values[72:102] = 1
    values[108:138] = 1
    fields = detect_events(
        values,
        np.full_like(values, 0.5),
        min_duration_hours=120,
        max_internal_gap_hours=6,
        join_gap_hours=48,
        xp=np,
    )
    assert (int(fields.events.start[0]), int(fields.events.stop[0]), int(fields.events.duration[0])) == (
        0,
        138,
        138,
    )


def test_audit_public_detection_defaults_match_hourly_method():
    parameters = inspect.signature(CUDAHeat.detect).parameters
    assert parameters["min_duration_hours"].default == 120
    assert parameters["max_internal_gap_hours"].default == 6
    assert parameters["join_gap_hours"].default == 48


def _xarray_hobday_reference(values: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate xarray implementation of the hourly 11-day/31-day algorithm."""
    source = xr.DataArray(values, dims=("time", "point"), coords={"time": time})
    offsets = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    day = offsets[source.time.dt.month.values - 1] + source.time.dt.day.values - 1
    hour = source.time.dt.hour.values
    raw_mean = np.full((366, 24, values.shape[1]), np.nan)
    raw_threshold = np.full_like(raw_mean, np.nan)
    for target_day in range(366):
        calendar_days = (target_day + np.arange(-5, 6)) % 366
        selected_days = np.isin(day, calendar_days)
        for target_hour in range(24):
            samples = source.isel(time=np.flatnonzero(selected_days & (hour == target_hour)))
            raw_mean[target_day, target_hour] = samples.mean("time", skipna=True).values
            raw_threshold[target_day, target_hour] = (
                samples.quantile(0.90, dim="time", skipna=True).values
            )
    # Hobday's leap-day convention is interpolation, not a separate sample pool.
    raw_mean[59] = np.nanmean(raw_mean[[58, 60]], axis=0)
    raw_threshold[59] = np.nanmean(raw_threshold[[58, 60]], axis=0)

    def smooth(data: np.ndarray) -> np.ndarray:
        array = xr.DataArray(data, dims=("day", "hour", "point"))
        repeated = xr.concat((array, array, array), dim="day")
        return repeated.rolling(day=31, center=True, min_periods=1).mean().isel(day=slice(366, 732)).values

    return smooth(raw_mean).reshape(HOURS_PER_LEAP_YEAR, values.shape[1]), smooth(
        raw_threshold
    ).reshape(HOURS_PER_LEAP_YEAR, values.shape[1])


@pytest.mark.parametrize("dtype, tolerance", [(np.float32, 4e-5), (np.float64, 1e-11)])
def test_audit_hobday_hourly_cycle_matches_independent_xarray(dtype, tolerance):
    time = np.arange(np.datetime64("2019-01-01"), np.datetime64("2021-01-01"), np.timedelta64(1, "h"))
    hour = np.arange(time.size)
    values = np.column_stack(
        (
            20 + 2 * np.sin(2 * np.pi * (hour % 24) / 24) + 0.3 * np.cos(hour / 400),
            17 + 1.5 * np.cos(2 * np.pi * (hour % 24) / 24) + 0.2 * np.sin(hour / 210),
        )
    ).astype(dtype)
    values[100:103, 0] = np.nan
    values[1000:1008, 1] = np.nan
    bins = hourly_climatology_index(xr.DataArray(time, dims="time"))
    fields = cpu_pre_event_fields(
        values,
        bins,
        np.arange(time.size),
        smooth_window_hours=31 * 24,
        sampling_window_half_width_days=5,
        smooth_same_hour=True,
    )
    reference_mean, reference_threshold = _xarray_hobday_reference(values, time)
    np.testing.assert_allclose(fields.climatology, reference_mean, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        fields.threshold_cycle, reference_threshold, atol=tolerance, rtol=tolerance
    )
    # Noon and midnight remain distinct: no smoothing operation blends clock hours.
    assert not np.allclose(fields.climatology[0], fields.climatology[12])


def test_audit_rejects_missing_timestamp_not_missing_observation():
    time = xr.DataArray(
        np.array([np.datetime64("2020-01-01T00"), np.datetime64("2020-01-01T02")]),
        dims="time",
    )
    with pytest.raises(ValueError, match="contiguous, strictly hourly"):
        validate_contiguous_hourly_time(time)
