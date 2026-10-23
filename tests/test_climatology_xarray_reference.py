"""Independent xarray regression checks for CUDAHeat hourly climatology."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cudaheat.climatology import (
    CALENDAR_MODE_DAY_OF_YEAR,
    CALENDAR_MODE_MONTH_DAY_HOUR,
    climatology_bin_count,
    cpu_pre_event_fields,
    hourly_climatology_index,
)


def _hourwave_reference(
    values: np.ndarray,
    time: np.ndarray,
    baseline_indices: np.ndarray,
    *,
    percentile: float,
    smooth_window_hours: int,
    smooth_min_periods: int,
) -> dict[str, np.ndarray]:
    """Independent xarray implementation of HourWave's threshold pipeline."""
    source = xr.DataArray(values, dims=("time", "point"), coords={"time": time})
    baseline = source.isel(time=baseline_indices)
    group = ((baseline.time.dt.dayofyear - 1) * 24 + baseline.time.dt.hour).rename(
        "climatology_hour"
    )
    grouped = baseline.groupby(group)
    climatology_raw = grouped.mean(dim="time", skipna=True)
    threshold_raw = grouped.quantile(percentile, dim="time", skipna=True).squeeze(drop=True)
    nbins = int(group.max().item()) + 1
    full_index = np.arange(nbins)
    # HourWave rolls over only populated ``groupby`` labels.  CUDAHeat returns
    # a dense cycle afterwards, so reindex only after the source-equivalent
    # rolling operation.
    climatology = _xarray_circular_mean(
        climatology_raw, "climatology_hour", smooth_window_hours, smooth_min_periods
    )
    threshold_cycle = _xarray_circular_mean(
        threshold_raw, "climatology_hour", smooth_window_hours, smooth_min_periods
    )
    climatology_raw = climatology_raw.reindex(climatology_hour=full_index)
    threshold_raw = threshold_raw.reindex(climatology_hour=full_index)
    climatology = climatology.reindex(climatology_hour=full_index)
    threshold_cycle = threshold_cycle.reindex(climatology_hour=full_index)
    time_group = ((source.time.dt.dayofyear - 1) * 24 + source.time.dt.hour).values
    climatology_by_time = _reindex_cycle_to_time(climatology, time_group, time)
    threshold = _reindex_cycle_to_time(threshold_cycle, time_group, time)
    return {
        "climatology_raw": climatology_raw.values,
        "climatology": climatology.values,
        "threshold_raw": threshold_raw.values,
        "threshold_cycle": threshold_cycle.values,
        "threshold": threshold.values,
        "anomaly": (source - climatology_by_time).values,
        "excess": (source - threshold).values,
        "exceedance": ((source > threshold).fillna(False)).values,
    }


def _xarray_circular_mean(
    values: xr.DataArray, dimension: str, window: int, min_periods: int
) -> xr.DataArray:
    if window <= 1:
        return values
    left = window // 2
    right = window - left - 1
    padded = xr.concat(
        [values.isel({dimension: slice(-left, None)}), values, values.isel({dimension: slice(0, right)})],
        dim=dimension,
    )
    return (
        padded.rolling({dimension: window}, center=True, min_periods=min_periods)
        .mean()
        .isel({dimension: slice(left, left + values.sizes[dimension])})
        .assign_coords({dimension: values[dimension]})
    )


def _reindex_cycle_to_time(cycle: xr.DataArray, bins: np.ndarray, time: np.ndarray) -> xr.DataArray:
    return (
        cycle.reindex(climatology_hour=bins)
        .rename({"climatology_hour": "time"})
        .assign_coords(time=time)
    )


def _hourly_values(start: str, end: str, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(np.datetime64(start), np.datetime64(end), np.timedelta64(1, "h"))
    hours = np.arange(time.size, dtype=np.float64)
    values = np.column_stack(
        (
            20 + np.sin(hours / 97),
            22 + np.cos(hours / 71),
            19 + np.sin(hours / 53) + np.cos(hours / 101),
        )
    ).astype(dtype)
    values[7::503, 0] = np.nan
    values[19::719, 1] = np.nan
    values[31::997, 2] = np.nan
    return time, values


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_dayofyear_cpu_matches_hourwave_xarray_with_leap_and_missing_values(dtype):
    time, values = _hourly_values("2019-01-01", "2021-01-01", dtype)
    time_coordinate = xr.DataArray(time, dims="time")
    bins = hourly_climatology_index(time_coordinate, calendar_mode=CALENDAR_MODE_DAY_OF_YEAR)
    rows = np.arange(time.size, dtype=np.intp)
    nbins = climatology_bin_count(bins, rows, calendar_mode=CALENDAR_MODE_DAY_OF_YEAR)
    fields = cpu_pre_event_fields(
        values,
        bins,
        rows,
        percentile=0.90,
        smooth_window_hours=25,
        nbins=nbins,
        smooth_min_periods=24,
        calendar_mode=CALENDAR_MODE_DAY_OF_YEAR,
        sampling_window_half_width_days=0,
        smooth_same_hour=False,
    )
    reference = _hourwave_reference(
        values,
        time,
        rows,
        percentile=0.90,
        smooth_window_hours=25,
        smooth_min_periods=24,
    )
    # Group reductions are mathematically equivalent but xarray/NumPy need
    # not use the same floating-point reduction order.  Float64 agreement is
    # therefore checked numerically rather than bitwise.
    tolerance = 3e-5 if dtype == np.float32 else 1e-10
    for name in ("climatology_raw", "climatology", "threshold_raw", "threshold_cycle", "threshold", "anomaly", "excess"):
        np.testing.assert_allclose(getattr(fields, name), reference[name], rtol=tolerance, atol=tolerance, equal_nan=True)
    np.testing.assert_array_equal(fields.exceedance, reference["exceedance"])


def test_dayofyear_nonleap_baseline_marks_unavailable_leap_day_nan():
    time, values = _hourly_values("2019-01-01", "2021-01-01", np.dtype("float64"))
    time_coordinate = xr.DataArray(time, dims="time")
    year = time_coordinate.dt.year.values
    rows = np.flatnonzero(year == 2019).astype(np.intp)
    bins = hourly_climatology_index(time_coordinate, calendar_mode=CALENDAR_MODE_DAY_OF_YEAR)
    nbins = climatology_bin_count(bins, rows, calendar_mode=CALENDAR_MODE_DAY_OF_YEAR)
    assert nbins == 365 * 24
    fields = cpu_pre_event_fields(
        values,
        bins,
        rows,
        smooth_window_hours=25,
        nbins=nbins,
        smooth_min_periods=24,
        calendar_mode=CALENDAR_MODE_DAY_OF_YEAR,
        sampling_window_half_width_days=0,
        smooth_same_hour=False,
    )
    reference = _hourwave_reference(
        values,
        time,
        rows,
        percentile=0.90,
        smooth_window_hours=25,
        smooth_min_periods=24,
    )
    leap_day_366 = (year == 2020) & (time_coordinate.dt.dayofyear.values == 366)
    assert np.isnan(fields.threshold[leap_day_366]).all()
    np.testing.assert_allclose(fields.threshold, reference["threshold"], equal_nan=True)
    np.testing.assert_array_equal(fields.exceedance, reference["exceedance"])


def test_irregular_baseline_coverage_matches_xarray_reference():
    time, values = _hourly_values("2019-01-01", "2021-01-01", np.dtype("float64"))
    time_coordinate = xr.DataArray(time, dims="time")
    bins = hourly_climatology_index(time_coordinate, calendar_mode=CALENDAR_MODE_DAY_OF_YEAR)
    # Emulate dropped observations across an otherwise multi-year baseline.
    rows = np.flatnonzero((bins % 97) != 0).astype(np.intp)
    nbins = climatology_bin_count(bins, rows, calendar_mode=CALENDAR_MODE_DAY_OF_YEAR)
    fields = cpu_pre_event_fields(
        values,
        bins,
        rows,
        percentile=0.90,
        smooth_window_hours=25,
        nbins=nbins,
        smooth_min_periods=24,
        calendar_mode=CALENDAR_MODE_DAY_OF_YEAR,
        sampling_window_half_width_days=0,
        smooth_same_hour=False,
    )
    reference = _hourwave_reference(
        values,
        time,
        rows,
        percentile=0.90,
        smooth_window_hours=25,
        smooth_min_periods=24,
    )
    assert np.isnan(fields.threshold_raw).any()
    for name in ("climatology_raw", "climatology", "threshold_raw", "threshold_cycle", "threshold"):
        np.testing.assert_allclose(getattr(fields, name), reference[name], rtol=1e-10, atol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(fields.exceedance, reference["exceedance"])


def test_calendar_modes_preserve_default_and_expose_hourwave_leap_shift():
    time = xr.DataArray(
        np.array(
            [
                np.datetime64("2019-03-01T12"),
                np.datetime64("2020-02-29T12"),
                np.datetime64("2020-03-01T12"),
            ]
        ),
        dims="time",
    )
    default = hourly_climatology_index(time)
    month_day = hourly_climatology_index(time, calendar_mode=CALENDAR_MODE_MONTH_DAY_HOUR)
    dayofyear = hourly_climatology_index(time, calendar_mode=CALENDAR_MODE_DAY_OF_YEAR)
    np.testing.assert_array_equal(default, month_day)
    assert month_day[0] == month_day[2]
    assert dayofyear[0] == dayofyear[1]
    assert dayofyear[0] != dayofyear[2]


def test_circular_smoothing_matches_hourwave_min_periods_at_year_boundary():
    values = np.full((12, 1), np.nan)
    values[[0, 1, -2, -1], 0] = (1, 3, 5, 7)
    source = xr.DataArray(values, dims=("climatology_hour", "point"))
    reference = _xarray_circular_mean(source, "climatology_hour", window=5, min_periods=3)
    from cudaheat.climatology import smooth_circular_cpu

    actual = smooth_circular_cpu(values, window=5, min_periods=3)
    np.testing.assert_allclose(actual, reference.values, equal_nan=True)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_gpu_matches_independent_xarray_reference_for_hourwave_calendar(dtype):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device unavailable")
    try:
        cp.full((1,), 0.0)
    except RuntimeError as exc:
        pytest.skip(f"CuPy kernel compilation unavailable: {exc}")
    from cudaheat.climatology import gpu_pre_event_fields

    time, values = _hourly_values("2019-01-01", "2021-01-01", dtype)
    time_coordinate = xr.DataArray(time, dims="time")
    bins = hourly_climatology_index(time_coordinate, calendar_mode=CALENDAR_MODE_DAY_OF_YEAR)
    rows = np.arange(time.size, dtype=np.intp)
    nbins = climatology_bin_count(bins, rows, calendar_mode=CALENDAR_MODE_DAY_OF_YEAR)
    gpu = gpu_pre_event_fields(
        cp.asarray(values),
        cp.asarray(bins),
        cp.asarray(rows),
        percentile=0.90,
        smooth_window_hours=25,
        nbins=nbins,
        smooth_min_periods=24,
        calendar_mode=CALENDAR_MODE_DAY_OF_YEAR,
        sampling_window_half_width_days=0,
        smooth_same_hour=False,
    )
    reference = _hourwave_reference(
        values,
        time,
        rows,
        percentile=0.90,
        smooth_window_hours=25,
        smooth_min_periods=24,
    )
    tolerance = 3e-5 if dtype == np.float32 else 1e-10
    for name in ("climatology_raw", "climatology", "threshold_raw", "threshold_cycle", "threshold", "anomaly", "excess"):
        np.testing.assert_allclose(cp.asnumpy(getattr(gpu, name)), reference[name], rtol=tolerance, atol=tolerance, equal_nan=True)
    np.testing.assert_array_equal(cp.asnumpy(gpu.exceedance), reference["exceedance"])
