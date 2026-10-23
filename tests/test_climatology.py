from __future__ import annotations

import numpy as np

from cudaheat.climatology import (
    HOURS_PER_LEAP_YEAR,
    cpu_pre_event_fields,
    hourly_climatology_index,
    smooth_circular_cpu,
)


def test_hourly_index_uses_calendar_month_day_hour(synthetic_sst):
    index = hourly_climatology_index(synthetic_sst.time)
    assert index[0] == 0
    assert index[1] == 1
    assert index[24] == 24
    assert index[-1] == 71
    assert index.max() < HOURS_PER_LEAP_YEAR


def test_circular_smoothing_wraps_end_to_start():
    values = np.full((8, 1), np.nan, dtype=np.float64)
    values[0] = 2
    values[-1] = 4
    smoothed = smooth_circular_cpu(values, 3)
    np.testing.assert_allclose(smoothed[0], [3])


def test_cpu_fields_keep_nans_local_to_each_point(synthetic_sst):
    values = synthetic_sst.analysed_sst.stack(point=("lat", "lon")).values
    bins = hourly_climatology_index(synthetic_sst.time)
    fields = cpu_pre_event_fields(
        values, bins, np.arange(values.shape[0]), smooth_window_hours=24 * 3
    )
    assert fields.exceedance.dtype == bool
    assert not fields.exceedance[5, 1]
    assert np.isfinite(fields.climatology[:72, 0]).any()
