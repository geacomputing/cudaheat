from __future__ import annotations

import numpy as np

from cudaheat.detection import detect_events


def test_detect_events_finds_and_filters_contiguous_runs_without_python_scans():
    values = np.array(
        [
            [2.0, 0.0],
            [2.0, 2.0],
            [np.nan, 2.0],
            [2.0, 2.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
        ]
    )
    fields = detect_events(
        values,
        np.ones_like(values),
        min_duration_hours=2,
        return_labels=True,
        xp=np,
    )

    np.testing.assert_array_equal(fields.events.point, [0, 0, 1])
    np.testing.assert_array_equal(fields.events.start, [0, 5, 1])
    np.testing.assert_array_equal(fields.events.stop, [2, 7, 4])
    np.testing.assert_array_equal(fields.events.duration, [2, 2, 3])
    np.testing.assert_array_equal(
        fields.labels,
        [
            [1, 0],
            [1, 3],
            [0, 3],
            [0, 3],
            [0, 0],
            [2, 0],
            [2, 0],
        ],
    )


def test_detect_events_does_not_keep_short_runs_or_nan_samples():
    values = np.array([[2.0], [np.nan], [2.0]])
    fields = detect_events(values, np.ones_like(values), min_duration_hours=2, xp=np)

    assert fields.events.start.size == 0
    assert fields.labels is None
