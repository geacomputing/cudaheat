from __future__ import annotations

import numpy as np

from cudaheat.validation import _CountAccumulator, _FloatAccumulator, _MaskAccumulator


def test_float_metrics_report_error_and_nan_locations():
    accumulator = _FloatAccumulator(relative_epsilon=1e-9)
    accumulator.update(
        np.array([[1.0, np.nan, 0.0]]),
        np.array([[1.25, 2.0, 0.25]]),
    )
    metrics = accumulator.result()

    assert metrics.compared_values == 2
    assert metrics.cpu_nan_count == 1
    assert metrics.gpu_nan_count == 0
    assert metrics.nan_location_disagreement == 1
    assert metrics.max_absolute_error == 0.25
    assert metrics.max_relative_error == 0.25


def test_mask_and_count_metrics_identify_false_classifications():
    mask = _MaskAccumulator(maximum_examples=2)
    counts = _CountAccumulator(maximum_examples=2)
    cpu_mask = np.array([[False, True], [False, True]])
    gpu_mask = np.array([[True, True], [False, False]])
    values = np.array([[1.01, 2.0], [1.0, 2.0]])
    threshold = np.array([[1.0, 1.0], [1.0, 1.0]])
    mask.update(
        cpu_mask,
        gpu_mask,
        values,
        threshold,
        values.astype("float32"),
        threshold.astype("float32"),
        np.array(["t0", "t1"]),
        np.array([10, 11]),
    )
    counts.update(cpu_mask, gpu_mask, np.array([10, 11]))

    mask_metrics = mask.result()
    count_metrics = counts.result()
    assert mask_metrics.disagreements == 2
    assert mask_metrics.false_positives == 1
    assert mask_metrics.false_negatives == 1
    assert len(mask_metrics.examples) == 2
    assert count_metrics.exact_matches == 0
    assert count_metrics.mismatched_points == 2
    assert count_metrics.max_absolute_difference == 1
