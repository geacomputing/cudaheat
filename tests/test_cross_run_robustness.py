"""Unit tests for the database-independent cross-run matching rules."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "cross_run_robustness.py"
SPEC = importlib.util.spec_from_file_location("cross_run_robustness", MODULE_PATH)
assert SPEC and SPEC.loader
robustness = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = robustness
SPEC.loader.exec_module(robustness)


def candidate(event_a: int, event_b: int, start_a: int, stop_a: int, start_b: int, stop_b: int):
    return robustness.Candidate(
        event_a, event_b, 9, start_a, stop_a, start_b, stop_b, stop_a - start_a, stop_b - start_b
    )


def test_split_is_distinguished_from_strict_one_to_one_match():
    # One ten-hour source event is represented by two five-hour comparison events.
    result = robustness.analyse_candidates(
        [candidate(1, 10, 0, 10, 0, 5), candidate(1, 11, 0, 10, 5, 10)],
        run_a="a", run_b="b", events_a=1, events_b=2, minimum_iou=0.5, coverage_threshold=0.8,
    )

    assert result.analysis.strict_detection_a_to_b == 1.0
    assert result.analysis.coverage_detection_a_to_b == 1.0
    assert result.analysis.one_to_one_edges == 0
    assert result.analysis.split_edges == 2
    assert result.analysis.merge_edges == 0


def test_overlap_below_threshold_is_not_silently_counted_as_match():
    result = robustness.analyse_candidates(
        [candidate(1, 2, 0, 10, 0, 4)],
        run_a="a", run_b="b", events_a=1, events_b=1, minimum_iou=0.5, coverage_threshold=0.8,
    )

    assert result.analysis.temporal_candidates == 1
    assert result.analysis.accepted_pairs == 0
    assert result.analysis.overlap_only_a == 1
    assert result.analysis.overlap_only_b == 1
    assert result.analysis.strict_detection_a_to_b == 0.0
    assert result.analysis.no_overlap_a == 0


def test_merge_is_distinguished_from_strict_one_to_one_match():
    # Two source events combine into one ten-hour comparison event.
    result = robustness.analyse_candidates(
        [candidate(1, 10, 0, 5, 0, 10), candidate(2, 10, 5, 10, 0, 10)],
        run_a="a", run_b="b", events_a=2, events_b=1, minimum_iou=0.5, coverage_threshold=0.8,
    )

    assert result.analysis.merge_edges == 2
    assert result.analysis.split_edges == 0
    assert result.analysis.strict_detection_a_to_b == 1.0
    assert result.analysis.strict_detection_b_to_a == 1.0


def test_descriptor_scores_are_symmetric_and_bounded():
    assert robustness.descriptor_similarity("heat_content", 10.0, 20.0) == 0.5
    assert robustness.descriptor_similarity("heat_content", 20.0, 10.0) == 0.5
    assert robustness.descriptor_similarity("daytime_strong_exceedance_fraction", 0.2, 0.7) == 0.5
    assert robustness.descriptor_similarity("heat_content", None, 1.0) is None


def test_noise_cluster_labels_are_not_treated_as_a_shared_family():
    assert robustness._label({("run", 1): "-1"}, "run", 1) is None
    assert robustness._label({("run", 1): "coastal"}, "run", 1) == "coastal"
