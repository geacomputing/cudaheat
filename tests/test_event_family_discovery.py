"""Pure numerical checks for the isolated event-family script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "scatterPlot" / "discover_event_families.py"
)
SPEC = importlib.util.spec_from_file_location("discover_event_families", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_scaling_and_pca_are_finite() -> None:
    values = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [40.0, 8.0]])
    scaled, median, scale = MODULE.robust_scale(values)
    _, _, ratios = MODULE.pca(scaled)
    assert np.allclose(median, [2.5, 6.5])
    assert np.all(scale > 0)
    assert np.isfinite(scaled).all()
    assert np.isclose(ratios.sum(), 1.0)


def test_density_graph_separates_two_dense_clouds() -> None:
    rng = np.random.default_rng(17)
    left = rng.normal(-3.0, 0.12, size=(200, 2))
    right = rng.normal(3.0, 0.12, size=(200, 2))
    noise = rng.uniform(-1.0, 1.0, size=(30, 2))
    labels, _, _ = MODULE.density_components(
        np.vstack((left, right, noise)), neighbours=15, density_quantile=0.75, min_cluster_size=80
    )
    assert labels[0] >= 0 and labels[250] >= 0
    assert labels[0] != labels[250]


def test_stability_metrics_score_identical_labels() -> None:
    labels = np.array([-1, -1, 0, 0, 1, 1])
    assert MODULE.adjusted_rand_index(labels, labels) == 1.0
    assert MODULE.cluster_jaccard(labels, labels) == 1.0
