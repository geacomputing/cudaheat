"""Fraction of event hours for which SST exceeds the time-varying threshold."""

from __future__ import annotations

import numpy as np


METADATA = {
    "id": "threshold_exceedance_fraction",
    "version": "2.0.0",
    "name": "Threshold exceedance fraction",
    "description": "Fraction of total event hours during which SST exceeds the local threshold.",
    "category": "thermal intensity",
    "output": {"primary": "value", "unit": "dimensionless", "range": [0.0, 1.0]},
    "formula_latex": r"F_{exc}=\frac{1}{N}\sum_{t=1}^{N}\mathbb{1}[SST_t>T_t]",
    "postgres": {
        "select_sql": "SELECT e.event_id, p.grid_point_index, e.start_index, e.stop_index, e.start_time, e.end_time, e.duration_hours FROM events AS e JOIN points AS p ON p.point_id = e.point_id WHERE e.run_id = %(run_id)s AND (p.grid_point_index, e.event_id) > (%(after_grid_point_index)s, %(after_event_id)s) ORDER BY p.grid_point_index, e.event_id LIMIT %(batch_size)s",
        "write_sql": "INSERT INTO event_enrichments (event_id, rule_id, rule_version, rule_sha256, result, applied_at) VALUES (%(event_id)s, %(rule_id)s, %(rule_version)s, %(rule_sha256)s, %(result)s::jsonb, %(applied_at)s) ON CONFLICT (event_id, rule_id, rule_version) DO NOTHING",
    },
}


def calculate(context):
    """Return the historical count-as-false threshold-exceedance metric."""
    valid = np.isfinite(context.sst) & np.isfinite(context.threshold)
    true_hours = context.count(valid & (context.sst > context.threshold))
    valid_hours = context.count(valid)
    total_hours = context.duration_hours
    with np.errstate(divide="ignore", invalid="ignore"):
        value = true_hours / total_hours
    return {
        "value": value,
        "true_hours": true_hours,
        "total_hours": total_hours,
        "valid_hours": valid_hours,
        "missing_hours": total_hours - valid_hours,
    }
