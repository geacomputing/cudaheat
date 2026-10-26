"""Fraction of valid nighttime event hours remaining above the threshold."""

from __future__ import annotations

import numpy as np


METADATA = {
    "id": "nighttime_retention_fraction",
    "version": "2.0.0",
    "name": "Nighttime retention fraction",
    "description": "Fraction of valid local-solar nighttime event hours with SST above threshold.",
    "category": "diurnal persistence",
    "output": {"primary": "value", "unit": "dimensionless", "range": [0.0, 1.0]},
    "formula_latex": r"R_{night}=\frac{N_{SST>T,\,night,\,valid}}{N_{night,\,valid}}",
    "postgres": {
        "select_sql": "SELECT e.event_id, p.grid_point_index, e.start_index, e.stop_index, e.start_time, e.end_time, e.duration_hours FROM events AS e JOIN points AS p ON p.point_id = e.point_id WHERE e.run_id = %(run_id)s AND (p.grid_point_index, e.event_id) > (%(after_grid_point_index)s, %(after_event_id)s) ORDER BY p.grid_point_index, e.event_id LIMIT %(batch_size)s",
        "write_sql": "INSERT INTO event_enrichments (event_id, rule_id, rule_version, rule_sha256, result, applied_at) VALUES (%(event_id)s, %(rule_id)s, %(rule_version)s, %(rule_sha256)s, %(result)s::jsonb, %(applied_at)s) ON CONFLICT (event_id, rule_id, rule_version) DO NOTHING",
    },
}


def calculate(context):
    """Evaluate local-solar night hours that occur inside each event interval only."""
    night = (context.local_hour >= 18.0) | (context.local_hour < 6.0)
    valid = np.isfinite(context.sst) & np.isfinite(context.threshold)
    valid_night = night & valid
    night_hours = context.count(night)
    valid_night_hours = context.count(valid_night)
    retained_night_hours = context.count(valid_night & (context.sst > context.threshold))
    with np.errstate(divide="ignore", invalid="ignore"):
        value = retained_night_hours / valid_night_hours
    return {
        "value": value,
        "night_hours": night_hours,
        "valid_night_hours": valid_night_hours,
        "retained_night_hours": retained_night_hours,
        "missing_night_hours": night_hours - valid_night_hours,
    }
