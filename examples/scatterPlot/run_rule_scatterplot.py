"""Render every enriched CUDAHeat event as one scatterplot marker.

Marker colour identifies the parent run (and therefore its parametric
configuration); marker area represents that event's duration. Edit the two
rule identifiers below to change the axes. The script has no input arguments
and writes the PNG next to itself.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from cudaheat import load_postgres_config


# Plot configuration: edit only these constants when changing the figure.
X_AXIS_RULE = "daytime_strong_exceedance_fraction"
Y_AXIS_RULE = "fraction_above_threshold"
NORMALIZE_AXES = True
DURATION_AREA_SCALE = 1.8
FETCH_BATCH_SIZE = 250_000
CLUSTER_GRID_SIZE = 90
OUTPUT_PATH = Path(__file__).with_name("event_rule_scatterplot.png")
POSTGRES_CONFIG = Path(__file__).resolve().parents[2] / "config" / "postgres.ini"


def _event_query() -> str:
    """Cluster every event into run-specific bins without sampling any events."""
    x_plot = "AVG(x_normalised)" if NORMALIZE_AXES else "AVG(x_value)"
    y_plot = "AVG(y_normalised)" if NORMALIZE_AXES else "AVG(y_value)"
    grid = CLUSTER_GRID_SIZE
    return f"""
        WITH event_values AS (
            SELECT
                event.run_id,
                event.duration_hours::double precision AS duration_hours,
                run.percentile,
                run.min_duration_hours,
                run.max_internal_gap_hours,
                run.join_gap_hours,
                NULLIF(x_rule.result ->> 'value', '')::double precision AS x_value,
                NULLIF(y_rule.result ->> 'value', '')::double precision AS y_value
            FROM events AS event
            JOIN runs AS run ON run.run_id = event.run_id
            JOIN event_enrichments AS x_rule
              ON x_rule.event_id = event.event_id AND x_rule.rule_id = %s
            JOIN event_enrichments AS y_rule
              ON y_rule.event_id = event.event_id AND y_rule.rule_id = %s
        ), finite_values AS (
            SELECT * FROM event_values WHERE x_value IS NOT NULL AND y_value IS NOT NULL
        ), normalised_values AS (
            SELECT *,
                CASE WHEN MAX(x_value) OVER () = MIN(x_value) OVER () THEN 0.5
                     ELSE (x_value - MIN(x_value) OVER ())
                          / (MAX(x_value) OVER () - MIN(x_value) OVER ()) END AS x_normalised,
                CASE WHEN MAX(y_value) OVER () = MIN(y_value) OVER () THEN 0.5
                     ELSE (y_value - MIN(y_value) OVER ())
                          / (MAX(y_value) OVER () - MIN(y_value) OVER ()) END AS y_normalised
            FROM finite_values
        ), clustered_values AS (
            SELECT *,
                LEAST({grid} - 1, GREATEST(0, FLOOR(x_normalised * {grid})::integer)) AS x_bin,
                LEAST({grid} - 1, GREATEST(0, FLOOR(y_normalised * {grid})::integer)) AS y_bin
            FROM normalised_values
        )
        SELECT
            run_id,
            COUNT(*) AS event_count,
            AVG(duration_hours) AS mean_duration_hours,
            percentile,
            min_duration_hours,
            max_internal_gap_hours,
            join_gap_hours,
            {x_plot} AS x_plot,
            {y_plot} AS y_plot
        FROM clustered_values
        GROUP BY run_id, percentile, min_duration_hours, max_internal_gap_hours,
                 join_gap_hours, x_bin, y_bin
        ORDER BY run_id, x_bin, y_bin
    """


def _run_colours(run_ids: list[object]) -> np.ndarray:
    """Assign a stable hue to each run without retaining the full catalogue."""
    hues = np.asarray(
        [int(str(run_id).replace("-", "")[:8], 16) / 0xFFFFFFFF for run_id in run_ids]
    )
    return plt.colormaps["hsv"](hues)


def _marker_areas(duration_hours: np.ndarray) -> np.ndarray:
    return np.clip(duration_hours * DURATION_AREA_SCALE, 1, 10.0)


def _configuration_label(parameters: tuple[float, int, int, int]) -> str:
    percentile, minimum_duration, internal_gap, join_gap = parameters
    return (
        f"q={percentile:.2f}; min={minimum_duration} h; "
        f"internal={internal_gap} h; join={join_gap} h"
    )


def main() -> None:
    import psycopg

    figure, (axis, legend_axis) = plt.subplots(
        1,
        2,
        figsize=(13.5, 7.5),
        constrained_layout=True,
        facecolor="white",
        gridspec_kw={"width_ratios": (4.5, 1)},
    )
    legend_axis.axis("off")
    configuration = load_postgres_config(POSTGRES_CONFIG)
    event_count = 0
    run_ids_seen: list[object] = []
    seen_run_ids: set[object] = set()
    run_parameters: dict[object, tuple[float, int, int, int]] = {}

    with psycopg.connect(**configuration.connection_kwargs()) as connection:
        with connection.cursor(name="cudaheat_event_scatter") as cursor:
            cursor.itersize = FETCH_BATCH_SIZE
            cursor.execute(_event_query(), (X_AXIS_RULE, Y_AXIS_RULE))
            while rows := cursor.fetchmany(FETCH_BATCH_SIZE):
                (
                    run_ids,
                    cluster_event_counts,
                    duration,
                    percentiles,
                    minimum_durations,
                    internal_gaps,
                    join_gaps,
                    x_values,
                    y_values,
                ) = zip(*rows, strict=True)
                axis.scatter(
                    np.asarray(x_values, dtype=float),
                    np.asarray(y_values, dtype=float),
                    s=_marker_areas(np.asarray(duration, dtype=float)),
                    c=_run_colours(list(run_ids)),
                    marker="o",
                    edgecolors="none",
                    alpha=0.22,
                    rasterized=True,
                )
                for run_id, percentile, minimum, internal, join in zip(
                    run_ids, percentiles, minimum_durations, internal_gaps, join_gaps, strict=True
                ):
                    if run_id not in seen_run_ids:
                        seen_run_ids.add(run_id)
                        run_ids_seen.append(run_id)
                        run_parameters[run_id] = (float(percentile), int(minimum), int(internal), int(join))
                event_count += int(np.sum(np.asarray(cluster_event_counts, dtype=np.int64)))

    x_label = X_AXIS_RULE.replace("_", " ")
    y_label = Y_AXIS_RULE.replace("_", " ")
    if NORMALIZE_AXES:
        x_label = f"Normalised {x_label}"
        y_label = f"Normalised {y_label}"
        axis.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axis.set(xlabel=x_label, ylabel=y_label)
    axis.set_title("CUDAHeat event-level enrichment space", weight="bold", pad=12)
    axis.grid(True, color="#D9E2EC", linewidth=0.75)
    axis.set_axisbelow(True)
    legend_handles = [
        Line2D(
            [], [], marker="o", linestyle="", markersize=5,
            markerfacecolor=_run_colours([run_id])[0], markeredgecolor="none",
            label=_configuration_label(run_parameters[run_id]),
        )
        for run_id in run_ids_seen
    ]

    legend_axis.legend(
            handles=legend_handles,
        loc="upper left",
        ncol=1,
            frameon=False,
            fontsize=7,
            handletextpad=0.35,
            columnspacing=0.8,
        title="Run parameters",
        title_fontsize=8,
        )
    figure.text(
        0.5,
        0.01,
        f"{event_count:,} events represented by dynamic run-specific clusters; colour identifies run and marker area is mean event duration.",
        ha="center",
        fontsize=8,
        color="#627D98",
    )
    figure.savefig(OUTPUT_PATH, dpi=240, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {OUTPUT_PATH} ({event_count:,} events)")


if __name__ == "__main__":
    main()
