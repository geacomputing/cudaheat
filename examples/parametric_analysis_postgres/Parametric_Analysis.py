#!/usr/bin/env python3

"""
CUDAHeat PostgreSQL — Parameter Sensitivity Analysis

Produces:

    postgres_analysis/sensitivity_analysis.png

Layout:

    A) Event frequency
    B) Mean event duration
    C) Events per active ocean point

All three upper panels use:
    X = Maximum internal gap [h]
    Y = Join gap [h]

Bottom:
    Annual median MHW days per ocean point
    across parameter configurations

No config_hash is required.

PostgreSQL:
    config/postgres.ini
    ~/.pgpass
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import psycopg

from cudaheat import load_postgres_config


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

POSTGRES_CONFIG = REPOSITORY_ROOT / "config" / "postgres.ini"

OUTDIR = ROOT / "postgres_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTDIR / "sensitivity_analysis.png"


# ============================================================
# DATABASE
# ============================================================

def connect():
    """Connect using config/postgres.ini and ~/.pgpass."""
    config = load_postgres_config(POSTGRES_CONFIG)
    return psycopg.connect(**config.connection_kwargs())


def query(conn, sql):
    return pd.read_sql_query(sql, conn)


# ============================================================
# LOAD RUNS
# ============================================================

def load_runs(conn):

    runs = query(
        conn,
        """
        SELECT
            run_id,
            created_at,
            percentile,
            min_duration_hours,
            max_internal_gap_hours,
            join_gap_hours,
            source_sha256
        FROM runs
        ORDER BY created_at
        """
    )

    print(f"Runs found: {len(runs)}")

    return runs


# ============================================================
# GLOBAL STATISTICS PER RUN
# ============================================================

def load_run_statistics(conn, runs):

    ids = tuple(str(x) for x in runs["run_id"])

    if not ids:
        raise RuntimeError("No runs found.")

    placeholders = ",".join(["%s"] * len(ids))

    sql = f"""
        SELECT
            run_id,

            COUNT(*) AS n_events,

            COUNT(DISTINCT point_id)
                AS active_points,

            AVG(duration_hours)
                AS mean_duration_h,

            PERCENTILE_CONT(0.5)
                WITHIN GROUP (
                    ORDER BY duration_hours
                ) AS median_duration_h,

            MAX(duration_hours)
                AS max_duration_h,

            SUM(duration_hours)
                AS total_event_hours

        FROM events

        WHERE run_id IN ({placeholders})

        GROUP BY run_id
    """

    with conn.cursor() as cur:

        cur.execute(sql, ids)

        rows = cur.fetchall()

        columns = [
            d.name
            for d in cur.description
        ]

    stats = pd.DataFrame(
        rows,
        columns=columns
    )

    stats["run_id"] = stats["run_id"].astype(str)

    runs = runs.copy()
    runs["run_id"] = runs["run_id"].astype(str)

    return runs.merge(
        stats,
        on="run_id",
        how="inner"
    )


# ============================================================
# ANNUAL MHW DAYS
# ============================================================

def load_annual_statistics(conn, runs):

    """
    For every run/year:

        1. Calculate MHW days independently for every point.
        2. Calculate the spatial median across active points.

    Result:
        median MHW days per ocean point [days/year]
    """

    ids = tuple(str(x) for x in runs["run_id"])

    if not ids:
        raise RuntimeError("No runs found.")

    placeholders = ",".join(["%s"] * len(ids))

    sql = f"""
        WITH point_year AS (

            SELECT
                run_id,
                point_id,

                EXTRACT(
                    YEAR FROM start_time
                )::integer AS year,

                SUM(duration_hours) / 24.0
                    AS mhw_days

            FROM events

            WHERE run_id IN ({placeholders})

            GROUP BY
                run_id,
                point_id,
                year
        )

        SELECT
            run_id,
            year,

            PERCENTILE_CONT(0.5)
                WITHIN GROUP (
                    ORDER BY mhw_days
                ) AS median_mhw_days

        FROM point_year

        GROUP BY
            run_id,
            year

        ORDER BY
            year,
            run_id
    """

    with conn.cursor() as cur:

        cur.execute(sql, ids)

        rows = cur.fetchall()

        columns = [
            d.name
            for d in cur.description
        ]

    annual = pd.DataFrame(
        rows,
        columns=columns
    )

    annual["run_id"] = annual["run_id"].astype(str)

    return annual


# ============================================================
# HELPERS
# ============================================================

def sorted_unique(series):

    return np.array(
        sorted(
            pd.Series(series)
            .dropna()
            .unique()
        ),
        dtype=float
    )


def make_grid(
    data,
    value_column,
    x_values,
    y_values
):

    grid = np.full(
        (
            len(y_values),
            len(x_values)
        ),
        np.nan
    )

    for iy, y in enumerate(y_values):

        for ix, x in enumerate(x_values):

            subset = data[
                (
                    data["max_internal_gap_hours"] == x
                )
                &
                (
                    data["join_gap_hours"] == y
                )
            ]

            if not subset.empty:

                grid[iy, ix] = (
                    subset[value_column]
                    .astype(float)
                    .mean()
                )

    return grid


def annotate_panel(ax, label):

    ax.text(
        -0.13,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False
    )


# ============================================================
# SENSITIVITY SURFACE
# ============================================================

def sensitivity_panel(
    fig,
    ax,
    data,
    value_column,
    title,
    colorbar_label,
    x_values,
    y_values,
):

    grid = make_grid(
        data,
        value_column,
        x_values,
        y_values
    )

    xx = np.arange(len(x_values))
    yy = np.arange(len(y_values))

    masked = np.ma.masked_invalid(grid)

    image = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest"
    )

    # --------------------------------------------------------
    # WHITE CONTOURS
    # --------------------------------------------------------

    valid = grid[np.isfinite(grid)]

    if (
        len(valid) >= 4
        and len(x_values) > 1
        and len(y_values) > 1
        and np.nanmax(valid) > np.nanmin(valid)
    ):

        try:

            X, Y = np.meshgrid(
                xx,
                yy
            )

            levels = np.linspace(
                np.nanmin(valid),
                np.nanmax(valid),
                5
            )[1:-1]

            contours = ax.contour(
                X,
                Y,
                grid,
                levels=levels,
                colors="white",
                linewidths=1.3,
                alpha=0.95
            )

            ax.clabel(
                contours,
                inline=True,
                fontsize=8,
                fmt="%.1f",
                colors="white"
            )

        except (ValueError, TypeError):
            pass

    # --------------------------------------------------------
    # AXES
    #
    # No white scatter markers.
    # --------------------------------------------------------

    ax.set_xticks(xx)

    ax.set_xticklabels(
        [
            f"{v:g}"
            for v in x_values
        ],
        fontsize=9
    )

    ax.set_yticks(yy)

    ax.set_yticklabels(
        [
            f"{v:g}"
            for v in y_values
        ],
        fontsize=9
    )

    ax.set_xlabel(
        "Maximum internal gap [h]",
        fontsize=11,
        labelpad=9
    )

    ax.set_ylabel(
        "Join gap [h]",
        fontsize=11,
        labelpad=9
    )

    ax.set_title(
        title,
        fontsize=12,
        pad=12
    )

    # --------------------------------------------------------
    # COLORBAR
    # --------------------------------------------------------

    cbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.030,
        fraction=0.046
    )

    cbar.set_label(
        colorbar_label,
        fontsize=10,
        labelpad=8
    )

    cbar.ax.tick_params(
        labelsize=9
    )


# ============================================================
# ANNUAL SENSITIVITY PANEL
# ============================================================

def annual_panel(
    ax,
    annual,
    runs
):

    annual = annual.merge(
        runs[
            [
                "run_id",
                "max_internal_gap_hours",
                "join_gap_hours"
            ]
        ],
        on="run_id",
        how="left"
    )

    annual["median_mhw_days"] = (
        annual["median_mhw_days"]
        .astype(float)
    )

    years = np.array(
        sorted(
            annual["year"]
            .unique()
        )
    )

    # --------------------------------------------------------
    # DISTRIBUTION ACROSS PARAMETER CONFIGURATIONS
    # --------------------------------------------------------

    grouped = (
        annual
        .groupby("year")[
            "median_mhw_days"
        ]
    )

    median = (
        grouped
        .median()
        .reindex(years)
    )

    q25 = (
        grouped
        .quantile(0.25)
        .reindex(years)
    )

    q75 = (
        grouped
        .quantile(0.75)
        .reindex(years)
    )

    # --------------------------------------------------------
    # COLOUR = MAXIMUM INTERNAL GAP
    # --------------------------------------------------------

    gap_values = sorted_unique(
        annual[
            "max_internal_gap_hours"
        ]
    )

    if len(gap_values) > 1:

        norm = plt.Normalize(
            gap_values.min(),
            gap_values.max()
        )

    else:

        value = (
            gap_values[0]
            if len(gap_values)
            else 0
        )

        norm = plt.Normalize(
            0,
            max(1, value)
        )

    cmap = plt.get_cmap("OrRd")

    # --------------------------------------------------------
    # SMALL HORIZONTAL OFFSETS
    # --------------------------------------------------------

    run_ids = list(
        annual["run_id"]
        .drop_duplicates()
    )

    offsets = np.linspace(
        -0.20,
        0.20,
        max(len(run_ids), 1)
    )

    offset_lookup = dict(
        zip(
            run_ids,
            offsets
        )
    )

    # --------------------------------------------------------
    # INDIVIDUAL CONFIGURATIONS
    # --------------------------------------------------------

    for row in annual.itertuples():

        gap = float(
            row.max_internal_gap_hours
        )

        ax.scatter(
            row.year
            + offset_lookup[row.run_id],
            row.median_mhw_days,
            s=23,
            c=[
                cmap(
                    norm(gap)
                )
            ],
            alpha=0.65,
            edgecolors="none",
            zorder=3
        )

    # --------------------------------------------------------
    # INTERQUARTILE RANGE
    # --------------------------------------------------------

    ax.fill_between(
        years,
        q25.values,
        q75.values,
        alpha=0.18,
        label="Interquartile range across configurations",
        zorder=1
    )

    # --------------------------------------------------------
    # MEDIAN
    # --------------------------------------------------------

    ax.plot(
        years,
        median.values,
        linewidth=2.2,
        marker="D",
        markersize=5,
        label="Median across configurations",
        zorder=4
    )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_title(
        "Annual MHW occurrence across parameter configurations",
        fontsize=12,
        pad=13
    )

    ax.set_xlabel(
        "Year",
        fontsize=12,
        labelpad=8
    )

    ax.set_ylabel(
        "Median MHW days per ocean point [days/year]",
        fontsize=11,
        labelpad=10
    )

    if len(years) <= 15:

        tick_step = 1

    elif len(years) <= 30:

        tick_step = 2

    else:

        tick_step = 5

    ax.set_xticks(
        years[::tick_step]
    )

    ax.tick_params(
        axis="both",
        labelsize=10
    )

    ax.grid(
        alpha=0.18
    )

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9
    )

    # --------------------------------------------------------
    # PARAMETER COLORBAR
    # --------------------------------------------------------

    sm = plt.cm.ScalarMappable(
        norm=norm,
        cmap=cmap
    )

    sm.set_array([])

    cbar = ax.figure.colorbar(
        sm,
        ax=ax,
        pad=0.018,
        fraction=0.022
    )

    cbar.set_label(
        "Maximum internal gap [h]",
        fontsize=10,
        labelpad=9
    )

    cbar.ax.tick_params(
        labelsize=9
    )


# ============================================================
# CREATE FIGURE
# ============================================================

def make_figure(
    stats,
    annual
):

    x_values = sorted_unique(
        stats[
            "max_internal_gap_hours"
        ]
    )

    y_values = sorted_unique(
        stats[
            "join_gap_hours"
        ]
    )

    if not len(x_values):

        raise RuntimeError(
            "No max_internal_gap_hours values."
        )

    if not len(y_values):

        raise RuntimeError(
            "No join_gap_hours values."
        )

    # --------------------------------------------------------
    # COMMON PARAMETERS
    # --------------------------------------------------------

    percentile = (
        stats["percentile"]
        .dropna()
        .unique()
    )

    min_duration = (
        stats["min_duration_hours"]
        .dropna()
        .unique()
    )

    subtitle_parts = []

    if len(percentile) == 1:

        subtitle_parts.append(
            f"p{percentile[0] * 100:g}"
        )

    if len(min_duration) == 1:

        subtitle_parts.append(
            f"minimum duration: "
            f"{min_duration[0]:g} h"
        )

    subtitle_parts.append(
        f"{len(stats)} runs"
    )

    subtitle = "  |  ".join(
        subtitle_parts
    )

    # --------------------------------------------------------
    # DERIVED METRICS
    # --------------------------------------------------------

    plot_stats = stats.copy()

    plot_stats[
        "mean_duration_days"
    ] = (
        plot_stats[
            "mean_duration_h"
        ]
        .astype(float)
        / 24.0
    )

    # C:
    # Mean number of detected events associated with each
    # ocean point that experienced at least one event.

    plot_stats[
        "events_per_active_point"
    ] = np.where(
        plot_stats[
            "active_points"
        ].astype(float) > 0,
        (
            plot_stats[
                "n_events"
            ].astype(float)
            /
            plot_stats[
                "active_points"
            ].astype(float)
        ),
        np.nan
    )

    # --------------------------------------------------------
    # LAYOUT
    #
    #       A       B       C
    #
    #   -------------------------
    #       annual time series
    #   -------------------------
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(18, 10)
    )

    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[
            1.0,
            1.20
        ],
        hspace=0.52,
        wspace=0.48
    )

    ax_a = fig.add_subplot(
        gs[0, 0]
    )

    ax_b = fig.add_subplot(
        gs[0, 1]
    )

    ax_c = fig.add_subplot(
        gs[0, 2]
    )

    ax_bottom = fig.add_subplot(
        gs[1, :]
    )

    # --------------------------------------------------------
    # A — EVENT FREQUENCY
    # --------------------------------------------------------

    sensitivity_panel(
        fig,
        ax_a,
        plot_stats,
        "n_events",
        "Event frequency",
        "Detected events",
        x_values,
        y_values
    )

    annotate_panel(
        ax_a,
        "A"
    )

    # --------------------------------------------------------
    # B — MEAN EVENT DURATION
    # --------------------------------------------------------

    sensitivity_panel(
        fig,
        ax_b,
        plot_stats,
        "mean_duration_days",
        "Mean event duration",
        "Days per event",
        x_values,
        y_values
    )

    annotate_panel(
        ax_b,
        "B"
    )

    # --------------------------------------------------------
    # C — EVENTS PER ACTIVE OCEAN POINT
    # --------------------------------------------------------

    sensitivity_panel(
        fig,
        ax_c,
        plot_stats,
        "events_per_active_point",
        "Events per active ocean point",
        "Events / active point",
        x_values,
        y_values
    )

    annotate_panel(
        ax_c,
        "C"
    )

    # --------------------------------------------------------
    # BOTTOM — ANNUAL SERIES
    #
    # Deliberately no panel letter.
    # --------------------------------------------------------

    annual_panel(
        ax_bottom,
        annual,
        plot_stats
    )

    # --------------------------------------------------------
    # MAIN TITLE
    # --------------------------------------------------------

    fig.suptitle(
        "CUDAHeat parameter sensitivity and temporal evolution",
        fontsize=17,
        y=0.985
    )

    fig.text(
        0.5,
        0.952,
        subtitle,
        ha="center",
        va="top",
        fontsize=11
    )

    # --------------------------------------------------------
    # FINAL SPACING
    # --------------------------------------------------------

    fig.subplots_adjust(
        top=0.88,
        bottom=0.08,
        left=0.07,
        right=0.94
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    fig.savefig(
        OUTPUT,
        dpi=350,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {OUTPUT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "CUDAHeat parameter sensitivity analysis"
    )

    print(
        f"PostgreSQL config: {POSTGRES_CONFIG}"
    )

    print(
        f"Output: {OUTPUT}"
    )

    with connect() as conn:

        runs = load_runs(
            conn
        )

        stats = load_run_statistics(
            conn,
            runs
        )

        annual = load_annual_statistics(
            conn,
            runs
        )

    print(
        f"Analysing {len(stats)} runs"
    )

    make_figure(
        stats,
        annual
    )

    print("Done.")


if __name__ == "__main__":
    main()