"""Minimal GPU/WebGL explorer for PostgreSQL-backed CUDAHeat event results.

The Streamlit/PyDeck bridge serializes layer data as JSON; PyDeck's typed-array
transport is documented as Jupyter-only.  This app therefore performs the only
large operation in PostgreSQL: all selected events are grouped into Cartesian
X/Y density cells before transfer. deck.gl renders those cells with WebGL.
"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from base64 import b64encode
from io import BytesIO
from pathlib import Path
import math
import os
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from sqlalchemy import URL, bindparam, create_engine, text
from sqlalchemy.engine import Engine


# Required CUDAHeat configuration path. Because this app is top-level, its
# repository-local fallback is one directory shallower than examples scripts.
POSTGRES_CONFIG = Path(__file__).resolve().parents[2] / "config" / "postgres.ini"
if not POSTGRES_CONFIG.is_file():
    POSTGRES_CONFIG = Path(__file__).resolve().parents[1] / "config" / "postgres.ini"

DEFAULT_GRID_BINS = 96
TIME_WINDOWS = {
    "Month (year-month)": "month",
    "Season (year-quarter)": "season",
    "Half-year (year-half)": "halfyear",
}
TEMPORAL_WINDOWS = {
    "Month-year": "month",
    "Season-year": "season",
    "Year": "year",
}
CHART_HEIGHT = 920
CHART_BACKGROUND = [10, 18, 31, 255]
AXIS_COLOUR = [255, 255, 255, 255]
TICK_COLOUR = [226, 232, 240, 255]
TIME_COLOUR_STOPS = np.asarray(
    [
        [68, 1, 84, 255],
        [59, 82, 139, 255],
        [33, 145, 140, 255],
        [94, 201, 98, 255],
        [253, 231, 37, 255],
    ],
    dtype=float,
)
PALETTE = np.asarray(
    [
        [0, 114, 178, 118],
        [213, 94, 0, 118],
        [0, 158, 115, 118],
        [204, 121, 167, 118],
        [230, 159, 0, 118],
        [86, 180, 233, 118],
        [0, 59, 92, 118],
        [122, 81, 149, 118],
    ],
    dtype=np.uint8,
)
COMPARISON_PALETTE = np.asarray([[56, 189, 248, 235], [251, 146, 60, 235]], dtype=np.uint8)
TEMPORAL_MARKERS = ("●", "■", "▲", "◆")
MARKER_ICON_ATLAS = "data:image/svg+xml;base64," + b64encode(
    b"""<svg xmlns="http://www.w3.org/2000/svg" width="256" height="64" viewBox="0 0 256 64">
    <circle cx="32" cy="32" r="23" fill="white"/>
    <rect x="73" y="9" width="46" height="46" fill="white"/>
    <path d="M160 7 L188 56 L132 56 Z" fill="white"/>
    <path d="M224 6 L250 32 L224 58 L198 32 Z" fill="white"/>
    </svg>"""
).decode("ascii")
MARKER_ICON_MAPPING = {
    name: {
        "x": index * 64,
        "y": 0,
        "width": 64,
        "height": 64,
        "anchorX": 32,
        "anchorY": 32,
        "mask": True,
    }
    for index, name in enumerate(("circle", "square", "triangle", "diamond"))
}


@dataclass(frozen=True)
class AxisLayout:
    """Cartesian axes and labels expressed in the same data space as deck.gl."""

    data_xmin: float
    data_xmax: float
    data_ymin: float
    data_ymax: float
    view_xmin: float
    view_xmax: float
    view_ymin: float
    view_ymax: float
    x_axis_y: float
    y_axis_x: float
    x_label_y: float
    y_label_x: float
    colourbar_x: float | None
    colourbar_label_x: float | None
    x_ticks: tuple[float, ...]
    y_ticks: tuple[float, ...]


def _setting(
    section: object, name: str, environment: str, default: str | None = None
) -> str | None:
    return os.environ.get(environment, section.get(name, fallback=default))  # type: ignore[union-attr]


@st.cache_resource
def engine() -> Engine:
    """Create one SQLAlchemy engine without storing credentials in the app."""
    parser = ConfigParser()
    if not parser.read(POSTGRES_CONFIG) or "postgres" not in parser:
        raise FileNotFoundError(f"PostgreSQL configuration not found: {POSTGRES_CONFIG}")
    section = parser["postgres"]
    password_variable = _setting(section, "password_env", "CUDAHEAT_POSTGRES_PASSWORD_ENV")
    password = os.environ.get(password_variable) if password_variable else None
    url = URL.create(
        "postgresql+psycopg",
        username=_setting(section, "username", "CUDAHEAT_POSTGRES_USER"),
        password=password,
        host=_setting(section, "host", "CUDAHEAT_POSTGRES_HOST"),
        port=int(_setting(section, "port", "CUDAHEAT_POSTGRES_PORT", "5432") or "5432"),
        database=_setting(section, "database", "CUDAHEAT_POSTGRES_DATABASE"),
        query={
            "sslmode": _setting(section, "sslmode", "CUDAHEAT_POSTGRES_SSLMODE", "prefer")
            or "prefer",
            "connect_timeout": _setting(
                section, "connect_timeout", "CUDAHEAT_POSTGRES_CONNECT_TIMEOUT", "10"
            )
            or "10",
            "application_name": "cudaheat_gpu_explorer",
        },
    )
    return create_engine(url, pool_pre_ping=True, pool_size=1, max_overflow=0)


@st.cache_data(ttl=60)
def catalogue(_engine: Engine) -> tuple[pd.DataFrame, tuple[str, ...]]:
    datasets = pd.read_sql(
        text(
            """
            SELECT r.run_id::text AS run_id, r.created_at, r.percentile,
                   r.min_duration_hours, r.max_internal_gap_hours, r.join_gap_hours,
                   COUNT(e.event_id) AS event_count
            FROM runs AS r JOIN events AS e ON e.run_id = r.run_id
            GROUP BY r.run_id, r.created_at, r.percentile, r.min_duration_hours,
                     r.max_internal_gap_hours, r.join_gap_hours
            ORDER BY r.created_at DESC
            """
        ),
        _engine,
    )
    variables = pd.read_sql(
        text(
            """
            SELECT DISTINCT rule_id AS variable
            FROM event_enrichments
            WHERE result ? 'value'
            ORDER BY rule_id
            """
        ),
        _engine,
    )["variable"].tolist()
    return datasets, tuple(["duration_hours", *variables])


def _axis_expression(variable: str, alias: str) -> tuple[str, str]:
    if variable == "duration_hours":
        return "e.duration_hours::double precision", ""
    return (
        f"NULLIF({alias}.result ->> 'value', '')::double precision",
        f"JOIN event_enrichments AS {alias} "
        f"ON {alias}.event_id = e.event_id AND {alias}.rule_id = :{alias}",
    )


def _time_group_expressions(window: str, *, event_alias: str = "e") -> tuple[str, str]:
    """Return safe SQL expressions for progressive calendar time bins."""
    timestamp = f"{event_alias}.start_time"
    if window == "month":
        return (
            f"DATE_TRUNC('month', {timestamp})",
            f"TO_CHAR({timestamp}, 'YYYY-MM')",
        )
    if window == "season":
        return (
            f"DATE_TRUNC('quarter', {timestamp})",
            f"TO_CHAR({timestamp}, 'YYYY') || '-Q' || EXTRACT(QUARTER FROM {timestamp})::integer",
        )
    if window == "halfyear":
        half = f"FLOOR((EXTRACT(MONTH FROM {timestamp}) - 1) / 6)"
        return (
            f"DATE_TRUNC('year', {timestamp}) + INTERVAL '6 months' * {half}",
            f"TO_CHAR({timestamp}, 'YYYY') || '-H' || ({half}::integer + 1)",
        )
    if window == "year":
        return f"DATE_TRUNC('year', {timestamp})", f"TO_CHAR({timestamp}, 'YYYY')"
    raise ValueError(f"unsupported time grouping window: {window}")


def _cluster_query(
    x_variable: str,
    y_variable: str,
    marker_variable: str | None,
    bins: int,
    normalise: bool,
    log_x: bool,
    log_y: bool,
    seasonal_trajectory: bool,
    time_window: str,
):
    x_expression, x_join = _axis_expression(x_variable, "x_rule")
    if x_variable == y_variable:
        y_expression, y_join = x_expression, ""
    else:
        y_expression, y_join = _axis_expression(y_variable, "y_rule")
    if marker_variable is None:
        marker_expression, marker_join = "e.duration_hours::double precision", ""
    elif marker_variable == x_variable:
        marker_expression, marker_join = x_expression, ""
    elif marker_variable == y_variable:
        marker_expression, marker_join = y_expression, ""
    else:
        marker_expression, marker_join = _axis_expression(marker_variable, "marker_rule")
    x_transform = "LN(x_value) / LN(10.0)" if log_x else "x_value"
    y_transform = "LN(y_value) / LN(10.0)" if log_y else "y_value"
    x_output = "AVG(x_normalised)" if normalise else "AVG(x_axis)"
    y_output = "AVG(y_normalised)" if normalise else "AVG(y_axis)"
    if seasonal_trajectory:
        final_select = f"""
        SELECT run_id, time_start, time_group,
               {x_output} AS x, {y_output} AS y,
               AVG(duration_hours) AS mean_duration_hours,
               AVG(marker_value) AS marker_value,
               COUNT(*)::bigint AS event_count
        FROM normalised_values
        GROUP BY run_id, time_start, time_group
        ORDER BY run_id, time_start
        """
    else:
        final_select = f"""
        SELECT run_id, x_bin, y_bin, {x_output} AS x, {y_output} AS y,
               AVG(duration_hours) AS mean_duration_hours,
               AVG(marker_value) AS marker_value,
               COUNT(*)::bigint AS event_count
        FROM binned_values
        GROUP BY run_id, x_bin, y_bin
        ORDER BY run_id, x_bin, y_bin
        """
    time_start, time_group = _time_group_expressions(time_window)
    query = text(
        f"""
        WITH event_values AS (
            SELECT e.run_id::text AS run_id,
                   {time_start} AS time_start,
                   {time_group} AS time_group,
                   e.duration_hours::double precision AS duration_hours,
                   {x_expression} AS x_value,
                   {y_expression} AS y_value,
                   {marker_expression} AS marker_value
            FROM events AS e
            {x_join}
            {y_join}
            {marker_join}
            WHERE e.run_id::text IN :run_ids
        ), finite_values AS (
            SELECT * FROM event_values
            WHERE x_value IS NOT NULL AND y_value IS NOT NULL AND marker_value IS NOT NULL
              AND ({str(not log_x).upper()} OR x_value > 0)
              AND ({str(not log_y).upper()} OR y_value > 0)
        ), transformed_values AS (
            SELECT *, {x_transform} AS x_axis, {y_transform} AS y_axis
            FROM finite_values
        ), bounds AS (
            SELECT MIN(x_axis) AS xmin, MAX(x_axis) AS xmax,
                   MIN(y_axis) AS ymin, MAX(y_axis) AS ymax
            FROM transformed_values
        ), normalised_values AS (
            SELECT f.*,
                   CASE WHEN b.xmax = b.xmin THEN 0.5
                        ELSE (f.x_axis - b.xmin) / (b.xmax - b.xmin) END AS x_normalised,
                   CASE WHEN b.ymax = b.ymin THEN 0.5
                        ELSE (f.y_axis - b.ymin) / (b.ymax - b.ymin) END AS y_normalised
            FROM transformed_values AS f CROSS JOIN bounds AS b
        ), binned_values AS (
            SELECT *,
                   LEAST({bins - 1}, GREATEST(0, FLOOR(x_normalised * {bins})::integer)) AS x_bin,
                   LEAST({bins - 1}, GREATEST(0, FLOOR(y_normalised * {bins})::integer)) AS y_bin
            FROM normalised_values
        )
        {final_select}
        """
    ).bindparams(bindparam("run_ids", expanding=True))
    parameters: dict[str, object] = {"run_ids": []}
    if x_variable != "duration_hours":
        parameters["x_rule"] = x_variable
    if y_variable != "duration_hours" and y_variable != x_variable:
        parameters["y_rule"] = y_variable
    if marker_variable not in (None, "duration_hours", x_variable, y_variable):
        parameters["marker_rule"] = marker_variable
    return query, parameters


@st.cache_data(ttl=60, show_spinner=False)
def clustered_points(
    _engine: Engine,
    run_ids: tuple[str, ...],
    x_variable: str,
    y_variable: str,
    marker_variable: str | None,
    bins: int,
    normalise: bool,
    log_x: bool,
    log_y: bool,
    seasonal_trajectory: bool,
    time_window: str,
) -> pd.DataFrame:
    query, parameters = _cluster_query(
        x_variable,
        y_variable,
        marker_variable,
        bins,
        normalise,
        log_x,
        log_y,
        seasonal_trajectory,
        time_window,
    )
    parameters["run_ids"] = list(run_ids)
    return pd.read_sql(query, _engine, params=parameters)


@st.cache_data(ttl=60, show_spinner=False)
def temporal_comparison_points(
    _engine: Engine,
    run_ids: tuple[str, str],
    metric_variable: str,
    time_window: str,
) -> pd.DataFrame:
    """Aggregate two runs into progressive calendar bins without trajectory geometry."""
    if metric_variable == "event_count":
        metric_expression, metric_join, metric_filter, value_expression = (
            "1.0::double precision",
            "",
            "",
            "COUNT(*)::double precision",
        )
    else:
        metric_expression, metric_join = _axis_expression(metric_variable, "metric_rule")
        metric_filter = "WHERE metric_value IS NOT NULL"
        value_expression = "AVG(metric_value)"
    time_start, time_group = _time_group_expressions(time_window)
    query = text(
        f"""
        WITH time_values AS (
            SELECT e.run_id::text AS run_id,
                   {time_start} AS time_start,
                   {time_group} AS time_group,
                   {metric_expression} AS metric_value
            FROM events AS e
            {metric_join}
            WHERE e.run_id::text IN :run_ids
        ), finite_values AS (
            SELECT * FROM time_values
            {metric_filter}
        )
        SELECT run_id, time_start, time_group,
               {value_expression} AS value,
               COUNT(*)::bigint AS event_count
        FROM finite_values
        GROUP BY run_id, time_start, time_group
        ORDER BY time_start, run_id
        """
    ).bindparams(bindparam("run_ids", expanding=True))
    parameters: dict[str, object] = {"run_ids": list(run_ids)}
    if metric_variable not in ("event_count", "duration_hours"):
        parameters["metric_rule"] = metric_variable
    return pd.read_sql(query, _engine, params=parameters)


def _finite_numeric_sql(column: str) -> str:
    """Portable finite-value predicate for PostgreSQL double precision values."""
    return (
        f"{column} IS NOT NULL AND {column} NOT IN "
        "('Infinity'::double precision, '-Infinity'::double precision, 'NaN'::double precision)"
    )


def _temporal_scatter_query(
    x_variable: str,
    y_variable: str,
    size_variable: str | None,
    colour_variable: str | None,
    time_window: str,
    log_x: bool,
    log_y: bool,
):
    """Aggregate event descriptors by run and progressive time group."""
    x_expression, x_join = _axis_expression(x_variable, "temporal_x_rule")
    if y_variable == x_variable:
        y_expression, y_join = x_expression, ""
    else:
        y_expression, y_join = _axis_expression(y_variable, "temporal_y_rule")
    if size_variable is None:
        size_expression, size_join = "NULL::double precision", ""
    elif size_variable == x_variable:
        size_expression, size_join = x_expression, ""
    elif size_variable == y_variable:
        size_expression, size_join = y_expression, ""
    else:
        size_expression, size_join = _axis_expression(size_variable, "temporal_size_rule")
    if colour_variable is None:
        colour_expression, colour_join = "NULL::double precision", ""
    elif colour_variable == x_variable:
        colour_expression, colour_join = x_expression, ""
    elif colour_variable == y_variable:
        colour_expression, colour_join = y_expression, ""
    elif colour_variable == size_variable:
        colour_expression, colour_join = size_expression, ""
    else:
        colour_expression, colour_join = _axis_expression(colour_variable, "temporal_colour_rule")
    time_start, time_group = _time_group_expressions(time_window)
    required = [_finite_numeric_sql("x_value"), _finite_numeric_sql("y_value")]
    if size_variable is not None:
        required.append(_finite_numeric_sql("size_value"))
    if colour_variable is not None:
        required.append(_finite_numeric_sql("colour_value"))
    x_axis = "LN(x_value) / LN(10.0)" if log_x else "x_value"
    y_axis = "LN(y_value) / LN(10.0)" if log_y else "y_value"
    size_output = "COUNT(*)::double precision" if size_variable is None else "AVG(size_value)"
    colour_output = "COUNT(*)::double precision" if colour_variable is None else "AVG(colour_value)"
    query = text(
        f"""
        WITH event_values AS (
            SELECT e.run_id::text AS run_id,
                   {time_start} AS time_start,
                   {time_group} AS time_group,
                   {x_expression} AS x_value,
                   {y_expression} AS y_value,
                   {size_expression} AS size_value,
                   {colour_expression} AS colour_value
            FROM events AS e
            {x_join}
            {y_join}
            {size_join}
            {colour_join}
            WHERE e.run_id::text IN :run_ids
        ), finite_values AS (
            SELECT *, {x_axis} AS x_axis, {y_axis} AS y_axis
            FROM event_values
            WHERE {" AND ".join(required)}
              AND ({str(not log_x).upper()} OR x_value > 0)
              AND ({str(not log_y).upper()} OR y_value > 0)
        )
        SELECT run_id, time_start, time_group,
               AVG(x_axis) AS x, AVG(y_axis) AS y,
               {size_output} AS size_value,
               {colour_output} AS colour_value,
               COUNT(*)::bigint AS event_count
        FROM finite_values
        GROUP BY run_id, time_start, time_group
        ORDER BY time_start, run_id
        """
    ).bindparams(bindparam("run_ids", expanding=True))
    parameters: dict[str, object] = {"run_ids": []}
    if x_variable != "duration_hours":
        parameters["temporal_x_rule"] = x_variable
    if y_variable not in ("duration_hours", x_variable):
        parameters["temporal_y_rule"] = y_variable
    if size_variable not in (None, "duration_hours", x_variable, y_variable):
        parameters["temporal_size_rule"] = size_variable
    if colour_variable not in (None, "duration_hours", x_variable, y_variable, size_variable):
        parameters["temporal_colour_rule"] = colour_variable
    return query, parameters


@st.cache_data(ttl=60, show_spinner=False)
def temporal_scatter_points(
    _engine: Engine,
    run_ids: tuple[str, ...],
    x_variable: str,
    y_variable: str,
    size_variable: str | None,
    colour_variable: str | None,
    time_window: str,
    log_x: bool,
    log_y: bool,
) -> pd.DataFrame:
    query, parameters = _temporal_scatter_query(
        x_variable,
        y_variable,
        size_variable,
        colour_variable,
        time_window,
        log_x,
        log_y,
    )
    parameters["run_ids"] = list(run_ids)
    return pd.read_sql(query, _engine, params=parameters)


def _dataset_label(row: pd.Series) -> str:
    return (
        f"q={row.percentile:.2f} | min={row.min_duration_hours} h | "
        f"gap={row.max_internal_gap_hours}/{row.join_gap_hours} h "
        f"({row.event_count:,} events)"
    )


def _display_name(variable: str, *, normalise: bool, log_scale: bool = False) -> str:
    """Turn persisted names into compact scientific axis labels."""
    if variable == "event_count":
        label = "Event count"
    elif variable == "duration_hours":
        label = "Duration (hours)"
    else:
        label = variable.replace("_", " ")
    if log_scale:
        label = f"log10 {label}"
    return f"Normalised {label}" if normalise else label


def _marker_radius(frame: pd.DataFrame, marker_variable: str | None) -> np.ndarray:
    """Map density or a third event descriptor to a legible pixel radius."""
    if marker_variable is None:
        return np.clip(np.sqrt(frame.event_count.to_numpy(dtype=float)), 1.0, 9.0)
    value_column = "marker_value" if "marker_value" in frame.columns else "size_value"
    values = frame[value_column].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(frame), 4.0)
    low, high = np.quantile(values[finite], (0.05, 0.95))
    if not high > low:
        return np.full(len(frame), 4.0)
    scaled = 2.0 + 7.0 * (values - low) / (high - low)
    return np.clip(np.nan_to_num(scaled, nan=2.0, posinf=9.0, neginf=2.0), 2.0, 9.0)


def _value_colours(values: np.ndarray, opacity: int) -> list[list[int]]:
    """Map numeric values onto the shared continuous scientific colour scale."""
    finite = np.isfinite(values)
    if not finite.any():
        return [[148, 163, 184, opacity] for _ in values]
    low, high = np.quantile(values[finite], (0.02, 0.98))
    if not high > low:
        scaled = np.full(len(values), 0.5)
    else:
        scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return [(*_time_colour(float(value))[:3], opacity) for value in scaled]


def _run_outline_colours(run_ids: Sequence[str]) -> dict[str, list[int]]:
    return {
        run_id: [*PALETTE[index % len(PALETTE)][:3].astype(int).tolist(), 255]
        for index, run_id in enumerate(run_ids)
    }


def _qualitative_contour_layers(layout: AxisLayout) -> list[pdk.Layer]:
    """Draw thin dashed semi-circular guide arcs; they encode no measurement."""
    x_radius = (layout.data_xmax - layout.data_xmin) / 2
    y_radius = layout.data_ymax - layout.x_axis_y
    if not x_radius > 0 or not y_radius > 0:
        return []
    # Centre the arcs on the horizontal-axis origin, giving upper semicircles
    # that span the plotted descriptor plane without implying severity values.
    origin_x = (layout.data_xmin + layout.data_xmax) / 2
    origin_y = layout.x_axis_y
    segments: list[dict[str, object]] = []
    for radius, colour in zip(
        (0.28, 0.46, 0.64, 0.82),
        (
            [148, 163, 184, 130],
            [148, 163, 184, 155],
            [226, 232, 240, 175],
            [255, 255, 255, 195],
        ),
        strict=True,
    ):
        angles = np.linspace(0.0, math.pi, 73)
        arc = np.column_stack(
            (
                origin_x + radius * x_radius * np.cos(angles),
                origin_y + radius * y_radius * np.sin(angles),
            )
        )
        # Individual short paths create a dashed line without a deck.gl
        # extension or an additional frontend dependency.
        for start in range(0, len(arc) - 3, 6):
            segments.append({"path": arc[start : start + 3].tolist(), "colour": colour})
    if not segments:
        return []
    return [
        pdk.Layer(
            "PathLayer",
            id="temporal-qualitative-contours",
            data=segments,
            get_path="path",
            get_color="colour",
            get_width=1,
            width_units="pixels",
            coordinate_system=pdk.types.String("cartesian"),
            pickable=False,
        )
    ]


def _time_colour(position: float) -> list[int]:
    """Interpolate a perceptually ordered dark-purple-to-yellow time scale."""
    clipped = min(max(position, 0.0), 1.0)
    scaled = clipped * (len(TIME_COLOUR_STOPS) - 1)
    index = min(int(scaled), len(TIME_COLOUR_STOPS) - 2)
    fraction = scaled - index
    colour = TIME_COLOUR_STOPS[index] * (1.0 - fraction) + TIME_COLOUR_STOPS[index + 1] * fraction
    return np.rint(colour).astype(int).tolist()


def _trajectory_groups(frame: pd.DataFrame) -> list[str]:
    """Return event-onset time groups in chronological order."""
    return [str(group) for group in frame.sort_values("time_start").time_group.drop_duplicates()]


def _trajectory_colours(groups: Sequence[str]) -> dict[str, list[int]]:
    """Map each chronological time group onto the continuous time scale."""
    denominator = max(len(groups) - 1, 1)
    return {group: _time_colour(index / denominator) for index, group in enumerate(groups)}


def _nice_ticks(lower: float, upper: float, *, target: int = 6) -> tuple[float, ...]:
    """Return readable, inclusive tick locations without a plotting dependency."""
    span = upper - lower
    if not math.isfinite(span) or span <= 0:
        return (lower,)
    raw_step = span / max(target - 1, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    fraction = raw_step / magnitude
    multiplier = next(value for value in (1.0, 2.0, 2.5, 5.0, 10.0) if fraction <= value)
    step = multiplier * magnitude
    start = math.ceil(lower / step) * step
    end = math.floor(upper / step) * step
    count = max(0, int(round((end - start) / step))) + 1
    return tuple(float(round(start + index * step, 14)) for index in range(count))


def _tick_label(value: float) -> str:
    """Keep deck.gl tick labels useful from tiny fractions through durations."""
    magnitude = abs(value)
    if magnitude and (magnitude >= 10_000 or magnitude < 0.001):
        return f"{value:.2e}"
    if magnitude >= 100:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _axis_layout(frame: pd.DataFrame, *, show_time_colourbar: bool = False) -> AxisLayout:
    """Allocate room for data-space axes, numerical ticks, and both labels."""
    xmin, xmax = float(frame.x.min()), float(frame.x.max())
    ymin, ymax = float(frame.y.min()), float(frame.y.max())
    x_span = max(xmax - xmin, max(abs(xmin), abs(xmax), 1.0) * 0.02)
    y_span = max(ymax - ymin, max(abs(ymin), abs(ymax), 1.0) * 0.02)
    data_xmin, data_xmax = xmin - 0.03 * x_span, xmax + 0.03 * x_span
    data_ymin, data_ymax = ymin - 0.03 * y_span, ymax + 0.03 * y_span
    x_span = data_xmax - data_xmin
    y_span = data_ymax - data_ymin
    y_axis_x = data_xmin - 0.07 * x_span
    x_axis_y = data_ymin - 0.08 * y_span
    colourbar_x = data_xmax + 0.11 * x_span if show_time_colourbar else None
    colourbar_label_x = data_xmax + 0.20 * x_span if show_time_colourbar else None
    return AxisLayout(
        data_xmin=data_xmin,
        data_xmax=data_xmax,
        data_ymin=data_ymin,
        data_ymax=data_ymax,
        view_xmin=y_axis_x - 0.22 * x_span,
        view_xmax=(data_xmax + 0.30 * x_span)
        if show_time_colourbar
        else (data_xmax + 0.04 * x_span),
        view_ymin=x_axis_y - 0.20 * y_span,
        view_ymax=data_ymax + 0.05 * y_span,
        x_axis_y=x_axis_y,
        y_axis_x=y_axis_x,
        x_label_y=x_axis_y - 0.13 * y_span,
        y_label_x=y_axis_x - 0.17 * x_span,
        colourbar_x=colourbar_x,
        colourbar_label_x=colourbar_label_x,
        x_ticks=_nice_ticks(xmin, xmax),
        y_ticks=_nice_ticks(ymin, ymax),
    )


def _view_state(layout: AxisLayout) -> pdk.ViewState:
    x_span = max(layout.view_xmax - layout.view_xmin, 1e-12)
    y_span = max(layout.view_ymax - layout.view_ymin, 1e-12)
    # OrthographicView accepts independent X/Y zoom levels. This is essential
    # for scientific variables with very different numeric ranges (e.g., event
    # duration in hours versus a unit-interval normalized rule).
    return pdk.ViewState(
        target=[
            (layout.view_xmin + layout.view_xmax) / 2,
            (layout.view_ymin + layout.view_ymax) / 2,
            0,
        ],
        zoom=0,
        zoom_x=math.log2(900.0 / x_span),
        zoom_y=math.log2(620.0 / y_span),
    )


def _layers(
    frame: pd.DataFrame,
    run_ids: tuple[str, ...],
    run_labels: Mapping[str, str],
    colour_by_dataset: bool,
    seasonal_trajectory: bool,
) -> list[pdk.Layer]:
    if seasonal_trajectory:
        run_id = run_ids[0]
        subset = frame.copy()
        subset["run_label"] = run_labels.get(run_id, run_id)
        return [
            pdk.Layer(
                "ScatterplotLayer",
                id=f"run-{run_id}",
                data=subset,
                get_position="[x, y]",
                get_radius="radius",
                get_fill_color="colour",
                coordinate_system=pdk.types.String("cartesian"),
                radius_units="pixels",
                radius_min_pixels=3,
                radius_max_pixels=11,
                opacity=0.88,
                pickable=True,
                auto_highlight=True,
                stroked=True,
                get_line_color=[255, 255, 255, 225],
                line_width_min_pixels=1,
            )
        ]
    layers: list[pdk.Layer] = []
    for index, run_id in enumerate(run_ids):
        subset = frame.loc[frame.run_id == run_id].copy()
        if subset.empty:
            continue
        subset["run_label"] = run_labels.get(run_id, run_id)
        colour = PALETTE[index % len(PALETTE)] if colour_by_dataset else PALETTE[0]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id=f"run-{run_id}",
                data=subset,
                get_position="[x, y]",
                get_radius="radius",
                get_fill_color=colour.tolist(),
                # Streamlit's deck.gl frontend accepts the literal coordinate
                # system name (not the enum values from older deck.gl APIs).
                # ``pdk.types.String`` prevents pydeck from treating it as a
                # JavaScript expression during JSON serialization.
                coordinate_system=pdk.types.String("cartesian"),
                radius_units="pixels",
                radius_min_pixels=1,
                radius_max_pixels=7,
                opacity=0.65,
                pickable=True,
                auto_highlight=True,
                stroked=False,
            )
        )
    return layers


def _comparison_layers(
    frame: pd.DataFrame, run_ids: tuple[str, str], run_labels: Mapping[str, str]
) -> list[pdk.Layer]:
    """Draw two independent temporal series: lines describe time, not paths in descriptor space."""
    cartesian = pdk.types.String("cartesian")
    layers: list[pdk.Layer] = []
    for index, run_id in enumerate(run_ids):
        subset = frame.loc[frame.run_id == run_id].sort_values("time_start").copy()
        if subset.empty:
            continue
        subset["run_label"] = run_labels.get(run_id, run_id)
        colour = COMPARISON_PALETTE[index].tolist()
        layers.extend(
            [
                pdk.Layer(
                    "PathLayer",
                    id=f"comparison-line-{index}",
                    data=[{"path": subset.loc[:, ["x", "y"]].to_numpy().tolist()}],
                    get_path="path",
                    get_color=colour,
                    get_width=2.5,
                    width_units="pixels",
                    coordinate_system=cartesian,
                    pickable=False,
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    id=f"comparison-points-{index}",
                    data=subset,
                    get_position="[x, y]",
                    get_radius="radius",
                    get_fill_color=colour,
                    get_line_color=[255, 255, 255, 230],
                    line_width_min_pixels=1,
                    radius_units="pixels",
                    radius_min_pixels=4,
                    radius_max_pixels=10,
                    coordinate_system=cartesian,
                    pickable=True,
                    auto_highlight=True,
                    stroked=True,
                ),
            ]
        )
    return layers


def _streamline_path(frame: pd.DataFrame, *, steps_per_segment: int = 10) -> list[list[float]]:
    """Smooth chronological centroids with a Catmull–Rom-equivalent Hermite path."""
    points = frame.sort_values("time_start").loc[:, ["x", "y"]].to_numpy(dtype=float)
    if len(points) < 2:
        return points.tolist()
    path: list[list[float]] = []
    for index in range(len(points) - 1):
        before = points[max(index - 1, 0)]
        start = points[index]
        end = points[index + 1]
        after = points[min(index + 2, len(points) - 1)]
        tangent_start = 0.5 * (end - before)
        tangent_end = 0.5 * (after - start)
        for step in range(steps_per_segment):
            time = step / steps_per_segment
            time_squared, time_cubed = time * time, time * time * time
            point = (
                (2 * time_cubed - 3 * time_squared + 1) * start
                + (time_cubed - 2 * time_squared + time) * tangent_start
                + (-2 * time_cubed + 3 * time_squared) * end
                + (time_cubed - time_squared) * tangent_end
            )
            path.append(point.tolist())
    path.append(points[-1].tolist())
    return path


def _streamline_arrows(frame: pd.DataFrame) -> list[dict[str, list[list[float]]]]:
    """Build arrowhead triangles that point from each time group to the next."""
    points = frame.sort_values("time_start").loc[:, ["x", "y"]].to_numpy(dtype=float)
    if len(points) < 2:
        return []
    x_span = max(float(frame.x.max() - frame.x.min()), 1e-12)
    y_span = max(float(frame.y.max() - frame.y.min()), 1e-12)
    length = 0.030
    width = 0.005
    arrows: list[dict[str, list[list[float]]]] = []
    for start, end in zip(points[:-1], points[1:], strict=True):
        direction = np.asarray([(end[0] - start[0]) / x_span, (end[1] - start[1]) / y_span])
        magnitude = float(np.linalg.norm(direction))
        if magnitude <= np.finfo(float).eps:
            continue
        unit = direction / magnitude
        along = np.asarray([unit[0] * x_span, unit[1] * y_span])
        normal = np.asarray([-unit[1] * x_span, unit[0] * y_span])
        tip = start + 0.78 * (end - start)
        base = tip - length * along
        arrows.append(
            {
                "polygon": [
                    tip.tolist(),
                    (base + width * normal).tolist(),
                    (base - width * normal).tolist(),
                ]
            }
        )
    return arrows


def _streamline_layers(frame: pd.DataFrame) -> list[pdk.Layer]:
    """Render a smooth white streamline and directional arrowheads."""
    path = _streamline_path(frame)
    if len(path) < 2:
        return []
    cartesian = pdk.types.String("cartesian")
    layers = [
        pdk.Layer(
            "PathLayer",
            id="seasonal-streamline",
            data=[{"path": path}],
            get_path="path",
            get_color=[255, 255, 255, 230],
            get_width=0.5,
            width_units="pixels",
            coordinate_system=cartesian,
            pickable=False,
        )
    ]
    arrows = _streamline_arrows(frame)
    if arrows:
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                id="seasonal-streamline-arrows",
                data=arrows,
                get_polygon="polygon",
                get_fill_color=[255, 255, 255, 245],
                coordinate_system=cartesian,
                pickable=False,
            )
        )
    return layers


def _time_colourbar_layers(layout: AxisLayout, groups: Sequence[str]) -> list[pdk.Layer]:
    """Render a compact, labelled continuous time colourbar in data coordinates."""
    if not groups or layout.colourbar_x is None or layout.colourbar_label_x is None:
        return []
    segment_count = 64
    y_span = layout.data_ymax - layout.data_ymin
    segments = []
    for index in range(segment_count):
        fraction = index / max(segment_count - 1, 1)
        y_start = layout.data_ymin + fraction * y_span
        y_end = layout.data_ymin + min((index + 1) / segment_count, 1.0) * y_span
        segments.append(
            {
                "source": [layout.colourbar_x, y_start],
                "target": [layout.colourbar_x, y_end],
                "colour": _time_colour(fraction),
            }
        )
    labels = [
        {"position": [layout.colourbar_label_x, layout.data_ymin], "text": groups[0]},
        {"position": [layout.colourbar_label_x, layout.data_ymax], "text": groups[-1]},
        {
            "position": [
                layout.colourbar_label_x + 0.065 * (layout.data_xmax - layout.data_xmin),
                (layout.data_ymin + layout.data_ymax) / 2,
            ],
            "text": "Event onset time",
            "angle": 90,
        },
    ]
    cartesian = pdk.types.String("cartesian")
    return [
        pdk.Layer(
            "LineLayer",
            id="seasonal-time-colourbar",
            data=segments,
            get_source_position="source",
            get_target_position="target",
            get_color="colour",
            get_width=16,
            width_units="pixels",
            coordinate_system=cartesian,
            pickable=False,
        ),
        pdk.Layer(
            "TextLayer",
            id="seasonal-time-colourbar-labels",
            data=labels,
            get_position="position",
            get_text="text",
            get_angle="angle",
            get_size=16,
            get_color=AXIS_COLOUR,
            size_units="pixels",
            coordinate_system=cartesian,
            pickable=False,
        ),
    ]


def _axis_layers(
    layout: AxisLayout,
    x_label: str,
    y_label: str,
    *,
    x_tick_labels: Mapping[int, str] | None = None,
) -> list[pdk.Layer]:
    """Build non-pickable deck.gl layers for axes, tick marks and labels."""
    axis_lines = [
        {
            "source": [layout.y_axis_x, layout.x_axis_y],
            "target": [layout.data_xmax, layout.x_axis_y],
        },
        {
            "source": [layout.y_axis_x, layout.x_axis_y],
            "target": [layout.y_axis_x, layout.data_ymax],
        },
    ]
    x_tick_size = 0.022 * (layout.data_ymax - layout.data_ymin)
    y_tick_size = 0.018 * (layout.data_xmax - layout.data_xmin)
    tick_lines = [
        {"source": [tick, layout.x_axis_y], "target": [tick, layout.x_axis_y + x_tick_size]}
        for tick in layout.x_ticks
    ] + [
        {"source": [layout.y_axis_x, tick], "target": [layout.y_axis_x + y_tick_size, tick]}
        for tick in layout.y_ticks
    ]
    x_text = [
        {
            "position": [tick, layout.x_axis_y - 0.055 * (layout.data_ymax - layout.data_ymin)],
            "text": x_tick_labels.get(round(tick), _tick_label(tick))
            if x_tick_labels
            else _tick_label(tick),
        }
        for tick in layout.x_ticks
    ] + [
        {
            "position": [(layout.data_xmin + layout.data_xmax) / 2, layout.x_label_y],
            "text": x_label,
        }
    ]
    y_tick_text = [
        {
            "position": [layout.y_axis_x - 0.035 * (layout.data_xmax - layout.data_xmin), tick],
            "text": _tick_label(tick),
        }
        for tick in layout.y_ticks
    ]
    y_axis_text = [
        {
            "position": [layout.y_label_x, (layout.data_ymin + layout.data_ymax) / 2],
            "text": y_label,
            "angle": 90,
        }
    ]
    cartesian = pdk.types.String("cartesian")
    return [
        pdk.Layer(
            "LineLayer",
            id="plot-axes",
            data=axis_lines,
            get_source_position="source",
            get_target_position="target",
            get_color=AXIS_COLOUR,
            get_width=2,
            width_units="pixels",
            coordinate_system=cartesian,
            pickable=False,
        ),
        pdk.Layer(
            "LineLayer",
            id="plot-ticks",
            data=tick_lines,
            get_source_position="source",
            get_target_position="target",
            get_color=TICK_COLOUR,
            get_width=1,
            width_units="pixels",
            coordinate_system=cartesian,
            pickable=False,
        ),
        pdk.Layer(
            "TextLayer",
            id="plot-x-text",
            data=x_text,
            get_position="position",
            get_text="text",
            get_size=18,
            get_color=AXIS_COLOUR,
            size_units="pixels",
            coordinate_system=cartesian,
            pickable=False,
        ),
        pdk.Layer(
            "TextLayer",
            id="plot-y-ticks",
            data=y_tick_text,
            get_position="position",
            get_text="text",
            get_size=16,
            get_color=AXIS_COLOUR,
            size_units="pixels",
            coordinate_system=cartesian,
            pickable=False,
        ),
        pdk.Layer(
            "TextLayer",
            id="plot-y-label",
            data=y_axis_text,
            get_position="position",
            get_text="text",
            get_angle="angle",
            get_size=20,
            get_color=AXIS_COLOUR,
            size_units="pixels",
            coordinate_system=cartesian,
            pickable=False,
        ),
    ]


def _selection_rows(event: Any) -> list[dict[str, Any]]:
    """Extract selected density cells from Streamlit's PyDeck event object."""
    try:
        layers = event.selection.objects
    except (AttributeError, KeyError):
        return []
    rows: list[dict[str, Any]] = []
    for layer_id, objects in layers.items():
        if str(layer_id).startswith(
            ("run-", "comparison-points-", "temporal-points", "temporal-run-")
        ):
            rows.extend(dict(item) for item in objects)
    return rows


def _render_png(
    frame: pd.DataFrame,
    run_ids: tuple[str, ...],
    run_labels: Mapping[str, str],
    colour_by_dataset: bool,
    x_label: str,
    y_label: str,
    seasonal_trajectory: bool,
) -> bytes:
    """Render the exact currently transferred density cells as a portable PNG."""
    from matplotlib import colors, patches
    import matplotlib.pyplot as plt

    background = "#0a121f"
    figure, axis = plt.subplots(figsize=(12.5, 8.3), constrained_layout=True, facecolor=background)
    axis.set_facecolor(background)
    if seasonal_trajectory:
        ordered = frame.sort_values("time_start")
        streamline = np.asarray(_streamline_path(ordered))
        axis.plot(
            streamline[:, 0], streamline[:, 1], color="white", linewidth=1.8, alpha=0.9, zorder=1
        )
        for start, end in zip(
            ordered.loc[:, ["x", "y"]].to_numpy()[:-1],
            ordered.loc[:, ["x", "y"]].to_numpy()[1:],
            strict=True,
        ):
            delta = end - start
            axis.add_patch(
                patches.FancyArrowPatch(
                    start + 0.54 * delta,
                    start + 0.80 * delta,
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.5,
                    color="white",
                    zorder=2,
                )
            )
        for row in ordered.itertuples():
            colour = np.asarray(row.colour[:3]) / 255.0
            axis.scatter(
                row.x,
                row.y,
                s=float(row.radius**2),
                color=colour,
                edgecolors="white",
                linewidths=0.7,
                zorder=2,
            )
            axis.annotate(
                row.time_group,
                (row.x, row.y),
                xytext=(5, 5),
                textcoords="offset points",
                color="white",
                fontsize=8,
            )
        groups = _trajectory_groups(ordered)
        colourmap = colors.ListedColormap(TIME_COLOUR_STOPS[:, :3] / 255.0)
        normalizer = colors.Normalize(vmin=0, vmax=max(len(groups) - 1, 1))
        colourbar = figure.colorbar(
            plt.cm.ScalarMappable(norm=normalizer, cmap=colourmap), ax=axis, pad=0.03
        )
        colourbar.set_ticks([0, max(len(groups) - 1, 1)], labels=[groups[0], groups[-1]])
        colourbar.set_label("Event onset time", color="white", fontsize=13)
        colourbar.ax.tick_params(colors="white", labelsize=11)
        colourbar.outline.set_edgecolor("white")
    else:
        for index, run_id in enumerate(run_ids):
            subset = frame.loc[frame.run_id == run_id]
            if subset.empty:
                continue
            colour = PALETTE[index % len(PALETTE)] if colour_by_dataset else PALETTE[0]
            axis.scatter(
                subset.x,
                subset.y,
                s=np.square(subset.radius.to_numpy(dtype=float)),
                color=colour[:3] / 255.0,
                alpha=float(colour[3]) / 255.0,
                edgecolors="none",
                rasterized=True,
                label=run_labels.get(run_id, run_id),
            )
    axis.set(
        title=(
            "CUDAHeat temporal trajectory"
            if seasonal_trajectory
            else "CUDAHeat event density scatter"
        ),
        xlabel=x_label,
        ylabel=y_label,
    )
    axis.tick_params(colors="white", labelsize=12)
    axis.xaxis.label.set_color("white")
    axis.yaxis.label.set_color("white")
    axis.title.set_color("white")
    axis.xaxis.label.set_size(15)
    axis.yaxis.label.set_size(15)
    axis.title.set_size(18)
    for spine in axis.spines.values():
        spine.set_color("white")
    axis.grid(True, color="#64748b", linewidth=0.75, alpha=0.55)
    axis.set_axisbelow(True)
    if not seasonal_trajectory and len(run_ids) > 1 and colour_by_dataset:
        legend = axis.legend(fontsize=8, frameon=False, title="Detection run")
        plt.setp(legend.get_texts(), color="white")
        legend.get_title().set_color("white")
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=220)
    plt.close(figure)
    return buffer.getvalue()


def _render_temporal_png(
    frame: pd.DataFrame,
    run_ids: tuple[str, ...],
    run_labels: Mapping[str, str],
    x_label: str,
    y_label: str,
    colour_label: str,
    colour_by_time: bool,
    ordered_groups: Sequence[str],
    show_contours: bool,
    show_legend: bool,
) -> bytes:
    """Export the current temporal marks with the same run shapes and colours."""
    from matplotlib import colors, lines
    import matplotlib.pyplot as plt

    background = "#0a121f"
    figure, axis = plt.subplots(figsize=(14, 9), constrained_layout=True, facecolor=background)
    axis.set_facecolor(background)
    outline_colours = _run_outline_colours(run_ids)
    markers = ("o", "s", "^", "D")
    for index, run_id in enumerate(run_ids):
        subset = frame.loc[frame.run_id == run_id]
        if subset.empty:
            continue
        fill = np.asarray(subset.colour.to_list(), dtype=float)[:, :3] / 255.0
        edge = np.asarray(outline_colours[run_id][:3], dtype=float) / 255.0
        axis.scatter(
            subset.x,
            subset.y,
            s=np.square(subset.radius.to_numpy(dtype=float) * 2.2),
            marker=markers[index % len(markers)],
            c=fill,
            edgecolors=edge,
            linewidths=1.0,
            label=run_labels.get(run_id, run_id),
            zorder=3,
        )
    if show_contours:
        xmin, xmax = axis.get_xlim()
        ymin, ymax = axis.get_ylim()
        origin_x, origin_y = (xmin + xmax) / 2, ymin
        x_radius, y_radius = (xmax - xmin) / 2, ymax - ymin
        angles = np.linspace(0.0, math.pi, 160)
        for radius in (0.28, 0.46, 0.64, 0.82):
            axis.plot(
                origin_x + radius * x_radius * np.cos(angles),
                origin_y + radius * y_radius * np.sin(angles),
                color="#cbd5e1",
                alpha=0.55,
                linewidth=0.8,
                linestyle=(0, (3, 4)),
                zorder=1,
            )
    colourmap = colors.ListedColormap(TIME_COLOUR_STOPS[:, :3] / 255.0)
    if colour_by_time:
        normalizer = colors.Normalize(vmin=0, vmax=max(len(ordered_groups) - 1, 1))
        ticks = [0, max(len(ordered_groups) - 1, 1)]
        ticklabels = [ordered_groups[0], ordered_groups[-1]]
    else:
        values = frame.severity_value.to_numpy(dtype=float)
        low, high = np.quantile(values[np.isfinite(values)], (0.02, 0.98))
        normalizer = colors.Normalize(vmin=low, vmax=high)
        ticks = [low, high]
        ticklabels = [f"{low:.4g}", f"{high:.4g}"]
    colourbar = figure.colorbar(plt.cm.ScalarMappable(norm=normalizer, cmap=colourmap), ax=axis)
    colourbar.set_ticks(ticks, labels=ticklabels)
    colourbar.set_label(colour_label, color="white")
    colourbar.ax.tick_params(colors="white")
    colourbar.outline.set_edgecolor("white")
    if show_legend:
        legend_handles = [
            lines.Line2D(
                [],
                [],
                marker=markers[index % len(markers)],
                linestyle="",
                markerfacecolor="#f8fafc",
                markeredgecolor=np.asarray(outline_colours[run_id][:3], dtype=float) / 255.0,
                markersize=8,
                label=run_labels.get(run_id, run_id),
            )
            for index, run_id in enumerate(run_ids)
        ]
        legend = axis.legend(
            handles=legend_handles, title="Detection run", frameon=False, fontsize=8
        )
        plt.setp(legend.get_texts(), color="white")
        legend.get_title().set_color("white")
    axis.set(title="CUDAHeat temporal analysis", xlabel=x_label, ylabel=y_label)
    axis.tick_params(colors="white")
    axis.xaxis.label.set_color("white")
    axis.yaxis.label.set_color("white")
    axis.title.set_color("white")
    for spine in axis.spines.values():
        spine.set_color("white")
    axis.grid(True, color="#64748b", linewidth=0.6, alpha=0.4)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=220, facecolor=background)
    plt.close(figure)
    return buffer.getvalue()


def _time_window_label(window: str) -> str:
    return next(label for label, value in TIME_WINDOWS.items() if value == window)


def _descriptor_panel(
    db: Engine,
    selected_runs: tuple[str, ...],
    run_labels: Mapping[str, str],
    x_variable: str,
    y_variable: str,
    marker_variable: str | None,
    bins: int,
    normalise: bool,
    log_x: bool,
    log_y: bool,
    colour_by_dataset: bool,
    grouped_over_time: bool,
    time_window: str,
) -> None:
    with st.spinner("Aggregating selected events in PostgreSQL…"):
        frame = clustered_points(
            db,
            selected_runs,
            x_variable,
            y_variable,
            marker_variable,
            bins,
            normalise,
            log_x,
            log_y,
            grouped_over_time,
            time_window,
        )
    selected_event_count = int(frame.event_count.sum()) if not frame.empty else 0
    st.sidebar.metric("Plotted events", f"{selected_event_count:,}")
    st.sidebar.caption(
        f"{len(frame):,} time groups"
        if grouped_over_time
        else f"{len(frame):,} WebGL density cells"
    )
    if frame.empty:
        st.info("No paired finite values are available for this selection.")
        return

    frame["radius"] = _marker_radius(frame, marker_variable)
    trajectory_groups: list[str] = []
    if grouped_over_time:
        trajectory_groups = _trajectory_groups(frame)
        trajectory_colours = _trajectory_colours(trajectory_groups)
        frame["colour"] = [trajectory_colours[str(group)] for group in frame.time_group]
    x_label = _display_name(x_variable, normalise=normalise, log_scale=log_x)
    y_label = _display_name(y_variable, normalise=normalise, log_scale=log_y)
    marker_label = (
        "Density (event count)"
        if marker_variable is None
        else _display_name(marker_variable, normalise=False)
    )
    marker_tooltip = (
        "Events in cell: {event_count}"
        if marker_variable is None
        else f"Mean {marker_label}: {{marker_value}}"
    )
    time_tooltip = "Time group: {time_group}<br/>" if grouped_over_time else ""
    layout = _axis_layout(frame, show_time_colourbar=grouped_over_time)
    data_layers = _layers(frame, selected_runs, run_labels, colour_by_dataset, grouped_over_time)
    if grouped_over_time:
        data_layers = [*_streamline_layers(frame), *data_layers]
    deck = pdk.Deck(
        layers=[
            *data_layers,
            *_axis_layers(layout, x_label, y_label),
            *_time_colourbar_layers(layout, trajectory_groups),
        ],
        views=[pdk.View(type="OrthographicView", controller=True, flip_y=False)],
        initial_view_state=_view_state(layout),
        map_provider=None,
        map_style=None,
        parameters={"clearColor": CHART_BACKGROUND},
        tooltip={
            "html": (
                "<b>{run_label}</b><br/>"
                f"{time_tooltip}{x_label}: {{x}}<br/>{y_label}: {{y}}<br/>"
                f"{marker_tooltip}<br/>Mean duration: {{mean_duration_hours}} h"
            )
        },
    )
    event = st.pydeck_chart(
        deck,
        width="stretch",
        height=CHART_HEIGHT,
        on_select="rerun",
        selection_mode="single-object",
        key="cudaheat-density-scatter",
        alt=f"Interactive density scatter of {x_label} against {y_label}",
    )
    selected = _selection_rows(event)
    if selected:
        cell = selected[0]
        st.subheader("Selected time group" if grouped_over_time else "Selected density cell")
        details_row = {
            "run": cell.get("run_label", cell.get("run_id")),
            x_label: cell.get("x"),
            y_label: cell.get("y"),
            "events represented": cell.get("event_count"),
            "mean duration (hours)": cell.get("mean_duration_hours"),
            f"mean {marker_label}": cell.get("event_count")
            if marker_variable is None
            else cell.get("marker_value"),
        }
        if grouped_over_time:
            details_row["time group"] = cell.get("time_group")
        else:
            details_row["x bin"] = cell.get("x_bin")
            details_row["y bin"] = cell.get("y_bin")
        st.dataframe(pd.DataFrame([details_row]), hide_index=True, width="stretch")

    export_signature = (
        selected_runs,
        x_variable,
        y_variable,
        marker_variable,
        bins,
        normalise,
        log_x,
        log_y,
        colour_by_dataset,
        grouped_over_time,
        time_window,
    )
    if st.session_state.get("cudaheat_export_signature") != export_signature:
        st.session_state.pop("cudaheat_export_png", None)
        st.session_state["cudaheat_export_signature"] = export_signature
    if st.button("Generate PNG image", help="Render this exact plot with Python/Matplotlib."):
        with st.spinner("Rendering PNG in Python…"):
            st.session_state["cudaheat_export_png"] = _render_png(
                frame,
                selected_runs,
                run_labels,
                colour_by_dataset,
                x_label,
                y_label,
                grouped_over_time,
            )
    if png := st.session_state.get("cudaheat_export_png"):
        st.download_button(
            "Download current PNG",
            data=png,
            file_name="cudaheat_plot.png",
            mime="image/png",
        )
    st.caption(
        f"X: {x_label}; Y: {y_label}. Marker size: {marker_label}. "
        + (
            f"One point per {_time_window_label(time_window).lower()}; the colourbar and arrowed streamline run forward in time."
            if grouped_over_time
            else "Pan/zoom are deck.gl Cartesian controls."
        )
    )


def _temporal_comparison_panel(
    db: Engine,
    labels: Mapping[str, str],
    variables: Sequence[str],
    selected_runs: tuple[str, ...],
    run_labels: Mapping[str, str],
) -> None:
    st.subheader("Temporal comparison")
    st.caption(
        "Compare two runs across consecutive calendar windows; no descriptor-space trajectories."
    )
    if len(labels) < 2:
        st.info("At least two runs are required for temporal comparison.")
        return

    run_options = list(labels)
    defaults = [label for label in run_options if labels[label] in selected_runs][:2]
    for label in run_options:
        if len(defaults) == 2:
            break
        if label not in defaults:
            defaults.append(label)
    comparison_labels = st.multiselect(
        "Runs to compare",
        run_options,
        default=defaults,
        max_selections=2,
        help="Selected runs remain checked in this dropdown.",
        key="cudaheat-temporal-runs",
    )
    if len(comparison_labels) != 2:
        st.info("Select exactly two runs to compare their temporal evolution.")
        return
    comparison_runs = tuple(labels[label] for label in comparison_labels)
    time_window_label = st.selectbox(
        "Grouping window",
        tuple(TIME_WINDOWS),
        index=1,
        key="cudaheat-temporal-window",
    )
    metric_variable = st.selectbox(
        "Temporal metric",
        ("event_count", *variables),
        format_func=lambda variable: _display_name(variable, normalise=False),
        key="cudaheat-temporal-metric",
    )
    time_window = TIME_WINDOWS[time_window_label]
    with st.spinner("Aggregating run histories in PostgreSQL…"):
        frame = temporal_comparison_points(db, comparison_runs, metric_variable, time_window)
    if frame.empty:
        st.info("No finite temporal values are available for these runs.")
        return

    ordered_times = (
        frame.loc[:, ["time_start", "time_group"]].drop_duplicates().sort_values("time_start")
    )
    time_index = {value: index for index, value in enumerate(ordered_times.time_start)}
    frame = frame.copy()
    frame["x"] = frame.time_start.map(time_index).astype(float)
    frame["y"] = frame.value.astype(float)
    frame["radius"] = np.clip(np.sqrt(frame.event_count.to_numpy(dtype=float)), 4.0, 9.0)
    tick_indices = np.unique(
        np.rint(np.linspace(0, len(ordered_times) - 1, min(7, len(ordered_times)))).astype(int)
    )
    x_tick_labels = {index: str(ordered_times.iloc[index].time_group) for index in tick_indices}
    layout = _axis_layout(frame)
    layout = AxisLayout(
        **{**layout.__dict__, "x_ticks": tuple(float(index) for index in tick_indices)}
    )
    metric_label = _display_name(metric_variable, normalise=False)
    deck = pdk.Deck(
        layers=[
            *_comparison_layers(frame, comparison_runs, run_labels),
            *_axis_layers(
                layout,
                _time_window_label(time_window),
                metric_label,
                x_tick_labels=x_tick_labels,
            ),
        ],
        views=[pdk.View(type="OrthographicView", controller=True, flip_y=False)],
        initial_view_state=_view_state(layout),
        map_provider=None,
        map_style=None,
        parameters={"clearColor": CHART_BACKGROUND},
        tooltip={
            "html": (
                "<b>{run_label}</b><br/>Time: {time_group}<br/>"
                f"{metric_label}: {{value}}<br/>Events represented: {{event_count}}"
            )
        },
    )
    event = st.pydeck_chart(
        deck,
        width="stretch",
        height=CHART_HEIGHT,
        on_select="rerun",
        selection_mode="single-object",
        key="cudaheat-temporal-comparison",
        alt=f"Temporal comparison of {metric_label}",
    )
    legend_columns = st.columns(2)
    for index, run_id in enumerate(comparison_runs):
        red, green, blue, _ = COMPARISON_PALETTE[index]
        legend_columns[index].markdown(
            f"<span style='color: rgb({red}, {green}, {blue})'>●</span> {run_labels.get(run_id, run_id)}",
            unsafe_allow_html=True,
        )
    selected = _selection_rows(event)
    if selected:
        point = selected[0]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "run": point.get("run_label", point.get("run_id")),
                        "time group": point.get("time_group"),
                        metric_label: point.get("value"),
                        "events represented": point.get("event_count"),
                    }
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def _temporal_analysis_panel(
    db: Engine,
    labels: Mapping[str, str],
    variables: Sequence[str],
    run_labels: Mapping[str, str],
) -> None:
    """Interactive descriptor scatter of progressive time-group centroids."""
    st.subheader("Temporal analysis")
    st.caption(
        "Controls rerun automatically. Each point summarizes one run in one calendar time group."
    )
    selected_labels = st.multiselect(
        "Runs",
        list(labels),
        default=list(labels)[:1],
        max_selections=4,
        help="Selected runs remain visibly checked in this dropdown.",
        key="temporal-analysis-runs",
    )
    if len(selected_labels) not in (1, 3, 4):
        st.info("Select one run, or exactly three or four runs for comparison.")
        return
    run_ids = tuple(labels[label] for label in selected_labels)
    variable_columns = st.columns(4)
    x_variable = variable_columns[0].selectbox("X field", variables, key="temporal-analysis-x")
    y_variable = variable_columns[1].selectbox(
        "Y field", variables, index=min(1, len(variables) - 1), key="temporal-analysis-y"
    )
    size_choice = variable_columns[2].selectbox(
        "Size field", ("Event count", *variables), key="temporal-analysis-size"
    )
    colour_choice = variable_columns[3].selectbox(
        "Colour / severity field",
        ("Time group (chronological)", "Event count", *variables),
        key="temporal-analysis-colour",
    )
    options, appearance = st.columns(2)
    with options:
        time_label = st.selectbox(
            "Time grouping", tuple(TEMPORAL_WINDOWS), key="temporal-analysis-window"
        )
        log_x = st.checkbox("Logarithmic X", key="temporal-analysis-log-x")
        log_y = st.checkbox("Logarithmic Y", key="temporal-analysis-log-y")
        show_contours = st.checkbox(
            "Add thin qualitative contour arcs",
            help="Dashed semi-circular guide arcs centred on the horizontal-axis origin; they do not encode a quantitative value.",
            key="temporal-analysis-contours",
        )
    with appearance:
        marker_scale = st.slider(
            "Marker size scale", 0.3, 3.0, 1.0, 0.1, key="temporal-analysis-marker-scale"
        )
        opacity = st.slider("Marker opacity", 30, 255, 180, 5, key="temporal-analysis-opacity")
        show_legend = st.checkbox("Show legend", value=True, key="temporal-analysis-legend")

    size_variable = None if size_choice == "Event count" else size_choice
    colour_by_time = colour_choice == "Time group (chronological)"
    colour_variable = None if colour_by_time or colour_choice == "Event count" else colour_choice
    with st.spinner("Updating temporal descriptor groups…"):
        frame = temporal_scatter_points(
            db,
            run_ids,
            x_variable,
            y_variable,
            size_variable,
            colour_variable,
            TEMPORAL_WINDOWS[time_label],
            log_x,
            log_y,
        )
    if frame.empty:
        st.info("No finite events match the selected fields and log-scale constraints.")
        return
    frame = frame.copy()
    time_groups = (
        frame.loc[:, ["time_start", "time_group"]].drop_duplicates().sort_values("time_start")
    )
    ordered_groups = time_groups.time_group.astype(str).tolist()
    positions = {group: index for index, group in enumerate(ordered_groups)}
    visible_range = (ordered_groups[0], ordered_groups[-1])
    if len(ordered_groups) > 1:
        selected_range = st.select_slider(
            "Visible time range",
            options=ordered_groups,
            value=(ordered_groups[0], ordered_groups[-1]),
            key=f"temporal-analysis-range-{TEMPORAL_WINDOWS[time_label]}",
            help="Restrict the scatter to a contiguous range of chronological time groups.",
        )
        first, last = (positions[selected_range[0]], positions[selected_range[1]])
        frame = frame.loc[frame.time_group.astype(str).map(positions).between(first, last)].copy()
        visible_range = selected_range
    frame["radius"] = np.clip(_marker_radius(frame, size_variable) * marker_scale, 1.0, 18.0)
    if colour_by_time:
        denominator = max(len(ordered_groups) - 1, 1)
        frame["severity_value"] = frame.time_group.astype(str).map(positions).astype(float)
        frame["colour"] = [
            _time_colour(value / denominator)[:3] + [opacity]
            for value in frame.severity_value.to_numpy(dtype=float)
        ]
    else:
        frame["severity_value"] = frame.colour_value.astype(float)
        frame["colour"] = _value_colours(frame.severity_value.to_numpy(dtype=float), opacity)
    outline_colours = _run_outline_colours(run_ids)
    frame["line_colour"] = [outline_colours[run_id] for run_id in frame.run_id]
    frame["run_label"] = [run_labels.get(run_id, run_id) for run_id in frame.run_id]
    marker_symbols = {
        run_id: TEMPORAL_MARKERS[index % len(TEMPORAL_MARKERS)]
        for index, run_id in enumerate(run_ids)
    }
    marker_icons = {
        run_id: ("circle", "square", "triangle", "diamond")[index % 4]
        for index, run_id in enumerate(run_ids)
    }
    frame["marker_symbol"] = [marker_symbols[run_id] for run_id in frame.run_id]
    frame["marker_icon"] = [marker_icons[run_id] for run_id in frame.run_id]
    frame["marker_size"] = np.clip(frame.radius.to_numpy(dtype=float) * 2.2, 7.0, 38.0)
    x_label = _display_name(x_variable, normalise=False, log_scale=log_x)
    y_label = _display_name(y_variable, normalise=False, log_scale=log_y)
    size_label = (
        "Event count" if size_variable is None else _display_name(size_variable, normalise=False)
    )
    colour_label = (
        "Chronological time group"
        if colour_by_time
        else (
            "Event count"
            if colour_variable is None
            else _display_name(colour_variable, normalise=False)
        )
    )
    layout = _axis_layout(frame)
    temporal_layers: list[pdk.Layer] = []
    for run_id in run_ids:
        subset = frame.loc[frame.run_id == run_id].copy()
        temporal_layers.append(
            pdk.Layer(
                "IconLayer",
                id=f"temporal-run-{run_id}",
                data=subset,
                icon_atlas=MARKER_ICON_ATLAS,
                icon_mapping=MARKER_ICON_MAPPING,
                get_icon="marker_icon",
                get_position="[x, y]",
                get_size="marker_size",
                get_color="colour",
                size_units="pixels",
                size_min_pixels=7,
                size_max_pixels=38,
                coordinate_system=pdk.types.String("cartesian"),
                pickable=True,
                auto_highlight=True,
            )
        )
    colour_tooltip = (
        "Chronological time colour: {time_group}<br/>"
        if colour_by_time
        else f"{colour_label}: {{colour_value}}<br/>"
    )
    deck = pdk.Deck(
        layers=[
            *(_qualitative_contour_layers(layout) if show_contours else []),
            *temporal_layers,
            *_axis_layers(layout, x_label, y_label),
        ],
        views=[pdk.View(type="OrthographicView", controller=True, flip_y=False)],
        initial_view_state=_view_state(layout),
        map_provider=None,
        map_style=None,
        parameters={"clearColor": CHART_BACKGROUND},
        tooltip={
            "html": (
                "<b>{run_label}</b><br/>Time: {time_group}<br/>"
                f"{x_label}: {{x}}<br/>{y_label}: {{y}}<br/>"
                f"{size_label}: {{size_value}}<br/>{colour_tooltip}"
                "Events represented: {event_count}"
            )
        },
    )
    event = st.pydeck_chart(
        deck,
        height=CHART_HEIGHT,
        width="stretch",
        on_select="rerun",
        selection_mode="single-object",
        key="temporal-analysis-deck",
        alt=f"Temporal scatter of {x_label} and {y_label}",
    )
    if show_legend:
        run_items = " ".join(
            (
                f"<span style='color: rgb({colour[0]}, {colour[1]}, {colour[2]}); font-size: 1.1rem'>"
                f"{marker_symbols[run_id]}</span> "
                f"{run_labels.get(run_id, run_id)}"
            )
            for run_id, colour in outline_colours.items()
        )
        if colour_by_time:
            colour_legend = (
                f"<div style='margin-top: .35rem'><b>Chronological time:</b> {ordered_groups[0]}"
                f"<span style='float:right'>{ordered_groups[-1]}</span></div>"
            )
        else:
            finite = frame.severity_value.to_numpy(dtype=float)
            low, high = np.quantile(finite[np.isfinite(finite)], (0.02, 0.98))
            colour_legend = (
                f"<div style='margin-top: .35rem'><b>{colour_label}:</b> {low:.4g}"
                f"<span style='float:right'>{high:.4g}</span></div>"
            )
        st.markdown(
            f"""
            <div style="border: 1px solid #475569; padding: .55rem .75rem; margin-top: .4rem">
              <div><b>Run outlines:</b> {run_items}</div>
              {colour_legend}
              <div style="height: 12px; border-radius: 3px; background: linear-gradient(90deg,
                rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37));"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    selected = _selection_rows(event)
    if selected:
        point = selected[0]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "run": point.get("run_label", point.get("run_id")),
                        "time group": point.get("time_group"),
                        x_label: point.get("x"),
                        y_label: point.get("y"),
                        size_label: point.get("size_value"),
                        colour_label: point.get("time_group")
                        if colour_by_time
                        else point.get("colour_value"),
                        "events represented": point.get("event_count"),
                    }
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    export_signature = (
        run_ids,
        x_variable,
        y_variable,
        size_choice,
        colour_choice,
        TEMPORAL_WINDOWS[time_label],
        log_x,
        log_y,
        marker_scale,
        opacity,
        visible_range,
        show_contours,
        show_legend,
    )
    if st.session_state.get("temporal_png_signature") != export_signature:
        st.session_state.pop("temporal_png", None)
        st.session_state["temporal_png_signature"] = export_signature
    if st.button("Generate PNG image", help="Render the current temporal view with Python."):
        with st.spinner("Rendering temporal PNG…"):
            st.session_state["temporal_png"] = _render_temporal_png(
                frame,
                run_ids,
                run_labels,
                x_label,
                y_label,
                colour_label,
                colour_by_time,
                ordered_groups,
                show_contours,
                show_legend,
            )
    if png := st.session_state.get("temporal_png"):
        st.download_button(
            "Download temporal PNG",
            data=png,
            file_name="cudaheat_temporal_analysis.png",
            mime="image/png",
        )
    st.caption(
        f"{len(frame):,} time-group points; {int(frame.event_count.sum()):,} events represented. "
        + ("Dashed arcs are qualitative guides only." if show_contours else "")
    )


def main() -> None:
    st.set_page_config(page_title="CUDAHeat GPU Explorer", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stDeckGlJsonChart"], [data-testid="stDeckGlJsonChart"] canvas { background: #0a121f; }
        [data-testid="stDeckGlJsonChart"] { border: 1px solid #475569; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("CUDAHeat GPU Explorer")
    db = engine()
    datasets, variables = catalogue(db)
    labels = {_dataset_label(row): row.run_id for _, row in datasets.iterrows()}
    run_labels = {run_id: label for label, run_id in labels.items()}

    with st.sidebar:
        st.header("Descriptor-space selection")
        selected_labels = st.multiselect("Datasets", list(labels), default=list(labels)[:1])
        x_variable = st.selectbox("X variable", variables, index=0)
        y_variable = st.selectbox("Y variable", variables, index=min(1, len(variables) - 1))
        log_x = st.checkbox("Log scale X")
        log_y = st.checkbox("Log scale Y")
        marker_choice = st.selectbox("Marker size", ("Density (event count)", *variables))
        normalise = st.checkbox("Normalise axes")
        colour_by_dataset = st.checkbox("Colour by dataset", value=True)
        bins = st.slider("Density resolution", 32, 192, DEFAULT_GRID_BINS, 16)
        if log_x or log_y:
            st.caption("Log axes retain only strictly positive values.")

    selected_runs = tuple(labels[label] for label in selected_labels)
    grouped_over_time = False
    time_window = "season"
    if len(selected_runs) == 1:
        with st.sidebar:
            st.divider()
            grouped_over_time = st.checkbox(
                "Group over time",
                help="Create chronological month, season, or half-year centroids and connect them.",
            )
            if grouped_over_time:
                grouping_label = st.selectbox(
                    "Grouping window",
                    tuple(TIME_WINDOWS),
                    index=1,
                    key="cudaheat-trajectory-window",
                )
                time_window = TIME_WINDOWS[grouping_label]
                st.caption("Density resolution is ignored in grouped mode.")
    marker_variable = None if marker_choice == "Density (event count)" else marker_choice

    explorer_tab, temporal_tab = st.tabs(["Explorer", "Temporal analysis"])
    with explorer_tab:
        if not selected_runs:
            st.info("Select at least one dataset.")
        else:
            _descriptor_panel(
                db,
                selected_runs,
                run_labels,
                x_variable,
                y_variable,
                marker_variable,
                bins,
                normalise,
                log_x,
                log_y,
                colour_by_dataset,
                grouped_over_time,
                time_window,
            )
    with temporal_tab:
        _temporal_analysis_panel(db, labels, variables, run_labels)


if __name__ == "__main__":
    main()
