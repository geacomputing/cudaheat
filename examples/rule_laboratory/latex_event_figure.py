"""Export one persisted CUDAHeat event as a two-column TikZ figure.

The exporter is read-only with respect to PostgreSQL.  It writes a CSV data
file and a LaTeX fragment containing a ``figure*`` environment: the left panel
plots SST, climatology and threshold; the right panel records the detection
configuration and values of all discovered enrichment rules.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import xarray as xr

from cudaheat import discover_rules
from cudaheat.climatology import hourly_climatology_index
from cudaheat.enrichment import _enrich_catalogue
from rule_lab import (
    DEFAULT_POSTGRES_CONFIG,
    DEFAULT_RULE_DIRECTORY,
    PROJECT_ROOT,
    _format_baseline,
    _format_hours,
    _format_percentile,
    _format_value,
    _metadata_path,
    _metadata_text,
    _point_columns,
    load_stored_catalogue,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "LaTex" / "tikz" / "test"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, type=UUID)
    parser.add_argument("--point-index", required=True, type=int)
    parser.add_argument(
        "--event-id",
        type=int,
        help="optional CUDAHeat event index; otherwise the latest event at the point is used",
    )
    parser.add_argument("--postgres-config", type=Path, default=DEFAULT_POSTGRES_CONFIG)
    parser.add_argument("--rule-directory", type=Path, default=DEFAULT_RULE_DIRECTORY)
    parser.add_argument("--padding-hours", type=int, default=12)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--label-prefix", default="fig:event")
    arguments = parser.parse_args()
    if arguments.padding_hours < 0:
        parser.error("--padding-hours must be non-negative")
    if not arguments.rule_directory.is_dir():
        parser.error(f"rule directory does not exist: {arguments.rule_directory}")
    return arguments


def _tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _date_text(value: np.datetime64) -> str:
    return np.datetime_as_string(value.astype("datetime64[s]"), unit="s").replace("T", " ")


def _to_celsius(values: np.ndarray, units: object) -> np.ndarray:
    """Convert Kelvin fields while preserving sources already expressed in Celsius."""
    normalized_units = str(units or "").strip().lower().replace("°", "")
    finite = values[np.isfinite(values)]
    is_kelvin = normalized_units in {"k", "kelvin", "degk", "degree_kelvin"}
    if not normalized_units and finite.size:
        is_kelvin = float(np.nanmedian(finite)) > 200.0
    return values - 273.15 if is_kelvin else values


def _tick_options(times: np.ndarray) -> str:
    """Use midday minor ticks for short events and readable coarser ticks otherwise."""
    span_hours = max(1.0, (times[-1] - times[0]) / np.timedelta64(1, "h"))
    if span_hours <= 72:
        return "xtick distance=0.5, minor x tick num=1,"
    if span_hours <= 14 * 24:
        return "xtick distance=1, minor x tick num=1,"
    if span_hours <= 60 * 24:
        return "xtick distance=3, minor x tick num=1,"
    return "xtick distance=14, minor x tick num=1,"


def _month_span_caption(start: str, end: str) -> str:
    """Return a concise month-and-year note for every exported event."""
    start_month = datetime.fromisoformat(start).strftime("%B %Y")
    end_month = datetime.fromisoformat(end).strftime("%B %Y")
    if start_month == end_month:
        return f" Event period: {start_month}."
    return f" Event period: {start_month} to {end_month}."


def _select_event(catalogue: xr.Dataset, event_id: int | None) -> tuple[xr.Dataset, int]:
    event_values = np.asarray(catalogue["event"].values, dtype=np.int64)
    if event_id is not None:
        positions = np.flatnonzero(event_values == event_id)
        if not positions.size:
            raise LookupError(f"event {event_id} is not present at the requested point")
        position = int(positions[0])
    else:
        starts = np.asarray(catalogue["start_index"].values, dtype=np.int64)
        position = int(np.argmax(starts))
    return catalogue.isel(event=[position]), int(event_values[position])


def _event_series(
    catalogue: xr.Dataset,
    metadata: Mapping[str, Any],
    *,
    padding_hours: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return timestamp-aligned SST, threshold and climatology for one event."""
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ImportError("event figure export requires zarr") from exc

    source_path = _metadata_path(metadata, "source_path")
    cache_path = _metadata_path(metadata, "climatology_cache_path")
    variable = _metadata_text(metadata, "variable")
    time_dimension = str(metadata.get("time_dimension", "time"))
    calendar_mode = str(metadata.get("calendar_mode", "month_day_hour"))
    point_index = int(catalogue["point_index"].values[0])
    start = int(catalogue["start_index"].values[0])
    stop = int(catalogue["stop_index"].values[0])

    source = xr.open_dataset(source_path, engine="h5netcdf", cache=False)
    try:
        if variable not in source or "point" not in source or time_dimension not in source:
            raise ValueError("optimized source is missing the required event-plot fields")
        timestamps = np.asarray(source[time_dimension].values, dtype="datetime64[ns]")
        source_column = int(_point_columns(
            np.asarray(source["point"].values, dtype=np.int64),
            np.asarray([point_index], dtype=np.int64),
            "optimized source",
        )[0])
        left, right = max(0, start - padding_hours), min(timestamps.size, stop + padding_hours)
        source_units = source[variable].attrs.get("units")
        sst = np.asarray(source[variable].isel({time_dimension: slice(left, right), "point": source_column}).values, dtype=float)
        bins = hourly_climatology_index(
            xr.DataArray(timestamps[left:right], dims=(time_dimension,)), calendar_mode=calendar_mode
        )
    finally:
        source.close()

    cache = zarr.open_group(str(cache_path), mode="r")
    cache_column = int(_point_columns(
        np.asarray(cache["point"][:], dtype=np.int64), np.asarray([point_index], dtype=np.int64), "climatology cache"
    )[0])
    valid = (bins >= 0) & (bins < cache["threshold"].shape[0])
    threshold = np.full(right - left, np.nan)
    climatology = np.full(right - left, np.nan)
    threshold[valid] = np.asarray(cache["threshold"][:, cache_column])[bins[valid]]
    climatology[valid] = np.asarray(cache["climatology"][:, cache_column])[bins[valid]]
    return timestamps[left:right], _to_celsius(sst, source_units), _to_celsius(threshold, source_units), _to_celsius(climatology, source_units), start - left, stop - left


def _write_data(path: Path, times: np.ndarray, sst: np.ndarray, threshold: np.ndarray, climatology: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "sst", "threshold", "climatology"))
        for timestamp, *values in zip(times, sst, threshold, climatology, strict=True):
            writer.writerow([_date_text(timestamp), *["" if not np.isfinite(value) else f"{value:.8g}" for value in values]])


def _panel_rows(
    metadata: Mapping[str, Any], catalogue: xr.Dataset, rules: Sequence[Any], enriched: xr.Dataset
) -> tuple[list[tuple[str, str]], list[tuple[str, Any]], list[tuple[str, str]]]:
    """Separate normalized and dimensional rule outputs for the side panel."""
    settings = [
        #("Event", str(int(catalogue["event"].values[0]))),
        #("Point", str(int(catalogue["point_index"].values[0]))),
        ("Duration", _format_hours(catalogue["duration_hours"].values[0])),
        ("Baseline", _format_baseline(metadata.get("baseline"))),
        ("Percentile", _format_percentile(metadata.get("percentile"))),
        #("Min. duration", _format_hours(metadata.get("min_duration_hours"))),
        ("Internal gap", _format_hours(metadata.get("max_internal_gap_hours"))),
        ("Join gap", _format_hours(metadata.get("join_gap_hours"))),
    ]
    normalized: list[tuple[str, Any]] = []
    dimensional: list[tuple[str, str]] = []
    for rule in rules:
        if rule.output_name in enriched:
            symbol = str(rule.metadata.get("symbol") or "").strip()
            label = rf"${symbol}$" if symbol else _tex_escape(_truncate_rule_id(rule.id))
            output = rule.metadata.get("output", {})
            rule_range = output.get("range", []) if isinstance(output, Mapping) else []
            unit = str(output.get("unit", "")).strip().lower() if isinstance(output, Mapping) else ""
            value = enriched[rule.output_name].values[0]
            if unit in {"dimensionless", "1", "-"} and list(rule_range) == [0.0, 1.0]:
                normalized.append((label, value))
            else:
                dimensional.append((label, _format_value(value)))
    return settings, normalized, dimensional


def _truncate_rule_id(rule_id: str, limit: int = 20) -> str:
    """Keep fallback rule names within the narrow figure side panel."""
    return rule_id if len(rule_id) <= limit else f"{rule_id[:limit - 1]}…"


def _normalized_bar_plot(rows: Sequence[tuple[str, Any]]) -> str:
    """Create one coordinate-consistent 0--1 PGFPlots bar chart."""
    finite_rows: list[tuple[str, float]] = []
    for label, value in rows:
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fraction):
            finite_rows.append((label, min(1.0, max(0.0, fraction))))
    if not finite_rows:
        return r"\footnotesize\textit{No finite normalized metrics.}"

    positions = ",".join(str(index) for index in range(len(finite_rows)))
    coordinates = " ".join(
        f"({value:.4f},{index})" for index, (_, value) in enumerate(finite_rows)
    )
    label_nodes = "\n".join(
        rf"\node[anchor=east,xshift=-3pt,font=\scriptsize,text=CUDAHeatSlate] at (axis cs:0,{index}) {{{label}}};"
        for index, (label, _) in enumerate(finite_rows)
    )
    height = max(1.65, 0.52 * len(finite_rows) + 0.50)
    return rf"""
\begin{{tikzpicture}}
\begin{{axis}}[
  xbar, width=\linewidth, height={height:.2f}cm,
  xmin=0, xmax=1, xtick={{0,0.5,1}}, minor x tick num=1,
  ytick={{{positions}}}, yticklabels={{}},
  y dir=reverse, enlarge y limits=0.30,
  axis x line*=bottom, axis y line=none,
  ytick style={{draw=none}}, clip=false,
  tick label style={{font=\scriptsize}},
  xmajorgrids, grid style={{draw=CUDAHeatGrid}},
  bar width=3.5pt,
]
\addplot[draw=none,fill=gray!60] coordinates {{{coordinates}}};
{label_nodes}
\end{{axis}}
\end{{tikzpicture}}""".strip()


def _write_tex(
    path: Path,
    data_path: Path,
    run_id: UUID,
    event_id: int,
    settings: Sequence[tuple[str, str]],
    normalized_rules: Sequence[tuple[str, Any]],
    dimensional_rules: Sequence[tuple[str, str]],
    ymin: float,
    ymax: float,
    event_start: str,
    event_end: str,
    label_prefix: str,
    tick_options: str,
    month_span_caption: str,
) -> None:
    latex_data_path = data_path.relative_to(PROJECT_ROOT / "LaTex").as_posix()
    parameter_rows = "\n".join(
        rf"\textcolor{{CUDAHeatSlate}}{{{_tex_escape(name)}}} & {_tex_escape(value)} \\" for name, value in settings
    )
    normalized_plot = _normalized_bar_plot(normalized_rules)
    dimensional_rows = "\n".join(
        rf"\textcolor{{CUDAHeatSlate}}{{{name}}} & {_tex_escape(value)} \\" for name, value in dimensional_rules
    ) or r"\multicolumn{2}{@{}l@{}}{None} \\"
    label = f"{label_prefix}:{event_id}:point-{settings[1][1]}"
    path.write_text(
        rf"""% Generated by latex_event_figure.py; requires tikz, pgfplots and class/cudaheat-colors.tex.
\begin{{figure*}}[tbp]
\centering
\begin{{minipage}}[t]{{0.72\textwidth}}\vspace{{0pt}}
\centering
\begin{{tikzpicture}}[baseline=(current bounding box.north)]
\begin{{axis}}[
  width=\linewidth, height=0.25\textheight,
  date coordinates in=x, xticklabel=\day.\month, xticklabel style={{rotate=45,anchor=east}},
  xlabel={{Date/Time (UTC)}}, ylabel={{SST ($^\circ$C)}},
  axis lines=left,
  grid=major, grid style={{draw=CUDAHeatGrid}},
  {tick_options}
  ymin={ymin:.8g}, ymax={ymax:.8g},
  legend style={{at={{(0.5,1.03)}},anchor=south,legend columns=3,draw=none,font=\small}},
]
\addplot[SSTLine,thick,no markers] table[x=time,y=sst,col sep=comma] {{{latex_data_path}}};
\addlegendentry{{SST}}
\addplot[ThresholdLine,thin,opacity=1,no markers] table[x=time,y=threshold,col sep=comma] {{{latex_data_path}}};
\addlegendentry{{Threshold}}
\addplot[ClimatologyLine,thin,no markers] table[x=time,y=climatology,col sep=comma] {{{latex_data_path}}};
\addlegendentry{{Climatology}}
\end{{axis}}
\end{{tikzpicture}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.24\textwidth}}\vspace{{0pt}}
\footnotesize
\textbf{{Detection settings}}\par\vspace{{2pt}}
\begin{{tabular}}{{@{{}}ll@{{}}}}
{parameter_rows}
\end{{tabular}}
\par\vspace{{5pt}}\textbf{{Normalized metrics}}\par\vspace{{2pt}}
{normalized_plot}
\par\vspace{{5pt}}\textbf{{Dimensional metrics}}\par\vspace{{2pt}}
\begin{{tabular}}{{@{{}}ll@{{}}}}
{dimensional_rows}
\end{{tabular}}
\end{{minipage}}
\caption{{Hourly SST, climatological threshold and climatological mean for CUDAHeat event {event_id} (run \texttt{{{run_id}}}).{_tex_escape(month_span_caption)}}}
\label{{{label}}}
\end{{figure*}}
""",
        encoding="utf-8",
    )


def main() -> None:
    arguments = parse_arguments()
    metadata, all_events = load_stored_catalogue(
        arguments.run_id,
        postgres_config=arguments.postgres_config,
        event_ids=[arguments.event_id] if arguments.event_id is not None else None,
        point_indices=[arguments.point_index],
        max_events=0,
    )
    catalogue, event_id = _select_event(all_events, arguments.event_id)
    rules = discover_rules(arguments.rule_directory)
    enriched, _, _ = _enrich_catalogue(catalogue, metadata, rules)
    times, sst, threshold, climatology, start, stop = _event_series(catalogue, metadata, padding_hours=arguments.padding_hours)
    values = np.concatenate([sst[np.isfinite(sst)], threshold[np.isfinite(threshold)], climatology[np.isfinite(climatology)]])
    if not values.size:
        raise ValueError("event plot contains no finite SST, threshold, or climatology values")
    spread = max(float(values.max() - values.min()), 0.1)
    ymin, ymax = float(values.min() - 0.08 * spread), float(values.max() + 0.08 * spread)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"event-{event_id:06d}_point-{arguments.point_index}"
    data_path = arguments.output_directory / f"{stem}.csv"
    tex_path = arguments.output_directory / f"{stem}.tex"
    _write_data(data_path, times, sst, threshold, climatology)
    event_start, event_end = _date_text(times[start]), _date_text(times[stop - 1])
    settings, normalized_rules, dimensional_rules = _panel_rows(metadata, catalogue, rules, enriched)
    _write_tex(
        tex_path, data_path, arguments.run_id, event_id,
        settings, normalized_rules, dimensional_rules, ymin, ymax,
        event_start, event_end, arguments.label_prefix,
        _tick_options(times),
        _month_span_caption(event_start, event_end),
    )
    print(f"Generated: {tex_path}\nData: {data_path}\nEvent: {event_id} | point: {arguments.point_index}")


if __name__ == "__main__":
    main()
