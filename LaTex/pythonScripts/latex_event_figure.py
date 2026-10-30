from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import xarray as xr

from cudaheat import discover_rules, load_postgres_config
from cudaheat.climatology import hourly_climatology_index
from cudaheat.enrichment import _enrich_catalogue
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSTGRES_CONFIG = PROJECT_ROOT / "config" / "postgres.ini"
DEFAULT_RULE_DIRECTORY = PROJECT_ROOT / "examples" / "rule_laboratory" / "rules"

LATEX_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIRECTORY = LATEX_ROOT / "tikz" / "events"

printFlag = False

def load_stored_catalogue(
    run_id: UUID,
    *,
    postgres_config: Path,
    event_ids: Sequence[int] | None,
    point_indices: Sequence[int] | None,
    max_events: int,
) -> tuple[dict[str, Any], xr.Dataset]:
    import psycopg

    config = load_postgres_config(postgres_config)
    with psycopg.connect(**config.connection_kwargs()) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT metadata FROM runs WHERE run_id = %s", (run_id,))
            row = cursor.fetchone()
            if row is None:
                raise LookupError(f"run does not exist: {run_id}")
            if not isinstance(row[0], Mapping):
                raise ValueError("persisted run metadata is malformed")
            metadata = dict(row[0])
            clauses = ["e.run_id = %s"]
            values: list[Any] = [run_id]
            if event_ids:
                clauses.append("e.cudaheat_event_index = ANY(%s)")
                values.append(list(event_ids))
            if point_indices:
                clauses.append("p.grid_point_index = ANY(%s)")
                values.append(list(point_indices))
            limit = "" if max_events == 0 else " LIMIT %s"
            if max_events:
                values.append(max_events)
            cursor.execute(
                f"""
                SELECT e.cudaheat_event_index, p.grid_point_index, e.start_index,
                       e.stop_index, e.duration_hours, e.start_time, e.end_time
                FROM events AS e
                JOIN points AS p ON p.point_id = e.point_id
                WHERE {" AND ".join(clauses)}
                ORDER BY e.cudaheat_event_index{limit}
                """,
                values,
            )
            rows = cursor.fetchall()
    if not rows:
        raise LookupError("no events matched the requested run and filters")
    columns = list(zip(*rows, strict=True))
    catalogue = xr.Dataset(
        {
            "point_index": ("event", np.asarray(columns[1], dtype=np.int64)),
            "start_index": ("event", np.asarray(columns[2], dtype=np.intp)),
            "stop_index": ("event", np.asarray(columns[3], dtype=np.intp)),
            "duration_hours": ("event", np.asarray(columns[4], dtype=np.intp)),
            "start_time": ("event", np.asarray(columns[5], dtype="datetime64[ns]")),
            "end_time": ("event", np.asarray(columns[6], dtype="datetime64[ns]")),
        },
        coords={"event": np.asarray(columns[0], dtype=np.int64)},
    )
    return metadata, catalogue


def _format_baseline(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{value[0]}–{value[1]}"
    return "unknown"


def _format_percentile(value: Any) -> str:
    try:
        return f"{float(value) * 100:g}%"
    except (TypeError, ValueError):
        return "unknown"


def _format_hours(value: Any) -> str:
    try:
        return f"{int(value)} h"
    except (TypeError, ValueError):
        return "unknown"


def _format_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "missing"
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def _metadata_path(metadata: Mapping[str, Any], name: str) -> Path:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"persisted run metadata has no {name!r}")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"persisted {name} does not exist: {path}")
    return path


def _metadata_text(metadata: Mapping[str, Any], name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"persisted run metadata has no {name!r}")
    return value


def _point_columns(points: np.ndarray, requested: np.ndarray, label: str) -> np.ndarray:
    order = np.argsort(points)
    locations = np.searchsorted(points[order], requested)
    found = locations < points.size
    found[found] &= points[order][locations[found]] == requested[found]
    if not np.all(found):
        missing = np.unique(requested[~found])[:10].tolist()
        raise LookupError(f"event points are absent from the {label}: {missing}")
    return order[locations]


def _parse_run(value: str) -> tuple[UUID, int, int | None]:
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("expected RUN_UUID:POINT_INDEX[:EVENT_ID]")
    try:
        run_id = UUID(parts[0])
        point_index = int(parts[1])
        event_id = int(parts[2]) if len(parts) == 3 else None
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(f"invalid run specification: {value}") from exc
    if point_index < 0 or (event_id is not None and event_id < 0):
        raise argparse.ArgumentTypeError("point and event indices must be nonnegative")
    return run_id, point_index, event_id


def _parse_batch(value: str) -> list[tuple[UUID, int, int | None]]:
    match = re.fullmatch(r"([0-9a-fA-F-]{36})\[([^\]]+)\]", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError("expected UUID[POINT:EVENT,POINT:EVENT,...]")
    try:
        run_id = UUID(match.group(1))
        selections = []
        for item in match.group(2).split(","):
            pair = item.strip().split(":")
            if len(pair) != 2:
                raise ValueError("each selection must be POINT:EVENT")
            point_index, event_id = map(int, pair)
            if point_index < 0 or event_id < 0:
                raise ValueError("point and event indices must be nonnegative")
            selections.append((run_id, point_index, event_id))
        return selections
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export multiple CUDAHeat events as TikZ figures")
    parser.add_argument("input", nargs="?", type=_parse_batch, metavar="UUID[POINT:EVENT,...]")
    parser.add_argument("output", nargs="?", type=Path, metavar="OUTPUT_DIRECTORY")
    parser.add_argument("--run", action="append", type=_parse_run, metavar="UUID:POINT[:EVENT]")
    parser.add_argument("--postgres-config", type=Path, default=DEFAULT_POSTGRES_CONFIG)
    parser.add_argument("--rule-directory", type=Path, default=DEFAULT_RULE_DIRECTORY)
    parser.add_argument("--padding-hours", type=int, default=0)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--label-prefix", default="fig:event")
    args = parser.parse_args()
    if args.input is None and not args.run:
        parser.error("provide UUID[POINT:EVENT,...] or at least one --run")
    if args.output is not None and args.input is None:
        parser.error("positional OUTPUT_DIRECTORY requires positional input")
    args.run = (args.input or []) + (args.run or [])
    if args.output is not None:
        args.output_directory = args.output
    if args.padding_hours < 0:
        parser.error("--padding-hours must be non-negative")
    args.postgres_config = args.postgres_config.expanduser().resolve()
    args.rule_directory = args.rule_directory.expanduser().resolve()
    args.output_directory = args.output_directory.expanduser().resolve()
    if not args.postgres_config.is_file():
        parser.error(f"postgres config does not exist: {args.postgres_config}")
    if not args.rule_directory.is_dir():
        parser.error(f"rule directory does not exist: {args.rule_directory}")
    if not args.output_directory.is_relative_to(LATEX_ROOT):
        parser.error(f"output directory must be inside {LATEX_ROOT}")
    return args


def _tex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _date_text(value: np.datetime64) -> str:
    return np.datetime_as_string(value.astype("datetime64[s]"), unit="s").replace("T", " ")


def _to_celsius(values: np.ndarray, units: object) -> np.ndarray:
    normalized_units = str(units or "").strip().lower().replace("°", "")
    finite = values[np.isfinite(values)]
    is_kelvin = normalized_units in {"k", "kelvin", "degk", "degree_kelvin"}
    if not normalized_units and finite.size:
        is_kelvin = float(np.nanmedian(finite)) > 200.0
    return values - 273.15 if is_kelvin else values


def _tick_options(times: np.ndarray) -> str:
    span_hours = max(1.0, (times[-1] - times[0]) / np.timedelta64(1, "h"))
    if span_hours <= 72:
        return "xtick distance=0.5, minor x tick num=1,"
    if span_hours <= 14 * 24:
        return "xtick distance=1, minor x tick num=1,"
    if span_hours <= 60 * 24:
        return "xtick distance=3, minor x tick num=1,"
    return "xtick distance=14, minor x tick num=1,"


def _month_span_caption(start: str, end: str) -> str:
    start_month = datetime.fromisoformat(start).strftime("%B %Y")
    end_month = datetime.fromisoformat(end).strftime("%B %Y")
    if start_month == end_month:
        return f" Event period: {start_month}."
    return f" Event period: {start_month} to {end_month}."


def _select_event(catalogue: xr.Dataset, event_id: int | None) -> tuple[xr.Dataset, int]:
    event_values = np.asarray(catalogue["event"].values, dtype=np.int64)
    if not event_values.size:
        raise LookupError("no events found for requested point")
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
    import zarr

    source_path = _metadata_path(metadata, "source_path")
    cache_path = _metadata_path(metadata, "climatology_cache_path")
    variable = _metadata_text(metadata, "variable")
    time_dimension = str(metadata.get("time_dimension", "time"))
    calendar_mode = str(metadata.get("calendar_mode", "month_day_hour"))
    point_index = int(catalogue["point_index"].values[0])
    start = int(catalogue["start_index"].values[0])
    stop = int(catalogue["stop_index"].values[0])

    with xr.open_dataset(source_path, engine="h5netcdf", cache=False) as source:
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
        sst = np.asarray(
            source[variable].isel({time_dimension: slice(left, right), "point": source_column}).values,
            dtype=float,
        )
        bins = hourly_climatology_index(
            xr.DataArray(timestamps[left:right], dims=(time_dimension,)), calendar_mode=calendar_mode
        )

    cache = zarr.open_group(str(cache_path), mode="r")
    cache_column = int(_point_columns(
        np.asarray(cache["point"][:], dtype=np.int64),
        np.asarray([point_index], dtype=np.int64),
        "climatology cache",
    )[0])
    valid = (bins >= 0) & (bins < cache["threshold"].shape[0])
    threshold = np.full(right - left, np.nan)
    climatology = np.full(right - left, np.nan)
    threshold_cycle = np.asarray(cache["threshold"][:, cache_column])
    climatology_cycle = np.asarray(cache["climatology"][:, cache_column])
    threshold[valid] = threshold_cycle[bins[valid]]
    climatology[valid] = climatology_cycle[bins[valid]]
    return (
        timestamps[left:right],
        _to_celsius(sst, source_units),
        _to_celsius(threshold, source_units),
        _to_celsius(climatology, source_units),
        start - left,
        stop - left,
    )


def _write_data(
    path: Path,
    times: np.ndarray,
    sst: np.ndarray,
    threshold: np.ndarray,
    climatology: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "sst", "threshold", "climatology"))
        for timestamp, *values in zip(times, sst, threshold, climatology, strict=True):
            writer.writerow([
                _date_text(timestamp),
                *["" if not np.isfinite(value) else f"{value:.8g}" for value in values],
            ])


def _truncate_rule_id(rule_id: str, limit: int = 20) -> str:
    return rule_id if len(rule_id) <= limit else f"{rule_id[:limit - 1]}…"


def _panel_rows(
    metadata: Mapping[str, Any],
    catalogue: xr.Dataset,
    rules: Sequence[Any],
    enriched: xr.Dataset,
) -> tuple[list[tuple[str, str]], list[tuple[str, Any]], list[tuple[str, str]]]:
    settings = [
        ("Duration", _format_hours(catalogue["duration_hours"].values[0])),
        ("Baseline", _format_baseline(metadata.get("baseline"))),
        ("Percentile", _format_percentile(metadata.get("percentile"))),
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


def _normalized_bar_plot(rows: Sequence[tuple[str, Any]]) -> str:
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
    point_index: int,
    event_id: int,
    settings: Sequence[tuple[str, str]],
    normalized_rules: Sequence[tuple[str, Any]],
    dimensional_rules: Sequence[tuple[str, str]],
    ymin: float,
    ymax: float,
    label_prefix: str,
    tick_options: str,
    month_span_caption: str,
) -> None:
    latex_data_path = data_path.resolve().relative_to(LATEX_ROOT).as_posix()
    parameter_rows = "\n".join(
        rf"\textcolor{{CUDAHeatSlate}}{{{_tex_escape(name)}}} & {_tex_escape(value)} \\"
        for name, value in settings
    )
    normalized_plot = _normalized_bar_plot(normalized_rules)
    dimensional_rows = "\n".join(
        rf"\textcolor{{CUDAHeatSlate}}{{{name}}} & {_tex_escape(value)} \\"
        for name, value in dimensional_rules
    ) or r"\multicolumn{2}{@{}l@{}}{None} \\"
    label = f"{label_prefix}:{run_id}:{event_id}:point-{point_index}"
    path.write_text(
        rf"""\begin{{figure*}}[tbp]
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


def _export_event(
    run_id: UUID,
    point_index: int,
    requested_event_id: int | None,
    args: argparse.Namespace,
    rules: Sequence[Any],
) -> tuple[Path, Path, int]:
    metadata, all_events = load_stored_catalogue(
        run_id,
        postgres_config=args.postgres_config,
        event_ids=[requested_event_id] if requested_event_id is not None else None,
        point_indices=[point_index],
        max_events=0,
    )
    catalogue, event_id = _select_event(all_events, requested_event_id)
    enriched, _, _ = _enrich_catalogue(catalogue, metadata, rules)
    times, sst, threshold, climatology, start, stop = _event_series(
        catalogue, metadata, padding_hours=args.padding_hours
    )
    values = np.concatenate([
        sst[np.isfinite(sst)],
        threshold[np.isfinite(threshold)],
        climatology[np.isfinite(climatology)],
    ])
    if not values.size:
        raise ValueError("event plot contains no finite SST, threshold, or climatology values")
    spread = max(float(values.max() - values.min()), 0.1)
    ymin, ymax = float(values.min() - 0.08 * spread), float(values.max() + 0.08 * spread)
    stem = f"event-{event_id:06d}_point-{point_index}"
    data_path = args.output_directory / f"{stem}.csv"
    tex_path = args.output_directory / f"{stem}.tex"
    _write_data(data_path, times, sst, threshold, climatology)
    event_start, event_end = _date_text(times[start]), _date_text(times[stop - 1])
    settings, normalized_rules, dimensional_rules = _panel_rows(metadata, catalogue, rules, enriched)
    _write_tex(
        tex_path, data_path, run_id, point_index, event_id,
        settings, normalized_rules, dimensional_rules,
        ymin, ymax, args.label_prefix, _tick_options(times),
        _month_span_caption(event_start, event_end),
    )
    return tex_path, data_path, event_id


def main() -> None:
    args = parse_arguments()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    rules = discover_rules(args.rule_directory)
    failures = []
    for run_id, point_index, requested_event_id in dict.fromkeys(args.run):
        try:
            tex_path, data_path, event_id = _export_event(
                run_id, point_index, requested_event_id, args, rules
            )
        except Exception as exc:
            failures.append((run_id, point_index, requested_event_id, exc))
            print(f"FAILED run={run_id} point={point_index} event={requested_event_id}: {exc}")
            continue
        if printFlag:    
            print(f"Generated: {tex_path}\nData: {data_path}\nEvent: {event_id} | point: {point_index}")
    if failures:
        raise SystemExit(f"{len(failures)} of {len(dict.fromkeys(args.run))} exports failed")


if __name__ == "__main__":
    main()