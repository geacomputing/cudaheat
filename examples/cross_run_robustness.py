#!/usr/bin/env python3
"""Conservative, traceable event matching between CUDAHeat PostgreSQL runs.

Events are eligible only when they share a persisted ``point_id`` and overlap
in time.  The primary relation is deliberately strict: temporal IoU >= 0.50.
The report keeps three cases separate:

* no temporal overlap (not detected at that point/time),
* temporal overlap below the IoU threshold (ambiguous, not a match), and
* one-to-one, split, merge, or complex qualifying relations.

``coverage_detection`` is reported separately from strict IoU detection.  It
allows a source event to be covered by several non-overlapping events in the
other run, but requires 80% coverage by default.  It must not be substituted
for the strict metric when comparing taxonomies.

The script never changes PostgreSQL.  It writes summary CSV files by default;
``--write-matches`` additionally writes every accepted relation, retaining
both event IDs and point IDs for physical follow-up.  Cluster labels can be
supplied from a *shared* taxonomy CSV (``event_id,cluster`` or
``run_id,event_id,cluster``) to calculate classification stability.

Examples
--------
Compare one selected run against every compatible run::

    python examples/cross_run_robustness.py --reference-run UUID --plot

Inspect every compatible pair (potentially expensive for large catalogues)::

    python examples/cross_run_robustness.py --all-pairs --plot
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from cudaheat import load_postgres_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSTGRES_CONFIG = PROJECT_ROOT / "config" / "postgres.ini"
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).with_name("cross_run_robustness")
DEFAULT_DESCRIPTOR_RULES = (
    "heat_content",
    "daytime_strong_exceedance_fraction",
    "nocturnal_below_climatology_fraction",
    "nocturnal_below_threshold_fraction",
    "fraction_above_threshold",
)
FRACTION_RULES = frozenset(DEFAULT_DESCRIPTOR_RULES[1:])


@dataclass(frozen=True, slots=True)
class Run:
    """Persisted detection configuration needed to establish comparability."""

    run_id: str
    created_at: str
    source_sha256: str
    climatology_hash: str | None
    variable: str
    baseline_start: int | None
    baseline_end: int | None
    percentile: float
    smooth_window_hours: int
    sampling_window_half_width_days: int | None
    same_hour_smoothing: bool | None
    calendar_mode: str | None
    min_duration_hours: int
    max_internal_gap_hours: int
    join_gap_hours: int
    event_count: int

    @property
    def label(self) -> str:
        return (
            f"gap={self.max_internal_gap_hours}h, join={self.join_gap_hours}h, "
            f"min={self.min_duration_hours}h"
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """A same-point, temporally overlapping event pair using half-open indices."""

    event_a: int
    event_b: int
    point_id: int
    start_a: int
    stop_a: int
    start_b: int
    stop_b: int
    duration_a: int
    duration_b: int

    @property
    def overlap_hours(self) -> int:
        return max(0, min(self.stop_a, self.stop_b) - max(self.start_a, self.start_b))

    @property
    def temporal_iou(self) -> float:
        overlap = self.overlap_hours
        union = self.duration_a + self.duration_b - overlap
        return overlap / union if union else 0.0

    @property
    def duration_similarity(self) -> float:
        high = max(self.duration_a, self.duration_b)
        return min(self.duration_a, self.duration_b) / high if high else 0.0

    @property
    def onset_stability(self) -> float:
        scale = max(self.duration_a, self.duration_b, 1)
        return 1.0 - min(abs(self.start_a - self.start_b) / scale, 1.0)

    @property
    def termination_stability(self) -> float:
        scale = max(self.duration_a, self.duration_b, 1)
        return 1.0 - min(abs(self.stop_a - self.stop_b) / scale, 1.0)


@dataclass(frozen=True, slots=True)
class PairAnalysis:
    """Directional and symmetric robustness results for one run pair."""

    run_a: str
    run_b: str
    events_a: int
    events_b: int
    temporal_candidates: int
    accepted_pairs: int
    matched_a: int
    matched_b: int
    overlap_only_a: int
    overlap_only_b: int
    no_overlap_a: int
    no_overlap_b: int
    coverage_detected_a: int
    coverage_detected_b: int
    strict_detection_a_to_b: float
    strict_detection_b_to_a: float
    coverage_detection_a_to_b: float
    coverage_detection_b_to_a: float
    mean_temporal_iou: float | None
    mean_onset_stability: float | None
    mean_termination_stability: float | None
    mean_duration_similarity: float | None
    one_to_one_edges: int
    split_edges: int
    merge_edges: int
    complex_edges: int
    descriptor_robustness: float | None = None
    labelled_pairs: int = 0
    classification_robustness: float | None = None


@dataclass(frozen=True, slots=True)
class PairInternals:
    """Traceability state kept in memory until optional detail export."""

    analysis: PairAnalysis
    accepted: tuple[Candidate, ...]
    best_by_a: Mapping[int, Candidate]
    best_by_b: Mapping[int, Candidate]
    edge_kind: Mapping[tuple[int, int], str]


def _finite_mean(values: Iterable[float]) -> float | None:
    data = [value for value in values if math.isfinite(value)]
    return sum(data) / len(data) if data else None


def _best(current: Candidate | None, candidate: Candidate) -> Candidate:
    """Prefer the most similar temporal relation, with a stable ID tie-break."""
    if current is None:
        return candidate
    current_key = (current.temporal_iou, current.overlap_hours, -current.event_b)
    candidate_key = (candidate.temporal_iou, candidate.overlap_hours, -candidate.event_b)
    return candidate if candidate_key > current_key else current


def analyse_candidates(
    candidates: Iterable[Candidate],
    *,
    run_a: str,
    run_b: str,
    events_a: int,
    events_b: int,
    minimum_iou: float,
    coverage_threshold: float,
) -> PairInternals:
    """Classify candidate relations without querying a database.

    Event intervals are half-open index ranges.  Events within one CUDAHeat
    run/point do not overlap, so summing all candidate intersections gives
    exact union coverage for a source event.  Coverage is capped defensively
    should malformed external catalogues violate that invariant.
    """
    overlap_a: dict[int, int] = defaultdict(int)
    overlap_b: dict[int, int] = defaultdict(int)
    duration_a: dict[int, int] = {}
    duration_b: dict[int, int] = {}
    accepted: list[Candidate] = []
    degree_a: dict[int, int] = defaultdict(int)
    degree_b: dict[int, int] = defaultdict(int)
    best_by_a: dict[int, Candidate] = {}
    best_by_b: dict[int, Candidate] = {}
    temporal_candidates = 0

    for candidate in candidates:
        overlap = candidate.overlap_hours
        if overlap <= 0:
            continue
        temporal_candidates += 1
        overlap_a[candidate.event_a] += overlap
        overlap_b[candidate.event_b] += overlap
        duration_a[candidate.event_a] = candidate.duration_a
        duration_b[candidate.event_b] = candidate.duration_b
        if candidate.temporal_iou < minimum_iou:
            continue
        accepted.append(candidate)
        degree_a[candidate.event_a] += 1
        degree_b[candidate.event_b] += 1
        best_by_a[candidate.event_a] = _best(best_by_a.get(candidate.event_a), candidate)
        best_by_b[candidate.event_b] = _best(best_by_b.get(candidate.event_b), candidate)

    edge_kind: dict[tuple[int, int], str] = {}
    kind_counts = defaultdict(int)
    for candidate in accepted:
        degree_left = degree_a[candidate.event_a]
        degree_right = degree_b[candidate.event_b]
        if degree_left == degree_right == 1:
            kind = "one_to_one"
        elif degree_left > 1 and degree_right == 1:
            kind = "split"
        elif degree_left == 1 and degree_right > 1:
            kind = "merge"
        else:
            kind = "complex"
        edge_kind[(candidate.event_a, candidate.event_b)] = kind
        kind_counts[kind] += 1

    covered_a = sum(
        min(overlap, duration_a[event_id]) / duration_a[event_id] >= coverage_threshold
        for event_id, overlap in overlap_a.items()
    )
    covered_b = sum(
        min(overlap, duration_b[event_id]) / duration_b[event_id] >= coverage_threshold
        for event_id, overlap in overlap_b.items()
    )
    best_relations = tuple(best_by_a.values())
    analysis = PairAnalysis(
        run_a=run_a,
        run_b=run_b,
        events_a=events_a,
        events_b=events_b,
        temporal_candidates=temporal_candidates,
        accepted_pairs=len(accepted),
        matched_a=len(degree_a),
        matched_b=len(degree_b),
        overlap_only_a=len(overlap_a) - len(degree_a),
        overlap_only_b=len(overlap_b) - len(degree_b),
        no_overlap_a=events_a - len(overlap_a),
        no_overlap_b=events_b - len(overlap_b),
        coverage_detected_a=covered_a,
        coverage_detected_b=covered_b,
        strict_detection_a_to_b=len(degree_a) / events_a if events_a else 0.0,
        strict_detection_b_to_a=len(degree_b) / events_b if events_b else 0.0,
        coverage_detection_a_to_b=covered_a / events_a if events_a else 0.0,
        coverage_detection_b_to_a=covered_b / events_b if events_b else 0.0,
        mean_temporal_iou=_finite_mean(x.temporal_iou for x in best_relations),
        mean_onset_stability=_finite_mean(x.onset_stability for x in best_relations),
        mean_termination_stability=_finite_mean(x.termination_stability for x in best_relations),
        mean_duration_similarity=_finite_mean(x.duration_similarity for x in best_relations),
        one_to_one_edges=kind_counts["one_to_one"],
        split_edges=kind_counts["split"],
        merge_edges=kind_counts["merge"],
        complex_edges=kind_counts["complex"],
    )
    return PairInternals(
        analysis=analysis,
        accepted=tuple(accepted),
        best_by_a=best_by_a,
        best_by_b=best_by_b,
        edge_kind=edge_kind,
    )


def descriptor_similarity(rule_id: str, left: float | None, right: float | None) -> float | None:
    """A bounded, scale-aware numerical-descriptor stability score.

    Fractions use one minus absolute difference.  Positive extensive values
    such as heat content use the ratio of the smaller to the larger value;
    that treats a two-fold increase and decrease symmetrically.  Other values
    use a symmetric absolute-difference score.  Missing/non-finite values are
    explicitly excluded from the descriptor mean rather than imputed.
    """
    if left is None or right is None or not (math.isfinite(left) and math.isfinite(right)):
        return None
    if rule_id in FRACTION_RULES:
        return max(0.0, min(1.0, 1.0 - abs(left - right)))
    if rule_id == "heat_content" and left >= 0.0 and right >= 0.0:
        high = max(left, right)
        return 1.0 if high == 0.0 else min(left, right) / high
    scale = abs(left) + abs(right)
    return 1.0 if scale == 0.0 else max(0.0, 1.0 - abs(left - right) / scale)


def _run_query() -> str:
    return """
        SELECT r.run_id::text, r.created_at::text, r.source_sha256,
               r.climatology_hash, r.variable, r.baseline_start, r.baseline_end,
               r.percentile, r.smooth_window_hours,
               r.sampling_window_half_width_days, r.same_hour_smoothing,
               r.calendar_mode, r.min_duration_hours, r.max_internal_gap_hours,
               r.join_gap_hours, COUNT(e.event_id)::bigint AS event_count
        FROM runs AS r
        LEFT JOIN events AS e ON e.run_id = r.run_id
        GROUP BY r.run_id
        ORDER BY r.created_at, r.run_id
    """


def load_runs(connection: Any) -> list[Run]:
    with connection.cursor() as cursor:
        cursor.execute(_run_query())
        return [Run(*row) for row in cursor.fetchall()]


def compatible(left: Run, right: Run) -> bool:
    """Only compare catalogues with the same physical input and climatology."""
    return (
        left.source_sha256 == right.source_sha256
        and left.climatology_hash == right.climatology_hash
        and left.variable == right.variable
        and left.baseline_start == right.baseline_start
        and left.baseline_end == right.baseline_end
        and left.percentile == right.percentile
        and left.smooth_window_hours == right.smooth_window_hours
        and left.sampling_window_half_width_days == right.sampling_window_half_width_days
        and left.same_hour_smoothing == right.same_hour_smoothing
        and left.calendar_mode == right.calendar_mode
    )


def iter_candidates(
    connection: Any, run_a: str, run_b: str, *, batch_size: int
) -> Iterable[Candidate]:
    """Stream same-point temporal overlaps; no full catalogue is transferred."""
    query = """
        SELECT a.event_id, b.event_id, a.point_id,
               a.start_index, a.stop_index, b.start_index, b.stop_index,
               a.duration_hours, b.duration_hours
        FROM events AS a
        JOIN events AS b
          ON b.point_id = a.point_id
         AND b.end_time >= a.start_time
         AND a.end_time >= b.start_time
        WHERE a.run_id = %s AND b.run_id = %s
        ORDER BY a.event_id, b.event_id
    """
    with connection.cursor(name=f"cudaheat_match_{run_a[:8]}_{run_b[:8]}") as cursor:
        cursor.itersize = batch_size
        cursor.execute(query, (run_a, run_b))
        while rows := cursor.fetchmany(batch_size):
            for row in rows:
                yield Candidate(*(int(value) for value in row))


def _chunks(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_descriptors(
    connection: Any, event_ids: Iterable[int], rules: Sequence[str]
) -> dict[int, dict[str, float]]:
    """Load only identifiers used by representative accepted matches."""
    identifiers = sorted(set(event_ids))
    values: dict[int, dict[str, float]] = defaultdict(dict)
    if not identifiers or not rules:
        return values
    query = """
        SELECT event_id, rule_id, NULLIF(result ->> 'value', '')::double precision
        FROM event_enrichments
        WHERE event_id = ANY(%s) AND rule_id = ANY(%s)
    """
    with connection.cursor() as cursor:
        for chunk in _chunks(identifiers, 20_000):
            cursor.execute(query, (list(chunk), list(rules)))
            for event_id, rule_id, value in cursor:
                if value is not None and math.isfinite(float(value)):
                    values[int(event_id)][str(rule_id)] = float(value)
    return values


def load_event_context(connection: Any, event_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
    """Fetch human-readable times and coordinates only for optional detail export."""
    identifiers = sorted(set(event_ids))
    context: dict[int, dict[str, Any]] = {}
    if not identifiers:
        return context
    query = """
        SELECT e.event_id, p.latitude, p.longitude, e.start_time::text, e.end_time::text
        FROM events AS e JOIN points AS p ON p.point_id = e.point_id
        WHERE e.event_id = ANY(%s)
    """
    with connection.cursor() as cursor:
        for chunk in _chunks(identifiers, 20_000):
            cursor.execute(query, (list(chunk),))
            for event_id, latitude, longitude, start_time, end_time in cursor:
                context[int(event_id)] = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_time": str(start_time),
                    "end_time": str(end_time),
                }
    return context


def load_cluster_labels(path: Path | None) -> dict[tuple[str | None, int], str]:
    """Read shared taxonomy labels without imposing a dependency on pandas."""
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"event_id", "cluster"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} needs columns event_id,cluster[,run_id]")
        labels: dict[tuple[str | None, int], str] = {}
        for row in reader:
            event_id = int(row["event_id"])
            run_id = (row.get("run_id") or "").strip() or None
            labels[(run_id, event_id)] = row["cluster"]
    return labels


def _label(labels: Mapping[tuple[str | None, int], str], run_id: str, event_id: int) -> str | None:
    label = labels.get((run_id, event_id), labels.get((None, event_id)))
    # Density algorithms conventionally use -1 for noise.  It is an explicit
    # non-assignment, not an event family, so counting noise/noise would
    # falsely inflate classification robustness.
    if label is None or label.strip().lower() in {"", "-1", "noise", "unassigned", "none", "nan"}:
        return None
    return label


def enrich_analysis(
    internals: PairInternals,
    descriptors: Mapping[int, Mapping[str, float]],
    rules: Sequence[str],
    labels: Mapping[tuple[str | None, int], str],
) -> PairInternals:
    """Add descriptor and optional classification stability to pair results."""
    scores: list[float] = []
    labelled = 0
    same_label = 0
    for candidate in internals.best_by_a.values():
        per_rule = [
            score
            for rule in rules
            if (score := descriptor_similarity(
                rule,
                descriptors.get(candidate.event_a, {}).get(rule),
                descriptors.get(candidate.event_b, {}).get(rule),
            )) is not None
        ]
        if per_rule:
            scores.append(sum(per_rule) / len(per_rule))
        label_a = _label(labels, internals.analysis.run_a, candidate.event_a)
        label_b = _label(labels, internals.analysis.run_b, candidate.event_b)
        if label_a is not None and label_b is not None:
            labelled += 1
            same_label += label_a == label_b
    analysis = PairAnalysis(
        **{
            **asdict(internals.analysis),
            "descriptor_robustness": _finite_mean(scores),
            "labelled_pairs": labelled,
            "classification_robustness": same_label / labelled if labelled else None,
        }
    )
    return PairInternals(
        analysis=analysis,
        accepted=internals.accepted,
        best_by_a=internals.best_by_a,
        best_by_b=internals.best_by_b,
        edge_kind=internals.edge_kind,
    )


def descriptor_report(
    internals: PairInternals,
    descriptors: Mapping[int, Mapping[str, float]],
    rules: Sequence[str],
) -> list[dict[str, Any]]:
    """Expose each descriptor score; never hide a weak/constant component in a mean."""
    rows: list[dict[str, Any]] = []
    for rule in rules:
        scores = [
            score
            for candidate in internals.best_by_a.values()
            if (score := descriptor_similarity(
                rule,
                descriptors.get(candidate.event_a, {}).get(rule),
                descriptors.get(candidate.event_b, {}).get(rule),
            )) is not None
        ]
        rows.append(
            {
                "run_a": internals.analysis.run_a,
                "run_b": internals.analysis.run_b,
                "rule_id": rule,
                "representative_matched_events": len(internals.best_by_a),
                "valid_descriptor_pairs": len(scores),
                "mean_descriptor_similarity": _finite_mean(scores),
            }
        )
    return rows


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    records = list(rows)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_manifest(
    directory: Path,
    arguments: argparse.Namespace,
    rules: Sequence[str],
    pairs: Sequence[tuple[Run, Run]],
    runs: Sequence[Run],
) -> None:
    """Record thresholds and selection so results remain reproducible."""
    incompatible_pairs = [
        {"run_a": left.run_id, "run_b": right.run_id}
        for left, right in combinations(runs, 2)
        if not compatible(left, right)
    ]
    manifest = {
        "minimum_temporal_iou": arguments.minimum_iou,
        "coverage_threshold": arguments.coverage_threshold,
        "descriptor_rules": list(rules),
        "cluster_labels": str(arguments.cluster_labels) if arguments.cluster_labels else None,
        "selected_pairs": [{"run_a": left.run_id, "run_b": right.run_id} for left, right in pairs],
        "incompatible_pairs_excluded": incompatible_pairs,
        "method": "same point_id + positive temporal overlap; half-open event indices",
    }
    with (directory / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def write_match_details(
    directory: Path,
    internals: PairInternals,
    descriptors: Mapping[int, Mapping[str, float]],
    rules: Sequence[str],
    event_context: Mapping[int, Mapping[str, Any]],
) -> Path:
    """Persist every qualifying relation, including its traceable IDs."""
    rows: list[dict[str, Any]] = []
    for candidate in internals.accepted:
        row: dict[str, Any] = {
            "run_a": internals.analysis.run_a,
            "run_b": internals.analysis.run_b,
            "event_a": candidate.event_a,
            "event_b": candidate.event_b,
            "point_id": candidate.point_id,
            "start_index_a": candidate.start_a,
            "stop_index_a": candidate.stop_a,
            "start_index_b": candidate.start_b,
            "stop_index_b": candidate.stop_b,
            "duration_a_hours": candidate.duration_a,
            "duration_b_hours": candidate.duration_b,
            "overlap_hours": candidate.overlap_hours,
            "temporal_iou": candidate.temporal_iou,
            "onset_stability": candidate.onset_stability,
            "termination_stability": candidate.termination_stability,
            "duration_similarity": candidate.duration_similarity,
            "relation": internals.edge_kind[(candidate.event_a, candidate.event_b)],
        }
        for side, event_id in (("a", candidate.event_a), ("b", candidate.event_b)):
            context = event_context.get(event_id, {})
            row[f"start_time_{side}"] = context.get("start_time")
            row[f"end_time_{side}"] = context.get("end_time")
        point_context = event_context.get(candidate.event_a, {})
        row["latitude"] = point_context.get("latitude")
        row["longitude"] = point_context.get("longitude")
        per_rule = []
        for rule in rules:
            score = descriptor_similarity(
                rule,
                descriptors.get(candidate.event_a, {}).get(rule),
                descriptors.get(candidate.event_b, {}).get(rule),
            )
            row[f"descriptor_{rule}"] = score
            if score is not None:
                per_rule.append(score)
        row["descriptor_similarity"] = _finite_mean(per_rule)
        rows.append(row)
    output = directory / (
        f"accepted_matches_{_safe_name(internals.analysis.run_a)}"
        f"__{_safe_name(internals.analysis.run_b)}.csv"
    )
    write_csv(output, rows)
    return output


def plot_robustness(directory: Path, analyses: Sequence[PairAnalysis], runs: Mapping[str, Run]) -> Path | None:
    """Write a compact directional strict-detection matrix when matplotlib exists."""
    if not analyses:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None
    ordered = sorted(
        {entry.run_a for entry in analyses} | {entry.run_b for entry in analyses},
        key=lambda run_id: (runs[run_id].max_internal_gap_hours, runs[run_id].join_gap_hours),
    )
    position = {run_id: index for index, run_id in enumerate(ordered)}
    values = np.full((len(ordered), len(ordered)), np.nan)
    for entry in analyses:
        values[position[entry.run_a], position[entry.run_b]] = entry.strict_detection_a_to_b
        values[position[entry.run_b], position[entry.run_a]] = entry.strict_detection_b_to_a
    figure, axis = plt.subplots(figsize=(max(7, len(ordered) * 0.68), max(6, len(ordered) * 0.6)))
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis")
    labels = [f"g{runs[key].max_internal_gap_hours}/j{runs[key].join_gap_hours}" for key in ordered]
    axis.set(
        xticks=range(len(ordered)), yticks=range(len(ordered)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Comparison configuration", ylabel="Reference configuration",
        title="Strict event detection robustness (temporal IoU ≥ 0.50)",
    )
    axis.tick_params(axis="x", rotation=55)
    figure.colorbar(image, ax=axis, label="fraction of reference events matched")
    figure.tight_layout()
    output = directory / "strict_detection_matrix.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def summary_row(entry: PairAnalysis, runs: Mapping[str, Run]) -> dict[str, Any]:
    """Keep parameter changes beside robustness metrics for standalone CSV use."""
    row = asdict(entry)
    for side, run_id in (("a", entry.run_a), ("b", entry.run_b)):
        run = runs[run_id]
        row[f"min_duration_{side}_hours"] = run.min_duration_hours
        row[f"max_internal_gap_{side}_hours"] = run.max_internal_gap_hours
        row[f"join_gap_{side}_hours"] = run.join_gap_hours
    return row


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--postgres-config", type=Path, default=DEFAULT_POSTGRES_CONFIG)
    parser.add_argument("--reference-run", help="UUID of the reference run")
    parser.add_argument("--comparison-run", action="append", default=[], help="UUID to compare; repeatable")
    parser.add_argument("--all-pairs", action="store_true", help="analyse every compatible unordered pair")
    parser.add_argument("--minimum-iou", type=float, default=0.50)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--descriptor-rule", action="append", default=[], help="enrichment rule ID; repeatable")
    parser.add_argument(
        "--no-descriptors",
        action="store_true",
        help="skip enrichment retrieval (useful for a fast IoU-threshold sensitivity run)",
    )
    parser.add_argument("--cluster-labels", type=Path, help="shared taxonomy CSV: event_id,cluster[,run_id]")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--fetch-batch-size", type=int, default=100_000)
    parser.add_argument("--write-matches", action="store_true", help="write all qualifying event-level matches")
    parser.add_argument("--plot", action="store_true", help="write strict-detection matrix PNG")
    arguments = parser.parse_args()
    if not 0.0 < arguments.minimum_iou <= 1.0:
        parser.error("--minimum-iou must be in (0, 1]")
    if not 0.0 < arguments.coverage_threshold <= 1.0:
        parser.error("--coverage-threshold must be in (0, 1]")
    if arguments.fetch_batch_size < 1:
        parser.error("--fetch-batch-size must be positive")
    if arguments.all_pairs and (arguments.reference_run or arguments.comparison_run):
        parser.error("--all-pairs cannot be combined with explicit run selection")
    if arguments.comparison_run and not arguments.reference_run:
        parser.error("--comparison-run requires --reference-run")
    if arguments.no_descriptors and arguments.descriptor_rule:
        parser.error("--no-descriptors cannot be combined with --descriptor-rule")
    return arguments


def select_pairs(arguments: argparse.Namespace, runs: Sequence[Run]) -> list[tuple[Run, Run]]:
    by_id = {run.run_id: run for run in runs}
    if arguments.all_pairs:
        return [(left, right) for left, right in combinations(runs, 2) if compatible(left, right)]
    if not arguments.reference_run:
        raise ValueError("choose --reference-run or --all-pairs; no implicit reference is scientifically neutral")
    try:
        reference = by_id[arguments.reference_run]
    except KeyError as exc:
        raise ValueError(f"reference run was not found: {arguments.reference_run}") from exc
    comparison_ids = arguments.comparison_run or [run.run_id for run in runs if run.run_id != reference.run_id]
    pairs: list[tuple[Run, Run]] = []
    for comparison_id in comparison_ids:
        try:
            comparison = by_id[comparison_id]
        except KeyError as exc:
            raise ValueError(f"comparison run was not found: {comparison_id}") from exc
        if not compatible(reference, comparison):
            raise ValueError(
                f"runs are not comparable (source/climatology settings differ): "
                f"{reference.run_id}, {comparison.run_id}"
            )
        pairs.append((reference, comparison))
    return pairs


def main() -> None:
    arguments = parse_arguments()
    import psycopg

    configuration = load_postgres_config(arguments.postgres_config)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    rules = () if arguments.no_descriptors else tuple(arguments.descriptor_rule or DEFAULT_DESCRIPTOR_RULES)
    labels = load_cluster_labels(arguments.cluster_labels)
    with psycopg.connect(**configuration.connection_kwargs()) as connection:
        runs = load_runs(connection)
        pairs = select_pairs(arguments, runs)
        if not pairs:
            raise RuntimeError("no compatible run pairs selected")
        print(f"Analysing {len(pairs)} pair(s); strict temporal IoU >= {arguments.minimum_iou:.2f}.")
        run_map = {run.run_id: run for run in runs}
        analysed: list[PairAnalysis] = []
        category_rows: list[dict[str, Any]] = []
        descriptor_rows: list[dict[str, Any]] = []
        for number, (left, right) in enumerate(pairs, start=1):
            internals = analyse_candidates(
                iter_candidates(connection, left.run_id, right.run_id, batch_size=arguments.fetch_batch_size),
                run_a=left.run_id,
                run_b=right.run_id,
                events_a=left.event_count,
                events_b=right.event_count,
                minimum_iou=arguments.minimum_iou,
                coverage_threshold=arguments.coverage_threshold,
            )
            representative_ids = {
                event_id
                for candidate in internals.best_by_a.values()
                for event_id in (candidate.event_a, candidate.event_b)
            }
            descriptors = load_descriptors(connection, representative_ids, rules)
            internals = enrich_analysis(internals, descriptors, rules, labels)
            analysed.append(internals.analysis)
            descriptor_rows.extend(descriptor_report(internals, descriptors, rules))
            category_rows.append(
                {
                    "run_a": left.run_id,
                    "run_b": right.run_id,
                    "one_to_one_edges": internals.analysis.one_to_one_edges,
                    "split_edges": internals.analysis.split_edges,
                    "merge_edges": internals.analysis.merge_edges,
                    "complex_edges": internals.analysis.complex_edges,
                }
            )
            if arguments.write_matches:
                detail_descriptors = descriptors
                missing_ids = {
                    event_id
                    for candidate in internals.accepted
                    for event_id in (candidate.event_a, candidate.event_b)
                    if event_id not in detail_descriptors
                }
                if missing_ids:
                    detail_descriptors = {**detail_descriptors, **load_descriptors(connection, missing_ids, rules)}
                event_context = load_event_context(connection, {
                    event_id
                    for candidate in internals.accepted
                    for event_id in (candidate.event_a, candidate.event_b)
                })
                write_match_details(
                    arguments.output_dir, internals, detail_descriptors, rules, event_context
                )
            print(
                f"[{number}/{len(pairs)}] {left.label} -> {right.label}: "
                f"strict={internals.analysis.strict_detection_a_to_b:.3f}, "
                f"IoU={internals.analysis.mean_temporal_iou or 0.0:.3f}, "
                f"accepted={internals.analysis.accepted_pairs:,}"
            )
    write_csv(arguments.output_dir / "runs.csv", (asdict(run) for run in runs))
    write_csv(arguments.output_dir / "pair_summary.csv", (summary_row(entry, run_map) for entry in analysed))
    write_csv(arguments.output_dir / "match_categories.csv", category_rows)
    write_csv(arguments.output_dir / "descriptor_summary.csv", descriptor_rows)
    write_manifest(arguments.output_dir, arguments, rules, pairs, runs)
    if arguments.plot:
        output = plot_robustness(arguments.output_dir, analysed, run_map)
        if output is not None:
            print(f"Saved: {output}")
    print(f"Saved summaries: {arguments.output_dir}")


if __name__ == "__main__":
    main()
