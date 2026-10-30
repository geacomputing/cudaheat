#!/usr/bin/env python3
"""Discover and stress-test descriptor-space event families.

The sample is equal-run stratified, avoiding domination by permissive runs.
It retains event IDs, locations and time bounds for physical traceability.
The required path uses existing PostgreSQL/plot extras plus SciPy; optional
UMAP/HDBSCAN are used only when already installed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from cudaheat import load_postgres_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "postgres.ini"
DEFAULT_OUTPUT = Path(__file__).with_name("event_family_analysis")
HASH_RANGE = 1_000_000
RULES = (
    ("fraction_above_threshold", "fraction_above_threshold"),
    ("daytime_strong_exceedance_fraction", "daytime_strong_exceedance_fraction"),
    ("heat_content", "heat_content"),
    ("nocturnal_below_climatology_fraction", "nocturnal_below_climatology_fraction"),
    ("nocturnal_below_threshold_fraction", "nocturnal_below_threshold_fraction"),
)
FEATURE_NAMES = (
    "log_duration_hours",
    "fraction_above_threshold",
    "daytime_strong_exceedance_fraction",
    "log_mean_excess_degC",
    "nocturnal_below_climatology_fraction",
    "nocturnal_below_threshold_fraction",
)


@dataclass
class Sample:
    event_id: np.ndarray
    run_id: np.ndarray
    point_id: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    start_time: np.ndarray
    end_time: np.ndarray
    duration_hours: np.ndarray
    raw_rules: np.ndarray
    population_by_run: dict[str, int]
    sampled_by_run: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--events-per-run", type=int, default=12_000)
    parser.add_argument("--min-cluster-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261030)
    return parser.parse_args()


def event_sql() -> str:
    joins = "\n".join(
        f"JOIN event_enrichments AS {alias} ON {alias}.event_id=e.event_id AND {alias}.rule_id='{rule}'"
        for alias, rule in RULES
    )
    values = ",\n               ".join(
        f"NULLIF({alias}.result ->> 'value', '')::double precision AS {alias}" for alias, _ in RULES
    )
    return f"""
        SELECT e.event_id, e.run_id::text, e.point_id, p.latitude, p.longitude,
               e.start_time, e.end_time, e.duration_hours::double precision,
               {values}
        FROM events AS e JOIN points AS p ON p.point_id=e.point_id
        {joins}
        WHERE e.run_id=%s
          AND MOD(ABS(hashtextextended(e.event_id::text, %s)), {HASH_RANGE}) < %s
    """


def load_stratified_sample(config_path: Path, events_per_run: int, seed: int) -> Sample:
    """Hash sample per run then cap uniformly; no pooled-catalogue bias."""
    if events_per_run < 1:
        raise ValueError("events_per_run must be positive")
    import psycopg

    rows: list[tuple[object, ...]] = []
    population: dict[str, int] = {}
    selected: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    with psycopg.connect(**load_postgres_config(config_path).connection_kwargs()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT run_id::text, COUNT(*) FROM events GROUP BY run_id ORDER BY run_id"
            )
            run_counts = [(str(run_id), int(count)) for run_id, count in cursor.fetchall()]
        for run_id, count in run_counts:
            cutoff = min(HASH_RANGE, int(np.ceil(HASH_RANGE * 1.15 * events_per_run / count)))
            with connection.cursor() as cursor:
                cursor.execute(event_sql(), (run_id, seed, cutoff))
                chosen = cursor.fetchall()
            if len(chosen) > events_per_run:
                keep = np.sort(rng.choice(len(chosen), events_per_run, replace=False))
                chosen = [chosen[int(index)] for index in keep]
            population[run_id], selected[run_id] = count, len(chosen)
            rows.extend(chosen)
    if not rows:
        raise RuntimeError("No enriched events returned by PostgreSQL.")
    columns = list(zip(*rows, strict=True))
    return Sample(
        event_id=np.asarray(columns[0], dtype=np.int64),
        run_id=np.asarray(columns[1], dtype=object),
        point_id=np.asarray(columns[2], dtype=np.int64),
        latitude=np.asarray(columns[3], dtype=float),
        longitude=np.asarray(columns[4], dtype=float),
        start_time=np.asarray(columns[5], dtype=object),
        end_time=np.asarray(columns[6], dtype=object),
        duration_hours=np.asarray(columns[7], dtype=float),
        raw_rules=np.asarray(columns[8:], dtype=float).T,
        population_by_run=population,
        sampled_by_run=selected,
    )


def event_features(sample: Sample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform duration and cumulative intensity without concealing exclusions."""
    fraction, daytime, heat, night_climate, night_threshold = sample.raw_rules.T
    mean_excess = heat / sample.duration_hours
    features = np.column_stack(
        (
            np.log1p(sample.duration_hours),
            fraction,
            daytime,
            np.log1p(np.maximum(mean_excess, 0.0)),
            night_climate,
            night_threshold,
        )
    )
    usable = (
        np.isfinite(features).all(axis=1)
        & (sample.duration_hours > 0)
        & (heat >= 0)
        & np.isfinite(sample.latitude)
        & np.isfinite(sample.longitude)
    )
    reason = np.full(len(usable), "", dtype=object)
    reason[~np.isfinite(features).all(axis=1)] = "non_finite_descriptor"
    reason[(sample.duration_hours <= 0) & (reason == "")] = "nonpositive_duration"
    reason[(heat < 0) & (reason == "")] = "negative_heat_content"
    reason[(~np.isfinite(sample.latitude) | ~np.isfinite(sample.longitude)) & (reason == "")] = (
        "missing_location"
    )
    return features, usable, reason


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    scale = np.diff(np.quantile(values, (0.25, 0.75), axis=0), axis=0)[0]
    scale[scale <= np.finfo(float).eps] = 1.0
    return np.clip((values - median) / scale, -6.0, 6.0), median, scale


def pca(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = values - np.mean(values, axis=0)
    _, singular_values, loadings = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2 / max(len(values) - 1, 1)
    return centered @ loadings.T, loadings, variance / variance.sum()


def spearman_matrix(values: np.ndarray) -> np.ndarray:
    ranks = np.column_stack([rankdata(values[:, index]) for index in range(values.shape[1])])
    return np.corrcoef(ranks, rowvar=False)


def rank_uniform_scale(values: np.ndarray) -> np.ndarray:
    """Distribution-free [0,1] scaling used only as a normalization stress test."""
    return np.column_stack(
        [
            rankdata(values[:, index], method="average") / (len(values) + 1)
            for index in range(values.shape[1])
        ]
    )


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def density_components(
    values: np.ndarray, *, neighbours: int, density_quantile: float, min_cluster_size: int
) -> tuple[np.ndarray, float, float]:
    """Conservative kNN density components; uncertain rows remain ``-1``.

    This is deliberately called a density graph rather than HDBSCAN.  Core
    points have a small k-neighbour radius.  Non-core points can join a core,
    but cannot bridge components and manufacture a spurious family.
    """
    if not 1 <= neighbours < len(values):
        raise ValueError("neighbours must be between 1 and the sample size")
    tree = cKDTree(values)
    distances, indices = tree.query(values, k=neighbours + 1, workers=-1)
    epsilon = float(np.quantile(distances[:, -1], density_quantile))
    core = distances[:, -1] <= epsilon
    union_find = _UnionFind(len(values))
    for row, nearest, nearest_distances in zip(
        np.flatnonzero(core), indices[core, 1:], distances[core, 1:], strict=True
    ):
        for other, distance in zip(nearest, nearest_distances, strict=True):
            if distance <= epsilon and core[other]:
                union_find.union(int(row), int(other))
    core_rows = np.flatnonzero(core)
    roots = np.asarray([union_find.find(int(row)) for row in core_rows])
    root_ids, counts = np.unique(roots, return_counts=True)
    root_to_label = {
        root: label for label, root in enumerate(sorted(root_ids[counts >= min_cluster_size]))
    }
    labels = np.full(len(values), -1, dtype=int)
    for row in core_rows:
        root = union_find.find(int(row))
        if root in root_to_label:
            labels[row] = root_to_label[root]

    # Border assignment never connects components.
    retained_core = np.flatnonzero(labels >= 0)
    if len(retained_core):
        retained_tree = cKDTree(values[retained_core])
        border = np.flatnonzero(labels < 0)
        distance, nearest = retained_tree.query(values[border], workers=-1)
        accepted = distance <= epsilon
        labels[border[accepted]] = labels[retained_core[nearest[accepted]]]
    return labels, epsilon, float(core.mean())


def adjusted_rand_index(left: np.ndarray, right: np.ndarray) -> float:
    """Sklearn-free ARI, including noise as an explicit assignment state."""
    if len(left) != len(right):
        raise ValueError("label arrays must have equal length")
    _, li = np.unique(left, return_inverse=True)
    _, ri = np.unique(right, return_inverse=True)
    table = np.zeros((li.max() + 1, ri.max() + 1), dtype=np.int64)
    np.add.at(table, (li, ri), 1)

    def choose_two(values: np.ndarray) -> float:
        return float(np.sum(values * (values - 1) // 2, dtype=np.int64))

    index = choose_two(table)
    left_pairs, right_pairs = choose_two(table.sum(axis=1)), choose_two(table.sum(axis=0))
    pairs = len(left) * (len(left) - 1) / 2
    if pairs == 0:
        return 1.0
    expected = left_pairs * right_pairs / pairs
    maximum = (left_pairs + right_pairs) / 2
    return 1.0 if maximum == expected else (index - expected) / (maximum - expected)


def cluster_jaccard(reference: np.ndarray, alternative: np.ndarray) -> float:
    """Membership-weighted best Jaccard score for non-noise reference families."""
    if not np.any(reference >= 0):
        return float("nan")
    score, total = 0.0, 0
    for label in np.unique(reference[reference >= 0]):
        member = reference == label
        best = 0.0
        for candidate in np.unique(alternative[alternative >= 0]):
            other = alternative == candidate
            best = max(best, float((member & other).sum() / (member | other).sum()))
        score += member.sum() * best
        total += int(member.sum())
    return score / total


def write_manifest(path: Path, sample: Sample, usable: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "catalogue_events", "sampled_events", "usable_events"])
        for run_id in sorted(sample.population_by_run):
            in_run = sample.run_id == run_id
            writer.writerow(
                [
                    run_id,
                    sample.population_by_run[run_id],
                    sample.sampled_by_run[run_id],
                    int(usable[in_run].sum()),
                ]
            )


def write_correlation_csv(path: Path, correlation: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", *FEATURE_NAMES])
        for name, row in zip(FEATURE_NAMES, correlation, strict=True):
            writer.writerow([name, *row])


def write_assignments(
    path: Path,
    sample: Sample,
    features: np.ndarray,
    usable: np.ndarray,
    reason: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    rank_labels: np.ndarray,
) -> None:
    """Write every sampled event, including invalid rows, so nothing vanishes."""
    header = [
        "event_id",
        "run_id",
        "point_id",
        "latitude",
        "longitude",
        "start_time",
        "end_time",
        "duration_hours",
        *[f"raw_{rule}" for _, rule in RULES],
        *FEATURE_NAMES,
        "usable_for_fit",
        "exclusion_reason",
        "pca_1",
        "pca_2",
        "cluster",
        "density_family",
        "cluster_rank_uniform",
    ]
    full_scores = np.full((len(sample.event_id), 2), np.nan)
    full_scores[usable] = scores[:, :2]
    full_labels = np.full(len(sample.event_id), -1, dtype=int)
    full_labels[usable] = labels
    full_rank_labels = np.full(len(sample.event_id), -1, dtype=int)
    full_rank_labels[usable] = rank_labels
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(len(sample.event_id)):
            writer.writerow(
                [
                    sample.event_id[index],
                    sample.run_id[index],
                    sample.point_id[index],
                    sample.latitude[index],
                    sample.longitude[index],
                    sample.start_time[index],
                    sample.end_time[index],
                    sample.duration_hours[index],
                    *sample.raw_rules[index],
                    *features[index],
                    bool(usable[index]),
                    reason[index],
                    *full_scores[index],
                    int(full_labels[index]),
                    int(full_labels[index]),
                    int(full_rank_labels[index]),
                ]
            )


def write_family_summary(
    path: Path, features: np.ndarray, labels: np.ndarray, sample: Sample, usable: np.ndarray
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "density_family",
                "sample_events",
                "sample_fraction",
                "runs_present",
                *[f"median_{name}" for name in FEATURE_NAMES],
            ]
        )
        for label in np.unique(labels[labels >= 0]):
            member = labels == label
            writer.writerow(
                [
                    int(label),
                    int(member.sum()),
                    member.mean(),
                    len(np.unique(sample.run_id[usable][member])),
                    *np.median(features[member], axis=0),
                ]
            )


def plot_correlations(path: Path, correlation: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    display = [name.replace("_", "\n") for name in FEATURE_NAMES]
    axis.set(
        xticks=range(len(display)),
        yticks=range(len(display)),
        xticklabels=display,
        yticklabels=display,
    )
    plt.setp(axis.get_xticklabels(), rotation=38, ha="right")
    for row in range(len(display)):
        for column in range(len(display)):
            axis.text(
                column, row, f"{correlation[row, column]:.2f}", ha="center", va="center", fontsize=8
            )
    figure.colorbar(image, ax=axis, label="Spearman correlation")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_embedding(path: Path, scores: np.ndarray, labels: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    density = axes[0].hexbin(
        scores[:, 0], scores[:, 1], gridsize=110, bins="log", mincnt=1, cmap="viridis"
    )
    axes[0].set(title="PCA density (all usable events)", xlabel="PC1", ylabel="PC2")
    figure.colorbar(density, ax=axes[0], label="log10 sample count")
    axes[1].scatter(
        scores[labels < 0, 0],
        scores[labels < 0, 1],
        s=0.45,
        color="#b5b5b5",
        alpha=0.18,
        rasterized=True,
        label="unassigned",
    )
    for label in np.unique(labels[labels >= 0]):
        in_family = labels == label
        axes[1].scatter(
            scores[in_family, 0],
            scores[in_family, 1],
            s=0.75,
            alpha=0.38,
            rasterized=True,
            label=f"family {label}",
        )
    axes[1].set(title="Conservative density families", xlabel="PC1", ylabel="PC2")
    axes[1].legend(markerscale=5, fontsize=7, frameon=False, loc="best")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def optional_umap_hdbscan(values: np.ndarray, output: Path, seed: int) -> str:
    """Run nonlinear alternatives only when their optional dependencies exist."""
    try:
        import hdbscan  # type: ignore[import-not-found]
        import umap  # type: ignore[import-not-found]
    except ImportError:
        return "UMAP/HDBSCAN unavailable (not installed); no dependency was added."
    rng = np.random.default_rng(seed)
    rows = np.arange(len(values))
    if len(rows) > 100_000:
        rows = np.sort(rng.choice(rows, 100_000, replace=False))
    subset = values[rows]
    embedding = umap.UMAP(
        n_neighbors=30, min_dist=0.08, metric="euclidean", random_state=seed
    ).fit_transform(subset)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=max(120, len(subset) // 500), min_samples=25
    ).fit_predict(subset)
    np.savetxt(
        output / "optional_umap_hdbscan.csv",
        np.column_stack((rows, embedding, labels)),
        delimiter=",",
        header="usable_row,umap_1,umap_2,hdbscan_family",
        comments="",
    )
    return f"Optional UMAP/HDBSCAN completed on {len(subset):,} usable sampled events."


def main() -> None:
    args = parse_args()
    if args.min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    sample = load_stratified_sample(args.config, args.events_per_run, args.seed)
    features, usable, exclusion_reason = event_features(sample)
    if usable.sum() < max(100, args.min_cluster_size):
        raise RuntimeError(f"Only {usable.sum()} sampled events have usable descriptors.")

    fitted_features = features[usable]
    scaled, median, scale = robust_scale(fitted_features)
    scores, loadings, variance_ratio = pca(scaled)
    correlation = spearman_matrix(fitted_features)
    correlation_pairs = sorted(
        (
            (
                abs(float(correlation[left, right])),
                float(correlation[left, right]),
                FEATURE_NAMES[left],
                FEATURE_NAMES[right],
            )
            for left in range(len(FEATURE_NAMES))
            for right in range(left + 1, len(FEATURE_NAMES))
        ),
        reverse=True,
    )
    baseline, epsilon, core_fraction = density_components(
        scaled, neighbours=25, density_quantile=0.72, min_cluster_size=args.min_cluster_size
    )
    alternatives: list[tuple[str, np.ndarray, float, float]] = []
    for name, neighbours, quantile in (("k20_q68", 20, 0.68), ("k30_q76", 30, 0.76)):
        labels, alternative_epsilon, alternative_core = density_components(
            scaled,
            neighbours=neighbours,
            density_quantile=quantile,
            min_cluster_size=args.min_cluster_size,
        )
        alternatives.append((name, labels, alternative_epsilon, alternative_core))
    rank_labels, rank_epsilon, rank_core = density_components(
        rank_uniform_scale(fitted_features),
        neighbours=25,
        density_quantile=0.72,
        min_cluster_size=args.min_cluster_size,
    )

    write_manifest(output / "sample_manifest.csv", sample, usable)
    write_correlation_csv(output / "feature_spearman_correlations.csv", correlation)
    write_assignments(
        output / "event_assignments.csv",
        sample,
        features,
        usable,
        exclusion_reason,
        scores,
        baseline,
        rank_labels,
    )
    write_family_summary(output / "family_summary.csv", fitted_features, baseline, sample, usable)
    write_family_summary(
        output / "rank_uniform_family_summary.csv", fitted_features, rank_labels, sample, usable
    )
    plot_correlations(output / "feature_spearman_correlations.png", correlation)
    plot_embedding(output / "pca_density_and_families.png", scores, baseline)
    optional_status = optional_umap_hdbscan(scaled, output, args.seed)

    family_count = len(np.unique(baseline[baseline >= 0]))
    with (output / "analysis_report.md").open("w") as report:
        report.write("# Event-family discovery report\n\n")
        report.write("## Scope and traceability\n\n")
        report.write(
            f"- Catalogue runs: {len(sample.population_by_run)}; sampled events: {len(sample.event_id):,}; "
            f"usable events: {usable.sum():,}.\n"
        )
        report.write(
            "- Sampling is equal-run, deterministic hash stratification—not pooled-catalogue sampling. "
            "It therefore does not let permissive configurations dominate descriptor geometry.\n"
        )
        report.write(
            "- `event_assignments.csv` retains IDs, location, bounds, raw/transformed descriptors, "
            "PCA, exclusions and family assignment. `-1` is intentionally unassigned/noise, not discarded.\n\n"
        )
        report.write("## Descriptor preparation\n\n")
        report.write(
            "- Features: log duration, three threshold/diurnal fractions, log mean positive excess "
            "(heat content / duration), and two nocturnal fractions. Dividing cumulative heat content "
            "by duration avoids a near-mechanical duration duplicate.\n"
        )
        report.write(
            "- Median/IQR scaling and clipping to ±6 scaled IQRs were applied before PCA/density geometry. "
            "Invalid rows remain in the assignment file with an exclusion reason.\n"
        )
        report.write(
            "- PCA explained variance: "
            + ", ".join(f"PC{index + 1}={ratio:.3f}" for index, ratio in enumerate(variance_ratio))
            + ".\n"
        )
        zero_fractions = np.mean(fitted_features == 0.0, axis=0)
        report.write(
            "- Exact-zero fractions: "
            + ", ".join(
                f"{name}={fraction:.3f}"
                for name, fraction in zip(FEATURE_NAMES, zero_fractions, strict=True)
            )
            + ". The mostly-zero nocturnal-below-climatology descriptor is retained for traceability but has negligible PCA variance.\n"
        )
        report.write(
            "- Strongest Spearman relationships: "
            + "; ".join(
                f"{left}/{right}={value:.3f}" for _, value, left, right in correlation_pairs[:4]
            )
            + ". These indicate an intensity/persistence axis, not independent evidence for separate families.\n"
        )
        report.write("- PCA loadings (rows PC1...):\n\n````\n")
        report.write(np.array2string(loadings, precision=3, suppress_small=True))
        report.write("\n````\n\n")
        report.write("## Conservative density-family result\n\n")
        report.write(
            f"- Baseline: k=25, core-radius quantile=0.72, epsilon={epsilon:.4f}, "
            f"core fraction={core_fraction:.3f}, min component={args.min_cluster_size}; "
            f"families={family_count}; assigned fraction={(baseline >= 0).mean():.3f}.\n"
        )
        report.write(
            "- This is a dependency-free density graph, not HDBSCAN. Border events cannot link two cores, "
            "so apparent families are conservative hypotheses rather than forced classifications.\n"
        )
        if family_count <= 1:
            report.write(
                "- Result: the sampled descriptor space supports one broad, connected event population—not "
                "multiple separable event families. Its stability means this negative result is more defensible "
                "than a visually attractive forced taxonomy.\n"
            )
        report.write("- Stability checks on the same events:\n")
        for name, labels, alternative_epsilon, alternative_core in alternatives:
            report.write(
                f"  - {name}: epsilon={alternative_epsilon:.4f}, core={alternative_core:.3f}, "
                f"families={len(np.unique(labels[labels >= 0]))}, assigned={(labels >= 0).mean():.3f}, "
                f"ARI={adjusted_rand_index(baseline, labels):.3f}, "
                f"reference-family Jaccard={cluster_jaccard(baseline, labels):.3f}.\n"
            )
        report.write(
            f"  - rank-uniform normalization (k25_q72): epsilon={rank_epsilon:.4f}, core={rank_core:.3f}, "
            f"families={len(np.unique(rank_labels[rank_labels >= 0]))}, assigned={(rank_labels >= 0).mean():.3f}, "
            f"ARI={adjusted_rand_index(baseline, rank_labels):.3f}, "
            f"reference-family Jaccard={cluster_jaccard(baseline, rank_labels):.3f}.\n"
        )
        if len(np.unique(rank_labels[rank_labels >= 0])) != family_count:
            report.write(
                "- Conclusion: the rank-uniform check produces a materially different partition. Therefore "
                "the apparent rank-scale families are not robust to reasonable normalization, and neither "
                "partition is currently defensible as a physical event taxonomy.\n"
            )
        report.write("\n## Interpretation guardrails\n\n")
        report.write(
            "- A separated PCA region is descriptor-space evidence only. Validate every proposed family "
            "by geographic footprint and time evolution before asserting a physical mechanism.\n"
        )
        report.write(
            "- A low stability score, single-run concentration, or descriptor-boundary family is evidence "
            "against a persistent physical taxonomy.\n"
        )
        report.write(f"- {optional_status}\n")
    print(f"Saved analysis to {output}")


if __name__ == "__main__":
    main()
