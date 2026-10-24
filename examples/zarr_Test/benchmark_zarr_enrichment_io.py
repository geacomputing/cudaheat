"""Standalone benchmark of Zarr reads used during event enrichment.

Run directly from any working directory:
    python benchmark_zarr_enrichment_io.py

This script reads only ``climatology`` and ``threshold``.  It does not import
CUDAHeat, Dask, or PostgreSQL.
"""

from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import zarr


# ---------------------------------------------------------------------------
# Edit only this section for a different machine, store, or event sample.
# ---------------------------------------------------------------------------

CACHE_DIRECTORY = Path(
    "/home/user/Documents/Gea/cudaheat/.cudaheat-cache/"
    "91f72e631413c69bb2ef4bb2ba28c5475c0c8b9e535ffbd900855f27322f7e11"
)

ZARR_HASHES = (
    "dbad02e351ebff033a984302bfed253af96712b7d58b2eb69598df6443f95d85",
    "920e04990eb129a37e884f907094c39d68e1874a17d5378ab3abbc513762ff45",
)

# Example event requests.  Replace these with point indices and half-open
# climatology-hour intervals exported from your catalogue, if desired.
# The intervals are positions along the climatology_hour axis: [start, stop).
EVENT_POINT_INDICES = np.array([120, 865, 2_410, 8_905, 17_640], dtype=np.intp)
EVENT_INTERVALS_BY_POINT = (
    ((0, 24),),
    ((480, 600),),
    ((1_200, 1_224), (3_600, 3_672)),
    ((240, 264), (1_440, 1_464), (5_760, 5_784)),
    ((24, 48), (840, 864), (2_400, 2_424), (7_200, 7_224)),
)

# Synthetic selections used to characterize the store.  These are deliberately
# deterministic, so repeated executions use identical point/time selections.
POINT_COUNTS = (1, 10, 100, 500, 1_000, 5_000)
REPEATED_READS = 3
INDIVIDUAL_READ_MAX_POINTS = 1_000


def store_paths() -> tuple[Path, ...]:
    """Return the two explicitly configured Zarr stores."""
    return tuple(
        CACHE_DIRECTORY / f"sst_8_years.cudaheat.clim-{cache_hash}.zarr"
        for cache_hash in ZARR_HASHES
    )


def hours_from_intervals(intervals: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Expand half-open [start, stop) intervals to climatology-hour positions."""
    return np.concatenate(
        [np.arange(start, stop, dtype=np.intp) for start, stop in intervals]
    )


def time_indices(pattern: str, length: int, n_hours: int) -> np.ndarray:
    """Return n_hours positions with a selected temporal access pattern."""
    if n_hours > length:
        raise ValueError(f"requested {n_hours} hours, but the store has only {length}")

    if pattern == "continuous":
        start = (length - n_hours) // 2
        return np.arange(start, start + n_hours, dtype=np.intp)

    if pattern == "several_separated":
        blocks = 5
        block_length = max(1, n_hours // blocks)
        starts = np.linspace(0, length - block_length, blocks, dtype=np.intp)
        return np.concatenate(
            [np.arange(start, start + block_length, dtype=np.intp) for start in starts]
        )[:n_hours]

    if pattern == "many_short_separated":
        block_length = 6
        blocks = (n_hours + block_length - 1) // block_length
        starts = np.linspace(0, length - block_length, blocks, dtype=np.intp)
        return np.concatenate(
            [np.arange(start, start + block_length, dtype=np.intp) for start in starts]
        )[:n_hours]

    if pattern == "strongly_intermittent":
        rng = np.random.default_rng(20_261_025 + n_hours)
        return rng.choice(length, size=n_hours, replace=False).astype(np.intp)

    raise ValueError(f"unknown time pattern: {pattern}")


def point_indices(pattern: str, count: int, n_points: int) -> np.ndarray:
    """Return point selections with different spatial distributions."""
    if count > n_points:
        raise ValueError(f"requested {count} points, but the store has only {n_points}")

    if pattern == "contiguous":
        start = max(0, min(1_024, n_points - count))
        return np.arange(start, start + count, dtype=np.intp)

    if pattern == "moderately_scattered":
        # Evenly spaced, hence scattered but deterministic and unique.
        return np.linspace(0, n_points - 1, count, dtype=np.intp)

    if pattern == "random_scattered":
        rng = np.random.default_rng(20_261_025 + count)
        return rng.choice(n_points, size=count, replace=False).astype(np.intp)

    raise ValueError(f"unknown point pattern: {pattern}")


def read_batched(climatology, threshold, hours: np.ndarray, points: np.ndarray) -> int:
    """One orthogonal, rectangular read: all requested hours × all points."""
    climatology_values = climatology.oindex[hours, points]
    threshold_values = threshold.oindex[hours, points]
    return climatology_values.size + threshold_values.size


def read_point_by_point(climatology, threshold, hours: np.ndarray, points: np.ndarray) -> int:
    """The same rectangular request, but as many small point reads."""
    values_retrieved = 0
    for point in points:
        one_point = np.array([point], dtype=np.intp)
        climatology_values = climatology.oindex[hours, one_point]
        threshold_values = threshold.oindex[hours, one_point]
        values_retrieved += climatology_values.size + threshold_values.size
    return values_retrieved


def read_event_requests(climatology, threshold, n_hours: int) -> int:
    """Read the configured intervals for their respective points, event by event."""
    values_retrieved = 0
    for point, intervals in zip(EVENT_POINT_INDICES, EVENT_INTERVALS_BY_POINT, strict=True):
        hours = hours_from_intervals(intervals)
        if np.any(hours >= n_hours):
            raise ValueError("an EVENT_INTERVALS_BY_POINT interval is outside climatology_hour")
        one_point = np.array([point], dtype=np.intp)
        climatology_values = climatology.oindex[hours, one_point]
        threshold_values = threshold.oindex[hours, one_point]
        values_retrieved += climatology_values.size + threshold_values.size
    return values_retrieved


def timed_read(read_function) -> tuple[float, int]:
    """Time a Zarr read. Zarr returns NumPy arrays, so the I/O is complete here."""
    started = perf_counter()
    values_retrieved = read_function()
    return perf_counter() - started, values_retrieved


def benchmark(read_function) -> tuple[float, float, float, int]:
    """Return first-read, first-repeat, median-repeat seconds, and value count."""
    first_seconds, values_retrieved = timed_read(read_function)
    repeated_seconds = [timed_read(read_function)[0] for _ in range(REPEATED_READS)]
    return first_seconds, repeated_seconds[0], median(repeated_seconds), values_retrieved


def add_row(
    rows: list[dict[str, object]],
    store_name: str,
    mode: str,
    points: int,
    hours: int,
    spatial: str,
    temporal: str,
    dtype: np.dtype,
    result: tuple[float, float, float, int],
) -> None:
    first_seconds, repeated_seconds, median_seconds, values_retrieved = result
    output_mib = values_retrieved * dtype.itemsize / 1024**2
    rows.append(
        {
            "store": store_name,
            "mode": mode,
            "points": points,
            "hours": hours,
            "spatial": spatial,
            "temporal": temporal,
            "first_ms": first_seconds * 1_000,
            "repeat_ms": repeated_seconds * 1_000,
            "median_ms": median_seconds * 1_000,
            "values": values_retrieved,
            "mib_s": output_mib / median_seconds if median_seconds else float("inf"),
        }
    )


def run_rectangular_case(
    rows: list[dict[str, object]],
    store_name: str,
    climatology,
    threshold,
    points: np.ndarray,
    hours: np.ndarray,
    spatial: str,
    temporal: str,
) -> None:
    add_row(
        rows,
        store_name,
        "batched",
        len(points),
        len(hours),
        spatial,
        temporal,
        climatology.dtype,
        benchmark(lambda: read_batched(climatology, threshold, hours, points)),
    )
    if len(points) <= INDIVIDUAL_READ_MAX_POINTS:
        add_row(
            rows,
            store_name,
            "point_by_point",
            len(points),
            len(hours),
            spatial,
            temporal,
            climatology.dtype,
            benchmark(lambda: read_point_by_point(climatology, threshold, hours, points)),
        )


def print_table(rows: list[dict[str, object]]) -> None:
    print("\nCompact results (first read is only cold-ish; OS cache is not cleared):")
    print(
        f"{'store':10} {'mode':14} {'points':>6} {'hours':>6} "
        f"{'spatial':21} {'temporal':24} {'first ms':>10} {'repeat ms':>10} "
        f"{'median ms':>10} {'values':>12} {'MiB/s':>10}"
    )
    for row in rows:
        print(
            f"{row['store']:10} {row['mode']:14} {row['points']:6d} {row['hours']:6d} "
            f"{row['spatial']:21} {row['temporal']:24} {row['first_ms']:10.2f} "
            f"{row['repeat_ms']:10.2f} {row['median_ms']:10.2f} {row['values']:12d} "
            f"{row['mib_s']:10.1f}"
        )


def main() -> None:
    rows: list[dict[str, object]] = []

    for path in store_paths():
        if not path.is_dir():
            raise FileNotFoundError(f"Zarr store does not exist: {path}")

        # Opening is intentionally outside every timed section.
        group = zarr.open_group(path, mode="r")
        climatology = group["climatology"]
        threshold = group["threshold"]
        n_hours, n_points = climatology.shape
        if threshold.shape != climatology.shape:
            raise ValueError(f"shape mismatch in {path}: {climatology.shape} vs {threshold.shape}")
        if len(EVENT_POINT_INDICES) != len(EVENT_INTERVALS_BY_POINT):
            raise ValueError("EVENT_POINT_INDICES and EVENT_INTERVALS_BY_POINT must have equal lengths")
        if np.any(EVENT_POINT_INDICES < 0) or np.any(EVENT_POINT_INDICES >= n_points):
            raise ValueError("an EVENT_POINT_INDICES value is outside the point dimension")

        store_name = path.name.split("clim-")[1][:8]
        print(f"\nStore: {path}")
        print(f"climatology/threshold shape={climatology.shape}, chunks={climatology.chunks}")

        # 1 + 2: all requested point counts and all spatial distributions,
        # using a 30-day continuous event.
        thirty_days = time_indices("continuous", n_hours, 30 * 24)
        for count in POINT_COUNTS:
            for spatial in ("contiguous", "moderately_scattered", "random_scattered"):
                points = point_indices(spatial, count, n_points)
                run_rectangular_case(
                    rows, store_name, climatology, threshold, points, thirty_days,
                    spatial, "continuous_30d",
                )

        # 3: continuous events from one day through a long (180-day) event.
        random_points = point_indices("random_scattered", 1_000, n_points)
        for label, hours in (
            ("continuous_24h", 24),
            ("continuous_5d", 5 * 24),
            ("continuous_30d", 30 * 24),
            ("continuous_90d", 90 * 24),
            ("continuous_180d", 180 * 24),
        ):
            run_rectangular_case(
                rows, store_name, climatology, threshold, random_points,
                time_indices("continuous", n_hours, hours), "random_scattered", label,
            )

        # 4: the same total number of requested hours, accessed intermittently.
        for temporal in (
            "continuous",
            "several_separated",
            "many_short_separated",
            "strongly_intermittent",
        ):
            hours = time_indices(temporal, n_hours, 30 * 24)
            run_rectangular_case(
                rows, store_name, climatology, threshold, random_points, hours,
                "random_scattered", temporal,
            )

        # Actual, editable event-like input at the top of this file.
        event_hours = sum(len(hours_from_intervals(intervals)) for intervals in EVENT_INTERVALS_BY_POINT)
        add_row(
            rows,
            store_name,
            "event_by_point",
            len(EVENT_POINT_INDICES),
            event_hours,
            "provided_points",
            "provided_intervals",
            climatology.dtype,
            benchmark(lambda: read_event_requests(climatology, threshold, n_hours)),
        )

    print_table(rows)
    print("\nThroughput is requested output payload / median read time, not physical disk bytes.")


if __name__ == "__main__":
    main()
