#!/usr/bin/env python3

"""
Standalone benchmark:

PostgreSQL
    -> 100,000 spatially ordered events
    -> Zarr climatology + threshold
    -> fake enrichment
    -> timing
"""

import time
from pathlib import Path

import numpy as np
import psycopg
import zarr


# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "cudaheat",
    "user": "olddog",
    "passfile": str(Path.home() / ".pgpass"),
}

RUN_ID = "14962e13-1425-43b9-ad1b-333f0bc40e02"

ZARR_PATH = Path(
    "/home/user/Documents/Gea/cudaheat/.cudaheat-cache/"
    "91f72e631413c69bb2ef4bb2ba28c5475c0c8b9e535ffbd900855f27322f7e11/"
    "sst_8_years.cudaheat.clim-"
    "920e04990eb129a37e884f907094c39d68e1874a17d5378ab3abbc513762ff45.zarr"
)

BATCH_SIZES = [
    30000,
    50000,
    100000,
    150000,
]

MAX_EVENTS = 200_000

FAKE_COMPUTE_SECONDS = 1.05


# ============================================================
# ZARR
# ============================================================

def open_zarr_store():

    print(f"Zarr path: {ZARR_PATH}")

    if not ZARR_PATH.exists():
        raise FileNotFoundError(
            f"Zarr store not found:\n{ZARR_PATH}"
        )

    root = zarr.open_group(
        str(ZARR_PATH),
        mode="r",
    )

    print("\nZarr opened successfully.\n")

    for name in root.array_keys():
        arr = root[name]

        print(
            f"  {name:24s} "
            f"shape={arr.shape} "
            f"chunks={arr.chunks} "
            f"dtype={arr.dtype}"
        )

    if "climatology" not in root:
        raise KeyError("climatology not found")

    if "threshold" not in root:
        raise KeyError("threshold not found")

    return root


# ============================================================
# POSTGRESQL
# ============================================================

def fetch_events(conn, batch_size, offset):

    sql = """
        SELECT
            e.event_id,
            p.grid_point_index,
            e.start_time,
            e.end_time
        FROM events e
        JOIN points p
            ON p.point_id = e.point_id
        WHERE e.run_id = %s
        ORDER BY
            p.grid_point_index,
            e.start_time
        LIMIT %s
        OFFSET %s
    """

    t0 = time.perf_counter()

    with conn.cursor() as cur:

        cur.execute(
            sql,
            (
                RUN_ID,
                batch_size,
                offset,
            ),
        )

        rows = cur.fetchall()

    elapsed = time.perf_counter() - t0

    return rows, elapsed


# ============================================================
# ZARR READ
# ============================================================

def read_zarr_batch(
    climatology,
    threshold,
    rows,
):

    point_indices = np.fromiter(
        (row[1] for row in rows),
        dtype=np.int64,
    )

    # Multiple events may belong to the same point.
    point_indices = np.unique(point_indices)

    t0 = time.perf_counter()

    clim = climatology[:, point_indices]
    thres = threshold[:, point_indices]

    elapsed = time.perf_counter() - t0

    return (
        clim,
        thres,
        len(point_indices),
        elapsed,
    )


# ============================================================
# FAKE ENRICHMENT
# ============================================================

def fake_compute(clim, thres):

    t0 = time.perf_counter()

    # Force actual access to arrays.
    _ = float(np.nanmean(clim))
    _ = float(np.nanmean(thres))

    # Simulated enrichment.
    time.sleep(FAKE_COMPUTE_SECONDS)

    return time.perf_counter() - t0


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(
    conn,
    climatology,
    threshold,
    batch_size,
):

    offset = 0

    total_events = 0
    total_points = 0
    batches = 0

    postgres_time = 0.0
    zarr_time = 0.0
    compute_time = 0.0

    wall_start = time.perf_counter()

    while total_events < MAX_EVENTS:

        remaining = MAX_EVENTS - total_events

        current_size = min(
            batch_size,
            remaining,
        )

        # ----------------------------------------------------
        # PostgreSQL
        # ----------------------------------------------------

        rows, elapsed = fetch_events(
            conn,
            current_size,
            offset,
        )

        postgres_time += elapsed

        if not rows:
            break

        # ----------------------------------------------------
        # Zarr
        # ----------------------------------------------------

        (
            clim,
            thres,
            n_points,
            elapsed,
        ) = read_zarr_batch(
            climatology,
            threshold,
            rows,
        )

        zarr_time += elapsed

        # ----------------------------------------------------
        # Fake enrichment
        # ----------------------------------------------------

        elapsed = fake_compute(
            clim,
            thres,
        )

        compute_time += elapsed

        # ----------------------------------------------------
        # Accounting
        # ----------------------------------------------------

        n_events = len(rows)

        total_events += n_events
        total_points += n_points

        offset += n_events
        batches += 1

    total_time = time.perf_counter() - wall_start

    return {
        "batch_size": batch_size,
        "batches": batches,
        "events": total_events,
        "points": total_points,
        "postgres": postgres_time,
        "zarr": zarr_time,
        "compute": compute_time,
        "total": total_time,
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(r):

    print()
    print("==============================================")
    print(f"BATCH SIZE = {r['batch_size']:,}")
    print("==============================================")

    print(f"Events processed : {r['events']:,}")
    print(f"Batches          : {r['batches']:,}")
    print(f"Points read      : {r['points']:,}")

    print()

    print(f"PostgreSQL       : {r['postgres']:.3f} s")
    print(f"Zarr I/O         : {r['zarr']:.3f} s")
    print(f"Fake enrichment  : {r['compute']:.3f} s")

    print("----------------------------------------------")

    print(f"TOTAL            : {r['total']:.3f} s")

    if r["total"]:

        throughput = r["events"] / r["total"]

        print(
            f"Throughput       : "
            f"{throughput:,.0f} events/s"
        )

        print()

        print("Time distribution:")

        print(
            f"  PostgreSQL : "
            f"{100 * r['postgres'] / r['total']:.1f}%"
        )

        print(
            f"  Zarr       : "
            f"{100 * r['zarr'] / r['total']:.1f}%"
        )

        print(
            f"  Compute    : "
            f"{100 * r['compute'] / r['total']:.1f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================================")
    print("POSTGRESQL -> ZARR -> FAKE ENRICHMENT")
    print("==============================================")
    print()

    root = open_zarr_store()

    climatology = root["climatology"]
    threshold = root["threshold"]

    print()
    print("Selected arrays:")

    print(
        f"  climatology : "
        f"shape={climatology.shape}, "
        f"chunks={climatology.chunks}"
    )

    print(
        f"  threshold   : "
        f"shape={threshold.shape}, "
        f"chunks={threshold.chunks}"
    )

    print()

    print("Connecting to PostgreSQL...")
    print(
        f"Using password file: "
        f"{DB_CONFIG['passfile']}"
    )

    results = []

    with psycopg.connect(**DB_CONFIG) as conn:

        print("Connected.")

        for batch_size in BATCH_SIZES:

            result = benchmark(
                conn,
                climatology,
                threshold,
                batch_size,
            )

            results.append(result)

            print_result(result)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print()
    print("==============================================")
    print("SUMMARY")
    print("==============================================")

    print(
        f"{'Batch':>10} "
        f"{'Batches':>9} "
        f"{'PG':>10} "
        f"{'Zarr':>10} "
        f"{'Compute':>10} "
        f"{'Total':>10} "
        f"{'Events/s':>12}"
    )

    print("-" * 78)

    for r in results:

        throughput = (
            r["events"] / r["total"]
            if r["total"]
            else 0
        )

        print(
            f"{r['batch_size']:>10,} "
            f"{r['batches']:>9,} "
            f"{r['postgres']:>9.3f}s "
            f"{r['zarr']:>9.3f}s "
            f"{r['compute']:>9.3f}s "
            f"{r['total']:>9.3f}s "
            f"{throughput:>12,.0f}"
        )

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()