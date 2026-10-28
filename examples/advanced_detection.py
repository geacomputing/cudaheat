"""Run a small parameter sweep and persist each CUDAHeat catalogue."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from cudaheat import CUDAHeat, DuplicateRunError, load_postgres_config, persist_detection_result

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/home/user/Documents/PhD.cy/lazy_loading/cudaHeat/sst_8_years.nc")
DEFAULT_POSTGRES_CONFIG = PROJECT_ROOT / "config/postgres.ini"

# Ordered so a short test run varies gap/join settings immediately. All share
# the same climatology configuration and therefore reuse one Zarr product.
TEST_CONFIGURATIONS = (
    (120, 14, 24),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100, help="number of test runs (default: 100)")
    parser.add_argument("--offset", type=int, default=0, help="first configuration index (default: 0)")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--postgres-config", type=Path, default=DEFAULT_POSTGRES_CONFIG)
    arguments = parser.parse_args()
    if arguments.limit < 1 or arguments.offset < 0:
        parser.error("--limit must be positive and --offset must be non-negative")
    if not arguments.source.is_file():
        parser.error(f"source NetCDF does not exist: {arguments.source}")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    configurations = TEST_CONFIGURATIONS[arguments.offset : arguments.offset + arguments.limit]
    if not configurations:
        raise ValueError("--offset is beyond the available test configurations")

    postgres = load_postgres_config(arguments.postgres_config)
    print(f"Running {len(configurations)} parametric configurations from raw input: {arguments.source}")

    for number, (minimum_duration, internal_gap, join_gap) in enumerate(configurations, start=1):
        started = time.perf_counter()
        try:
            # Pass only the raw input. CUDAHeat handles central optimized-NetCDF
            # and climatology-Zarr discovery, validation, creation, and reuse.
            with CUDAHeat(
                arguments.source,
                variable="analysed_sst",
                timing_logging=False,
                batch_size="guess",
            ) as heat:
                result = heat.detect(
                    baseline=(2019, 2023),
                    percentile=0.90,
                    min_duration_hours=minimum_duration,
                    max_internal_gap_hours=internal_gap,
                    join_gap_hours=join_gap,
                    enrich=True,
                )
                batch_size = heat.batch_size

            detection_enrichment_seconds = time.perf_counter() - started
            gpu_enrichment_seconds = float(result.metadata["enrichment_timing"]["gpu_seconds"])
            run_id = persist_detection_result(result, postgres)

        except DuplicateRunError as exc:
            print(f"[{number}/{len(configurations)}] duplicate run: {exc.existing_run_id}")
            continue
        except Exception as exc:
            print(f"[{number}/{len(configurations)}] FAILED: {exc}")
            continue

        total_seconds = time.perf_counter() - started
        print(
            f"[{number}/{len(configurations)}] saved {run_id} | "
            f"min={minimum_duration}h gap={internal_gap}h join={join_gap}h | "
            f"detect+enrich={detection_enrichment_seconds:.1f}s "
            f"gpu-enrich={gpu_enrichment_seconds:.1f}s total={total_seconds:.1f}s | "
            f"batch={batch_size} zarr={result.metadata['climatology_cache_status']} "
            f"clim={result.metadata['climatology_hash']}"
        )


if __name__ == "__main__":
    main()
