"""Detect contiguous threshold-exceedance events in an hourly SST NetCDF file."""

from __future__ import annotations

from itertools import product

import time

import cupy as cp
import matplotlib.pyplot as plt

from cudaheat import CUDAHeat, DuplicateRunError, load_postgres_config, persist_detection_result
from cudaheat.preprocessing import optimize_netcdf


#import matplotlib
#matplotlib.use("qtagg")
#import matplotlib.pyplot as plt

#PATH = "/home/user/Documents/PhD.cy/lazy_loading/Python.Scripts/sst_7_years.nc"
#PATH = "/home/user/Documents/PhD.cy/lazy_loading/cudaHeat/sst_8_years.nc"
PATH = "/home/user/Documents/Gea/cudaheat/benchmarks/extend_NETCDF_create40_years_timespan/sst_40_years.nc"

if False: 
    report = optimize_netcdf(
        PATH, 
        "/home/user/Documents/Gea/cudaheat/benchmarks/extend_NETCDF_create40_years_timespan/sst_40yrs_optimzed.nc",
        variable="analysed_sst",
    )
    print(report)





if True:
    final = "/home/user/Documents/Gea/cudaheat/examples/sst_cudaheat.nc"
    config = load_postgres_config(
        "/home/user/Documents/Gea/cudaheat/config/postgres.ini"
    )

    # Fill the missing cells in the max-internal-gap × join-gap sensitivity grid.
    missing_configurations = tuple(
        product((1, 2, 4, 8, 16, 24, 36, 48), (3, 6, 12, 24, 48, 72))
    )

    for number, (max_internal_gap_hours, join_gap_hours) in enumerate(
        missing_configurations, start=1
    ):
        t0 = time.perf_counter()
        try:
            with CUDAHeat(
                final,
                variable="analysed_sst",
                timing_logging=False,
                input_optimization="auto",
                batch_size="guess",
            ) as hw:
                result = hw.detect(
                    baseline=(2019, 2023),
                    percentile=0.9,
                    min_duration_hours=120,
                    max_internal_gap_hours=max_internal_gap_hours,
                    join_gap_hours=join_gap_hours,
                )

            run_id = persist_detection_result(result, config)
        except DuplicateRunError as exc:
            print(f"[{number:02d}/{len(missing_configurations)}] Already saved: {exc.existing_run_id}")
        except Exception as exc:
            print(
                f"[{number:02d}/{len(missing_configurations)}] "
                f"FAILED gap={max_internal_gap_hours}, join={join_gap_hours}: {exc}"
            )
        else:
            elapsed = time.perf_counter() - t0
            print(
                f"[{number:02d}/{len(missing_configurations)}] saved {run_id} "
                f"gap={max_internal_gap_hours}, join={join_gap_hours} "
                f"({elapsed:.1f} s; batch={hw.batch_size})"
            )
        finally:
            cp.cuda.Stream.null.synchronize()
            plt.close("all")

