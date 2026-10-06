"""Detect contiguous threshold-exceedance events in an hourly SST NetCDF file."""

from __future__ import annotations

import argparse

from cudaheat import CUDAHeat
import time
import cupy as cp



PATH = "/home/user/Documents/PhD.cy/lazy_loading/Python.Scripts/sst_7_years.nc"
#PATH = "/home/user/Documents/PhD.cy/lazy_loading/cudaHeat/sst_8_years.nc"

t0 = time.perf_counter()


with CUDAHeat(PATH, variable="analysed_sst", dtype="float32", lazy=False) as hw:
    result = hw.detect(
        baseline=(2019, 2025),
        percentile=.90,
        smooth_window_hours=24*5,
        min_duration_hours=120,
        max_internal_gap_hours=12
    )

cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - t0
print()
print(f"Execution time: {elapsed:.3f} s")
print()
print()
if False: 
    print(result.data[["point_index", "start_time", "end_time", "duration_hours"]])
    print("\n\nEvent count per point:")
    print(result.event_count)



# year = 2023

# events_2023 = result.data[
#     ["point_index", "start_time", "end_time", "duration_hours", "event_count"]
# ].where(
#     result.data["start_time"].dt.year == year,
#     drop=True,
# )