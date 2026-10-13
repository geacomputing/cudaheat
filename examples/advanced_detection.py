"""Detect contiguous threshold-exceedance events in an hourly SST NetCDF file."""

from __future__ import annotations

import argparse

from cudaheat import CUDAHeat
import time
import cupy as cp

import numpy as np
import matplotlib.pyplot as plt

#import matplotlib
#matplotlib.use("qtagg")
#import matplotlib.pyplot as plt

PATH = "/home/user/Documents/PhD.cy/lazy_loading/Python.Scripts/sst_7_years.nc"
PATH = "/home/user/Documents/PhD.cy/lazy_loading/cudaHeat/sst_8_years.nc"


from cudaheat.preprocessing import optimize_netcdf


if False: 
    report = optimize_netcdf(
        PATH, 
        "sst_cudaheat.nc",
        variable="analysed_sst",
    )
    print(report)


t0 = time.perf_counter()

final= "/home/user/Documents/Gea/cudaheat/examples/sst_cudaheat.nc"
with CUDAHeat(final, variable="analysed_sst", timing_logging=False) as hw:
    result = hw.detect(
        baseline=(2019, 2023),
        percentile=.90,
        smooth_window_hours=24*7,
        min_duration_hours=120,
        max_internal_gap_hours=6, 
        join_gap_hours=48, 
        batch_size=512,
    )


cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - t0
print()
print(f"Execution time: {elapsed:.3f} s")
print()
print()