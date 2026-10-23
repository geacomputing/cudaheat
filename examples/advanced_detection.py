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

#PATH = "/home/user/Documents/PhD.cy/lazy_loading/Python.Scripts/sst_7_years.nc"
#PATH = "/home/user/Documents/PhD.cy/lazy_loading/cudaHeat/sst_8_years.nc"
PATH = "/home/user/Documents/Gea/cudaheat/benchmarks/extend_NETCDF_create40_years_timespan/sst_40_years.nc"

from cudaheat.preprocessing import optimize_netcdf


if False: 
    report = optimize_netcdf(
        PATH, 
        "/home/user/Documents/Gea/cudaheat/benchmarks/extend_NETCDF_create40_years_timespan/sst_40yrs_optimzed.nc",
        variable="analysed_sst",
    )
    print(report)





if True:
    t0 = time.perf_counter()

    final= "/home/user/Documents/Gea/cudaheat/examples/sst_cudaheat.nc"
    #final = "/home/user/Documents/Gea/cudaheat/benchmarks/extend_NETCDF_create40_years_timespan/sst_40yrs_optimzed.nc"

    with CUDAHeat(final,
                variable="analysed_sst",
                timing_logging=False,
                input_optimization="auto",
                batch_size="guess") as hw:
        result = hw.detect(
            baseline=(2019, 2023),
            percentile=.9,
            #smooth_window_hours=24*10,
            min_duration_hours=120,
            max_internal_gap_hours=0, 
            join_gap_hours=0, 
            retain_fields={"climatology", "threshold_cycle"}
        )


    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    print()
    print(f"Execution time    : {elapsed:.3f} s")
    print(f"Used batch Size   : {hw.batch_size}  [requested batch size: {hw.requested_batch_size}]")
    print(f"VRAM Safety Margin: {hw.vram_safety}")
    print()
    print()




    for pts in [2000, 4000, 6000, 8000, 10000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 22000]:
    
        series = result.event_timeseries(event_id=pts, pad_hours=0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        series.climatology.plot(ax=ax, label="climatology")
        series.sst.plot(ax=ax, label="sst")
        series.threshold.plot(ax=ax, label="threshold")
        plt.legend()
        plt.plot()