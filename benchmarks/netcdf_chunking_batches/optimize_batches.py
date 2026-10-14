

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
PATH = "/home/user/Documents/PhD.cy/lazy_loading/cudaHeat/sst_8_years.nc"


from cudaheat.preprocessing import optimize_netcdf



report = optimize_netcdf(
    PATH, 
    "optimized.nc",
    variable="analysed_sst",
)
print(report)