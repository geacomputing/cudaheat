"""Plot daily thermal amplitude from a cached climatology or threshold cycle.

The selected Zarr store is the only data source. For each point and calendar
day, the plotted value is the maximum minus minimum of the 24 cached hourly
climatology (or threshold) values. No actual SST time series is read.

Run from the repository root:
    python "examples/daily_amplitude/Create_daily amplitude_plot.py"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = PROJECT_ROOT / (
    ".cudaheat-cache/91f72e631413c69bb2ef4bb2ba28c5475c0c8b9e535ffbd900855f27322f7e11/"
    "sst_8_years.cudaheat.clim-920e04990eb129a37e884f907094c39d68e1874a17d5378ab3abbc513762ff45.zarr"
    #"sst_8_years.cudaheat.clim-dbad02e351ebff033a984302bfed253af96712b7d58b2eb69598df6443f95d85.zarr"
    
)

MAP_EXTENT = (20, 36.5, 30.5, 41)
# Wide enough to fill a 16:9 slide while preserving the map aspect.
SLIDE_MAP_EXTENT = (20, 38.2, 30.5, 40.7)


ds = xr.open_zarr(DEFAULT_STORE)["threshold"]
print(ds)

def plot_variability_map(score: xr.DataArray, output: Path) -> None:
    """Draw the exploratory full-domain threshold-amplitude swing map."""
    figure = plt.figure(figsize=(16, 9), facecolor="black")
    axis = figure.add_axes((0.07, 0.055, 0.90, 0.90), projection=ccrs.PlateCarree())
    axis.set_extent(SLIDE_MAP_EXTENT, crs=ccrs.PlateCarree())
    axis.set_aspect("equal", adjustable="box")
    axis.set_facecolor("black")
    axis.add_feature(cfeature.LAND.with_scale("10m"), facecolor="black", zorder=0)
    axis.coastlines(resolution="10m", linewidth=0.65, color="white", zorder=3)
    gridlines = axis.gridlines(
        draw_labels=True,
        xlocs=np.arange(20, 39, 2),
        ylocs=np.arange(31, 42, 2),
        linewidth=0.45,
        color="white",
        alpha=0.4,
        zorder=0,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 8, "color": "white"}
    gridlines.ylabel_style = {"size": 8, "color": "white"}

    axis.text(
        0.7,
        0.945,
        "MHW-threshold thermal-amplitude swing score\n"
        "0.5 × normalized amplitude + 0.5 × normalized stdv",
        color="white",
        fontsize=17,
        #fontweight="bold",
        ha="center",
        va="top",
        transform=axis.transAxes,
    )
    colour_axis = axis.inset_axes((0.475, 0.85, 0.45, 0.015))
    #colourbar = figure.colorbar(scatter, cax=colour_axis, orientation="horizontal")
    # colourbar.set_label("Normalized Score", color="white", fontsize=13, labelpad=2)
    # colourbar.ax.tick_params(colors="white", labelsize=14, length=3)
    # colourbar.outline.set_edgecolor("white")
    #output.parent.mkdir(parents=True, exist_ok=True)
    #figure.savefig(output, dpi=480, facecolor="black")
    #plt.close(figure)



