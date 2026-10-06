"""Run CUDAHeat and plot its source data and compact output on Cartopy maps.

Install the optional plotting dependencies first::

    pip install "cudaheat[plot]"
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from cudaheat import CUDAHeat

PATH = "/home/user/Documents/PhD.cy/lazy_loading/Python.Scripts/sst_7_years.nc"
OUTPUT_FIGURE = "cudaheat_sst_and_exceedance_count.png"


def _spatial_coordinate_names(hw: CUDAHeat) -> tuple[str, str]:
    """Return latitude and longitude dimension names for a conventional SST grid."""
    latitude = [name for name in hw.spatial_dims if "lat" in name.lower()]
    longitude = [name for name in hw.spatial_dims if "lon" in name.lower()]
    if len(latitude) != 1 or len(longitude) != 1:
        raise ValueError(
            "This map example needs one latitude and one longitude dimension; "
            f"found {hw.spatial_dims!r}."
        )
    return latitude[0], longitude[0]


def _decorate_map(axis: plt.Axes, extent: list[float]) -> None:
    """Apply the same readable geographic styling to each Cartopy panel."""
    axis.set_extent(extent, crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.LAND, facecolor="0.88", zorder=2)
    axis.coastlines(resolution="10m", linewidth=0.7, zorder=3)
    gridlines = axis.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.35,
        color="0.35",
        alpha=0.55,
        linestyle="--",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 8}
    gridlines.ylabel_style = {"size": 8}


with CUDAHeat(
    PATH, variable="analysed_sst", dtype="float32", vram_safety=0.30, devices=[0]
) as hw:
    result = hw.run(baseline=(2019, 2023), percentile=0.90, smooth_window_hours=120)
    print(f"Processed {hw.ntime} hours and {hw.npoints} points")
    print(result.exceedance_count)
    for batch in result.telemetry:
        print(batch.as_dict())
        print()

    count = result.exceedance_count

    print("\nExceedance statistics")
    print("---------------------")
    print("Minimum:", count.min().item())
    print("Maximum:", count.max().item())
    print("Mean:   ", count.mean().item())
    print("Nonzero:", (count > 0).sum().item())
    print("Total:  ", count.size)

    # CUDAHeat is an engine, not an xarray object, so it has no ``hw.plot()``.
    # Its retained source is an xarray.DataArray.  Select one time slice before
    # plotting: loading the complete 3-D seven-year SST field would be wasteful.
    # ``_source`` is an internal attribute; for stable production code, open
    # PATH separately with xarray and select the same variable/time slice.
    sst_snapshot = hw._source.isel({hw.time_dim: -1}).load()
    lat_name, lon_name = _spatial_coordinate_names(hw)
    extent = [
        float(sst_snapshot[lon_name].min()),
        float(sst_snapshot[lon_name].max()),
        float(sst_snapshot[lat_name].min()),
        float(sst_snapshot[lat_name].max()),
    ]

    # ``result.exceedance_count`` is an xarray.DataArray rebuilt on the
    # original (lat, lon) grid. It is compact: no time-by-space output was kept.
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        layout="constrained",
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    sst_snapshot.plot.pcolormesh(
        ax=axes[0],
        x=lon_name,
        y=lat_name,
        transform=ccrs.PlateCarree(),
        cmap="turbo",
        robust=True,
        cbar_kwargs={
            "label": f"SST ({sst_snapshot.attrs.get('units', 'unknown units')})"
        },
    )
    snapshot_time = str(sst_snapshot[hw.time_dim].values).replace("T", " ")[:16]
    axes[0].set_title(f"{hw.variable}: {snapshot_time}")
    _decorate_map(axes[0], extent)

    count.plot.pcolormesh(
        ax=axes[1],
        x=lon_name,
        y=lat_name,
        transform=ccrs.PlateCarree(),
        cmap="magma",
        robust=True,
        cbar_kwargs={"label": "Hours above threshold"},
    )
    axes[1].set_title("CUDAHeat threshold exceedance count")
    _decorate_map(axes[1], extent)

    figure.suptitle(
        "CUDAHeat SST input and preprocessing output", fontsize=15, fontweight="bold"
    )
    #figure.savefig(OUTPUT_FIGURE, dpi=200, bbox_inches="tight")
    print(f"Saved map: {OUTPUT_FIGURE}")
    plt.show()
