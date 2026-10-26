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
# Representative ocean targets spanning the western, central, southern, and
# eastern parts of the domain. They are sorted by their *resolved* longitude
# before plotting, rather than by these approximate target coordinates.
HEATMAP_TARGETS = (
    # Open-water locations (at least ~150 km from the nearest coast), ordered
    # to show the transition from the western basin to the far east.
    ("W", 20.5, 36.0),
    ("CW", 24.5, 33.5),
    ("C", 27.5, 33.5),
    ("CE", 29.0, 33.5),
    ("E", 31.5, 34.5),
    ("FE", 33.25, 34.2),
    ("FNE", 33.95, 35.75),
    #("EX", 34.35, 35.2),
)
MAP_EXTENT = (20, 36.5, 30.5, 41)
# Wide enough to fill a 16:9 slide while preserving the map aspect.
SLIDE_MAP_EXTENT = (20, 38.2, 30.5, 40.7)
MONTH_LENGTHS = np.array((31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31))
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--output", type=Path, help="PNG output path")
    parser.add_argument(
        "--variability-output",
        type=Path,
        help="PNG output for the exploratory full-domain variability map",
    )
    return parser.parse_args()


def resolve_locations(
    group: zarr.hierarchy.Group,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Resolve and order representative locations from west to east."""
    store_latitude = np.asarray(group["latitude"][:])
    store_longitude = np.asarray(group["longitude"][:])
    marks, target_longitude, target_latitude = zip(*HEATMAP_TARGETS, strict=True)
    positions = np.array(
        [
            np.argmin((store_longitude - longitude) ** 2 + (store_latitude - latitude) ** 2)
            for longitude, latitude in zip(target_longitude, target_latitude, strict=True)
        ],
        dtype=np.intp,
    )
    if np.unique(positions).size != positions.size:
        raise ValueError("Two representative targets resolve to the same Zarr point")
    order = np.argsort(store_longitude[positions], kind="stable")
    positions = positions[order]
    latitude = store_latitude[positions]
    longitude = store_longitude[positions]
    marks = tuple(np.asarray(marks)[order])
    labels = tuple(
        f"{mark} · {lon:.1f}°E, {lat:.1f}°N"
        for mark, lon, lat in zip(marks, longitude, latitude, strict=True)
    )
    return positions, latitude, longitude, labels, marks


def daily_thermal_amplitude(
    group: zarr.hierarchy.Group,
    field: str,
    *,
    day_start_hour: float = 6.0,
    day_end_hour: float = 18.0,
    local_solar_time: bool = True,
) -> xr.DataArray:
    """Return daily daytime-maximum minus nighttime-minimum temperature.

    Existing calls using only ``group`` and ``field`` remain valid. By default,
    daylight is 06:00–18:00 local solar time for each point; set
    ``local_solar_time=False`` to interpret the two hour limits in UTC.
    """
    if not (0 <= day_start_hour < 24 and 0 <= day_end_hour < 24):
        raise ValueError("day_start_hour and day_end_hour must be in [0, 24)")
    if day_start_hour == day_end_hour:
        raise ValueError("day_start_hour and day_end_hour must differ")

    positions, latitude, longitude, names, marks = resolve_locations(group)
    points = np.asarray(group["point"].oindex[positions])
    hourly = np.asarray(group[field].oindex[:, positions]).reshape(366, 24, len(points))
    utc_hour = np.arange(24, dtype=float)[:, None]
    clock_hour = utc_hour + longitude[None, :] / 15 if local_solar_time else utc_hour
    clock_hour = np.mod(clock_hour, 24)
    if day_start_hour < day_end_hour:
        day_mask = (clock_hour >= day_start_hour) & (clock_hour < day_end_hour)
    else:
        day_mask = (clock_hour >= day_start_hour) | (clock_hour < day_end_hour)
    night_mask = ~day_mask
    coordinates = {
        "dayofyear": np.arange(1, 367),
        "point": points,
        "location": ("point", list(names)),
        "mark": ("point", list(marks)),
        "latitude": ("point", latitude),
        "longitude": ("point", longitude),
    }
    return xr.DataArray(
        np.nanmax(np.where(day_mask[None, :, :], hourly, np.nan), axis=1)
        - np.nanmin(np.where(night_mask[None, :, :], hourly, np.nan), axis=1),
        dims=("dayofyear", "point"),
        coords=coordinates,
        name=f"{field}_daily_thermal_amplitude",
    )


def threshold_variability_score(
    group: zarr.hierarchy.Group,
    *,
    field: str = "threshold",
    point_batch_size: int = 1024,
) -> xr.DataArray:
    """Return an exploratory equal-weight threshold-amplitude swing score.

    The score is ``0.5 * robust-normalised P90 daily amplitude + 0.5 *
    robust-normalised temporal standard deviation`` at every ocean point.
    P90 avoids a single calendar-day maximum dominating the high-amplitude
    component; standard deviation identifies locations whose amplitude changes
    strongly through the threshold cycle.
    """
    points = np.asarray(group["point"][:])
    latitude = np.asarray(group["latitude"][:])
    longitude = np.asarray(group["longitude"][:])
    upper_amplitude = np.full(points.size, np.nan)
    deviation = np.full(points.size, np.nan)
    utc_hour = np.arange(24, dtype=float)[:, None]

    for start in range(0, points.size, point_batch_size):
        stop = min(start + point_batch_size, points.size)
        batch_longitude = longitude[start:stop]
        hourly = np.asarray(group[field][:, start:stop]).reshape(366, 24, stop - start)
        local_hour = np.mod(utc_hour + batch_longitude[None, :] / 15, 24)
        daytime = (local_hour >= 6) & (local_hour < 18)
        daily = (
            np.nanmax(np.where(daytime[None, :, :], hourly, np.nan), axis=1)
            - np.nanmin(np.where(~daytime[None, :, :], hourly, np.nan), axis=1)
        )
        upper_amplitude[start:stop] = np.nanpercentile(daily, 90, axis=0)
        deviation[start:stop] = np.nanstd(daily, axis=0)

    def normalise(values: np.ndarray) -> np.ndarray:
        lower, upper = np.nanpercentile(values, (5, 95))
        return np.clip((values - lower) / (upper - lower), 0, 1)

    score = 0.5 * normalise(upper_amplitude) + 0.5 * normalise(deviation)
    return xr.DataArray(
        score,
        dims="point",
        coords={
            "point": points,
            "latitude": ("point", latitude),
            "longitude": ("point", longitude),
        },
        name="threshold_thermal_amplitude_variability_score",
        attrs={
            "long_name": "Equal-weight robust-normalized P90 and temporal standard deviation of daily threshold thermal amplitude",
            "upper_amplitude_quantile": 0.9,
            "upper_amplitude_weight": 0.5,
            "standard_deviation_weight": 0.5,
        },
    )


def add_month_axis(axis: plt.Axes) -> None:
    """Add month labels above a 366-day calendar axis."""
    month_axis = axis.secondary_xaxis("top")
    month_starts = np.r_[1, 1 + np.cumsum(MONTH_LENGTHS[:-1])]
    month_axis.set_xticks(month_starts + (MONTH_LENGTHS - 1) / 2)
    month_axis.set_xticklabels(MONTH_NAMES, fontsize=11)
    month_axis.set_xlabel("Month", fontsize=12)


def plot_threshold_heatmap(amplitude: xr.DataArray, output: Path) -> None:
    """Draw threshold thermal amplitude for a west-to-east location transect."""
    figure = plt.figure(figsize=(26, 10), layout="constrained")
    figure.suptitle(
        "Eastward gradient in daily thermal amplitude at the MHW threshold",
        fontsize=20,
        fontweight="bold",
    )
    # The wider map column preserves the geographic aspect while matching the
    # heatmap panel height.
    grid = figure.add_gridspec(1, 3, width_ratios=(1.05, 0.055, 1.70), wspace=0.09)
    amplitude_axis = figure.add_subplot(grid[0])
    colour_axis = figure.add_subplot(grid[1])
    map_axis = figure.add_subplot(grid[2], projection=ccrs.PlateCarree())
    image = amplitude_axis.imshow(
        amplitude.values.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="Reds",
        vmin=float(np.nanpercentile(amplitude.values, 50)),
        vmax=float(1.1*np.nanmax(amplitude.values)),
        extent=(0.5, 366.5, -0.5, amplitude.sizes["point"] - 0.5),
    )
    amplitude_axis.set(
        xlim=(1, 366),
        xlabel="Day of year",
        ylabel="Sample location (west → east)",
        yticks=np.arange(amplitude.sizes["point"]),
        yticklabels=amplitude.location.values,
    )
    amplitude_axis.tick_params(axis="x", labelsize=11)
    amplitude_axis.tick_params(axis="y", labelsize=14)
    amplitude_axis.xaxis.label.set_size(16)
    amplitude_axis.yaxis.label.set_size(16)
    add_month_axis(amplitude_axis)
    colourbar = figure.colorbar(image, cax=colour_axis)
    colourbar.set_label("Threshold thermal amplitude (°C)", fontsize=12)
    colourbar.ax.tick_params(labelsize=10)

    map_axis.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.set_facecolor("#dcecf5")
    map_axis.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#efede7", zorder=0)
    map_axis.coastlines(resolution="10m", linewidth=0.7)
    map_axis.gridlines(draw_labels=False, linewidth=0.35, color="white", alpha=0.6, zorder= 10)



        
    special_names = {"E", "FE", "FNE"}

    for mark, lon, lat in zip(
        amplitude.mark.values,
        amplitude.longitude.values,
        amplitude.latitude.values,
        strict=True,
    ):
        is_special = mark in special_names

        map_axis.scatter(
            lon, lat,
            s=480 if is_special else 130,
            marker="*" if is_special else "s",
            color="#c80707" if is_special else "#0071B2",
            edgecolor="#00070B",
            linewidth=0.65,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        map_axis.annotate(
            mark,
            xy=(lon, lat),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
    map_axis.set_title("Sampled locations", fontsize=15, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


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
    scatter = axis.scatter(
        score.longitude,
        score.latitude,
        c=score,
        s=9,
        cmap="Reds",
        vmin=float(np.nanpercentile(score.values, 50)),
        vmax=float(np.nanmax(score.values)),
        linewidths=0,
        transform=ccrs.PlateCarree(),
        rasterized=True,
        zorder=4,
    )
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
    colourbar = figure.colorbar(scatter, cax=colour_axis, orientation="horizontal")
    colourbar.set_label("Normalized Score", color="white", fontsize=13, labelpad=2)
    colourbar.ax.tick_params(colors="white", labelsize=14, length=3)
    colourbar.outline.set_edgecolor("white")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=480, facecolor="black")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    if not arguments.store.is_dir():
        raise FileNotFoundError(f"Zarr store does not exist: {arguments.store}")
    group = zarr.open_group(arguments.store, mode="r")

    output = arguments.output or Path(__file__).with_name("climatology_threshold_thermal_amplitude.png")
    variability_output = arguments.variability_output or Path(__file__).with_name(
        "threshold_thermal_amplitude_variability_map.png"
    )
    threshold = daily_thermal_amplitude(group, "threshold")
    plot_threshold_heatmap(threshold, output)
    plot_variability_map(threshold_variability_score(group), variability_output)
    print(f"Saved {output}")
    print(f"Saved {variability_output}")
    print(threshold.to_pandas().describe())


if __name__ == "__main__":
    main()
