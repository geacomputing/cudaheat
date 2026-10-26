"""Draw a polar daily-fraction view from the cached CUDAHeat Zarr climatology.

The radius progresses from the earliest year (inside) to the latest year
(outside); theta is calendar day.  Each year has two adjacent bands: red is
the fraction of valid daily hours above the MHW threshold, while blue is the
fraction above climatology but not the threshold.

Run from the repository root:
    python examples/mhw_Pizza/mhw_pizza.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import xarray as xr
import zarr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path(__file__).with_name("mhw_pizza.png")
DEFAULT_LATITUDE = 36.5
DEFAULT_LONGITUDE = 37.0
DEFAULT_POINTS_PER_BATCH = 256
DEFAULT_DAY_TICK_INTERVAL = 15
MONTH_OFFSETS = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, help="climatology Zarr store (default: cached 2019–2023 store)")
    parser.add_argument("--source", type=Path, help="preprocessed NetCDF source paired with the store")
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--lat-min", type=float, help="southern rectangle bound; requires all rectangle bounds")
    parser.add_argument("--lat-max", type=float, help="northern rectangle bound; requires all rectangle bounds")
    parser.add_argument("--lon-min", type=float, help="western rectangle bound; requires all rectangle bounds")
    parser.add_argument("--lon-max", type=float, help="eastern rectangle bound; requires all rectangle bounds")
    parser.add_argument(
        "--points-per-batch",
        type=int,
        default=DEFAULT_POINTS_PER_BATCH,
        help="Zarr/NetCDF points processed together for a rectangle (default: 256)",
    )
    parser.add_argument(
        "--day-tick-interval",
        type=int,
        default=DEFAULT_DAY_TICK_INTERVAL,
        help="outer-ring tick interval in calendar days (default: 15)",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def find_store() -> Path:
    """Select the available cache whose baseline is 2019–2023."""
    stores = sorted((PROJECT_ROOT / ".cudaheat-cache").glob("**/*.zarr"))
    if not stores:
        raise FileNotFoundError("No Zarr climatology store was found in .cudaheat-cache")
    for store in stores:
        config = json.loads(zarr.open_group(store, mode="r").attrs["cudaheat_climatology_config"])
        if config.get("baseline") == [2019, 2023]:
            return store
    return stores[0]


def find_source(store: Path) -> Path:
    """Find the local preprocessed NetCDF placed beside this cache product."""
    sources = sorted(store.parent.glob("*.cudaheat.nc"))
    if not sources:
        raise FileNotFoundError(f"No preprocessed NetCDF source beside {store}")
    return sources[0]


def nearest_point(group: zarr.hierarchy.Group, latitude: float, longitude: float) -> tuple[int, float, float]:
    latitudes = np.asarray(group["latitude"][:])
    longitudes = np.asarray(group["longitude"][:])
    point = int(np.argmin((latitudes - latitude) ** 2 + (longitudes - longitude) ** 2))
    return point, float(latitudes[point]), float(longitudes[point])


def select_points(group: zarr.hierarchy.Group, arguments: argparse.Namespace) -> tuple[np.ndarray, str]:
    """Return either the nearest point or all points in an inclusive rectangle."""
    bounds = (arguments.lat_min, arguments.lat_max, arguments.lon_min, arguments.lon_max)
    if all(bound is None for bound in bounds):
        point, latitude, longitude = nearest_point(group, arguments.latitude, arguments.longitude)
        return np.array([point], dtype=np.intp), f"Point:\n{latitude:.3f}°N, {longitude:.3f}°E"
    if any(bound is None for bound in bounds):
        raise ValueError("--lat-min, --lat-max, --lon-min, and --lon-max must be supplied together")
    if arguments.lat_min > arguments.lat_max or arguments.lon_min > arguments.lon_max:
        raise ValueError("rectangle minimum bounds must not exceed maximum bounds")

    latitude = np.asarray(group["latitude"][:])
    longitude = np.asarray(group["longitude"][:])
    positions = np.flatnonzero(
        (latitude >= arguments.lat_min)
        & (latitude <= arguments.lat_max)
        & (longitude >= arguments.lon_min)
        & (longitude <= arguments.lon_max)
    ).astype(np.intp)
    if not positions.size:
        raise ValueError("the requested rectangle contains no valid Zarr grid points")
    description = (
        "Rectangle mean:\n"
        f"{arguments.lat_min:.2f}–{arguments.lat_max:.2f}°N\n"
        f"{arguments.lon_min:.2f}–{arguments.lon_max:.2f}°E\n"
        f"{positions.size:,} grid points"
    )
    return positions, description


def climatology_bins(time: xr.DataArray) -> np.ndarray:
    """Return CUDAHeat's fixed leap-calendar month/day/hour bin positions."""
    month = np.asarray(time.dt.month.values, dtype=np.int16)
    day = np.asarray(time.dt.day.values, dtype=np.int16)
    hour = np.asarray(time.dt.hour.values, dtype=np.int16)
    return ((MONTH_OFFSETS[month - 1] + day - 1) * 24 + hour).astype(np.intp)


def spatial_daily_fractions(
    sst: xr.DataArray,
    time: xr.DataArray,
    group: zarr.hierarchy.Group,
    positions: np.ndarray,
    points_per_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return daily fractions over one point or an equal-weight point rectangle.

    Fractions use all available point-hours, so a missing value at one grid
    point does not discard the rest of that day or rectangle. This is an
    equal-grid-cell mean, not a latitude/area-weighted mean.
    """
    if points_per_batch < 1:
        raise ValueError("--points-per-batch must be positive")
    years_per_hour = np.asarray(time.dt.year.values, dtype=int)
    month = np.asarray(time.dt.month.values, dtype=np.int16)
    day = np.asarray(time.dt.day.values, dtype=np.int16)
    calendar_day = (MONTH_OFFSETS[month - 1] + day - 1).astype(np.intp)
    years = np.arange(years_per_hour.min(), years_per_hour.max() + 1)
    year_row = years_per_hour - years[0]
    daily_index = year_row * 366 + calendar_day
    bins = climatology_bins(time)
    cells = years.size * 366
    valid_totals = np.zeros(cells, dtype=float)
    threshold_totals = np.zeros(cells, dtype=float)
    climate_only_totals = np.zeros(cells, dtype=float)

    for start in range(0, positions.size, points_per_batch):
        batch_positions = positions[start : start + points_per_batch]
        point_ids = np.asarray(group["point"].oindex[batch_positions])
        values = np.asarray(sst.sel(point=point_ids).load().values)
        climatology_cycle = np.asarray(group["climatology"].oindex[:, batch_positions])
        threshold_cycle = np.asarray(group["threshold"].oindex[:, batch_positions])
        climatology = climatology_cycle[bins]
        threshold = threshold_cycle[bins]
        valid = np.isfinite(values) & np.isfinite(climatology) & np.isfinite(threshold)
        over_threshold = valid & (values > threshold)
        climate_only = valid & (values > climatology) & ~over_threshold
        valid_totals += np.bincount(
            daily_index, weights=valid.sum(axis=1), minlength=cells
        )
        threshold_totals += np.bincount(
            daily_index, weights=over_threshold.sum(axis=1), minlength=cells
        )
        climate_only_totals += np.bincount(
            daily_index, weights=climate_only.sum(axis=1), minlength=cells
        )

    threshold_fraction = np.full(cells, np.nan, dtype=float)
    climate_only_fraction = np.full(cells, np.nan, dtype=float)
    np.divide(threshold_totals, valid_totals, out=threshold_fraction, where=valid_totals > 0)
    np.divide(climate_only_totals, valid_totals, out=climate_only_fraction, where=valid_totals > 0)
    return (
        years,
        threshold_fraction.reshape(years.size, 366),
        climate_only_fraction.reshape(years.size, 366),
    )


def draw_plot(
    years: np.ndarray,
    threshold_fraction: np.ndarray,
    climate_only_fraction: np.ndarray,
    selection_description: str,
    baseline: str,
    day_tick_interval: int,
    output: Path,
) -> None:
    if not 1 <= day_tick_interval <= 366:
        raise ValueError("--day-tick-interval must be between 1 and 366")
    theta_edges = np.linspace(0, 2 * np.pi, 367)
    month_angles = 2 * np.pi * MONTH_OFFSETS / 366
    red = plt.colormaps["Reds"].copy()
    blue = plt.colormaps["Oranges"].copy()
    red.set_bad((0, 0, 0, 0))
    blue.set_bad("#f8f8f8")
    norm = Normalize(0, 1)
    #secondary_colour_name = "gray" if blue.name.lower().startswith("gray") else "blue"
    secondary_colour_name = blue.name.lower()

    figure = plt.figure(figsize=(1.5*9, 1.5*11))
    axis = figure.add_axes((0.08, 0.30, 0.84, 0.60), projection="polar")
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_xticks(month_angles)
    axis.set_xticklabels(MONTH_NAMES, fontsize=15)
    axis.set_yticks([])
    axis.grid(False)
    axis.spines["polar"].set_color("#d7d7d7")

    inner_radius = 2.45
    band_width = 0.85
    between_year_gap = 0.14
    pitch = band_width + between_year_gap
    for row in range(years.size):
        start = inner_radius + row * pitch
        threshold_values = threshold_fraction[row : row + 1]
        climate_only_values = climate_only_fraction[row : row + 1]
        axis.pcolormesh(
            theta_edges,
            [start, start + band_width],
            np.ma.masked_where(
                ~np.isfinite(climate_only_values) | (threshold_values > 0), climate_only_values
            ),
            cmap=blue,
            norm=norm,
            shading="flat",
            edgecolors=(1, 1, 1, 0.24),
            linewidth=0.06,
            rasterized=True,
            alpha=0.6,
        )
        axis.pcolormesh(
            theta_edges,
            [start, start + band_width],
            np.ma.masked_where(~np.isfinite(threshold_values) | (threshold_values <= 0), threshold_values),
            cmap=red,
            norm=norm,
            shading="flat",
            edgecolors=(1, 1, 1, 0.24),
            linewidth=0.06,
            rasterized=True,
        )

    outer_radius = inner_radius + years.size * pitch - between_year_gap
    tick_ring_radius = outer_radius + 0.12
    tick_inner_radius = tick_ring_radius - 0.14
    axis.set_ylim(0, tick_ring_radius + 0.06)
    axis.set_yticks(np.r_[inner_radius + np.arange(years.size) * pitch, outer_radius])
    axis.set_yticklabels([])
    axis.grid(True, color="#8a8a8a", linewidth=0.65, alpha=1.0)
    tick_angles = 2 * np.pi * (np.arange(1, 367, day_tick_interval) - 1) / 366
    axis.plot(
        np.linspace(0, 2 * np.pi, 367),
        np.full(367, tick_ring_radius),
        color="#555555",
        linewidth=0.8,
        zorder=5,
    )
    for angle in tick_angles:
        axis.plot(
            (angle, angle),
            (tick_inner_radius, tick_ring_radius),
            color="#555555",
            linewidth=0.9,
            solid_capstyle="butt",
            zorder=5,
        )
    axis.bar(
        0,
        inner_radius,
        width=2 * np.pi,
        bottom=0,
        align="edge",
        color="white",
        edgecolor="none",
        zorder=3,
    )
    axis.text(
        0,
        0,
        f"{selection_description}",
        ha="center",
        va="center",
        fontsize=11,
        zorder=4,
    )
    axis.set_title(
        f"Daily Polar Heatmap of MHW Fractions — {years[0]}–{years[-1]} | baseline {baseline}\n"
        f"One band/year — red: above threshold; {secondary_colour_name}: above climatology-only",
        fontsize=17,
        pad=22,
    )

    left_bar = figure.add_axes((0.11, 0.14, 0.34, 0.018))
    right_bar = figure.add_axes((0.56, 0.14, 0.34, 0.018))
    red_bar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=red), cax=left_bar, orientation="horizontal")
    blue_bar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=blue), cax=right_bar, orientation="horizontal")
    for bar, label in (
        (red_bar, "Fraction of daily hours\nabove percentile threshold"),
        (blue_bar, "Fraction of daily hours above\nclimatology only (no threshold exceedance)"),
    ):
        bar.set_ticks(np.linspace(0, 1, 6))
        bar.set_ticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 6)])
        bar.ax.set_title(label, fontsize=13, pad=25)
    figure.text(0.28, 0.10, "Potential MHW event", ha="center", fontsize=16)
    figure.text(0.73, 0.10, "Not an event", ha="center", fontsize=16)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    store = arguments.store or find_store()
    source = arguments.source or find_source(store)
    if not store.is_dir():
        raise FileNotFoundError(f"Zarr store does not exist: {store}")
    if not source.is_file():
        raise FileNotFoundError(f"NetCDF source does not exist: {source}")

    group = zarr.open_group(store, mode="r")
    config = json.loads(group.attrs["cudaheat_climatology_config"])
    baseline = "–".join(map(str, config["baseline"]))
    positions, selection_description = select_points(group, arguments)
    with xr.open_dataset(source) as dataset:
        time = dataset.time.load()
        years, threshold_fraction, climate_only_fraction = spatial_daily_fractions(
            dataset[config["variable"]],
            time,
            group,
            positions,
            arguments.points_per_batch,
        )
    draw_plot(
        years,
        threshold_fraction,
        climate_only_fraction,
        selection_description,
        baseline,
        arguments.day_tick_interval,
        arguments.output,
    )
    print(f"Saved {arguments.output}")
    print(f"Store: {store.name}; {selection_description.replace(chr(10), '; ')}")


if __name__ == "__main__":
    main()
