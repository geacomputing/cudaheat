import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from matplotlib.path import Path

# ============================================================
# Load dataset
# ============================================================

final = "/home/user/Documents/PhD.cy/lazy_loading/cudaHeat/sst_8_years.nc"

ds = xr.open_dataset(final)

# ============================================================
# Plot configuration
# ============================================================

plt.close("all")
plt.style.use("default")

fig = plt.figure(
    figsize=(13.333333, 7.5),
    facecolor="white"
)

# ============================================================
# Main map — full figure, no margins
# ============================================================

ax = fig.add_axes(
    [0, 0, 1, 1],
    projection=ccrs.PlateCarree()
)

ax.set_aspect("auto")
ax.set_facecolor("white")

# ============================================================
# SST data and grid geometry
# ============================================================

sst = ds.analysed_sst.isel(time=0)

lon_name = "longitude" if "longitude" in sst.coords else "lon"
lat_name = "latitude" if "latitude" in sst.coords else "lat"

lons = sst[lon_name].values
lats = sst[lat_name].values

lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
lat_min, lat_max = float(np.min(lats)), float(np.max(lats))

# Grid dimensions
nx = len(lons)
ny = len(lats)

npoints = nx * ny

# Spatial resolution (degrees)
dx = float(np.median(np.abs(np.diff(lons))))
dy = float(np.median(np.abs(np.diff(lats))))

print(f"Grid: {nx} × {ny}")
print(f"Total points: {npoints:,}")
print(f"Resolution: {dx:.4f}° × {dy:.4f}°")

# ============================================================
# Main map extent
# ============================================================

ax.set_extent(
    [lon_min, lon_max, lat_min, lat_max],
    crs=ccrs.PlateCarree()
)

# ============================================================
# SST field
# ============================================================

im = sst.plot(
    ax=ax,
    x=lon_name,
    y=lat_name,
    transform=ccrs.PlateCarree(),
    cmap="Blues",
    add_colorbar=False,
    add_labels=False,
    zorder=1
)

# ============================================================
# Black continents
# ============================================================

ax.add_feature(
    cfeature.LAND.with_scale("10m"),
    facecolor="#050505",
    edgecolor="#6F6F6F",
    linewidth=0.4,
    zorder=2
)

# ============================================================
# Gridlines
# ============================================================

ax.gridlines(
    draw_labels=False,
    color="white",
    alpha=0.35,
    linewidth=0.5,
    zorder=3
)

# ============================================================
# Orthographic inset panel — progressive view of the study domain
# ============================================================

panel = fig.add_axes([0.69, 0.70, 0.29, 0.27], zorder=4)
panel.set_facecolor((0.015, 0.025, 0.04, 0.90))
panel.set_xticks([])
panel.set_yticks([])
for spine in panel.spines.values():
    spine.set_edgecolor("#688096")
    spine.set_linewidth(0.8)

panel.text(
    0.5,
    0.96,
    "STUDY DOMAIN CONTEXT",
    color="white",
    fontsize=7,
    fontweight="bold",
    ha="center",
    va="top",
    transform=panel.transAxes,
)

# Cyan distinguishes the SST domain from the red Cyprus study-area box.
domain_color = "#11FF00"
n = 100
domain_lon = np.concatenate([
    np.linspace(lon_min, lon_max, n),
    np.full(n, lon_max),
    np.linspace(lon_max, lon_min, n),
    np.full(n, lon_min),
])
domain_lat = np.concatenate([
    np.full(n, lat_min),
    np.linspace(lat_min, lat_max, n),
    np.full(n, lat_max),
    np.linspace(lat_max, lat_min, n),
])


def add_domain_globe(bounds, central_longitude, central_latitude, label, extent=None):
    globe = fig.add_axes(
        bounds,
        projection=ccrs.Orthographic(central_longitude, central_latitude),
        zorder=5
    )
    globe.set_facecolor("none")
    globe.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#DCEAF2", zorder=0)
    globe.add_feature(
        cfeature.LAND.with_scale("10m"),
        facecolor="black",
        edgecolor="#777777",
        linewidth=0.3,
        zorder=1,
    )
    globe.gridlines(color="gray", alpha=0.3, linewidth=0.5, zorder=2)
    globe.plot(
        domain_lon,
        domain_lat,
        transform=ccrs.PlateCarree(),
        color=domain_color,
        linewidth=1.7,
        zorder=10,
    )
    globe.spines["geo"].set_edgecolor("#9AB0C0")
    globe.spines["geo"].set_linewidth(0.7)
    if extent is None:
        globe.set_global()
    else:
        globe.set_extent(extent, crs=ccrs.PlateCarree())
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = np.column_stack([np.sin(theta), np.cos(theta)]) * 0.5 + 0.5
        globe.set_boundary(
            Path(circle),
            transform=globe.transAxes,
        )
    panel.text(
        (bounds[0] - 0.69 + bounds[2] / 2) / 0.29,
        0.05,
        label,
        color="#DCEAF2",
        fontsize=6,
        ha="center",
        va="bottom",
        transform=panel.transAxes,
    )
    return globe


# A global overview followed by an orthographic close-up of the Eastern Mediterranean.
overview_globe = add_domain_globe([0.705, 0.735, 0.125, 0.185], 10, 35, "OVERVIEW")
detail_globe = add_domain_globe(
    [0.842, 0.735, 0.125, 0.185],
    28,
    36,
    "EASTERN MEDITERRANEAN",
    extent=[5, 52, 12, 59],
)

# ============================================================
# Domain geometry label — shifted left to clear the inset panel
# ============================================================

geometry_text = (
    f"Grid: {nx} × {ny}\n"
    f"Points: {npoints:,} (LAND+WATER)\n"
    f"Δx: {dx:.4f}°\n"
    f"Δy: {dy:.4f}°\n"
    f"\n"
    f"Longitude: {lon_min:.2f}° – {lon_max:.2f}°\n"
    f"Latitude:  {lat_min:.2f}° – {lat_max:.2f}°"
)

ax.text(
    29.050,
    38.75,
    geometry_text,
    transform=ccrs.PlateCarree(),
    fontsize=11,
    fontweight="bold",
    color="white",
    fontfamily="monospace",
    va="center",
    ha="center",
    linespacing=1.4,
    bbox=dict(facecolor="#050505", edgecolor="none", alpha=0.9, pad=0.5),
    zorder=20
)


ax.text(
    30,
    38.775,
    "≈6km",
    transform=ccrs.PlateCarree(),
    fontsize=11,
    fontweight="bold",
    color="red",
    fontfamily="monospace",
    va="center",
    ha="center",
    linespacing=1.4,
    zorder=20
)




# ============================================================
# Study domain label — over Greece
# ============================================================

study_text = (
    " STUDY DOMAIN\n"
    "Eastern Mediterranean Sea\n"
    "8 years of hourly SST observations"
)

ax.text(
    21.95, 31.86,
    study_text,
    transform=ccrs.PlateCarree(),
    fontsize=11,
    #fontweight="bold",
    color="white",
    fontfamily="sans-serif",
    va="center",
    ha="center",
    linespacing=1.5,
    zorder=20
)


# ============================================================
# Future study area — Cyprus
# ============================================================

# Approximate geographical boundaries
cyprus_lon_min = 31.5
cyprus_lon_max = 35
cyprus_lat_min = 33.75
cyprus_lat_max = 36.0

# Rectangle
roi = Rectangle(
    (cyprus_lon_min, cyprus_lat_min),
    cyprus_lon_max - cyprus_lon_min,
    cyprus_lat_max - cyprus_lat_min,
    transform=ccrs.PlateCarree(),
    fill=False,
    edgecolor="red",
    linewidth=1.5,
    linestyle="-",
    zorder=15
)

ax.add_patch(roi)

# Label
if False:
    ax.text(
        33.5,
        33.3,
        "FUTURE STUDY AREA",
        transform=ccrs.PlateCarree(),
        fontsize=12,
        fontweight="bold",
        color="red",
        ha="center",
        va="top",
        zorder=20
    )

# ============================================================
# Save and display
# ============================================================

from pathlib import Path

output = Path(__file__).resolve().parent / "Map.png"


plt.savefig(
    output,
    dpi=500,
    transparent=False,
    bbox_inches=None,
    pad_inches=0
)

