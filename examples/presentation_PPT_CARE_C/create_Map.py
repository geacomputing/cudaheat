import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
    dpi=300,
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
    edgecolor="#555555",
    linewidth=0.4,
    zorder=2
)

# ============================================================
# Gridlines
# ============================================================

ax.gridlines(
    draw_labels=False,
    color="white",
    alpha=0.25,
    linewidth=0.5,
    zorder=3
)

# ============================================================
# Orthographic globe — upper-right
# ============================================================

ax_globe = fig.add_axes(
    [0.76, 0.74, 0.24, 0.24],
    projection=ccrs.Orthographic(10, 35)
)

ax_globe.set_global()
ax_globe.set_facecolor("none")

# Oceans
ax_globe.add_feature(
    cfeature.OCEAN.with_scale("10m"),
    facecolor="#DCEAF2",
    zorder=0
)

# Continents
ax_globe.add_feature(
    cfeature.LAND.with_scale("10m"),
    facecolor="black",
    edgecolor="#777777",
    linewidth=0.4,
    zorder=1
)

# Globe gridlines
ax_globe.gridlines(
    color="gray",
    alpha=0.3,
    linewidth=0.7,
    zorder=2
)

# Globe outline
ax_globe.spines["geo"].set_edgecolor("#777777")
ax_globe.spines["geo"].set_linewidth(0.8)

# ============================================================
# Red rectangle — SST domain on globe
# ============================================================

n = 100

lon = np.concatenate([
    np.linspace(lon_min, lon_max, n),
    np.full(n, lon_max),
    np.linspace(lon_max, lon_min, n),
    np.full(n, lon_min)
])

lat = np.concatenate([
    np.full(n, lat_min),
    np.linspace(lat_min, lat_max, n),
    np.full(n, lat_max),
    np.linspace(lat_max, lat_min, n)
])

ax_globe.plot(
    lon,
    lat,
    transform=ccrs.PlateCarree(),
    color="red",
    linewidth=2,
    zorder=10
)

# ============================================================
# Domain geometry label — over Turkey
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
    30.0,
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
    zorder=20
)




# ============================================================
# Study domain label — over Greece
# ============================================================

study_text = (
    "MARINE HEATWAVE (MHW) STUDY DOMAIN\n"
    "Eastern Mediterranean Sea\n"
    "8 years of hourly SST observations"
)

ax.text(
    22.35, 31.46,
    study_text,
    transform=ccrs.PlateCarree(),
    fontsize=7,
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

from matplotlib.patches import Rectangle

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

