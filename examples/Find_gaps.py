import numpy as np
import pandas as pd
import xarray as xr

PATH = "/home/user/Documents/PhD.cy/lazy_loading/cudaHeat/sst_8_years.nc"
VAR = "analysed_sst"


def chunk_slices(chunks):
    start = 0
    for size in chunks:
        yield slice(start, start + size)
        start += size


with xr.open_dataset(PATH, chunks="auto") as ds:
    sst = ds[VAR].transpose("time", "latitude", "longitude")

    times = pd.DatetimeIndex(sst.time.values)
    latitudes = sst.latitude.values
    longitudes = sst.longitude.values
    nlat = sst.sizes["latitude"]
    nlon = sst.sizes["longitude"]

    nan_counts = np.zeros((nlat, nlon), dtype=np.int64)
    nan_ranges = {}

    for lat_slice in chunk_slices(sst.chunksizes["latitude"]):
        for lon_slice in chunk_slices(sst.chunksizes["longitude"]):
            missing = (
                sst.isel(latitude=lat_slice, longitude=lon_slice)
                .isnull()
                .compute()
                .values
            )

            # Land = NaN throughout the complete time series: ignore it.
            ocean = ~missing.all(axis=0)

            counts = missing.sum(axis=0, dtype=np.int64)
            counts[~ocean] = 0
            nan_counts[lat_slice, lon_slice] = counts

            # Ocean cells having one or more NaN values.
            local_lat, local_lon = np.where(missing.any(axis=0) & ocean)

            for y, x in zip(local_lat, local_lon):
                missing_i = np.flatnonzero(missing[:, y, x])

                # Create contiguous NaN ranges: start, end, and duration in hours.
                split = np.flatnonzero(np.diff(missing_i) > 1)
                starts = np.r_[missing_i[0], missing_i[split + 1]]
                ends = np.r_[missing_i[split], missing_i[-1]]

                nan_ranges[(lat_slice.start + y, lon_slice.start + x)] = [
                    (times[start], times[end], int(end - start + 1))
                    for start, end in zip(starts, ends)
                ]

# One row per ocean point containing one or more NaNs.
nan_table = pd.DataFrame([
    {
        "point_index": lat_i * nlon + lon_i,
        "latitude": latitudes[lat_i],
        "longitude": longitudes[lon_i],
        "nan_count": nan_counts[lat_i, lon_i],
        "nan_ranges": nan_ranges[(lat_i, lon_i)],
    }
    for lat_i, lon_i in nan_ranges
]).sort_values("nan_count", ascending=False).reset_index(drop=True)

nan_table

export = nan_table.copy()

export["nan_ranges"] = export["nan_ranges"].apply(
    lambda ranges: [
        {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "hours": hours,
        }
        for start, end, hours in ranges
    ]
)

export.set_index("point_index").to_json(
    "nan_counts.json",
    orient="index",
    indent=2,
)