from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def synthetic_sst() -> xr.Dataset:
    time = np.arange(
        np.datetime64("2000-01-01T00"), np.datetime64("2000-01-04T00"), np.timedelta64(1, "h")
    )
    lat = np.array([34.0, 35.0])
    lon = np.array([32.0, 33.0, 34.0])
    seasonal = np.sin(np.arange(time.size)[:, None, None] / 12)
    gradient = lat[None, :, None] * 0.01 + lon[None, None, :] * 0.02
    values = (20 + seasonal + gradient).astype("float64")
    values[5, 0, 1] = np.nan
    return xr.Dataset(
        {"analysed_sst": (("time", "lat", "lon"), values)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
