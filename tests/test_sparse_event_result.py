from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cudaheat.result import EventDetectionResult, EventGrid


pytest.importorskip("dask.array")


def _result(metadata: dict[str, object] | None = None) -> EventDetectionResult:
    time = np.arange(
        np.datetime64("2000-01-01T00"),
        np.datetime64("2000-01-01T06"),
        np.timedelta64(1, "h"),
    )
    points = xr.DataArray(
        np.arange(4).reshape(2, 2),
        dims=("latitude", "longitude"),
        coords={"latitude": [34.0, 35.0], "longitude": [32.0, 33.0]},
    ).stack(point=("latitude", "longitude"))
    event_count = xr.DataArray(
        np.array([[0, 1], [1, 0]], dtype=np.uint64),
        dims=("latitude", "longitude"),
        coords={"latitude": [34.0, 35.0], "longitude": [32.0, 33.0]},
    )
    catalogue = xr.Dataset(
        {
            "point_index": ("event", [1, 2]),
            "start_index": ("event", [1, 3]),
            "stop_index": ("event", [3, 6]),
            "duration_hours": ("event", [2, 3]),
            "start_time": ("event", time[[1, 3]]),
            "end_time": ("event", time[[2, 5]]),
            "event_count": event_count,
        },
        coords={"event": [1, 2]},
    )
    grid = EventGrid(
        time=time,
        point_index=points.indexes["point"],
        point_indices=np.arange(4),
        time_dim="time",
    )
    return EventDetectionResult(data=catalogue, telemetry=(), metadata=metadata or {}, _grid=grid)


def test_sparse_catalogue_reconstructs_a_chunked_dense_event_cube():
    dense = _result().to_xarray(time_chunk_hours=2)

    assert hasattr(dense["event_id"].data, "compute")
    expected = np.zeros((6, 2, 2), dtype=np.uint32)
    expected[1:3, 0, 1] = 1
    expected[3:6, 1, 0] = 2
    np.testing.assert_array_equal(dense["event_id"].compute().values, expected)


def test_select_events_filters_the_cpu_catalogue_and_recounts_points():
    result = _result()
    selected = result.select_events(start="2000-01-01T04")

    np.testing.assert_array_equal(selected["event"].values, [2])
    np.testing.assert_array_equal(
        selected["event_count"].values, np.array([[0, 0], [1, 0]], dtype=np.uint64)
    )


def test_point_and_event_timeseries_reconstruct_only_one_source_point(tmp_path):
    time = np.arange(
        np.datetime64("2000-01-01T00"),
        np.datetime64("2000-01-01T06"),
        np.timedelta64(1, "h"),
    )
    values = np.arange(24, dtype=np.float32).reshape(6, 2, 2)
    source = xr.Dataset(
        {"analysed_sst": (("time", "latitude", "longitude"), values)},
        coords={"time": time, "latitude": [34.0, 35.0], "longitude": [32.0, 33.0]},
    )
    path = tmp_path / "source.nc"
    source.to_netcdf(path, engine="h5netcdf")
    result = _result(
        {
            "source_path": str(path),
            "variable": "analysed_sst",
            "dtype": "float32",
            "time_dimension": "time",
            "spatial_dimensions": ("latitude", "longitude"),
            "baseline": None,
            "percentile": 0.9,
            "smooth_window_hours": 1,
        }
    )

    point = result.point_timeseries(latitude=1, longitude=0)
    assert set(point.data_vars) == {"sst", "climatology", "threshold", "exceedance", "event_id"}
    np.testing.assert_array_equal(point["sst"].values, values[:, 1, 0])
    np.testing.assert_array_equal(point["event_id"].values, [0, 0, 0, 2, 2, 2])

    np.testing.assert_array_equal(result.point_events(latitude=1, longitude=0)["event"].values, [2])

    event = result.event_timeseries(2, pad_hours=1)
    np.testing.assert_array_equal(event["event_id"].values, [0, 2, 2, 2])
