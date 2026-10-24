from __future__ import annotations

import h5py
import numpy as np
import os
import pytest
import xarray as xr

from cudaheat.cache_paths import derived_cache_root
from cudaheat.io import NetCDFReader, is_optimized_point_layout
from cudaheat.preprocessing import (
    choose_chunk_shape,
    inspect_netcdf,
    optimize_netcdf,
    prepare_netcdf_input,
)
from cudaheat.result import EventDetectionResult, EventGrid


@pytest.fixture(autouse=True)
def _use_test_cache_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDAHEAT_CACHE_DIR", str(tmp_path / ".cudaheat-cache"))


def _source_dataset(*, dimensions: tuple[str, str, str] = ("time", "lat", "lon")) -> xr.Dataset:
    time = np.arange(
        np.datetime64("2001-01-01T00"), np.datetime64("2001-01-04T00"), np.timedelta64(1, "h")
    )
    values = np.arange(time.size * 2 * 3, dtype=np.float64).reshape(time.size, 2, 3)
    values[:, 0, 0] = np.nan  # permanent land
    values[7, 1, 2] = np.nan  # valid point with a partial gap
    canonical = xr.DataArray(
        values, dims=("time", "lat", "lon"), coords={"time": time, "lat": [10.0, 11.0], "lon": [20.0, 21.0, 22.0]}
    )
    return canonical.transpose(*dimensions).to_dataset(name="sst")


def test_inspection_recognizes_latitude_longitude_aliases_and_arbitrary_dimension_order(tmp_path):
    source = _source_dataset(dimensions=("lon", "time", "lat"))
    path = tmp_path / "input.nc"
    source.to_netcdf(path, engine="h5netcdf")

    inspection = inspect_netcdf(path, variable="sst")

    assert inspection.time_dim == "time"
    assert (inspection.latitude_dim, inspection.longitude_dim) == ("lat", "lon")
    assert inspection.dimensions == ("lon", "time", "lat")

    named = source.rename({"lat": "latitude", "lon": "longitude"})
    named_path = tmp_path / "named.nc"
    named.to_netcdf(named_path, engine="h5netcdf")
    named_inspection = inspect_netcdf(named_path, variable="sst")
    assert (named_inspection.latitude_coordinate, named_inspection.longitude_coordinate) == (
        "latitude",
        "longitude",
    )


def test_optimizer_compacts_land_preserves_gaps_and_topology(tmp_path):
    source = _source_dataset(dimensions=("lon", "time", "lat"))
    input_path = tmp_path / "input.nc"
    output_path = tmp_path / "optimized.nc"
    source.to_netcdf(input_path, engine="h5netcdf")

    report = optimize_netcdf(input_path, output_path, variable="sst", compression="none")

    assert report.retained_points == 5
    assert report.removed_points == 1
    assert report.points_with_gaps == 1
    assert report.missing_samples == 1
    assert report.validation_passed
    with xr.open_dataset(output_path) as optimized:
        assert optimized["sst"].dims == ("time", "point")
        assert is_optimized_point_layout(optimized["sst"], "time")
        np.testing.assert_array_equal(optimized["original_flat_index"], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(optimized["original_lat_index"], [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(optimized["original_lon_index"], [1, 2, 0, 1, 2])
        assert np.isnan(optimized["sst"].isel(time=7, point=4))

        reconstructed = np.full((source.sizes["time"], 2, 3), np.nan, dtype=np.float32)
        reconstructed[:, optimized["original_lat_index"], optimized["original_lon_index"]] = optimized["sst"].values
        np.testing.assert_allclose(reconstructed, source["sst"].transpose("time", "lat", "lon"), equal_nan=True)

    reader = NetCDFReader(output_path, variable="sst", lazy=True, spatial_chunk=2)
    try:
        stored = reader.dataset["sst"].transpose("time", "point")
        assert reader.rechunk_stacked(stored).chunks == stored.chunks
    finally:
        reader.close()


def test_optimizer_decodes_packed_values_once_and_writes_real_chunks(tmp_path):
    source = _source_dataset()
    source["sst"].encoding.update(
        {"dtype": "int16", "scale_factor": 0.1, "add_offset": 10.0, "_FillValue": -32768}
    )
    input_path = tmp_path / "packed.nc"
    output_path = tmp_path / "optimized.nc"
    source.to_netcdf(input_path, engine="h5netcdf")

    report = optimize_netcdf(input_path, output_path, variable="sst", compression="gzip")

    assert report.original_dtype == "int16"
    assert report.output_dtype == "float32"
    with xr.open_dataset(output_path) as optimized:
        expected = source["sst"].transpose("time", "lat", "lon").values.reshape(-1, 6)[:, 1:]
        np.testing.assert_allclose(optimized["sst"], expected, rtol=1e-6, atol=1e-6, equal_nan=True)
        assert "scale_factor" not in optimized["sst"].attrs
        assert "add_offset" not in optimized["sst"].attrs
    with h5py.File(output_path, "r") as physical:
        assert physical["sst"].chunks == report.output_chunks
        assert physical["sst"].compression == "gzip"


def test_chunk_selection_is_deterministic_and_bounded():
    shapes = ((8760, 5000, "float32"), (64080, 15000, "float32"), (350000, 500000, "float64"))
    for ntime, npoints, dtype in shapes:
        first = choose_chunk_shape(ntime, npoints, dtype, source_chunks=(24, 157, 115))
        second = choose_chunk_shape(ntime, npoints, dtype, source_chunks=(24, 157, 115))
        assert first == second
        assert 0 < first[0] <= ntime
        assert 0 < first[1] <= npoints
        assert first[0] * first[1] * np.dtype(dtype).itemsize <= 2 * 4 * 1024 * 1024


def test_compact_point_event_result_keeps_geographic_mapping():
    time = np.arange(np.datetime64("2001-01-01"), np.datetime64("2001-01-01T04"), np.timedelta64(1, "h"))
    points = xr.DataArray(np.arange(2), dims="point", coords={"point": [0, 1]}).indexes["point"]
    catalogue = xr.Dataset(
        {
            "point_index": ("event", [1]),
            "start_index": ("event", [1]),
            "stop_index": ("event", [3]),
            "duration_hours": ("event", [2]),
            "start_time": ("event", [time[1]]),
            "end_time": ("event", [time[2]]),
            "event_count": ("point", [0, 1]),
        },
        coords={"event": [1], "point": points},
    )
    result = EventDetectionResult(
        data=catalogue,
        telemetry=(),
        metadata={},
        _grid=EventGrid(
            time=time,
            point_index=points,
            point_indices=np.array([0, 1]),
            time_dim="time",
            point_coordinates={"latitude": np.array([10.0, 11.0]), "longitude": np.array([20.0, 21.0])},
        ),
    )

    dense = result.to_xarray(time_chunk_hours=2).compute()

    assert dense["event_id"].dims == ("time", "point")
    np.testing.assert_array_equal(dense["latitude"], [10.0, 11.0])
    np.testing.assert_array_equal(dense["event_id"].isel(point=1), [0, 1, 1, 0])


def test_cudaheat_consumes_optimized_point_layout(tmp_path):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA device unavailable")
    try:
        cp.full((1,), 0.0)
    except RuntimeError as exc:
        pytest.skip(f"CuPy kernel compilation unavailable: {exc}")
    from cudaheat import CUDAHeat

    source = _source_dataset()
    input_path = tmp_path / "input.nc"
    source.to_netcdf(input_path, engine="h5netcdf")

    with CUDAHeat(input_path, variable="sst", vram_safety=0.90) as engine:
        assert engine._optimized_point_layout
        assert engine.npoints == 5
        result = engine.run(smooth_window_hours=24 * 3, return_fields={"mask"})
        detection = engine.detect(smooth_window_hours=24 * 3)

    assert result.data["mask"].dims == ("time", "point")
    assert "original_flat_index" in result.data.coords
    assert "original_flat_index" in detection.data
    assert detection.metadata["climatology_cache_status"] == "created"
    cache_path = detection.metadata["climatology_cache_path"]
    assert cache_path is not None
    optimized = derived_cache_root() / detection.metadata["source_sha256"] / "input.cudaheat.nc"
    assert optimized.is_file()
    assert not input_path.with_name("input.cudaheat.nc").exists()

    with CUDAHeat(input_path, variable="sst", vram_safety=0.90) as engine:
        reused = engine.detect(smooth_window_hours=24 * 3)
        different_detection_controls = engine.detect(
            smooth_window_hours=24 * 3,
            min_duration_hours=2,
            max_internal_gap_hours=0,
            join_gap_hours=0,
        )

    assert reused.metadata["climatology_cache_status"] == "reused"
    assert reused.metadata["climatology_hash"] == detection.metadata["climatology_hash"]
    assert different_detection_controls.metadata["climatology_cache_status"] == "reused"
    assert different_detection_controls.metadata["climatology_hash"] == detection.metadata["climatology_hash"]
    xr.testing.assert_identical(reused.data, detection.data)


def test_auto_input_preparation_creates_then_reuses_a_validated_sibling(tmp_path):
    source = _source_dataset()
    input_path = tmp_path / "input.nc"
    source.to_netcdf(input_path, engine="h5netcdf")

    first = prepare_netcdf_input(input_path, variable="sst", dtype="float32")
    second = prepare_netcdf_input(input_path, variable="sst", dtype="float32")

    assert first.active_path != input_path
    assert first.active_path == derived_cache_root() / first.source_sha256 / "input.cudaheat.nc"
    assert first.active_path.is_file()
    assert first.report is not None
    assert not first.reused_cached_optimization
    assert second.active_path == first.active_path
    assert second.reused_cached_optimization


def test_auto_input_preparation_rebuilds_when_content_changes_without_stat_change(tmp_path):
    source = _source_dataset()
    input_path = tmp_path / "input.nc"
    source.to_netcdf(input_path, engine="h5netcdf")
    first = prepare_netcdf_input(input_path, variable="sst", dtype="float32")
    initial_stat = input_path.stat()

    with h5py.File(input_path, "r+") as raw:
        raw["sst"][0, 1, 1] += 1
    os.utime(input_path, ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns))

    rebuilt = prepare_netcdf_input(input_path, variable="sst", dtype="float32")

    assert rebuilt.active_path != first.active_path
    assert rebuilt.report is not None
    assert not rebuilt.reused_cached_optimization
    assert rebuilt.source_sha256 != first.source_sha256


def test_auto_input_preparation_refuses_cudaheat_suffix_chaining(tmp_path):
    path = tmp_path / "input.cudaheat.nc"
    _source_dataset().to_netcdf(path, engine="h5netcdf")

    with pytest.raises(ValueError, match="CUDAHeat-derived"):
        prepare_netcdf_input(path, variable="sst")
