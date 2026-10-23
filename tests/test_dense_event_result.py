from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cudaheat import CUDAHeat
import cudaheat.core as core

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover - hardware dependent
    pytest.skip("CUDA device unavailable", allow_module_level=True)
try:  # pragma: no cover - installation dependent
    cp.full((1,), 0.0)
except RuntimeError as exc:
    pytest.skip(f"CuPy kernel compilation unavailable: {exc}", allow_module_level=True)


def _event_source() -> tuple[xr.Dataset, int]:
    time = np.arange(
        np.datetime64("2000-01-01T00"),
        np.datetime64("2004-01-04T00"),
        np.timedelta64(1, "h"),
    )
    values = np.zeros((time.size, 2, 2), dtype=np.float64)
    event_start = int(np.searchsorted(time, np.datetime64("2003-01-02T00")))
    values[event_start : event_start + 2, 1, 1] = 3
    values[event_start + 3 : event_start + 5, 1, 1] = 3
    values[event_start + 24 : event_start + 26, 0, 0] = 3
    return (
        xr.Dataset(
            {"analysed_sst": (("time", "latitude", "longitude"), values)},
            coords={"time": time, "latitude": [34.0, 35.0], "longitude": [32.0, 33.0]},
        ),
        event_start,
    )


def test_dense_event_result_is_exact_and_cpu_only_after_source_context_closes(tmp_path, monkeypatch):
    source, gap_start = _event_source()
    path = tmp_path / "events.nc"
    source.to_netcdf(path, engine="h5netcdf")

    with CUDAHeat(path, variable="analysed_sst", dtype="float32") as engine:
        result = engine.detect(
            percentile=0.9,
            smooth_window_hours=1,
            min_duration_hours=2,
            max_internal_gap_hours=1,
        )
        original_telemetry = result.telemetry
        original_assembly_seconds = result.metadata["result_dataset_assembly_seconds"]

    # The source engine is closed.  Dense work starts only here.
    assert "threshold" not in result.data

    def gpu_reentry(*_args, **_kwargs):
        raise AssertionError("to_xarray must not re-enter CUDAHeat or CuPy")

    monkeypatch.setattr(core, "cupy", gpu_reentry)
    monkeypatch.setattr(CUDAHeat, "run", gpu_reentry)
    dense = result.to_xarray(time_chunk_hours=24)
    dense_repeat = result.to_dense(time_chunk_hours=24)
    xr.testing.assert_equal(dense.compute(), dense_repeat.compute())

    assert isinstance(dense, xr.Dataset)
    assert dense["event_id"].dtype == np.uint32
    assert set(dense.data_vars) == {"event_id", "event_count"}
    assert hasattr(dense["event_id"].data, "compute")
    assert result.telemetry == original_telemetry
    assert result.metadata["result_dataset_assembly_seconds"] == original_assembly_seconds
    assert dense.attrs["stop_index_semantics"].endswith("[start_index, stop_index).")

    event_id_by_point = dense["event_id"].stack(point=("latitude", "longitude"))
    for event_id, point_index, start, stop, duration in zip(
        result.data["event"].values,
        result.data["point_index"].values,
        result.data["start_index"].values,
        result.data["stop_index"].values,
        result.data["duration_hours"].values,
        strict=True,
    ):
        observed = np.flatnonzero(event_id_by_point.isel(point=int(point_index)).values == event_id)
        np.testing.assert_array_equal(observed, np.arange(start, stop))
        assert observed.size == duration

    # The one-hour allowed gap belongs to the retained event.
    point = dense.sel(latitude=35.0, longitude=33.0, method="nearest")
    assert int(point["event_id"].isel(time=gap_start + 2).compute()) > 0
