from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover - hardware dependent
    pytest.skip("CUDA device unavailable", allow_module_level=True)

from cudaheat.climatology import cpu_pre_event_fields, gpu_pre_event_fields, hourly_climatology_index
from cudaheat.core import BatchOutput, CUDAHeat
from cudaheat.memory import BatchTelemetry


def test_gpu_matches_cpu_reference_with_nan_and_float32(synthetic_sst):
    values = synthetic_sst.analysed_sst.stack(point=("lat", "lon")).values.astype("float32")
    bins = hourly_climatology_index(synthetic_sst.time)
    rows = np.arange(values.shape[0])
    cpu = cpu_pre_event_fields(values, bins, rows, smooth_window_hours=3)
    gpu = gpu_pre_event_fields(cp.asarray(values), cp.asarray(bins), cp.asarray(rows), smooth_window_hours=3)

    np.testing.assert_allclose(cp.asnumpy(gpu.climatology), cpu.climatology, rtol=2e-5, atol=2e-5, equal_nan=True)
    np.testing.assert_allclose(cp.asnumpy(gpu.threshold), cpu.threshold, rtol=2e-5, atol=2e-5, equal_nan=True)
    np.testing.assert_array_equal(cp.asnumpy(gpu.exceedance), cpu.exceedance)


def test_engine_stacks_space_preserves_all_time_and_returns_mask(synthetic_sst):
    with CUDAHeat(synthetic_sst, variable="analysed_sst", dtype="float32", vram_safety=0.30) as hw:
        assert hw.ntime == synthetic_sst.sizes["time"]
        assert hw.npoints == synthetic_sst.sizes["lat"] * synthetic_sst.sizes["lon"]
        batch = hw._load_spatial_batch(0, 2)
        assert batch.shape == (hw.ntime, 2)
        assert batch.dtype == np.float32

        result = hw.run(smooth_window_hours=3, return_fields={"mask"})
        assert result.metadata["time_was_split"] is False
        assert result.data["mask"].dims == ("time", "lat", "lon")
        assert result.data["mask"].dtype == bool
        assert len(result.telemetry) >= 1


def test_engine_opens_netcdf_then_loads_complete_time_by_spatial_batch(synthetic_sst, tmp_path):
    path = tmp_path / "synthetic-sst.nc"
    synthetic_sst.to_netcdf(path, engine="h5netcdf")
    with CUDAHeat(path, variable="analysed_sst", dtype="float32") as hw:
        batch = hw._load_spatial_batch(1, 4)
        assert batch.shape == (synthetic_sst.sizes["time"], 3)
        assert batch.dtype == np.float32


def test_engine_retries_same_full_time_range_with_smaller_spatial_batch(synthetic_sst, monkeypatch):
    with CUDAHeat(synthetic_sst, variable="analysed_sst", dtype="float32") as hw:
        calls = []

        def fake_execute(values, start, stop, *_):
            calls.append((start, stop, values.shape))
            if len(calls) == 1:
                raise cp.cuda.memory.OutOfMemoryError(1, 0)
            width = stop - start
            telemetry = BatchTelemetry(
                device_id=0, point_start=start, point_stop=stop, batch_size=width,
                dtype="float32", bytes_per_point=1, safety_fraction=0.30, oom_retries=1,
                free_before_bytes=1, used_before_bytes=1, free_after_bytes=1,
                used_after_bytes=1, peak_used_bytes=1, pool_used_bytes=1,
                transfer_to_device_seconds=0.0, gpu_compute_seconds=0.0,
                transfer_from_device_seconds=0.0,
            )
            fields = {
                "mask": np.zeros((hw.ntime, width), dtype=bool),
                "exceedance_count": np.zeros(width, dtype=np.uint64),
            }
            return BatchOutput(start, stop, fields, telemetry), telemetry

        monkeypatch.setattr(hw, "_execute_gpu_batch", fake_execute)
        result = hw.run(return_fields={"mask"})

    assert calls[0][:2] == (0, hw.npoints)
    assert calls[1][:2] == (0, hw.npoints // 2)
    assert calls[0][2][0] == calls[1][2][0] == hw.ntime
    assert result.metadata["time_was_split"] is False
