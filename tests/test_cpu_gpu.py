from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover - hardware dependent
    pytest.skip("CUDA device unavailable", allow_module_level=True)

from cudaheat.climatology import (
    CALENDAR_MODE_DAY_OF_YEAR,
    _nanpercentile_by_bin_gpu,
    _nanpercentile_by_bin_gpu_legacy,
    cpu_pre_event_fields,
    gpu_pre_event_fields,
    hourly_climatology_index,
)
from cudaheat.benchmark import CUDAHeatBenchmark
from cudaheat.core import BatchOutput, CUDAHeat
from cudaheat.detection import fill_internal_gaps, _fill_internal_gaps_legacy
from cudaheat.memory import BatchTelemetry
from cudaheat.validation import select_validation_points


def test_gpu_matches_cpu_reference_with_nan_and_float32(synthetic_sst):
    values = synthetic_sst.analysed_sst.stack(point=("lat", "lon")).values.astype("float32")
    bins = hourly_climatology_index(synthetic_sst.time)
    rows = np.arange(values.shape[0])
    cpu = cpu_pre_event_fields(values, bins, rows, smooth_window_hours=3)
    gpu = gpu_pre_event_fields(cp.asarray(values), cp.asarray(bins), cp.asarray(rows), smooth_window_hours=3)

    np.testing.assert_allclose(cp.asnumpy(gpu.climatology), cpu.climatology, rtol=2e-5, atol=2e-5, equal_nan=True)
    np.testing.assert_allclose(cp.asnumpy(gpu.threshold), cpu.threshold, rtol=2e-5, atol=2e-5, equal_nan=True)
    np.testing.assert_array_equal(cp.asnumpy(gpu.exceedance), cpu.exceedance)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_gpu_percentile_grouping_matches_legacy_for_ragged_bins_and_nans(dtype):
    rng = np.random.default_rng(91)
    bins = np.array([4, 1, 4, 1, 4, 7, 7, 4, 2, 1, 7, 2, 2, 2, 4, 1, 7], dtype=np.int32)
    values = rng.normal(size=(bins.size, 5)).astype(dtype)
    values[[1, 4, 12], [3, 0, 4]] = np.nan

    legacy = _nanpercentile_by_bin_gpu_legacy(
        cp.asarray(values), cp.asarray(bins), 9, 0.9, cp
    )
    optimized = _nanpercentile_by_bin_gpu(cp.asarray(values), cp.asarray(bins), 9, 0.9, cp)

    np.testing.assert_array_equal(cp.asnumpy(optimized), cp.asnumpy(legacy))


def test_gpu_gap_fill_float16_delta_matches_legacy_int32_delta():
    exceedance = np.ones((104, 3), dtype=bool)
    exceedance[[1, 7, 8, 44, 45, 46, 100], 0] = False
    exceedance[[11, 12, 50, 99], 1] = False
    exceedance[[0, 30, 31, 32, 103], 2] = False

    legacy = _fill_internal_gaps_legacy(cp.asarray(exceedance), 2, xp=cp)
    optimized = fill_internal_gaps(cp.asarray(exceedance), 2, xp=cp)

    np.testing.assert_array_equal(cp.asnumpy(optimized), cp.asnumpy(legacy))


def test_engine_stacks_space_preserves_all_time_and_returns_mask(synthetic_sst):
    with CUDAHeat(synthetic_sst, variable="analysed_sst", dtype="float32", vram_safety=0.30) as hw:
        assert hw.ntime == synthetic_sst.sizes["time"]
        assert hw.npoints == synthetic_sst.sizes["lat"] * synthetic_sst.sizes["lon"]
        batch = hw._load_spatial_batch(0, 2)
        assert batch.shape == (hw.ntime, 2)
        assert batch.dtype == np.float32

        result = hw.run(smooth_window_hours=3, return_fields={"mask", "threshold_raw"})
        assert result.metadata["time_was_split"] is False
        assert result.data["mask"].dims == ("time", "lat", "lon")
        assert result.data["mask"].dtype == bool
        assert result.data["threshold_raw"].dims == ("climatology_hour", "lat", "lon")
        assert len(result.telemetry) >= 1


def test_engine_dayofyear_mode_uses_hourwave_bins_and_metadata(synthetic_sst):
    with CUDAHeat(
        synthetic_sst,
        variable="analysed_sst",
        dtype="float32",
        calendar_mode=CALENDAR_MODE_DAY_OF_YEAR,
    ) as hw:
        assert hw._resolve_smooth_window_hours(None) == 15 * 24
        result = hw.run(smooth_window_hours=3, return_fields={"climatology", "threshold"})

    assert result.data["climatology"].sizes["climatology_hour"] == synthetic_sst.sizes["time"]
    assert result.metadata["calendar_mode"] == CALENDAR_MODE_DAY_OF_YEAR
    assert result.metadata["smooth_min_periods"] == 3


def test_engine_opens_netcdf_then_loads_complete_time_by_spatial_batch(synthetic_sst, tmp_path):
    path = tmp_path / "synthetic-sst.nc"
    synthetic_sst.to_netcdf(path, engine="h5netcdf")
    with CUDAHeat(path, variable="analysed_sst", dtype="float32") as hw:
        batch = hw._load_spatial_batch(1, 4)
        assert batch.shape == (synthetic_sst.sizes["time"], 3)
        assert batch.dtype == np.float32


def test_engine_retries_same_full_time_range_with_smaller_spatial_batch(synthetic_sst, monkeypatch):
    with CUDAHeat(
        synthetic_sst, variable="analysed_sst", dtype="float32", batch_size=6
    ) as hw:
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


def test_targeted_validation_selection_is_seeded_and_keeps_complete_time(synthetic_sst):
    with CUDAHeat(synthetic_sst, variable="analysed_sst") as hw:
        first = select_validation_points(hw, points=3, random_seed=17)
        second = select_validation_points(hw, points=3, random_seed=17)
        result = hw.validate(points=3, random_seed=17, smooth_window_hours=3)

    np.testing.assert_array_equal(first, second)
    assert result.ntime == synthetic_sst.sizes["time"]
    assert len(result.point_indices) == 3
    assert result.mask.total_elements == result.ntime * 3
    assert result.exceedance_count.total_points == 3


def test_benchmark_runs_repeats_with_complete_time_series(synthetic_sst):
    with CUDAHeat(synthetic_sst, variable="analysed_sst") as hw:
        benchmark = CUDAHeatBenchmark(hw)
        records = benchmark.run(
            repeats=2,
            warmup=0,
            point_counts=(2,),
            batch_sizes=(1,),
            smooth_window_hours=3,
        )

    assert len(records) == 2
    assert all(record.ntime == synthetic_sst.sizes["time"] for record in records)
    assert all(record.cpu_compute_seconds is not None for record in records)
    assert all(record.gpu_end_to_end_seconds >= record.gpu_compute_seconds for record in records)
