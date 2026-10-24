from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr
import zarr

from cudaheat.cache_paths import derived_cache_root
from cudaheat.climatology_cache import open_climatology_cache


@pytest.fixture(autouse=True)
def _use_test_cache_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDAHEAT_CACHE_DIR", str(tmp_path / ".cudaheat-cache"))


def _open_cache(source, *, percentile: float = 0.9, source_sha256: str = "a" * 64):
    return open_climatology_cache(
        source,
        source_sha256=source_sha256,
        variable="sst",
        dtype="float32",
        baseline=(2001, 2003),
        percentile=percentile,
        smooth_window_hours=24 * 31,
        smooth_min_periods=24 * 31,
        sampling_window_half_width_days=5,
        same_hour_smoothing=True,
        calendar_mode="month_day_hour",
        nbins=24,
        point_values=np.array([0, 1, 2]),
        point_coordinates={
            "latitude": np.array([30.0, 31.0, 32.0]),
            "longitude": np.array([20.0, 21.0, 22.0]),
            "original_flat_index": np.array([4, 5, 6]),
        },
    )


def test_climatology_cache_is_atomic_reusable_and_has_complete_provenance(tmp_path):
    source = tmp_path / "sst_40_years.nc"
    source.write_bytes(b"raw netcdf identity is held separately in production")

    first = _open_cache(source)
    assert first.path.name.startswith("sst_40_years.cudaheat.clim-")
    assert first.path.suffix == ".zarr"
    assert first.path.parent == derived_cache_root() / ("a" * 64)
    assert first.needs_write
    assert not first.path.exists()

    climatology = np.arange(72, dtype=np.float32).reshape(24, 3)
    threshold = climatology + np.float32(1)
    first.write_batch(0, 2, {"climatology": climatology[:, :2], "threshold_cycle": threshold[:, :2]})
    first.write_batch(2, 3, {"climatology": climatology[:, 2:], "threshold_cycle": threshold[:, 2:]})
    assert first.commit() == "created"

    group = zarr.open_group(str(first.path), mode="r")
    assert group.attrs["cudaheat_complete"]
    assert group.attrs["cudaheat_source_sha256"] == "a" * 64
    config = json.loads(group.attrs["cudaheat_climatology_config"])
    assert config == {
        "algorithm": "cudaheat-hourly-hobday-v1",
        "baseline": [2001, 2003],
        "calendar_mode": "month_day_hour",
        "dtype": "float32",
        "nbins": 24,
        "percentile": 0.9,
        "same_hour_smoothing": True,
        "sampling_window_half_width_days": 5,
        "smooth_min_periods": 744,
        "smooth_window_hours": 744,
        "source_sha256": "a" * 64,
        "variable": "sst",
    }
    np.testing.assert_array_equal(group["latitude"][:], [30.0, 31.0, 32.0])
    with xr.open_zarr(first.path) as dataset:
        assert dataset["climatology"].dims == ("climatology_hour", "point")
        assert dataset["threshold"].dims == ("climatology_hour", "point")

    reused = _open_cache(source)
    assert reused.path == first.path
    assert reused.status == "reused"
    assert not reused.needs_write
    fields = reused.read_batch(0, 3)
    np.testing.assert_array_equal(fields["climatology"], climatology)
    np.testing.assert_array_equal(fields["threshold_cycle"], threshold)


def test_climatology_cache_hash_changes_only_with_climatology_inputs(tmp_path):
    source = tmp_path / "sst.nc"
    source.write_bytes(b"source")

    first = _open_cache(source)
    different_percentile = _open_cache(source, percentile=0.95)
    different_source = _open_cache(source, source_sha256="b" * 64)
    try:
        assert first.climatology_hash != different_percentile.climatology_hash
        assert first.climatology_hash != different_source.climatology_hash
        # Detection-only controls deliberately are not accepted by this cache API,
        # so duration and gap settings cannot change this deterministic identity.
    finally:
        first.abort()
        different_percentile.abort()
        different_source.abort()
