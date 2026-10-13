from __future__ import annotations

import numpy as np
import xarray as xr

from util.extend_netcdf_backward import extend_backward, extend_file_backward


def _hourly_dataset(
    start: str,
    end: str,
    *,
    spatial: bool = False,
    leading_invalid_hours: int = 0,
) -> xr.Dataset:
    time = np.arange(np.datetime64(start), np.datetime64(end), np.timedelta64(1, "h"))
    values = np.arange(time.size, dtype=np.float32)
    if spatial:
        values = np.stack((values, values + 100), axis=1)[:, :, None]
        values[:, 1, 0] = np.nan
        values[12, 0, 0] = np.nan
        dimensions = ("time", "lat", "lon")
        coordinates = {"time": time, "lat": [34.0, 35.0], "lon": [32.0]}
    else:
        dimensions = ("time",)
        coordinates = {"time": time}
    if leading_invalid_hours:
        values[:leading_invalid_hours] = np.nan
    return xr.Dataset(
        {
            "analysed_sst": (dimensions, values, {"units": "degC", "long_name": "SST"}),
            "quality": ("time", np.arange(time.size, dtype=np.int16)),
        },
        coords=coordinates,
        attrs={"title": "source dataset"},
    )


def _synthetic_count(ds: xr.Dataset) -> int:
    return int(ds["synthetic_benchmark_time"].sum().item())


def _daily_amplitude(values: xr.DataArray) -> float:
    daily_mean = values.resample(time="1D").mean()
    mean_at_time = daily_mean.reindex(time=values.time, method="ffill")
    return float(np.abs(values - mean_at_time).mean().item())


def test_extension_is_hourly_unique_and_preserves_original_data() -> None:
    original = _hourly_dataset("2021-01-01", "2022-01-02", spatial=True)
    extended = extend_backward(
        original,
        variable="analysed_sst",
        fake_start="2020-01-01",
        window="24H",
    )

    synthetic_steps = _synthetic_count(extended)
    assert extended.time.values[0] == np.datetime64("2020-01-01T00")
    assert np.all(np.diff(extended.time.values) == np.timedelta64(1, "h"))
    assert len(np.unique(extended.time.values)) == extended.sizes["time"]
    assert np.all(extended.synthetic_benchmark_time.values[:synthetic_steps] == 1)
    assert np.all(extended.synthetic_benchmark_time.values[synthetic_steps:] == 0)

    recovered = extended.drop_vars("synthetic_benchmark_time").isel(time=slice(synthetic_steps, None))
    recovered.attrs = dict(original.attrs)
    xr.testing.assert_identical(recovered, original)
    assert "not a scientific reconstruction" in extended.attrs["cudaheat_synthetic_benchmark_notice"]


def test_template_starts_at_first_valid_timestep_and_preserves_masked_cells() -> None:
    original = _hourly_dataset(
        "2021-01-01", "2021-01-04", spatial=True, leading_invalid_hours=2
    )
    extended = extend_backward(
        original,
        variable="analysed_sst",
        fake_start="2020-12-30",
        window="24H",
    )

    added = _synthetic_count(extended)
    first_full_block = extended.analysed_sst.isel(time=slice(added - 24, added))
    expected_template = original.analysed_sst.isel(time=slice(2, 26))
    np.testing.assert_allclose(first_full_block.values, expected_template.values, equal_nan=True)
    assert np.isnan(extended.analysed_sst.isel(lat=1, lon=0).values).all()


def test_noise_is_deterministic_for_a_fixed_seed() -> None:
    original = _hourly_dataset("2021-01-01", "2021-01-04", spatial=True)
    kwargs = dict(
        variable="analysed_sst",
        fake_start="2020-12-30",
        window="24H",
        noise=0.05,
        seed=42,
    )
    first = extend_backward(original, **kwargs)
    second = extend_backward(original, **kwargs)
    np.testing.assert_allclose(first.analysed_sst.values, second.analysed_sst.values, equal_nan=True)


def test_partial_final_replication_block_uses_the_original_template() -> None:
    original = _hourly_dataset("2020-01-03", "2020-01-05")
    extended = extend_backward(
        original,
        variable="analysed_sst",
        fake_start="2019-12-31T22",
        window="24H",
    )

    added = _synthetic_count(extended)
    assert added == 50
    values = extended.analysed_sst.isel(time=slice(0, added)).values
    template = original.analysed_sst.isel(time=slice(0, 24)).values
    np.testing.assert_array_equal(values[:2], template[:2])
    np.testing.assert_array_equal(values[2:26], template)
    np.testing.assert_array_equal(values[26:50], template)


def test_leap_day_is_present_without_timestamp_gaps_or_duplicates() -> None:
    original = _hourly_dataset("2021-01-01", "2022-01-01")
    extended = extend_backward(
        original,
        variable="analysed_sst",
        fake_start="2019-01-01",
        window="1Y",
    )

    leap_day = np.datetime64("2020-02-29")
    assert np.count_nonzero(extended.time.values.astype("datetime64[D]") == leap_day) == 24
    assert np.all(np.diff(extended.time.values) == np.timedelta64(1, "h"))
    assert len(np.unique(extended.time.values)) == extended.sizes["time"]


def test_decay_reduces_daily_amplitude_without_shifting_daily_mean() -> None:
    time = np.arange(np.datetime64("2023-01-01"), np.datetime64("2025-01-01"), np.timedelta64(1, "h"))
    excursions = np.where(np.arange(time.size) % 24 < 12, 2.0, -2.0).astype(np.float32)
    original = xr.Dataset(
        {"analysed_sst": ("time", 10.0 + excursions)},
        coords={"time": time},
    )
    extended = extend_backward(
        original,
        variable="analysed_sst",
        fake_start="2021-01-01",
        window="1Y",
        decay=0.5,
    )

    rightmost_synthetic_year = extended.analysed_sst.sel(time=slice("2022-01-01", "2022-12-31T23"))
    oldest_synthetic_year = extended.analysed_sst.sel(time=slice("2021-01-01", "2021-12-31T23"))
    assert np.isclose(_daily_amplitude(rightmost_synthetic_year), 1.5, atol=1e-6)
    assert np.isclose(_daily_amplitude(oldest_synthetic_year), 1.0, atol=1e-6)
    assert np.isclose(float(rightmost_synthetic_year.mean().item()), 10.0, atol=1e-6)
    assert np.isclose(float(oldest_synthetic_year.mean().item()), 10.0, atol=1e-6)


def test_extension_writes_a_distinct_netcdf_file(tmp_path) -> None:
    original = _hourly_dataset("2021-01-01", "2021-01-04")
    source = tmp_path / "source.nc"
    output = tmp_path / "extended.nc"
    original.to_netcdf(source)

    summary = extend_file_backward(
        source,
        output,
        variable="analysed_sst",
        fake_start="2020-12-31",
        window="24H",
    )
    assert output.exists()
    assert summary["Output file"] == str(output)
    with xr.open_dataset(output) as reloaded:
        assert _synthetic_count(reloaded) == 24
        assert reloaded.attrs["cudaheat_synthetic_benchmark_extension"] == "backward_replication"



# python util/extend_netcdf_backward.py input.nc output_extended.nc \
#   --variable analysed_sst --fake-start 1980-01-01 \
#   --window 1Y --noise 0.05 --decay 0.5 --seed 42