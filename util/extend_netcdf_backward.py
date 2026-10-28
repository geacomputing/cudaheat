"""Create a synthetic backward NetCDF extension for CUDAHeat benchmarks.

This utility replicates a window from the beginning of an existing valid
record.  It is for computational benchmarking only and is not a scientific
reconstruction of historical SST.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


_WINDOW_PATTERN = re.compile(r"^(?P<count>[1-9][0-9]*)(?P<unit>[YMDH])$", re.IGNORECASE)
_SYNTHETIC_MARKER = "synthetic_benchmark_time"
_DAYS_PER_YEAR = 365.2425


@dataclass(frozen=True)
class Window:
    """A calendar window accepted by the extension utility."""

    count: int
    unit: str

    @property
    def years(self) -> float:
        if self.unit == "Y":
            return float(self.count)
        if self.unit == "M":
            return self.count / 12.0
        if self.unit == "D":
            return self.count / _DAYS_PER_YEAR
        return self.count / (24.0 * _DAYS_PER_YEAR)

    def __str__(self) -> str:
        return f"{self.count}{self.unit}"


def extend_backward(
    ds: xr.Dataset,
    *,
    variable: str,
    fake_start: str | np.datetime64 | datetime,
    window: str = "1Y",
    noise: float = 0.0,
    decay: float = 0.0,
    seed: int | None = None,
    time_dim: str | None = None,
) -> xr.Dataset:
    """Prepend a synthetic, independently replicated benchmark section.

    Parameters
    ----------
    ds:
        Source dataset. It is never mutated.
    variable:
        Variable whose valid-record start, optional noise, and optional
        daily-amplitude decay define the synthetic extension.
    fake_start:
        First timestamp in the desired synthetic section.
    window:
        Calendar replication window such as ``"1Y"``, ``"6M"``, ``"30D"``,
        or ``"24H"``.
    noise:
        Gaussian noise standard deviation in the variable's units, applied
        independently to each synthetic block.
    decay:
        Daily-amplitude reduction in variable units per synthetic year. It
        preserves each daily mean while shrinking its excursions where
        possible; it is not a uniform temperature offset.
    seed:
        Optional seed for deterministic independent block-level noise.
    time_dim:
        Optional time-dimension override.

    Returns
    -------
    xarray.Dataset
        A new dataset with a ``synthetic_benchmark_time`` marker variable.

    Notes
    -----
    The generated section is synthetic benchmark data only. It must not be
    interpreted as a historical or scientific SST reconstruction.
    """
    if variable not in ds:
        raise KeyError(f"variable {variable!r} is not present in the dataset")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    if decay < 0:
        raise ValueError("decay must be non-negative")
    if _SYNTHETIC_MARKER in ds:
        raise ValueError(f"dataset already contains reserved variable {_SYNTHETIC_MARKER!r}")

    selected = ds[variable]
    resolved_time_dim = _resolve_time_dim(selected, time_dim)
    time_values = np.asarray(ds[resolved_time_dim].values)
    cadence = _validate_regular_time_axis(time_values, resolved_time_dim)
    uses_numpy_datetime = np.issubdtype(time_values.dtype, np.datetime64)
    original_start = time_values[0]
    parsed_start = _parse_fake_start(fake_start, original_start, uses_numpy_datetime)
    synthetic_count = _synthetic_count(parsed_start, original_start, cadence)
    if synthetic_count < 1:
        raise ValueError("fake_start must be earlier than the dataset's first timestamp")

    spec = _parse_window(window)
    valid_index = _first_valid_index(selected, resolved_time_dim)
    template_stop = _template_stop_index(
        time_values,
        valid_index,
        _calendar_shift(time_values[valid_index], spec, uses_numpy_datetime),
        cadence,
    )
    template = ds.isel({resolved_time_dim: slice(valid_index, template_stop)})
    template_length = template.sizes[resolved_time_dim]
    if template_length < 1:
        raise ValueError("replication window contains no source timesteps")

    synthetic_times = _synthetic_times(parsed_start, synthetic_count, cadence, uses_numpy_datetime)
    blocks = _synthetic_blocks(
        template,
        variable=variable,
        time_dim=resolved_time_dim,
        synthetic_times=synthetic_times,
        decay=decay,
        noise=noise,
        seed=seed,
        window_years=spec.years,
    )
    result = xr.concat(
        [*blocks, ds],
        dim=resolved_time_dim,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
    )
    _restore_metadata(result, ds)
    marker = np.concatenate(
        (
            np.ones(synthetic_count, dtype=np.int8),
            np.zeros(ds.sizes[resolved_time_dim], dtype=np.int8),
        )
    )
    result[_SYNTHETIC_MARKER] = xr.DataArray(
        marker,
        dims=(resolved_time_dim,),
        coords={resolved_time_dim: result[resolved_time_dim]},
        attrs={
            "long_name": "Synthetic CUDAHeat benchmark extension marker",
            "flag_values": np.array([0, 1], dtype=np.int8),
            "flag_meanings": "original synthetic_benchmark",
            "comment": "Synthetic benchmark data only; not a historical SST reconstruction.",
        },
    )
    result.attrs.update(
        {
            "cudaheat_synthetic_benchmark_extension": "backward_replication",
            "cudaheat_synthetic_benchmark_notice": (
                "Synthetic benchmark data only; not a scientific reconstruction of historical SST."
            ),
            "cudaheat_synthetic_variable": variable,
            "cudaheat_synthetic_original_start": _format_time(original_start),
            "cudaheat_synthetic_start": _format_time(parsed_start),
            "cudaheat_synthetic_timesteps_added": synthetic_count,
            "cudaheat_synthetic_replication_window": str(spec),
            "cudaheat_synthetic_noise_sigma": float(noise),
            "cudaheat_synthetic_amplitude_decay_per_year": float(decay),
            "cudaheat_synthetic_rng_seed": "none" if seed is None else int(seed),
        }
    )
    return result


def extend_file_backward(
    input_path: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write a synthetic backward extension to a distinct NetCDF file."""
    source = Path(input_path)
    output = Path(output_path)
    overwrite = kwargs.pop("overwrite", False)
    if source.resolve() == output.resolve():
        raise ValueError("output_path must differ from input_path; the source is never modified")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass overwrite=True to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(source) as ds:
        extended = extend_backward(ds, **kwargs)
        extended.to_netcdf(output, mode="w")
        summary = extension_summary(extended, output)
    return summary


def extension_summary(ds: xr.Dataset, output_path: str | Path | None = None) -> dict[str, Any]:
    """Return the completion report associated with an extended dataset."""
    time_dim = _marker_time_dim(ds)
    synthetic_steps = int(ds[_SYNTHETIC_MARKER].sum().item())
    original_steps = ds.sizes[time_dim] - synthetic_steps
    return {
        "Original start": ds.attrs["cudaheat_synthetic_original_start"],
        "Synthetic start": ds.attrs["cudaheat_synthetic_start"],
        "Original number of timesteps": original_steps,
        "Synthetic timesteps added": synthetic_steps,
        "Replication window": ds.attrs["cudaheat_synthetic_replication_window"],
        "Noise sigma": ds.attrs["cudaheat_synthetic_noise_sigma"],
        "Decay": ds.attrs["cudaheat_synthetic_amplitude_decay_per_year"],
        "Output number of timesteps": ds.sizes[time_dim],
        "Output file": None if output_path is None else str(output_path),
    }


def _synthetic_blocks(
    template: xr.Dataset,
    *,
    variable: str,
    time_dim: str,
    synthetic_times: np.ndarray,
    decay: float,
    noise: float,
    seed: int | None,
    window_years: float,
) -> list[xr.Dataset]:
    template_length = template.sizes[time_dim]
    number_of_blocks = int(np.ceil(len(synthetic_times) / template_length))
    child_sequences = np.random.SeedSequence(seed).spawn(number_of_blocks)
    blocks: list[xr.Dataset] = []
    cursor = len(synthetic_times)
    backward_block = 0
    while cursor:
        start = max(0, cursor - template_length)
        length = cursor - start
        source = template.isel({time_dim: slice(0, length)}).copy(deep=True)
        source = source.assign_coords({time_dim: synthetic_times[start:cursor]})
        backward_block += 1
        transformed = _transform_synthetic_block(
            source,
            variable=variable,
            time_dim=time_dim,
            noise=noise,
            decay_reduction=decay * window_years * backward_block,
            rng=np.random.default_rng(child_sequences[backward_block - 1]),
        )
        blocks.append(transformed)
        cursor = start
    blocks.reverse()
    return blocks


def _transform_synthetic_block(
    block: xr.Dataset,
    *,
    variable: str,
    time_dim: str,
    noise: float,
    decay_reduction: float,
    rng: np.random.Generator,
) -> xr.Dataset:
    values = block[variable]
    transformed = values
    if decay_reduction:
        transformed = _decay_daily_amplitude(values, time_dim, decay_reduction)
    if noise:
        random_values = rng.normal(0.0, noise, size=values.shape)
        if np.issubdtype(values.dtype, np.floating):
            random_values = random_values.astype(values.dtype, copy=False)
        random_noise = xr.DataArray(
            random_values,
            dims=values.dims,
            coords={dim: values[dim] for dim in values.dims if dim in values.coords},
        )
        transformed = transformed + xr.where(values.notnull(), random_noise, 0)
    if np.issubdtype(values.dtype, np.floating):
        transformed = transformed.astype(values.dtype)
    transformed.attrs = dict(values.attrs)
    block[variable] = transformed
    return block


def _decay_daily_amplitude(values: xr.DataArray, time_dim: str, reduction: float) -> xr.DataArray:
    """Reduce mean absolute daily excursions while retaining the daily mean."""
    daily_mean = values.resample({time_dim: "1D"}).mean(skipna=True)
    mean_at_time = daily_mean.reindex({time_dim: values[time_dim]}, method="ffill")
    daily_amplitude = np.abs(values - mean_at_time).resample({time_dim: "1D"}).mean(
        skipna=True
    )
    amplitude_at_time = daily_amplitude.reindex({time_dim: values[time_dim]}, method="ffill")
    scale = xr.where(
        amplitude_at_time > 0,
        xr.where(amplitude_at_time > reduction, 1.0 - reduction / amplitude_at_time, 0.0),
        0.0,
    )
    return mean_at_time + (values - mean_at_time) * scale


def _restore_metadata(result: xr.Dataset, original: xr.Dataset) -> None:
    result.attrs = dict(original.attrs)
    for name in original.data_vars:
        result[name].attrs = dict(original[name].attrs)
    for name in original.coords:
        if name in result.coords:
            result.coords[name].attrs = dict(original.coords[name].attrs)


def _resolve_time_dim(source: xr.DataArray, requested: str | None) -> str:
    if requested is not None:
        if requested not in source.dims:
            raise ValueError(f"time dimension {requested!r} is not present in {source.name!r}")
        return requested
    named = [dimension for dimension in source.dims if dimension.lower() == "time"]
    if len(named) == 1:
        return named[0]
    datetime_dimensions = [
        dimension
        for dimension in source.dims
        if dimension in source.coords
        and (
            np.issubdtype(source[dimension].dtype, np.datetime64)
            or source[dimension].dtype == object
        )
    ]
    if len(datetime_dimensions) == 1:
        return datetime_dimensions[0]
    raise ValueError("could not identify a time dimension; pass time_dim explicitly")


def _validate_regular_time_axis(time_values: np.ndarray, time_dim: str) -> Any:
    if time_values.ndim != 1 or len(time_values) < 2:
        raise ValueError(f"{time_dim!r} must contain at least two timestamps")
    differences = np.diff(time_values)
    cadence = differences[0]
    if any(delta != cadence for delta in differences):
        raise ValueError("synthetic extension requires a strictly regular time axis")
    if not _is_positive_duration(cadence):
        raise ValueError("time coordinate must be strictly increasing")
    return cadence


def _is_positive_duration(value: Any) -> bool:
    if isinstance(value, np.timedelta64):
        return value > np.timedelta64(0, "ns")
    return value > timedelta(0)


def _first_valid_index(values: xr.DataArray, time_dim: str) -> int:
    spatial_dims = [dimension for dimension in values.dims if dimension != time_dim]
    valid_by_time = values.notnull()
    if spatial_dims:
        valid_by_time = valid_by_time.any(dim=spatial_dims)
    indices = np.flatnonzero(np.asarray(valid_by_time.values, dtype=bool))
    if not len(indices):
        raise ValueError(f"selected variable {values.name!r} contains no valid values")
    return int(indices[0])


def _parse_window(value: str) -> Window:
    match = _WINDOW_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError("window must use a positive integer followed by Y, M, D, or H (for example 1Y)")
    return Window(count=int(match["count"]), unit=match["unit"].upper())


def _parse_fake_start(value: Any, reference: Any, uses_numpy_datetime: bool) -> Any:
    if uses_numpy_datetime:
        return pd.Timestamp(value).to_datetime64()
    parsed = pd.Timestamp(value).to_pydatetime()
    constructor = type(reference)
    arguments = (
        parsed.year,
        parsed.month,
        parsed.day,
        parsed.hour,
        parsed.minute,
        parsed.second,
        parsed.microsecond,
    )
    calendar = getattr(reference, "calendar", None)
    try:
        return constructor(*arguments, calendar=calendar)
    except TypeError:
        return constructor(*arguments)


def _calendar_shift(value: Any, window: Window, uses_numpy_datetime: bool) -> Any:
    if uses_numpy_datetime:
        timestamp = pd.Timestamp(value)
        if window.unit == "Y":
            return (timestamp + pd.DateOffset(years=window.count)).to_datetime64()
        if window.unit == "M":
            return (timestamp + pd.DateOffset(months=window.count)).to_datetime64()
        if window.unit == "D":
            return (timestamp + pd.DateOffset(days=window.count)).to_datetime64()
        return (timestamp + pd.DateOffset(hours=window.count)).to_datetime64()
    if window.unit == "H":
        return value + timedelta(hours=window.count)
    if window.unit == "D":
        return value + timedelta(days=window.count)
    months = window.count * (12 if window.unit == "Y" else 1)
    return _shift_months(value, months)


def _shift_months(value: Any, months: int) -> Any:
    total_month = value.year * 12 + (value.month - 1) + months
    year, month_index = divmod(total_month, 12)
    month = month_index + 1
    first_of_month = value.replace(year=year, month=month, day=1)
    if month == 12:
        next_month = first_of_month.replace(year=year + 1, month=1)
    else:
        next_month = first_of_month.replace(month=month + 1)
    last_day = (next_month - timedelta(days=1)).day
    return value.replace(year=year, month=month, day=min(value.day, last_day))


def _template_stop_index(
    time_values: np.ndarray,
    valid_index: int,
    template_end: Any,
    cadence: Any,
) -> int:
    stop = next((index for index, value in enumerate(time_values) if value >= template_end), None)
    if stop is None:
        if time_values[-1] + cadence != template_end:
            raise ValueError("input record is shorter than the requested replication window")
        stop = len(time_values)
    if stop <= valid_index:
        raise ValueError("input record is shorter than the requested replication window")
    return stop


def _synthetic_count(fake_start: Any, original_start: Any, cadence: Any) -> int:
    elapsed = original_start - fake_start
    if elapsed <= 0:
        return 0
    count = int(round(float(elapsed / cadence)))
    if fake_start + count * cadence != original_start:
        raise ValueError("fake_start must align with the input time cadence")
    return count


def _synthetic_times(
    fake_start: Any,
    count: int,
    cadence: Any,
    uses_numpy_datetime: bool,
) -> np.ndarray:
    if uses_numpy_datetime:
        return fake_start + np.arange(count, dtype=np.int64) * cadence
    return np.asarray([fake_start + index * cadence for index in range(count)], dtype=object)


def _marker_time_dim(ds: xr.Dataset) -> str:
    if _SYNTHETIC_MARKER not in ds:
        raise KeyError(f"{_SYNTHETIC_MARKER!r} is not present in the dataset")
    dimensions = ds[_SYNTHETIC_MARKER].dims
    if len(dimensions) != 1:
        raise ValueError(f"{_SYNTHETIC_MARKER!r} must have exactly one dimension")
    return dimensions[0]


def _format_time(value: Any) -> str:
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value, unit="s"))
    return str(value.isoformat())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Existing NetCDF source file")
    parser.add_argument("output", type=Path, help="New NetCDF output file")
    parser.add_argument("--variable", required=True, help="Variable to replicate and transform")
    parser.add_argument("--fake-start", required=True, help="First synthetic timestamp, e.g. 1980-01-01")
    parser.add_argument("--window", default="1Y", help="Replication window, e.g. 1Y or 6M")
    parser.add_argument("--noise", type=float, default=0.0, help="Gaussian noise sigma")
    parser.add_argument("--decay", type=float, default=0.0, help="Daily-amplitude decay per year")
    parser.add_argument("--seed", type=int, help="Reproducible random seed")
    parser.add_argument("--time-dim", help="Explicit time dimension")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    summary = extend_file_backward(
        args.input,
        args.output,
        variable=args.variable,
        fake_start=args.fake_start,
        window=args.window,
        noise=args.noise,
        decay=args.decay,
        seed=args.seed,
        time_dim=args.time_dim,
        overwrite=args.overwrite,
    )
    for label, value in summary.items():
        print(f"{label}: {value}")


if __name__ == "__main__":
    _main()
