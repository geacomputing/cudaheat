"""Standalone spatial-scaling benchmark for the 40-year CUDAHeat SST dataset.

The already compact ``time × point`` input is deliberately used as the source:
it has no permanent-land points and avoids repeating grid flattening for every
scale.  ``optimize_netcdf`` is intentionally not used here because it accepts
a latitude/longitude grid, not an already flattened point layout.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import h5netcdf
import h5py
import matplotlib.pyplot as plt
import numpy as np

from cudaheat import CUDAHeat
from cudaheat.memory import cupy


# ---------------------------------------------------------------------------
# Hard-coded experiment configuration. Edit here; this script has no CLI.

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent
ORIGINAL_GRID_SOURCE = (
    BENCHMARKS_DIR / "extend_NETCDF_create40_years_timespan" / "sst_40_years.nc"
)
OPTIMIZED_POINT_SOURCE = (
    BENCHMARKS_DIR / "extend_NETCDF_create40_years_timespan" / "sst_40yrs_optimzed.nc"
)
# Keep generated NetCDF subsets, reports, and figures beside this script.
OUTPUT_DIRECTORY = SCRIPT_DIR
DATASET_DIRECTORY = OUTPUT_DIRECTORY / "datasets"

VARIABLE = "analysed_sst"
RANDOM_SEED = 20261014
REPETITIONS = 1
RUN_WARMUP = True

POINT_COUNTS = (2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000)
#POINT_COUNTS = (2000, 4000)

# These are the chunks of the inspected 40-year compact source. A one-year
# time chunk and 128 spatial points yield about 4.3 MiB per float32 SST chunk.
TIME_CHUNK = 8784
POINT_CHUNK = 128

BASELINE = (2019, 2030)
PERCENTILE = 0.90
SMOOTH_WINDOW_HOURS = 24 * 7
MIN_DURATION_HOURS = 120
MAX_INTERNAL_GAP_HOURS = 6
JOIN_GAP_HOURS = 48


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DATASET_DIRECTORY.mkdir(parents=True, exist_ok=True)

    original = inspect_file(ORIGINAL_GRID_SOURCE)
    source = inspect_file(OPTIMIZED_POINT_SOURCE)
    validate_source_layout(original, source)

    total_points = source["npoints"]
    selected_counts = [count for count in POINT_COUNTS if count <= total_points]
    if total_points not in selected_counts:
        selected_counts.append(total_points)
    if len(selected_counts) != len(POINT_COUNTS) + 1:
        print("Some requested point counts exceed the available ocean points and were skipped.")

    print(
        f"Using compact source {OPTIMIZED_POINT_SOURCE} with {total_points:,} valid ocean points; "
        f"the original grid source is inspected only for provenance."
    )
    print(
        f"Physical output chunks: time={TIME_CHUNK}, point={POINT_CHUNK}; "
        f"time steps={source['ntime']:,}."
    )

    # One permutation makes every smaller data set a strict subset of every
    # larger one. Sorting only changes storage order, not point membership.
    permutation = np.random.default_rng(RANDOM_SEED).permutation(total_points)
    datasets: dict[int, Path] = {}
    preparation: dict[int, dict[str, Any]] = {}
    for count in selected_counts:
        source_points = np.sort(permutation[:count])
        path = DATASET_DIRECTORY / dataset_name(count, total_points)
        preparation[count] = prepare_dataset(path, source, source_points)
        datasets[count] = path

    # A warm-up is intentionally excluded from result files. It also proves
    # that a prepared file can be opened by CUDAHeat before timed execution.
    if RUN_WARMUP:
        warmup_count = min(selected_counts)
        print(f"Warm-up: {warmup_count:,} points (excluded from measurements)")
        run_cudaheat(datasets[warmup_count])

    executions: list[dict[str, Any]] = []
    order_rng = np.random.default_rng(RANDOM_SEED + 1)
    execution_number = 0
    for repeat in range(1, REPETITIONS + 1):
        for count in order_rng.permutation(selected_counts):
            count = int(count)
            execution_number += 1
            print(
                f"Measurement {execution_number}/{len(selected_counts) * REPETITIONS}: "
                f"{count:,} points, repetition {repeat}/{REPETITIONS}",
                flush=True,
            )
            record = run_cudaheat(datasets[count])
            record.update(
                point_count=count,
                repetition=repeat,
                execution_number=execution_number,
                dataset=str(datasets[count]),
            )
            executions.append(record)

    summaries = summarise(executions, selected_counts)
    write_results(source, selected_counts, preparation, executions, summaries)
    make_figures(executions, summaries)
    print(f"Spatial-scaling artifacts written to {OUTPUT_DIRECTORY}")


def inspect_file(path: Path) -> dict[str, Any]:
    """Read storage metadata only; this never materializes SST values."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5netcdf.File(path, "r") as netcdf_file, h5py.File(path, "r") as hdf_file:
        if VARIABLE not in netcdf_file.variables:
            raise KeyError(f"{VARIABLE!r} is absent from {path}")
        netcdf_data = netcdf_file.variables[VARIABLE]
        data = hdf_file[VARIABLE]
        return {
            "path": path,
            "mtime_ns": path.stat().st_mtime_ns,
            "size_bytes": path.stat().st_size,
            "variable_dimensions": tuple(netcdf_data.dimensions),
            "shape": tuple(int(size) for size in data.shape),
            "dtype": np.dtype(data.dtype),
            "chunks": None if data.chunks is None else tuple(int(size) for size in data.chunks),
            "compression": data.compression,
            "compression_opts": data.compression_opts,
            "shuffle": bool(data.shuffle),
            "fletcher32": bool(data.fletcher32),
            "fillvalue": np.asarray(data.fillvalue).item(),
            "attrs": dict(netcdf_file.attrs),
            "ntime": int(data.shape[0]),
            "npoints": int(data.shape[1]) if data.ndim == 2 else 0,
        }


def validate_source_layout(original: dict[str, Any], source: dict[str, Any]) -> None:
    """Confirm the compact 40-year input is the appropriate benchmark source."""
    if original["shape"][:1] != source["shape"][:1]:
        raise ValueError("the original and optimized inputs do not have the same time length")
    if source["variable_dimensions"] != ("time", "point"):
        raise ValueError("the optimized source must use dimensions (time, point)")
    if source["attrs"].get("cudaheat_preprocessed") != "true":
        raise ValueError("the optimized source lacks CUDAHeat's preprocessing marker")
    if source["chunks"] != (TIME_CHUNK, POINT_CHUNK):
        raise ValueError(
            f"configured chunks {(TIME_CHUNK, POINT_CHUNK)} do not match source chunks "
            f"{source['chunks']}"
        )
    if source["dtype"] != np.dtype("float32"):
        raise ValueError(f"expected float32 source data, received {source['dtype']}")


def dataset_name(count: int, total_points: int) -> str:
    return "sst_full.nc" if count == total_points else f"sst_{count:05d}.nc"


def prepare_dataset(path: Path, source: dict[str, Any], source_points: np.ndarray) -> dict[str, Any]:
    """Reuse a verified subset or stream-write it in physical output chunks."""
    started = time.perf_counter()
    if validate_dataset(path, source, source_points, raise_on_error=False):
        return {
            "status": "reused",
            "seconds": time.perf_counter() - started,
            "size_bytes": path.stat().st_size,
        }

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        write_subset(temporary, source, source_points)
        validate_dataset(temporary, source, source_points, raise_on_error=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "status": "created",
        "seconds": time.perf_counter() - started,
        "size_bytes": path.stat().st_size,
    }


def write_subset(path: Path, source: dict[str, Any], source_points: np.ndarray) -> None:
    """Copy a subset without loading a full time-by-point cube into RAM."""
    with h5netcdf.File(OPTIMIZED_POINT_SOURCE, "r") as input_file, h5netcdf.File(
        path, "w"
    ) as output_file:
        for name, dimension in input_file.dimensions.items():
            output_file.dimensions[name] = (
                source_points.size if name == "point" else len(dimension)
            )
        output_file.attrs.update(dict(input_file.attrs))
        output_file.attrs.update(
            {
                "cudaheat_spatial_scaling_source": str(OPTIMIZED_POINT_SOURCE.resolve()),
                "cudaheat_spatial_scaling_source_mtime_ns": source["mtime_ns"],
                "cudaheat_spatial_scaling_seed": RANDOM_SEED,
                "cudaheat_spatial_scaling_point_count": int(source_points.size),
                "cudaheat_spatial_scaling_chunks": np.asarray(
                    (TIME_CHUNK, POINT_CHUNK), dtype=np.int64
                ),
            }
        )

        # All non-SST variables are coordinates/topology arrays. They are at
        # most one-dimensional here, so copying them is bounded and preserves
        # latitude, longitude, original-grid mappings, and the full time axis.
        for name, variable in input_file.variables.items():
            if name in {VARIABLE, "point"}:
                continue
            dimensions = tuple(variable.dimensions)
            if "time" in dimensions and "point" in dimensions:
                raise ValueError(f"unsupported auxiliary time-and-point variable {name!r}")
            values = np.asarray(variable[:])
            if "point" in dimensions:
                values = np.take(values, source_points, axis=dimensions.index("point"))
            coordinate = output_file.create_variable(name, dimensions, dtype=values.dtype)
            coordinate[:] = values
            coordinate.attrs.update(attributes_without_fill_value(variable.attrs))

        # CUDAHeat expects local compact point positions, while this auxiliary
        # coordinate preserves the selected positions in the 40-year source.
        point = output_file.create_variable("point", ("point",), dtype="i8")
        point[:] = np.arange(source_points.size, dtype=np.int64)
        point.attrs.update(attributes_without_fill_value(input_file.variables["point"].attrs))
        source_index = output_file.create_variable("source_point_index", ("point",), dtype="i8")
        source_index[:] = source_points
        source_index.attrs["long_name"] = "point index in the optimized 40-year source"

        input_sst = input_file.variables[VARIABLE]
        output_sst = output_file.create_variable(
            VARIABLE,
            ("time", "point"),
            dtype=input_sst.dtype,
            **storage_options(source),
        )
        output_sst.attrs.update(attributes_without_fill_value(input_sst.attrs))
        output_sst.attrs["coordinates"] = (
            "latitude longitude original_lat_index original_lon_index original_flat_index "
            "source_point_index"
        )

        for time_start in range(0, source["ntime"], TIME_CHUNK):
            time_stop = min(source["ntime"], time_start + TIME_CHUNK)
            for output_start in range(0, source_points.size, POINT_CHUNK):
                output_stop = min(source_points.size, output_start + POINT_CHUNK)
                selected = source_points[output_start:output_stop]
                values = np.ascontiguousarray(input_sst[time_start:time_stop, selected])
                output_sst[time_start:time_stop, output_start:output_stop] = values


def storage_options(source: dict[str, Any]) -> dict[str, Any]:
    """Preserve the inspected storage filters; the supplied source is uncompressed."""
    options: dict[str, Any] = {
        "chunks": (TIME_CHUNK, POINT_CHUNK),
        "fillvalue": source["fillvalue"],
    }
    if source["compression"] is None:
        return options
    if source["compression"] != "gzip":
        raise ValueError(f"unsupported source compression: {source['compression']!r}")
    options.update(
        compression="gzip",
        compression_opts=source["compression_opts"],
        shuffle=source["shuffle"],
        fletcher32=source["fletcher32"],
    )
    return options


def attributes_without_fill_value(attributes: Any) -> dict[str, Any]:
    """``_FillValue`` is supplied at variable creation and cannot be reset later."""
    return {name: value for name, value in attributes.items() if name != "_FillValue"}


def validate_dataset(
    path: Path,
    source: dict[str, Any],
    source_points: np.ndarray,
    *,
    raise_on_error: bool,
) -> bool:
    """Validate metadata exactly and SST values with bounded representative reads."""
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(OPTIMIZED_POINT_SOURCE, "r") as input_file, h5py.File(path, "r") as output_file:
            output_sst = output_file[VARIABLE]
            if output_sst.shape != (source["ntime"], source_points.size):
                raise ValueError("unexpected SST shape")
            if np.dtype(output_sst.dtype) != source["dtype"]:
                raise ValueError("SST dtype changed")
            if output_sst.chunks != (TIME_CHUNK, POINT_CHUNK):
                raise ValueError(f"unexpected physical chunks {output_sst.chunks}")
            if output_sst.compression != source["compression"]:
                raise ValueError("SST compression changed")
            if output_sst.compression_opts != source["compression_opts"]:
                raise ValueError("SST compression options changed")
            if bool(output_sst.shuffle) != source["shuffle"]:
                raise ValueError("SST shuffle filter changed")
            if bool(output_sst.fletcher32) != source["fletcher32"]:
                raise ValueError("SST checksum filter changed")
            if output_file.attrs.get("cudaheat_preprocessed") != "true":
                raise ValueError("CUDAHeat preprocessing marker is absent")
            if output_file.attrs.get("cudaheat_spatial_scaling_source") != str(
                OPTIMIZED_POINT_SOURCE.resolve()
            ):
                raise ValueError("subset was made from a different source")
            if integer_attribute(output_file.attrs, "cudaheat_spatial_scaling_source_mtime_ns") != source[
                "mtime_ns"
            ]:
                raise ValueError("subset source revision differs")
            if integer_attribute(output_file.attrs, "cudaheat_spatial_scaling_seed") != RANDOM_SEED:
                raise ValueError("subset random seed differs")
            if (
                integer_attribute(output_file.attrs, "cudaheat_spatial_scaling_point_count")
                != source_points.size
            ):
                raise ValueError("subset point count marker differs")
            if not np.array_equal(
                output_file.attrs.get("cudaheat_spatial_scaling_chunks"),
                np.asarray((TIME_CHUNK, POINT_CHUNK), dtype=np.int64),
            ):
                raise ValueError("subset chunk marker differs")
            if not np.array_equal(output_file["source_point_index"][:], source_points):
                raise ValueError("subset point selection differs")
            if not np.array_equal(output_file["time"][:], input_file["time"][:]):
                raise ValueError("time coordinate changed")
            if not np.array_equal(output_file["point"][:], np.arange(source_points.size)):
                raise ValueError("local point coordinate changed")
            for name in ("latitude", "longitude", "original_lat_index", "original_lon_index", "original_flat_index"):
                if name not in output_file:
                    raise ValueError(f"required coordinate {name!r} is absent")
                if not np.array_equal(output_file[name][:], input_file[name][source_points]):
                    raise ValueError(f"coordinate {name!r} changed")

            # The source marker guarantees selected points are already valid
            # ocean points. These bounded samples verify time-series values and
            # NaN semantics without reading a complete 40-year subset.
            time_samples = np.unique(np.array((0, source["ntime"] // 2, source["ntime"] - 1)))
            point_samples = np.unique(
                np.array((0, source_points.size // 3, (2 * source_points.size) // 3, source_points.size - 1))
            )
            for time_index in time_samples:
                expected = input_file[VARIABLE][time_index, source_points[point_samples]]
                actual = output_sst[time_index, point_samples]
                if not np.array_equal(expected, actual, equal_nan=True):
                    raise ValueError("representative SST values changed")
    except (KeyError, OSError, TypeError, ValueError) as error:
        if raise_on_error:
            raise RuntimeError(f"validation failed for {path}: {error}") from error
        return False
    return True


def integer_attribute(attributes: Any, name: str) -> int:
    """Read HDF5 scalar attributes regardless of scalar-versus-length-one storage."""
    value = np.asarray(attributes.get(name, -1))
    if value.size != 1:
        raise ValueError(f"attribute {name!r} is not scalar")
    return int(value.item())


def run_cudaheat(path: Path) -> dict[str, Any]:
    """Run one complete CUDAHeat detection measurement and aggregate telemetry."""
    started = time.perf_counter()
    with CUDAHeat(
        path,
        variable=VARIABLE,
        timing_logging=True,
        input_optimization="auto",
        batch_size="guess",
    ) as hw:
        result = hw.detect(
            baseline=BASELINE,
            percentile=PERCENTILE,
            smooth_window_hours=SMOOTH_WINDOW_HOURS,
            min_duration_hours=MIN_DURATION_HOURS,
            max_internal_gap_hours=MAX_INTERNAL_GAP_HOURS,
            join_gap_hours=JOIN_GAP_HOURS,
        )
        cp = cupy()
        cp.cuda.Stream.null.synchronize()
        npoints = hw.npoints
        ntime = hw.ntime
        gpu_name = hw.gpu_info.name
    elapsed = time.perf_counter() - started
    telemetry = result.telemetry
    if not telemetry:
        raise RuntimeError("CUDAHeat returned no batch telemetry")
    return {
        "npoints": int(npoints),
        "ntime": int(ntime),
        "observations": npoints * ntime,
        "total_seconds": elapsed,
        "host_read_seconds": sum(item.host_read_seconds for item in telemetry),
        "host_prepare_seconds": sum(item.host_prepare_seconds for item in telemetry),
        "transfer_to_device_seconds": sum(item.transfer_to_device_seconds for item in telemetry),
        "gpu_compute_seconds": sum(item.gpu_compute_seconds for item in telemetry),
        "transfer_from_device_seconds": sum(item.transfer_from_device_seconds for item in telemetry),
        "effective_batch_size": max(item.batch_size for item in telemetry),
        "spatial_batches": len(telemetry),
        "peak_used_vram_bytes": max(item.peak_used_bytes for item in telemetry),
        "throughput_point_hours_per_second": (npoints * ntime) / elapsed,
        "gpu_name": gpu_name,
    }


def summarise(executions: list[dict[str, Any]], point_counts: list[int]) -> list[dict[str, Any]]:
    """Compute mean and sample standard deviation for every spatial scale."""
    summaries: list[dict[str, Any]] = []
    numeric_metrics = (
        "total_seconds",
        "host_read_seconds",
        "host_prepare_seconds",
        "transfer_to_device_seconds",
        "gpu_compute_seconds",
        "transfer_from_device_seconds",
        "effective_batch_size",
        "spatial_batches",
        "peak_used_vram_bytes",
        "throughput_point_hours_per_second",
    )
    for count in sorted(point_counts):
        rows = [row for row in executions if row["point_count"] == count]
        summary: dict[str, Any] = {
            "point_count": count,
            "ntime": rows[0]["ntime"],
            "observations": rows[0]["observations"],
            "repetitions": len(rows),
        }
        for metric in numeric_metrics:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[f"mean_{metric}"] = float(np.mean(values))
            summary[f"std_{metric}"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summaries.append(summary)
    return summaries


def write_results(
    source: dict[str, Any],
    point_counts: list[int],
    preparation: dict[int, dict[str, Any]],
    executions: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    """Write every repeat plus the reproducibility metadata and summaries."""
    csv_path = OUTPUT_DIRECTORY / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(executions[0]))
        writer.writeheader()
        writer.writerows(executions)

    payload = {
        "source": {
            "original_grid": str(ORIGINAL_GRID_SOURCE),
            "optimized_point_source": str(OPTIMIZED_POINT_SOURCE),
            "source_size_bytes": source["size_bytes"],
            "source_mtime_ns": source["mtime_ns"],
            "ntime": source["ntime"],
            "valid_ocean_points": source["npoints"],
            "dtype": source["dtype"].name,
            "chunks": list(source["chunks"]),
            "compression": source["compression"],
            "compression_options": source["compression_opts"],
            "fillvalue": source["fillvalue"],
        },
        "selection": {
            "random_seed": RANDOM_SEED,
            "point_counts": point_counts,
            "selection": "first N positions of one random permutation, then sorted for I/O",
        },
        "preparation": {str(count): values for count, values in preparation.items()},
        "cudaheat": {
            "variable": VARIABLE,
            "batch_size": "guess",
            "input_optimization": "auto",
            "baseline": BASELINE,
            "percentile": PERCENTILE,
            "smooth_window_hours": SMOOTH_WINDOW_HOURS,
            "min_duration_hours": MIN_DURATION_HOURS,
            "max_internal_gap_hours": MAX_INTERNAL_GAP_HOURS,
            "join_gap_hours": JOIN_GAP_HOURS,
            "repetitions": REPETITIONS,
            "warmup_excluded": RUN_WARMUP,
        },
        "executions": executions,
        "summaries": summaries,
    }
    (OUTPUT_DIRECTORY / "results.json").write_text(
        json.dumps(json_ready(payload), indent=2), encoding="utf-8"
    )


def make_figures(executions: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    """Create the four requested figures from measured, not inferred, metrics."""
    x = np.asarray([row["point_count"] for row in summaries])
    by_points = {count: [row for row in executions if row["point_count"] == count] for count in x}

    figure, axis = plt.subplots(figsize=(8, 5))
    for count, rows in by_points.items():
        axis.scatter([count] * len(rows), [row["total_seconds"] for row in rows], color="tab:blue", alpha=0.6)
    axis.errorbar(
        x,
        [row["mean_total_seconds"] for row in summaries],
        yerr=[row["std_total_seconds"] for row in summaries],
        color="black",
        marker="o",
        label="mean ± sample standard deviation",
    )
    axis.set(xlabel="Valid ocean points", ylabel="Total execution time (s)", title="CUDAHeat spatial scaling")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "execution_time.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for metric, label in (
        ("host_read_seconds", "Host NetCDF read"),
        ("transfer_to_device_seconds", "CPU → GPU"),
        ("gpu_compute_seconds", "GPU compute"),
        ("transfer_from_device_seconds", "GPU → CPU"),
    ):
        axis.plot(x, [row[f"mean_{metric}"] for row in summaries], marker="o", label=label)
    axis.set(xlabel="Valid ocean points", ylabel="Measured time (s)", title="CUDAHeat pipeline anatomy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "pipeline_anatomy.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for count, rows in by_points.items():
        axis.scatter(
            [count] * len(rows),
            [row["throughput_point_hours_per_second"] for row in rows],
            color="tab:green",
            alpha=0.6,
        )
    axis.plot(
        x,
        [row["mean_throughput_point_hours_per_second"] for row in summaries],
        color="black",
        marker="o",
    )
    axis.set(
        xlabel="Valid ocean points",
        ylabel="Point-hours processed per second",
        title="CUDAHeat throughput",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "throughput.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for axis, metric, label, scale in (
        (axes[0], "effective_batch_size", "Effective batch size (points)", 1.0),
        (axes[1], "spatial_batches", "Spatial batches", 1.0),
        (axes[2], "peak_used_vram_bytes", "Peak GPU VRAM (MiB)", 2**20),
    ):
        axis.plot(x, [row[f"mean_{metric}"] / scale for row in summaries], marker="o")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[2].set_xlabel("Valid ocean points")
    figure.suptitle("CUDAHeat GPU memory and batching")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "gpu_memory.png", dpi=180)
    plt.close(figure)


def json_ready(value: Any) -> Any:
    """Convert NumPy values used in metadata into standard JSON values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
