# CUDAHeat

Experimental GPU-native preprocessing of hourly SST data for Marine Heatwave analysis. Event detection is intentionally not implemented yet.

## Architecture

```
NetCDF -> Xarray / CPU -> complete time x spatial batch -> CuPy / CUDA
```

Time is scientifically indivisible; space is computationally divisible. CUDAHeat stacks spatial dimensions into `point` and transfers full `(time, point_batch)` arrays. It never splits time. On an OOM it clears CuPy memory, halves the spatial batch, and retries the same complete time series. One series that cannot fit raises a clear exception.

## Current milestone

- Runtime NVIDIA VRAM discovery and conservative VRAM-aware point batching.
- Configurable 30% free-VRAM safety reserve.
- Explicit `float32` default or `float64` processing.
- NaN-aware hourly calendar-bin climatology, percentile thresholds, circular smoothing, anomaly, excess, and GPU exceedance mask.
- Per-batch CUDA-event compute, transfer, OOM-retry, and VRAM telemetry.
- Separate benchmark utilities with CSV/JSON export and Matplotlib plots.

The climatological cycle has 366 × 24 month/day/hour bins. Smoothing is circular and NaN-aware. NetCDF I/O stays on CPU through Xarray; CuPy performs GPU calculations. The `kernels` namespace is reserved for future profiling-justified CUDA event-detection kernels.

## Install

```bash
pip install -e '.[gpu,benchmark,test]'
```

Install the CuPy extra matching the installed CUDA driver. Base-package imports work without CuPy, but `CUDAHeat` requires an NVIDIA CUDA device.

## Minimal use

```python
from cudaheat import CUDAHeat

with CUDAHeat("sst.nc", variable="analysed_sst", dtype="float32", vram_safety=0.30, devices=[0]) as hw:
    print(hw.ntime, hw.npoints, hw.batch_size, hw.gpu_info)
    result = hw.run(baseline=(1982, 2011), percentile=0.90, smooth_window_hours=120)
    print(result.exceedance_count)

    # Explicit, host-memory-intensive output for manageable datasets only.
    mask = hw.mask(baseline=(1982, 2011))
```

`run()` returns compact per-point exceedance counts by default. For production-size outputs, request fields with `return_fields` and use `on_batch` to persist every spatial batch without retaining all `(time, point)` values in CPU RAM.

## Validation and benchmarks

`cudaheat.climatology.cpu_pre_event_fields` is the NumPy reference used for CPU/GPU tests, including NaNs and float32 tolerances. Run tests with `pytest`.

```python
from cudaheat.benchmark import CUDAHeatBenchmark

benchmark = CUDAHeatBenchmark(hw)
benchmark.run(repeats=3)
benchmark.export_json("benchmark-results/run.json")
```

Benchmarks retain real repeat measurements only. GPU compute timing uses CUDA events and is distinct from transfer and end-to-end timing.

## Roadmap

`detect()` is a documented placeholder for complete-time GPU event segmentation. The sequential first milestone is device-explicit; a future scheduler can distribute independent spatial batches across `devices=[0, 1, ...]` without rewriting the scientific core. Validate results, precision, and memory telemetry for every SST product before scientific use.
