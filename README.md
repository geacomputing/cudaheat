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
- Per-batch host-read, preparation, allocation, transfer, CUDA-event compute, result-assembly, OOM-retry, and VRAM telemetry.
- Explicit preallocated device-buffer transfers, avoiding an unnecessary host/device conversion in the H2D path.
- Separate benchmark utilities with CSV/JSON export, a timestamped campaign runner, and high-resolution Matplotlib figures (PNG/PDF/SVG).

The climatological cycle has 366 × 24 month/day/hour bins. Smoothing is circular and NaN-aware. NetCDF I/O stays on CPU through Xarray; CuPy performs GPU calculations. The `kernels` namespace is reserved for future profiling-justified CUDA event-detection kernels.

## Install

```bash
pip install -e '.[gpu,benchmark,test]'
```

For bounded-RAM Dask-backed NetCDF reads, install `.[lazy]` and pass
`lazy=True` to `CUDAHeat`. CUDAHeat still transfers complete time series and
uses spatial batches only.

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

Targeted validation keeps every time step for a seeded selection of valid ocean
points. It compares CPU float64 with GPU float32 (or float64) for raw and
smoothed climatology/thresholds, anomaly, excess, mask, and exceedance counts.
It returns a `ValidationResult` with numerical, NaN, threshold-boundary, and
count diagnostics, and can write JSON without large arrays.

```bash
python examples/validate_real_data.py sst.nc \
  --variable analysed_sst --baseline 2019 2025 --points 50
```

Use `--full-dataset` only for the optional full spatial validation; it still
batches space only and keeps every time series complete.

```python
from cudaheat.benchmark import CUDAHeatBenchmark

benchmark = CUDAHeatBenchmark(hw)
benchmark.run(repeats=3, warmup=1, point_counts=[100, 1000], gpu_dtypes=["float32"])
benchmark.export_json("benchmark-results/run.json")
```

Benchmarks retain real repeat measurements only. They separately report CPU
compute, host-to-device transfer, CUDA-event GPU compute, device-to-host
transfer, end-to-end timing, both speedups, throughput, and batch telemetry.
Spatial and controlled-batch scaling never shorten time. Use the real-data CLI
to create CSV, JSON, and actual-measurement Matplotlib plots:

```bash
python benchmarks/benchmark_real_data.py sst.nc \
  --variable analysed_sst --baseline 2019 2025 --dtype float32 --repeats 3
```

Outputs default to `benchmark-results/`; use `--output-dir` to select another
location. No multi-GPU performance result is generated until multi-GPU execution exists.

For a complete validation, dense scaling, batch-size, precision, figure, and
report campaign on a real dataset:

```bash
python benchmarks/run_real_data_campaign.py /path/to/sst.nc \
  --variable analysed_sst --baseline 2019 2025 \
  --percentile 0.90 --smooth-window-hours 120
```

Each campaign writes a non-overwriting timestamped directory containing raw
CSV/JSON repetitions, summaries, validation JSON, a Markdown report, and 12
large PNG/PDF/SVG figures. Warm-ups are excluded from the stored results.

## Measured real-data result

On an NVIDIA GeForce RTX 4070 Laptop GPU, CUDAHeat processed the provided
7-year `analysed_sst` dataset (64,080 hours × 18,055 points; 1.157 billion SST
values) without splitting time. These are mean float32-GPU / float64-CPU
measurements from the dense campaign:

| Metric | Measured result |
| --- | ---: |
| CPU float64 compute | 123.10 s |
| GPU float32 compute | 5.26 s |
| GPU float32 device end-to-end | 5.63 s |
| Complete GPU pipeline wall time | 7.73 s |
| Compute / device end-to-end speedup | 23.40× / 21.88× |
| Automatic batch / peak VRAM | 2,276 points / 5.33 GiB |

The explicit transfer path reduced measured H2D time from 4.924 s to 0.353 s
(14.0×); GPU computation is now the dominant device cost. GPU32 validation at
500 points across seeds 0, 42, and 123 retained at least 99.99911985% mask
agreement; a 1,000-point run retained 99.99911205%. GPU64 at 500 points
retained 99.99973783% agreement. Differences were threshold-boundary cases;
no NaN-location disagreements were observed.

## Roadmap

`detect()` is a documented placeholder for complete-time GPU event segmentation. The sequential first milestone is device-explicit; a future scheduler can distribute independent spatial batches across `devices=[0, 1, ...]` without rewriting the scientific core. Validate results, precision, and memory telemetry for every SST product before scientific use.
