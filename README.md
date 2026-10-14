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

For bounded-RAM Dask-backed NetCDF reads, pass `lazy=True` to `CUDAHeat`.
CUDAHeat still transfers complete time series and uses spatial batches only.

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

### Input and batch selection

For a NetCDF path, CUDAHeat now uses `input_optimization="auto"` by default.
It accepts an existing CUDAHeat `(time, point)` file unchanged; otherwise it
writes and validates a sibling such as `sst.cudaheat.nc`, never overwriting the
source file. The run metadata records both paths and the preparation result.

`batch_size="guess"` is the default. It immediately chooses the closest of
`512`, `1024`, `1536`, or `2048` to one quarter of the VRAM-safe ceiling.
Those values are multiples of CUDAHeat's standard 512-point optimized input
chunk. It does not benchmark, run trial CUDA work, or create a batch-size
cache. Set `batch_size` once on `CUDAHeat`; both `run()` and `detect()` use
that same engine-level value.

```python
# Default: optimize a path if needed, then make one immediate structural guess.
with CUDAHeat("sst.nc", variable="analysed_sst", batch_size="guess") as hw:
    result = hw.run(baseline=(1982, 2011))

# Fully reproducible explicit size.
with CUDAHeat("sst.nc", variable="analysed_sst", batch_size=512) as hw:
    result = hw.run(baseline=(1982, 2011))
```

Use `input_optimization="skip"` to retain direct grid input or
`input_optimization="require"` to reject an unoptimized file.

`run()` returns compact per-point exceedance counts by default. For production-size outputs, request fields with `return_fields` and use `on_batch` to persist every spatial batch without retaining all `(time, point)` values in CPU RAM.

`detect()` retains only a compact sparse event catalogue. `to_xarray()` and
`to_dense()` reconstruct a Dask-backed, chunked `event_id` cube from that
catalogue; they do not retain or allocate dense SST, threshold, or anomaly
fields. Zero means no event and positive values are catalogue IDs.

```python
dense = result.to_xarray(time_chunk_hours=168)
point = dense.isel(latitude=33, longitude=30).compute()
```

Do not call `.load()` or `.values` on the complete `dense` Dataset; that
deliberately materializes a full dense cube. Use the CPU-only
`result.select_events(...)` method for catalogue filtering, or write the
Dask-backed cube to a chunked store such as Zarr.

For an SST/threshold plot, reconstruct only the required point or event. This
reads one full time series and performs no CUDA work:

```python
event = result.event_timeseries(event_id=42, pad_hours=48)
event[["sst", "climatology", "threshold"]].to_array().plot.line(x="time")
```

`result.point_timeseries(latitude=33, longitude=30)` returns the full history
for one positional grid point; its keyword arguments follow `isel` indexing.
Use `result.point_events(latitude=33, longitude=30)` to get its event IDs from
the sparse catalogue without creating a dense cube.

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

When `batch_sizes` is omitted, a benchmark uses the engine's configured
`batch_size`. Pass `batch_sizes=[512, 1024]` to benchmark explicit sizes.

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
