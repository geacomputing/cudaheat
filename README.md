# CUDAHeat

CUDAHeat is a GPU-native Python package for hourly sea-surface-temperature
climatology, marine heatwave detection, sparse event results, enrichment, and
optional PostgreSQL persistence.

## Processing model

```
NetCDF / xarray -> complete time × spatial batch -> CuPy
```

CUDAHeat never partitions a time series. It stacks spatial dimensions into
`point`, transfers every time step for one spatial batch to the GPU, and
halves only the spatial batch after an out-of-memory error. A single series
that cannot fit raises `InsufficientVRAMError`.

The default GPU dtype is `float32`; `float64` is also supported. The default
VRAM reserve is 15% of currently free memory. `devices` accepts selected
device IDs, but batches currently execute sequentially on the first one.

## Install

```bash
pip install -e '.[gpu,test]'
```

Use `.[postgres]` for PostgreSQL persistence and `.[plot]` for plotting
examples. CuPy must match the installed CUDA driver. Importing the package does
not require CuPy, but creating `CUDAHeat` requires an available CUDA device.

## Basic use

```python
from cudaheat import CUDAHeat

with CUDAHeat("sst.nc", variable="analysed_sst", dtype="float32") as heat:
    result = heat.run(baseline=(1982, 2011))
    print(result.exceedance_count)
```

`run()` returns per-point exceedance counts by default. Use
`return_fields={"mask"}` or other documented fields only when the resulting
time-by-space output fits host memory. An `on_batch` callback can consume
requested fields one spatial batch at a time.

Convenience methods are:

- `climatology()` for the smoothed annual cycle.
- `thresholds()` for the timestamp-aligned threshold.
- `mask()` for an in-memory exceedance mask.
- `validate()` for CPU/GPU agreement diagnostics.

## Input preparation and batching

For a NetCDF path, `input_optimization="auto"` is the default. CUDAHeat uses
an existing `(time, point)` input unchanged or creates a validated compact
NetCDF cache outside the source file. The cache location is
`.cudaheat-cache/<source-sha>/` by default, or `CUDAHEAT_CACHE_DIR` when
set. The raw source SHA-256 and storage metadata prevent stale-cache reuse.

Use `input_optimization="skip"` to read the supplied layout directly, or
`"require"` to reject a non-optimized grid. `lazy=True` uses Dask-backed
NetCDF reads; it bounds I/O memory but does not change the complete-time GPU
processing rule.

`batch_size="guess"` chooses one of 512, 1024, 1536, or 2048 points near a
conservative VRAM target. An explicit positive batch size is also supported.

## Climatology and detection

The default climatology uses a 366 × 24 month/day/hour cycle, same-clock-hour
sampling over ±5 calendar days, leap-day interpolation, and 31-day (744-hour)
same-hour circular smoothing. A `dayofyear` calendar mode is also available.

```python
events = heat.detect(
    baseline=(1982, 2011),
    percentile=0.90,
    min_duration_hours=120,
    max_internal_gap_hours=6,
    join_gap_hours=48,
)
```

Detection retains raw exceedances, fills bounded internal gaps, finds
contiguous runs, applies the minimum duration, and joins nearby qualifying
runs. `detect()` creates or reuses a Zarr cache for the compact climatology
and threshold cycles when the input is a complete optimized path input.

The returned `EventDetectionResult` contains a sparse event catalogue. Use
`select_events()`, `point_events()`, `point_timeseries()`, or
`event_timeseries()` for targeted analysis. `to_xarray()` reconstructs a
Dask-backed, chunked dense event-ID field; avoid loading the entire field into
memory.

## Enrichment

Pass `enrich=True` to `detect()`, or call `cudaheat.enrich(result)` for
an existing result. CUDAHeat discovers YAML rules, validates and executes their
embedded calculations, and attaches event-aligned outputs plus rule
provenance. Rule directories are executable extensions and must be trusted.

The standalone enrichment path reads the optimized source and climatology
cache in storage-local batches. Integrated detection enrichment evaluates
rules on device-resident batch data and avoids a second source read.

## PostgreSQL persistence

```python
from cudaheat import load_postgres_config, persist_detection_result

run_id = persist_detection_result(events, load_postgres_config())
```

Copy `config/postgres.example.ini` to the ignored `config/postgres.ini`.
Passwords are resolved from the configured environment variable or standard
libpq configuration; they are not stored in the INI file. Persistence stores
run metadata, point identities, sparse events, and enrichment outputs—not SST
or dense climatology arrays.

## Package layout

- `core.py`: public GPU engine and orchestration.
- `climatology.py`, `detection.py`: numerical algorithms.
- `preprocessing.py`, `io.py`, `memory.py`, `climatology_cache.py`:
  input layout, bounded reads, VRAM policy, and cycle caching.
- `result.py`: sparse events and lazy dense reconstruction.
- `enrichment.py`, `postgres.py`: optional rule evaluation and storage.
- `validation.py`: CPU/GPU numerical diagnostics.

## Validation

With the GPU extra and an available CUDA device, run the suite with:

```bash
pytest -q
```

The suite creates its own small inputs, except for the real-data smoke-test
placeholder, which is skipped automatically because it requires a manually
provisioned dataset and CUDA device. For explicit real-data validation, use:

```bash
python examples/validate_real_data.py PATH_TO_SST.nc \
  --variable analysed_sst --baseline 2019 2025 --points 50
```

Validate each scientific SST product, precision choice, and memory setting
before interpreting results.
