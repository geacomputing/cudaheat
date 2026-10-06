# CUDAHeat Full-Pipeline Beta Benchmark

`benchmark_id`: `full_pipeline_beta`

## Scope

This campaign evaluates the integrated numerical mechanism only: lazy NetCDF ingestion, bounded host-memory spatial batching, H2D, GPU computation, D2H/results, and numerical validation. MHW taxonomy and event classification are excluded.

## Dataset and method

- Dataset: `/home/user/Documents/PhD.cy/lazy_loading/Python.Scripts/sst_7_years.nc`
- Scale: 18,055 spatial points × 64,080 hourly samples
- Processing: complete time series per point; spatial partitioning only; `lazy=True`
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU (7.61 GiB)
- Baseline / percentile / smoothing: [2019, 2022] / 90% / 120 h

## Largest measured performance configuration

- Points: 18,055
- GPU pipeline wall time: 7.901 s
- GPU end-to-end time: 3.744 s
- Pipeline speedup: 11.84×

## Numerical consistency

- GPU float32: 99.99882349% exceedance-mask agreement; 386 disagreements.
- GPU float64: 99.99985979% exceedance-mask agreement; 46 disagreements.

## Lazy versus eager isolated processes

- lazy: **complete**; peak private working memory 2.7884559631347656 GiB; process RSS 13.777179718017578 GiB; 
- eager: **complete**; peak private working memory 9.411808013916016 GiB; process RSS 18.18537139892578 GiB; 

`peak_private_working_gib` excludes file-backed mappings and is the primary
host-working-memory measure. `peak_rss_gib` is retained separately because
some NetCDF backends memory-map source files, which can inflate process RSS
without allocating equivalent private RAM.

## Dask chunk sensitivity and repeatability

- 64 points: 9.837 s pipeline wall; count hashes match: True.
- 256 points: 9.274 s pipeline wall; count hashes match: True.
- 1,024 points: 9.456 s pipeline wall; count hashes match: True.

The largest observed pipeline-wall coefficient of variation was 2.337%.

## Artifacts

- Metadata: `metadata/benchmark_metadata.json`
- Raw records: `raw/`
- Tables: `tables/`
- Presentation figures (PNG at 360 dpi, plus PDF/SVG): `figures/`
- Additional raw Dask chunk runs: `raw/dask_chunk_runs.csv`
- Repeatability table: `tables/repeatability.csv`
