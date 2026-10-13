# CUDAHeat GPU Pipeline Benchmark

This timestamped campaign contains GPU-only performance measurements, GPU
precision comparisons, VRAM/batch behaviour, a complete pipeline anatomy, and
untimed numerical validation. CPU performance timing, Dask chunk sensitivity,
and lazy-versus-eager comparisons are deliberately absent.

- Dataset: `/home/user/Documents/Gea/cudaheat/examples/sst_cudaheat.nc`
- Scope: Numerical preprocessing only; marine-heatwave taxonomy and event classification are excluded.
- Raw repeat measurements: `gpu_measurements.csv` and `.json`
- Derived tables: `*_summary.csv/json`, `pipeline_anatomy.csv/json`, and `repeatability.csv/json`
- Figures: `*.png` and `*.svg`

Every JSON file carries a concise description, purpose, methodology, and units.
