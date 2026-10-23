# Numerical audit

## A. Overall assessment

**PARTIAL — not yet ready for scientific production.** CPU climatology and
event segmentation now follow the stated hourly Hobday adaptation, but GPU
execution could not be validated in this environment and the event catalogue
does not provide the requested intensity/integral metrics.

## B. Problems discovered

- The production climatology used exact calendar-hour samples and a 120-hour
  consecutive-bin smoother. It omitted Hobday's centred ±5-day sample pool,
  used the wrong default smoothing width, and blended clock hours.
- Public `CUDAHeat.detect()` defaults were `1/0/0`, not the stated
  `120/6/48` hours.
- Missing timestamps were silently counted as adjacent hours, so durations and
  gap limits could be wrong.
- The event-rule test adapter was disconnected from production detection.

Scientific consequence: thresholds, exceedances, and event counts could be
materially biased. On four real SST points (68,016 hours; 2019–2023 baseline),
the old calculation gave 95 events / 29,425 event-hours versus 49 / 12,744
with the corrected method.

## C. Corrections implemented

- `src/cudaheat/climatology.py`: same-hour ±5-day pooling, 31-day same-hour
  circular smoothing, NaN-aware 29-Feb interpolation, and an hourly-axis
  validator.
- `src/cudaheat/core.py`, `validation.py`, `benchmark/runner.py`, `result.py`:
  use and preserve the corrected method; public detection defaults are now
  120/6/48 h.
- `tests/test_numerical_audit.py`: independent Xarray climatology reference,
  randomized detector reference, all requested gap boundaries, NaNs, and time
  gaps. Existing detection-rule tests now exercise production code.

Method basis: [Hobday et al. (2016)](https://wernberglab.org/wp-content/uploads/2013/03/2016_hobday-et-al-pio-mhw-definition.pdf).

## D. Validation results

- 34 CPU audit/climatology/detection tests passed; 1 GPU test skipped.
- 6 independent legacy-Xarray reference tests passed; 2 GPU tests skipped.
- 24 broader non-GPU tests passed; 1 GPU test skipped.
- Float32/float64 real-subset threshold maximum difference: `2.27e-05` °C;
  6 exceedance disagreements in 272,064 comparisons.
- GPU kernels could not compile because this environment lacks CUDA toolkit
  headers; therefore CPU/GPU agreement for the new method is **not verified**.

## E. Outstanding issues

- Event output supplies start/end/duration only. It does not implement or
  validate exceedance fraction, continuous-exceedance length, SST/anomaly
  summary metrics, or °C·h integrals.
- A NaN is a raw non-exceedance but is currently fillable when it lies in a
  bounded internal gap ≤ `max_internal_gap_hours`. Whether unavailable data
  may bridge an event needs an explicit scientific policy.
- The full suite also has pre-existing collection blockers: missing
  `util.extend_netcdf_backward` and a real-data test that executes an external
  dataset at import time. Re-run GPU and full-suite validation after repairing
  the CUDA installation and those test fixtures.
