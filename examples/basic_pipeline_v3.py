import time
from cudaheat import CUDAHeat
import cupy as cp
PATH = "/home/user/Documents/PhD.cy/lazy_loading/cudaHeat/sst_8_years.nc"

t0 = time.perf_counter()

with CUDAHeat(
    PATH,
    variable="analysed_sst",
    dtype="float32",
    batch_size="guess",
    timing_logging=False,
) as hw:

    result = hw.detect(
        baseline=(2019, 2023),
        percentile=0.90,
        min_duration_hours=120,
        max_internal_gap_hours=6,
        join_gap_hours=48,
    )


cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - t0

print(f"Execution time: {elapsed:.2f} s")
print(f"Batch size: {hw.batch_size}")
print(f"Climatology cache: {result.metadata['climatology_cache_status']}")
print(f"Zarr: {result.metadata['climatology_cache_path']}")