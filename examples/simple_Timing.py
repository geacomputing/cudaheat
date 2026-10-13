"""Time complete CUDAHeat runs for batch sizes 502–522."""

import time

import cupy as cp
from cudaheat import CUDAHeat

PATH = "/home/user/Documents/Gea/cudaheat/examples/sst_cudaheat.nc"

timings = []
steps = 10

for batch in range(512 - steps, 512 + steps + 1):
    cp.cuda.Device().synchronize()
    start = time.perf_counter()

    with CUDAHeat(
        PATH,
        variable="analysed_sst",
        timing_logging=False,
    ) as hw:
        result = hw.detect(
            baseline=(2019, 2023),
            percentile=0.90,
            smooth_window_hours=24 * 7,
            min_duration_hours=120,
            max_internal_gap_hours=6,
            join_gap_hours=48,
            batch_size=batch,
        )
        cp.cuda.Device().synchronize()

    elapsed = time.perf_counter() - start
    timings.append((batch, elapsed))
    print(f"Batch size {batch:4d} — execution time: {elapsed:.3f} s")
    del result

best_batch, best_time = min(timings, key=lambda item: item[1])
print(f"\nFastest measured: batch size {best_batch} — {best_time:.3f} s")