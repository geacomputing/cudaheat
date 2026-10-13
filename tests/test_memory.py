from cudaheat.memory import VRAMEstimator, reduced_batch_size


def test_vram_estimate_reserves_safety_margin_and_allows_only_spatial_batches():
    estimator = VRAMEstimator(ntime=100, itemsize=4, safety_fraction=0.30)
    estimate = estimator.estimate(free_bytes=100_000, npoints=10_000)

    assert estimate.budget_bytes == 70_000
    assert estimate.bytes_per_point > 100 * 4
    assert estimate.batch_size == min(10_000, 70_000 // estimate.bytes_per_point)


def test_vram_estimate_includes_fixed_cycle_and_float64_smoothing_workspace():
    estimator = VRAMEstimator(ntime=24, itemsize=4, cycle_length=100)
    # A short record still requires a full climatology-cycle workspace and
    # float64 prefix sums for numerically stable float32 smoothing.
    assert estimator.bytes_per_point > int((24 * (7 * 4 + 1) + 100 * 16) * 1.20)


def test_oom_recovery_only_reduces_spatial_batch_size():
    assert reduced_batch_size(9) == 4
    assert reduced_batch_size(2) == 1
    assert reduced_batch_size(1) == 1
