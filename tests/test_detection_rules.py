import numpy as np
import pytest

from cudaheat.detection import detect_events as cudaheat_detect_events

MIN_DURATION = 120
MAX_INTERNAL_GAP = 6
JOIN_GAP = 48


# ============================================================
# ADAPTER — CONNECT TO CUDAHeat DETECTION API
# ============================================================

def detect_events(exceedance, backend="cpu"):
    """Run the public vectorized detector through a one-point threshold field."""
    if backend == "cpu":
        xp = np
    elif backend == "gpu":
        xp = pytest.importorskip("cupy")
        if xp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
        try:
            xp.full((1, 1), 0.5)
        except RuntimeError as exc:
            pytest.skip(f"CuPy kernel compilation unavailable: {exc}")
    else:
        raise ValueError(f"unknown backend {backend!r}")
    values = xp.asarray(exceedance[:, None], dtype=xp.float64)
    fields = cudaheat_detect_events(
        values,
        xp.full(values.shape, 0.5),
        min_duration_hours=MIN_DURATION,
        max_internal_gap_hours=MAX_INTERNAL_GAP,
        join_gap_hours=JOIN_GAP,
        xp=xp,
    )
    starts = xp.asnumpy(fields.events.start) if backend == "gpu" else fields.events.start
    stops = xp.asnumpy(fields.events.stop) if backend == "gpu" else fields.events.stop
    durations = xp.asnumpy(fields.events.duration) if backend == "gpu" else fields.events.duration
    return [
        {"start": int(start), "end": int(stop - 1), "duration_hours": int(duration)}
        for start, stop, duration in zip(starts, stops, durations, strict=True)
    ]


# ============================================================
# SYNTHETIC DATA GENERATOR
# ============================================================

def make_series(length, intervals):
    """
    Generate Boolean exceedance time series.

    intervals:
        List of (start, end) pairs.
        Both indices are inclusive.
    """
    series = np.zeros(length, dtype=bool)

    for start, end in intervals:
        series[start:end + 1] = True

    return series


def run_detection(series, backend="cpu"):
    return detect_events(series, backend=backend)


def assert_event(event, start, end):
    assert event["start"] == start
    assert event["end"] == end
    assert event["duration_hours"] == end - start + 1


# ============================================================
# 1. MINIMUM DURATION
# ============================================================

@pytest.mark.parametrize(
    "duration,expected_count",
    [
        (119, 0),
        (120, 1),
        (121, 1),
    ],
)
def test_minimum_duration(duration, expected_count):

    series = make_series(
        200,
        [(0, duration - 1)],
    )

    events = run_detection(series)

    assert len(events) == expected_count

    if expected_count:
        assert_event(events[0], 0, duration - 1)


# ============================================================
# 2. INTERNAL GAP TOLERANCE
# ============================================================

@pytest.mark.parametrize(
    "gap,expected_count",
    [
        (5, 1),
        (6, 1),
        (7, 0),
    ],
)
def test_internal_gap(gap, expected_count):
    """
    Two 60-hour exceedance runs separated by a gap.

    A gap <= 6 produces one qualifying event.

    A gap > 6 splits the runs, neither of which qualifies.
    """

    first_start = 0
    first_end = 59

    second_start = 60 + gap
    second_end = second_start + 59

    series = make_series(
        200,
        [
            (first_start, first_end),
            (second_start, second_end),
        ],
    )

    events = run_detection(series)

    assert len(events) == expected_count

    if expected_count:
        assert_event(events[0], 0, second_end)


# ============================================================
# 3. EVENT JOINING
# ============================================================

@pytest.mark.parametrize(
    "gap,expected_count",
    [
        (47, 1),
        (48, 1),
        (49, 2),
    ],
)
def test_join_gap(gap, expected_count):
    """
    Two independently qualifying 120-hour events.

    Join when their separation is <= 48 hours.
    """

    first = (0, 119)

    second_start = 120 + gap
    second = (second_start, second_start + 119)

    series = make_series(
        350,
        [first, second],
    )

    events = run_detection(series)

    assert len(events) == expected_count

    if expected_count == 1:
        assert_event(events[0], 0, second[1])

    else:
        assert_event(events[0], *first)
        assert_event(events[1], *second)


# ============================================================
# 4. SHORT EVENTS MUST NOT BE JOINED
# ============================================================

def test_short_events_not_joined():
    """
    Two 100-hour exceedance runs separated by 24 hours.

    Neither run qualifies independently.

    They must NOT be joined into a valid event.
    """

    series = make_series(
        300,
        [
            (0, 99),
            (124, 223),
        ],
    )

    events = run_detection(series)

    assert len(events) == 0


# ============================================================
# 5. EXACTLY 120 HOURS INCLUDING INTERNAL GAP
# ============================================================

def test_duration_includes_internal_gap():
    """
    Total event duration = 120 hours.

    Includes a 6-hour internal interruption.
    """

    series = make_series(
        200,
        [
            (0, 59),
            (66, 119),
        ],
    )

    events = run_detection(series)

    assert len(events) == 1
    assert_event(events[0], 0, 119)


# ============================================================
# 6. JOINED EVENT DURATION
# ============================================================

def test_joined_duration():
    """
    Two 120-hour events separated by 48 hours.

    Expected joined duration:
        120 + 48 + 120 = 288 hours.
    """

    series = make_series(
        350,
        [
            (0, 119),
            (168, 287),
        ],
    )

    events = run_detection(series)

    assert len(events) == 1
    assert_event(events[0], 0, 287)


# ============================================================
# 7. MULTIPLE EVENTS
# ============================================================

def test_multiple_independent_events():

    series = make_series(
        600,
        [
            (0, 119),
            (200, 319),
            (400, 519),
        ],
    )

    events = run_detection(series)

    assert len(events) == 3

    assert_event(events[0], 0, 119)
    assert_event(events[1], 200, 319)
    assert_event(events[2], 400, 519)


# ============================================================
# 8. CPU / GPU EQUIVALENCE
# ============================================================

def test_cpu_gpu_equivalence():

    pytest.importorskip("cupy")

    series = make_series(
        600,
        [
            (0, 119),
            (160, 279),
            (400, 519),
        ],
    )

    cpu_events = run_detection(series, backend="cpu")
    gpu_events = run_detection(series, backend="gpu")

    assert cpu_events == gpu_events


# ============================================================
# 9. EVERY EVENT MUST SATISFY MINIMUM DURATION
# ============================================================

def test_all_events_satisfy_minimum_duration():

    rng = np.random.default_rng(42)

    series = rng.random(10000) > 0.15

    events = run_detection(series)

    for event in events:
        assert event["duration_hours"] >= MIN_DURATION
