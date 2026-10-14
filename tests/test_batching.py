from __future__ import annotations

from cudaheat.memory import BatchEstimate, guess_batch_size


def _estimate(batch_size: int) -> BatchEstimate:
    return BatchEstimate(
        batch_size=batch_size,
        bytes_per_point=1,
        budget_bytes=batch_size,
        available_bytes=batch_size,
        safety_fraction=0.15,
        float_arrays=7,
        workspace_multiplier=1.2,
    )


def test_guess_uses_nearest_structural_size_to_a_quarter_of_safe_vram():
    assert guess_batch_size(_estimate(2_201), npoints=22_083) == 512
    assert guess_batch_size(_estimate(6_000), npoints=22_083) == 1536


def test_guess_uses_safe_fallback_when_no_structural_size_fits():
    assert guess_batch_size(_estimate(20), npoints=20) == 20
