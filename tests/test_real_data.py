"""Placeholder for a manually provisioned real-data smoke test.

The original module ran a GPU pipeline against an absolute local path during
test collection.  Automated tests must remain self-contained; run real-data
validation explicitly with ``examples/validate_real_data.py`` instead.
"""

import pytest


@pytest.mark.skip(reason="requires a manually provisioned real dataset and CUDA device")
def test_real_data_smoke_test_is_manual_only() -> None:
    """Reserve the test name without executing external work during collection."""
