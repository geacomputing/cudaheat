from __future__ import annotations

from time import perf_counter

from util.timing import log_elapsed_time


def test_log_elapsed_time_reports_elapsed_time_and_call_site(capsys) -> None:
    elapsed = log_elapsed_time("test stage", perf_counter(), batch=3)

    output = capsys.readouterr().out
    assert elapsed >= 0
    assert "[CUDAHeat timing] test stage" in output
    assert "elapsed=" in output
    assert "source=" in output
    assert "test_log_elapsed_time_reports_elapsed_time_and_call_site" in output
    assert "batch=3" in output
