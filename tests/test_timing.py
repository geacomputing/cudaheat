from __future__ import annotations

from time import perf_counter

from cudaheat.timing import log_elapsed_time as core_log_elapsed_time
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


def test_core_timing_output_can_be_disabled(capsys) -> None:
    elapsed = core_log_elapsed_time("test stage", perf_counter(), enabled=False, batch=3)

    assert elapsed >= 0
    assert capsys.readouterr().out == ""
