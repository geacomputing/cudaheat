"""Small, dependency-free timing output for CUDAHeat utilities and runs."""

from __future__ import annotations

import inspect
from pathlib import Path
from time import perf_counter
from typing import Any


def log_elapsed_time(label: str, started_at: float, /, **details: Any) -> float:
    """Print a timing record and return its elapsed seconds.

    The record includes the caller's module, function, file, and line so it
    remains clear which part of a pipeline emitted it. ``started_at`` must be
    a :func:`time.perf_counter` value captured at the beginning of the work.
    """
    elapsed = perf_counter() - started_at
    source = _caller_location()
    context = " ".join(f"{name}={value}" for name, value in details.items())
    suffix = f" | {context}" if context else ""
  
    print(
        f"[CUDAHeat timing] {label} | elapsed={elapsed:.3f}s | source={source}{suffix}",
        flush=True,
    )
    return elapsed


def _caller_location() -> str:
    """Return the source location of the direct caller of ``log_elapsed_time``."""
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
        if caller is None:  # pragma: no cover - CPython always supplies the caller here.
            return "<unknown>"
        module = caller.f_globals.get("__name__", "<unknown>")
        function = caller.f_code.co_qualname
        filename = Path(caller.f_code.co_filename).name
        return f"{module}.{function} ({filename}:{caller.f_lineno})"
    finally:
        del frame
