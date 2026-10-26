"""Data records shared by the enrichment pilot modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RunReference:
    """Persisted identity of the source and climatology products for a run."""

    run_id: str
    source_path: Path
    source_sha256: str
    climatology_hash: str
    variable: str
    metadata: Mapping[str, Any]
    calendar_mode: str


@dataclass(frozen=True)
class Event:
    """One sparse, half-open event interval from PostgreSQL."""

    event_id: int
    grid_point_index: int
    start_time: Any
    end_time: Any
    start_index: int
    stop_index: int
    duration_hours: int


@dataclass(frozen=True)
class EventWindow:
    """Three aligned series used by the optional pilot benchmark and plot."""

    event: Event
    time: np.ndarray
    climatology: np.ndarray
    threshold: np.ndarray
    sst: np.ndarray
    latitude: float | None
    longitude: float | None
    timings_seconds: dict[str, float]
