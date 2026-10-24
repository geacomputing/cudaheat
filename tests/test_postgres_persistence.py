from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import numpy as np
import xarray as xr

from cudaheat.postgres import (
    PostgresConfig,
    _event_rows,
    _event_values,
    _point_rows,
    _run_row,
)
from cudaheat.result import EventDetectionResult, EventGrid


def _result(source: Path) -> EventDetectionResult:
    time = np.arange(
        np.datetime64("2000-01-01T00"),
        np.datetime64("2000-01-01T06"),
        np.timedelta64(1, "h"),
    )
    points = xr.DataArray(
        np.arange(4).reshape(2, 2),
        dims=("latitude", "longitude"),
        coords={"latitude": [34.0, 35.0], "longitude": [32.0, 33.0]},
    ).stack(point=("latitude", "longitude"))
    catalogue = xr.Dataset(
        {
            "point_index": ("event", [1, 2]),
            "start_index": ("event", [1, 3]),
            "stop_index": ("event", [3, 6]),
            "duration_hours": ("event", [2, 3]),
            "start_time": ("event", time[[1, 3]]),
            "end_time": ("event", time[[2, 5]]),
            "event_count": xr.DataArray(
                np.array([[0, 1], [1, 0]], dtype=np.uint64),
                dims=("latitude", "longitude"),
                coords={"latitude": [34.0, 35.0], "longitude": [32.0, 33.0]},
            ),
        },
        coords={"event": [1, 2]},
    )
    metadata = {
        "source_path": str(source),
        "original_source_path": str(source),
        "variable": "analysed_sst",
        "dtype": "float32",
        "baseline": (1982, 2011),
        "percentile": 0.9,
        "smooth_window_hours": 744,
        "smooth_min_periods": 1,
        "sampling_window_half_width_days": 5,
        "same_hour_smoothing": True,
        "calendar_mode": "month_day_hour",
        "min_duration_hours": 120,
        "max_internal_gap_hours": 6,
        "join_gap_hours": 48,
        "time_dimension": "time",
        "spatial_dimensions": ("latitude", "longitude"),
        "optimized_point_layout": False,
        "time_was_split": False,
        "vram_safety": 0.15,
        "devices": (0,),
        "requested_batch_size": "guess",
        "batch_size": 512,
        "batch_selection": "guess",
    }
    return EventDetectionResult(
        data=catalogue,
        telemetry=(),
        metadata=metadata,
        _grid=EventGrid(
            time=time,
            point_index=points.indexes["point"],
            point_indices=np.arange(4),
            time_dim="time",
        ),
    )


class _Jsonb:
    def __init__(self, value: object) -> None:
        self.value = value


def test_postgres_config_uses_environment_overrides_without_password(monkeypatch, tmp_path):
    config = tmp_path / "postgres.ini"
    config.write_text(
        "[postgres]\nhost = db.example\nport = 5432\ndatabase = cudaheat\n"
        "username = app\npassword_env = CUDAHEAT_POSTGRES_PASSWORD\n"
    )
    monkeypatch.setenv("CUDAHEAT_POSTGRES_PORT", "5433")
    monkeypatch.delenv("CUDAHEAT_POSTGRES_PASSWORD", raising=False)

    loaded = PostgresConfig.from_ini(config)

    assert loaded.port == 5433
    assert "password" not in loaded.connection_kwargs()
    assert loaded.connection_kwargs()["dbname"] == "cudaheat"


def test_sparse_result_becomes_run_point_and_event_records(tmp_path):
    source = tmp_path / "sst.nc"
    source.write_bytes(b"synthetic netcdf payload")
    result = _result(source)

    points, fingerprint = _point_rows(result)
    assert len(points) == 4
    assert points[1][1:4] == (1, 34.0, 33.0)
    assert len(fingerprint) == 64

    run = _run_row(result, uuid4(), _Jsonb, fingerprint)
    assert run["source_sha256"] == sha256(source.read_bytes()).hexdigest()
    assert run["baseline_start"] == 1982
    assert run["metadata"].value["baseline"] == [1982, 2011]
    assert len(run["run_fingerprint"]) == 64

    point_ids = {row[1]: position + 100 for position, row in enumerate(points)}
    events = _event_rows(_event_values(result), uuid4(), point_ids)
    assert [(row[2], row[3], row[4], row[7]) for row in events] == [
        (1, 1, 3, 2),
        (2, 3, 6, 3),
    ]
