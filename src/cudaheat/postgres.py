"""Minimal PostgreSQL persistence for sparse CUDAHeat detection results.

This module is deliberately separate from the numerical pipeline. Install the
optional ``postgres`` extra before calling :func:`persist_detection_result`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from .result import EventDetectionResult


_CONFIG_SECTION = "postgres"
_DEFAULT_PASSWORD_ENV = "CUDAHEAT_POSTGRES_PASSWORD"


class DuplicateRunError(RuntimeError):
    """Raised when an equivalent CUDAHeat detection run is already persisted."""

    def __init__(self, existing_run_id: UUID) -> None:
        self.existing_run_id = existing_run_id
        super().__init__(f"an equivalent CUDAHeat run already exists: {existing_run_id}")


@dataclass(frozen=True)
class PostgresConfig:
    """Non-secret PostgreSQL connection settings.

    Passwords are resolved from ``password_env`` only at connection time. If
    that variable is unset, libpq-compatible ``.pgpass`` configuration remains
    available to the driver.
    """

    host: str
    port: int
    database: str
    username: str
    password_env: str | None = _DEFAULT_PASSWORD_ENV
    sslmode: str = "prefer"
    connect_timeout: int = 10
    application_name: str = "cudaheat"

    @classmethod
    def from_ini(cls, path: str | Path) -> PostgresConfig:
        """Load a ``[postgres]`` INI file, with environment overrides."""
        source = Path(path)
        parser = ConfigParser()
        if not parser.read(source):
            raise FileNotFoundError(f"PostgreSQL configuration file not found: {source}")
        if _CONFIG_SECTION not in parser:
            raise ValueError(f"{source} must contain a [{_CONFIG_SECTION}] section")
        section = parser[_CONFIG_SECTION]

        def setting(name: str, env_name: str, default: str | None = None) -> str | None:
            return os.environ.get(env_name, section.get(name, fallback=default))

        host = setting("host", "CUDAHEAT_POSTGRES_HOST")
        database = setting("database", "CUDAHEAT_POSTGRES_DATABASE")
        username = setting("username", "CUDAHEAT_POSTGRES_USER")
        if not host or not database or not username:
            raise ValueError("PostgreSQL host, database, and username must be configured")

        port_text = setting("port", "CUDAHEAT_POSTGRES_PORT", "5432")
        timeout_text = setting("connect_timeout", "CUDAHEAT_POSTGRES_CONNECT_TIMEOUT", "10")
        password_env = setting("password_env", "CUDAHEAT_POSTGRES_PASSWORD_ENV")
        return cls(
            host=host,
            port=_positive_integer(port_text, "port"),
            database=database,
            username=username,
            password_env=password_env or None,
            sslmode=setting("sslmode", "CUDAHEAT_POSTGRES_SSLMODE", "prefer") or "prefer",
            connect_timeout=_positive_integer(timeout_text, "connect_timeout"),
            application_name=(
                setting("application_name", "CUDAHEAT_POSTGRES_APPLICATION_NAME", "cudaheat")
                or "cudaheat"
            ),
        )

    def connection_kwargs(self) -> dict[str, Any]:
        """Return driver kwargs without retaining or exposing a password."""
        kwargs: dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "dbname": self.database,
            "user": self.username,
            "sslmode": self.sslmode,
            "connect_timeout": self.connect_timeout,
            "application_name": self.application_name,
        }
        if self.password_env:
            password = os.environ.get(self.password_env)
            if password:
                kwargs["password"] = password
        return kwargs


def load_postgres_config(path: str | Path = "config/postgres.ini") -> PostgresConfig:
    """Load local PostgreSQL settings from an INI file."""
    return PostgresConfig.from_ini(path)


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id UUID PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        run_fingerprint CHAR(64) NOT NULL,
        source_path TEXT NOT NULL,
        original_source_path TEXT,
        source_sha256 CHAR(64) NOT NULL,
        source_checksum_algorithm TEXT NOT NULL,
        climatology_hash CHAR(64),
        variable TEXT NOT NULL,
        dtype TEXT,
        baseline_start SMALLINT,
        baseline_end SMALLINT,
        percentile DOUBLE PRECISION NOT NULL,
        smooth_window_hours INTEGER NOT NULL,
        smooth_min_periods INTEGER,
        sampling_window_half_width_days INTEGER,
        same_hour_smoothing BOOLEAN,
        calendar_mode TEXT,
        min_duration_hours INTEGER NOT NULL,
        max_internal_gap_hours INTEGER NOT NULL,
        join_gap_hours INTEGER NOT NULL,
        time_dimension TEXT,
        spatial_dimensions TEXT[] NOT NULL,
        optimized_point_layout BOOLEAN,
        time_was_split BOOLEAN,
        vram_safety DOUBLE PRECISION,
        device_ids INTEGER[] NOT NULL,
        requested_batch_size TEXT,
        batch_size INTEGER,
        batch_selection TEXT,
        cudaheat_version TEXT NOT NULL,
        git_commit TEXT,
        metadata JSONB NOT NULL
    )
    """,
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS run_fingerprint CHAR(64)",
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS climatology_hash CHAR(64)",
    """
    CREATE UNIQUE INDEX IF NOT EXISTS runs_run_fingerprint_unique_idx
    ON runs (run_fingerprint)
    WHERE run_fingerprint IS NOT NULL
    """,
    "CREATE INDEX IF NOT EXISTS runs_climatology_hash_idx ON runs (climatology_hash)",
    """
    CREATE TABLE IF NOT EXISTS points (
        point_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        grid_fingerprint CHAR(64) NOT NULL,
        grid_point_index BIGINT NOT NULL,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        original_lat_index INTEGER,
        original_lon_index INTEGER,
        original_flat_index BIGINT,
        UNIQUE (grid_fingerprint, grid_point_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        point_id BIGINT NOT NULL REFERENCES points(point_id),
        cudaheat_event_index BIGINT NOT NULL,
        start_index BIGINT NOT NULL,
        stop_index BIGINT NOT NULL,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP NOT NULL,
        duration_hours BIGINT NOT NULL,
        UNIQUE (run_id, cudaheat_event_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS events_run_point_time_idx ON events (run_id, point_id, start_time)",
)

_RUN_INSERT = """
INSERT INTO runs (
    run_id, run_fingerprint, source_path, original_source_path, source_sha256,
    source_checksum_algorithm, climatology_hash, variable, dtype, baseline_start, baseline_end,
    percentile, smooth_window_hours, smooth_min_periods,
    sampling_window_half_width_days, same_hour_smoothing, calendar_mode,
    min_duration_hours, max_internal_gap_hours, join_gap_hours, time_dimension,
    spatial_dimensions, optimized_point_layout, time_was_split, vram_safety,
    device_ids, requested_batch_size, batch_size, batch_selection,
    cudaheat_version, git_commit, metadata
) VALUES (
    %(run_id)s, %(run_fingerprint)s, %(source_path)s, %(original_source_path)s,
    %(source_sha256)s,
    %(source_checksum_algorithm)s, %(climatology_hash)s, %(variable)s, %(dtype)s, %(baseline_start)s,
    %(baseline_end)s, %(percentile)s, %(smooth_window_hours)s,
    %(smooth_min_periods)s, %(sampling_window_half_width_days)s,
    %(same_hour_smoothing)s, %(calendar_mode)s, %(min_duration_hours)s,
    %(max_internal_gap_hours)s, %(join_gap_hours)s, %(time_dimension)s,
    %(spatial_dimensions)s, %(optimized_point_layout)s, %(time_was_split)s,
    %(vram_safety)s, %(device_ids)s, %(requested_batch_size)s, %(batch_size)s,
    %(batch_selection)s, %(cudaheat_version)s, %(git_commit)s, %(metadata)s
)
"""

_EXISTING_RUN = "SELECT run_id FROM runs WHERE run_fingerprint = %s"

_POINT_UPSERT = """
INSERT INTO points (
    grid_fingerprint, grid_point_index, latitude, longitude,
    original_lat_index, original_lon_index, original_flat_index
) VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (grid_fingerprint, grid_point_index) DO UPDATE SET
    latitude = EXCLUDED.latitude,
    longitude = EXCLUDED.longitude,
    original_lat_index = EXCLUDED.original_lat_index,
    original_lon_index = EXCLUDED.original_lon_index,
    original_flat_index = EXCLUDED.original_flat_index
"""

_POINT_IDS_FOR_GRID = """
SELECT grid_point_index, point_id
FROM points
WHERE grid_fingerprint = %s AND grid_point_index = ANY(%s)
"""

_EVENT_INSERT = """
INSERT INTO events (
    run_id, point_id, cudaheat_event_index, start_index, stop_index,
    start_time, end_time, duration_hours
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


def create_postgres_schema(config: PostgresConfig) -> None:
    """Create the minimal CUDAHeat persistence schema if it is absent."""
    psycopg = _load_psycopg()
    with psycopg.connect(**config.connection_kwargs()) as connection:
        with connection.cursor() as cursor:
            _create_schema(cursor)


def persist_detection_result(
    result: EventDetectionResult,
    config: PostgresConfig,
) -> UUID:
    """Persist one completed detection result and return its new run ID.

    CUDAHeat result metadata supplies the raw-source SHA-256 when available;
    legacy results are checksummed at persistence time. This operation stores
    only run metadata, spatial identities, and sparse event rows.
    """
    if not isinstance(result, EventDetectionResult):
        raise TypeError("result must be an EventDetectionResult")

    psycopg = _load_psycopg()
    from psycopg.types.json import Jsonb

    point_rows, grid_fingerprint = _point_rows(result)
    run_id = uuid4()
    run_row = _run_row(result, run_id, Jsonb, grid_fingerprint)
    event_data = _event_values(result)

    with psycopg.connect(**config.connection_kwargs()) as connection:
        with connection.cursor() as cursor:
            _create_schema(cursor)
            cursor.execute(_EXISTING_RUN, (run_row["run_fingerprint"],))
            existing = cursor.fetchone()
            if existing is not None:
                existing_run_id = existing[0]
                if not isinstance(existing_run_id, UUID):
                    existing_run_id = UUID(str(existing_run_id))
                raise DuplicateRunError(existing_run_id)
            cursor.execute(_RUN_INSERT, run_row)
            cursor.executemany(_POINT_UPSERT, point_rows)
            cursor.execute(
                _POINT_IDS_FOR_GRID,
                (grid_fingerprint, [row[1] for row in point_rows]),
            )
            point_id_by_grid_index = {int(index): int(point_id) for index, point_id in cursor}
            if len(point_id_by_grid_index) != len(point_rows):
                raise RuntimeError(
                    "could not resolve every CUDAHeat point to a PostgreSQL point_id"
                )
            cursor.executemany(
                _EVENT_INSERT,
                _event_rows(event_data, run_id, point_id_by_grid_index),
            )
    return run_id


def _create_schema(cursor: Any) -> None:
    for statement in _SCHEMA_STATEMENTS:
        cursor.execute(statement)


def _run_row(
    result: EventDetectionResult,
    run_id: UUID,
    jsonb_type: Any,
    grid_fingerprint: str,
) -> dict[str, Any]:
    metadata = _json_safe(result.metadata)
    active_source_path = metadata.get("source_path")
    if not isinstance(active_source_path, str) or not active_source_path:
        raise ValueError("result metadata must contain a source_path for checksum persistence")
    source_path = metadata.get("original_source_path") or active_source_path
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("result metadata original_source_path must be a non-empty path when supplied")
    checksum = _metadata_sha256(metadata) or _sha256_file(source_path)
    baseline_start, baseline_end = _baseline_years(metadata.get("baseline"))
    return {
        "run_id": run_id,
        "run_fingerprint": _run_fingerprint(metadata, checksum, grid_fingerprint),
        "source_path": source_path,
        "original_source_path": metadata.get("original_source_path"),
        "source_sha256": checksum,
        "source_checksum_algorithm": "sha256",
        "climatology_hash": _optional_sha256(metadata.get("climatology_hash")),
        "variable": _required_text(metadata, "variable"),
        "dtype": metadata.get("dtype"),
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "percentile": _required_float(metadata, "percentile"),
        "smooth_window_hours": _required_int(metadata, "smooth_window_hours"),
        "smooth_min_periods": _optional_int(metadata.get("smooth_min_periods")),
        "sampling_window_half_width_days": _optional_int(
            metadata.get("sampling_window_half_width_days")
        ),
        "same_hour_smoothing": _optional_bool(metadata.get("same_hour_smoothing")),
        "calendar_mode": metadata.get("calendar_mode"),
        "min_duration_hours": _required_int(metadata, "min_duration_hours"),
        "max_internal_gap_hours": _required_int(metadata, "max_internal_gap_hours"),
        "join_gap_hours": _required_int(metadata, "join_gap_hours"),
        "time_dimension": metadata.get("time_dimension"),
        "spatial_dimensions": [str(value) for value in metadata.get("spatial_dimensions", ())],
        "optimized_point_layout": _optional_bool(metadata.get("optimized_point_layout")),
        "time_was_split": _optional_bool(metadata.get("time_was_split")),
        "vram_safety": _optional_float(metadata.get("vram_safety")),
        "device_ids": [int(value) for value in metadata.get("devices", ())],
        "requested_batch_size": _optional_text(metadata.get("requested_batch_size")),
        "batch_size": _optional_int(metadata.get("batch_size")),
        "batch_selection": _optional_text(metadata.get("batch_selection")),
        "cudaheat_version": _cudaheat_version(),
        "git_commit": _git_commit(),
        "metadata": jsonb_type(metadata),
    }


def _point_rows(result: EventDetectionResult) -> tuple[list[tuple[Any, ...]], str]:
    grid = result._grid
    point_indices = np.asarray(grid.point_indices, dtype=np.int64)
    if point_indices.ndim != 1 or point_indices.size == 0:
        raise ValueError("result grid must contain one or more point indices")
    if np.unique(point_indices).size != point_indices.size:
        raise ValueError("result grid point indices must be unique")

    latitude = _point_coordinate(grid, ("latitude", "lat"))
    longitude = _point_coordinate(grid, ("longitude", "lon"))
    if latitude is None or longitude is None:
        raise ValueError("result grid must provide latitude and longitude coordinates")
    original_lat = _point_coordinate(grid, ("original_lat_index",))
    original_lon = _point_coordinate(grid, ("original_lon_index",))
    original_flat = _point_coordinate(grid, ("original_flat_index",))
    fingerprint = _grid_fingerprint(
        point_indices, latitude, longitude, original_lat, original_lon, original_flat
    )
    rows = [
        (
            fingerprint,
            int(point_indices[position]),
            _optional_float(latitude[position]),
            _optional_float(longitude[position]),
            _optional_int(None if original_lat is None else original_lat[position]),
            _optional_int(None if original_lon is None else original_lon[position]),
            _optional_int(None if original_flat is None else original_flat[position]),
        )
        for position in range(point_indices.size)
    ]
    return rows, fingerprint


def _point_coordinate(grid: Any, names: Iterable[str]) -> np.ndarray | None:
    expected_size = grid.point_indices.size
    for name in names:
        values = grid.point_coordinates.get(name)
        if values is not None:
            array = np.asarray(values)
            if array.shape == (expected_size,):
                return array
        point_index = grid.point_index
        index_names = getattr(point_index, "names", ())
        if name in index_names:
            return np.asarray(point_index.get_level_values(name))
    return None


def _event_values(result: EventDetectionResult) -> dict[str, np.ndarray]:
    catalogue = result.data
    required = (
        "point_index",
        "start_index",
        "stop_index",
        "start_time",
        "end_time",
        "duration_hours",
    )
    missing = [name for name in required if name not in catalogue]
    if missing:
        raise ValueError(f"result event catalogue is missing required field(s): {missing!r}")
    event_ids = np.asarray(catalogue["event"].values, dtype=np.int64)
    values = {name: np.asarray(catalogue[name].values) for name in required}
    sizes = {event_ids.size, *(array.size for array in values.values())}
    if len(sizes) != 1:
        raise ValueError("event catalogue columns must have the same length")
    values["event"] = event_ids
    return values


def _event_rows(
    values: Mapping[str, np.ndarray], run_id: UUID, point_id_by_grid_index: Mapping[int, int]
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for event_id, point_index, start, stop, start_time, end_time, duration in zip(
        values["event"],
        values["point_index"],
        values["start_index"],
        values["stop_index"],
        values["start_time"],
        values["end_time"],
        values["duration_hours"],
        strict=True,
    ):
        grid_point_index = int(point_index)
        try:
            point_id = point_id_by_grid_index[grid_point_index]
        except KeyError as exc:
            raise ValueError(
                f"event references unavailable point index {grid_point_index}"
            ) from exc
        rows.append(
            (
                run_id,
                point_id,
                int(event_id),
                int(start),
                int(stop),
                _timestamp(start_time),
                _timestamp(end_time),
                int(duration),
            )
        )
    return rows


def _grid_fingerprint(*arrays: np.ndarray | None) -> str:
    digest = sha256()
    for array in arrays:
        if array is None:
            digest.update(b"<none>")
            continue
        values = np.ascontiguousarray(np.asarray(array))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"cannot checksum missing NetCDF source: {source}")
    digest = sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_sha256(metadata: Mapping[str, Any]) -> str | None:
    algorithm = metadata.get("source_checksum_algorithm")
    if algorithm not in (None, "sha256"):
        raise ValueError("only SHA-256 source checksums can be persisted")
    return _optional_sha256(metadata.get("source_sha256"))


def _optional_sha256(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("SHA-256 values must be 64-character hexadecimal strings")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("SHA-256 values must be hexadecimal strings") from exc
    return value.lower()


def _run_fingerprint(
    metadata: Mapping[str, Any], source_sha256: str, grid_fingerprint: str
) -> str:
    """Hash stable scientific inputs, excluding runtime-only telemetry."""
    parameter_names = (
        "variable",
        "dtype",
        "baseline",
        "percentile",
        "smooth_window_hours",
        "smooth_min_periods",
        "sampling_window_half_width_days",
        "same_hour_smoothing",
        "calendar_mode",
        "climatology_hash",
        "min_duration_hours",
        "max_internal_gap_hours",
        "join_gap_hours",
        "time_dimension",
        "spatial_dimensions",
        "optimized_point_layout",
        "time_was_split",
        "vram_safety",
        "devices",
        "requested_batch_size",
        "batch_size",
        "batch_selection",
        "retained_fields",
    )
    payload = {
        "source_sha256": source_sha256,
        "grid_fingerprint": grid_fingerprint,
        "parameters": {name: metadata.get(name) for name in parameter_names},
    }
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _timestamp(value: Any) -> datetime:
    timestamp = np.datetime64(value, "us")
    if np.isnat(timestamp):
        raise ValueError("event timestamps must not be NaT")
    converted = timestamp.astype("datetime64[us]").item()
    if not isinstance(converted, datetime):  # pragma: no cover - NumPy-version guard
        raise TypeError(f"could not convert event timestamp {value!r} to datetime")
    return converted


def _baseline_years(value: Any) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("baseline metadata must be None or a two-year sequence")
    return int(value[0]), int(value[1])


def _positive_integer(value: str | None, name: str) -> int:
    try:
        converted = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"PostgreSQL {name} must be an integer") from exc
    if converted < 1:
        raise ValueError(f"PostgreSQL {name} must be positive")
    return converted


def _required_text(metadata: Mapping[str, Any], name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"result metadata must contain a non-empty {name!r}")
    return value


def _required_int(metadata: Mapping[str, Any], name: str) -> int:
    value = _optional_int(metadata.get(name))
    if value is None:
        raise ValueError(f"result metadata must contain {name!r}")
    return value


def _required_float(metadata: Mapping[str, Any], name: str) -> float:
    value = _optional_float(metadata.get(name))
    if value is None:
        raise ValueError(f"result metadata must contain {name!r}")
    return value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _json_safe(value: Any) -> Any:
    """Convert NumPy and path values in metadata to JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.datetime64):
        return str(value)
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _cudaheat_version() -> str:
    from . import __version__

    return __version__


def _git_commit() -> str | None:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _load_psycopg() -> Any:
    try:
        import psycopg
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PostgreSQL persistence needs the optional dependency; install CUDAHeat with "
            "`pip install -e '.[postgres]'`."
        ) from exc
    return psycopg
