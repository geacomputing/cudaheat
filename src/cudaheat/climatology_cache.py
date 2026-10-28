"""Local sibling Zarr cache for CUDAHeat climatology cycles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import numpy as np

from .cache_paths import climatology_zarr_path

_CACHE_SCHEMA_VERSION = 1
_CLIMATOLOGY_ALGORITHM = "cudaheat-hourly-hobday-v1"
_POINT_COORDINATES = (
    "latitude",
    "longitude",
    "original_lat_index",
    "original_lon_index",
    "original_flat_index",
)


@dataclass
class ClimatologyCache:
    """A validated existing cache or an atomic, in-progress cache writer."""

    path: Path
    climatology_hash: str
    source_sha256: str
    status: str
    _group: Any
    _dtype: np.dtype[Any]
    _nbins: int
    _npoints: int
    _config_json: str
    _point_values: np.ndarray
    _temporary_path: Path | None = None

    @property
    def needs_write(self) -> bool:
        return self._temporary_path is not None

    def read_batch(self, start: int, stop: int) -> dict[str, np.ndarray]:
        """Read one complete climatological cycle for a spatial batch."""
        if self.needs_write:
            raise RuntimeError("a new climatology cache cannot be read before it is committed")
        _validate_batch_slice(start, stop, self._npoints)
        return {
            "climatology": np.asarray(self._group["climatology"][:, start:stop]),
            "threshold_cycle": np.asarray(self._group["threshold"][:, start:stop]),
        }

    def write_batch(self, start: int, stop: int, fields: Mapping[str, np.ndarray]) -> None:
        """Write one GPU-produced cycle batch into the temporary Zarr store."""
        if not self.needs_write:
            raise RuntimeError("cannot write to a reused climatology cache")
        _validate_batch_slice(start, stop, self._npoints)
        width = stop - start
        climatology = _cycle_values(fields, "climatology", self._dtype, self._nbins, width)
        threshold = _cycle_values(fields, "threshold_cycle", self._dtype, self._nbins, width)
        self._group["climatology"][:, start:stop] = climatology
        self._group["threshold"][:, start:stop] = threshold

    def commit(self) -> str:
        """Publish a completely written cache with one atomic directory rename."""
        if not self.needs_write:
            return self.status
        temporary = self._temporary_path
        if temporary is None:  # pragma: no cover - type narrowing guard
            raise RuntimeError("missing temporary climatology cache path")
        self._group.attrs["cudaheat_complete"] = True
        _zarr().consolidate_metadata(str(temporary))
        if self.path.exists():
            if _valid_cache(
                self.path,
                climatology_hash=self.climatology_hash,
                config_json=self._config_json,
                dtype=self._dtype,
                nbins=self._nbins,
                point_values=self._point_values,
            ):
                self.abort()
                self.status = "reused"
                return self.status
            raise RuntimeError(f"incompatible climatology cache appeared at {self.path}")
        os.replace(temporary, self.path)
        self._temporary_path = None
        self.status = "created"
        return self.status

    def abort(self) -> None:
        """Discard only the incomplete temporary cache."""
        if self._temporary_path is not None:
            shutil.rmtree(self._temporary_path, ignore_errors=True)
            self._temporary_path = None


def open_climatology_cache(
    source_path: str | Path,
    *,
    source_sha256: str,
    variable: str,
    dtype: str | np.dtype[Any],
    baseline: tuple[int, int] | None,
    percentile: float,
    smooth_window_hours: int,
    smooth_min_periods: int,
    sampling_window_half_width_days: int,
    same_hour_smoothing: bool,
    calendar_mode: str,
    nbins: int,
    point_values: np.ndarray,
    point_coordinates: Mapping[str, np.ndarray],
) -> ClimatologyCache:
    """Open a compatible sibling cache or create an unpublished temporary one."""
    resolved_source = Path(source_path).resolve()
    value_dtype = np.dtype(dtype)
    points = np.asarray(point_values, dtype=np.int64)
    if points.ndim != 1 or points.size == 0 or np.unique(points).size != points.size:
        raise ValueError("point_values must be a non-empty one-dimensional unique integer array")
    if nbins < 1:
        raise ValueError("nbins must be positive")
    payload = {
        "algorithm": _CLIMATOLOGY_ALGORITHM,
        "calendar_mode": str(calendar_mode),
        "dtype": value_dtype.name,
        "baseline": None if baseline is None else [int(baseline[0]), int(baseline[1])],
        "percentile": float(percentile),
        "smooth_window_hours": int(smooth_window_hours),
        "smooth_min_periods": int(smooth_min_periods),
        "sampling_window_half_width_days": int(sampling_window_half_width_days),
        "same_hour_smoothing": bool(same_hour_smoothing),
        "nbins": int(nbins),
        "source_sha256": str(source_sha256),
        "variable": str(variable),
    }
    config_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    climatology_hash = sha256(config_json.encode("utf-8")).hexdigest()
    destination = climatology_zarr_path(resolved_source, source_sha256, climatology_hash)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _valid_cache(
        destination,
        climatology_hash=climatology_hash,
        config_json=config_json,
        dtype=value_dtype,
        nbins=nbins,
        point_values=points,
    ):
        group = _zarr().open_group(str(destination), mode="r")
        return ClimatologyCache(
            destination,
            climatology_hash,
            str(source_sha256),
            "reused",
            group,
            value_dtype,
            nbins,
            points.size,
            config_json,
            points,
        )
    if destination.exists():
        _remove_invalid_cache(destination)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    group = _create_cache(
        temporary,
        source_path=resolved_source,
        source_sha256=str(source_sha256),
        climatology_hash=climatology_hash,
        config_json=config_json,
        dtype=value_dtype,
        nbins=nbins,
        point_values=points,
        point_coordinates=point_coordinates,
    )
    return ClimatologyCache(
        destination,
        climatology_hash,
        str(source_sha256),
        "created",
        group,
        value_dtype,
        nbins,
        points.size,
        config_json,
        points,
        temporary,
    )


def _create_cache(
    path: Path,
    *,
    source_path: Path,
    source_sha256: str,
    climatology_hash: str,
    config_json: str,
    dtype: np.dtype[Any],
    nbins: int,
    point_values: np.ndarray,
    point_coordinates: Mapping[str, np.ndarray],
) -> Any:
    zarr = _zarr()
    from numcodecs import Blosc

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    group = zarr.open_group(str(path), mode="w")
    group.attrs.update(
        {
            "cudaheat_climatology_cache": "true",
            "cudaheat_climatology_schema_version": _CACHE_SCHEMA_VERSION,
            "cudaheat_complete": False,
            "cudaheat_climatology_hash": climatology_hash,
            "cudaheat_climatology_config": config_json,
            "cudaheat_source_path": str(source_path),
            "cudaheat_source_sha256": source_sha256,
            "cudaheat_source_checksum_algorithm": "sha256",
            "cudaheat_climatology_algorithm": _CLIMATOLOGY_ALGORITHM,
        }
    )
    npoints = point_values.size
    point_chunk = min(256, npoints)
    climatology = group.create_dataset(
        "climatology",
        shape=(nbins, npoints),
        chunks=(nbins, point_chunk),
        dtype=dtype,
        compressor=compressor,
        fill_value=np.nan,
    )
    climatology.attrs["_ARRAY_DIMENSIONS"] = ["climatology_hour", "point"]
    threshold = group.create_dataset(
        "threshold",
        shape=(nbins, npoints),
        chunks=(nbins, point_chunk),
        dtype=dtype,
        compressor=compressor,
        fill_value=np.nan,
    )
    threshold.attrs["_ARRAY_DIMENSIONS"] = ["climatology_hour", "point"]
    climatology_hour = group.create_dataset(
        "climatology_hour",
        data=np.arange(nbins, dtype=np.int32),
        chunks=(min(nbins, 1024),),
        compressor=compressor,
    )
    climatology_hour.attrs["_ARRAY_DIMENSIONS"] = ["climatology_hour"]
    point = group.create_dataset(
        "point",
        data=point_values,
        chunks=(point_chunk,),
        compressor=compressor,
    )
    point.attrs["_ARRAY_DIMENSIONS"] = ["point"]
    for name in _POINT_COORDINATES:
        values = point_coordinates.get(name)
        if values is None:
            continue
        coordinate = np.asarray(values)
        if coordinate.shape != (npoints,) or coordinate.dtype.kind not in "iuf":
            continue
        point_coordinate = group.create_dataset(
            name, data=coordinate, chunks=(point_chunk,), compressor=compressor
        )
        point_coordinate.attrs["_ARRAY_DIMENSIONS"] = ["point"]
    return group


def _valid_cache(
    path: Path,
    *,
    climatology_hash: str,
    config_json: str,
    dtype: np.dtype[Any],
    nbins: int,
    point_values: np.ndarray,
) -> bool:
    if not path.is_dir():
        return False
    try:
        group = _zarr().open_group(str(path), mode="r")
        attrs = group.attrs
        if (
            attrs.get("cudaheat_climatology_cache") != "true"
            or attrs.get("cudaheat_climatology_schema_version") != _CACHE_SCHEMA_VERSION
            or not bool(attrs.get("cudaheat_complete"))
            or attrs.get("cudaheat_climatology_hash") != climatology_hash
            or attrs.get("cudaheat_climatology_config") != config_json
        ):
            return False
        npoints = point_values.size
        for name in ("climatology", "threshold"):
            array = group[name]
            if array.shape != (nbins, npoints) or np.dtype(array.dtype) != dtype:
                return False
        if group["climatology_hour"].shape != (nbins,) or group["point"].shape != (npoints,):
            return False
        return np.array_equal(np.asarray(group["point"][:]), point_values)
    except (KeyError, OSError, ValueError, TypeError):
        return False


def _remove_invalid_cache(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _cycle_values(
    fields: Mapping[str, np.ndarray],
    name: str,
    dtype: np.dtype[Any],
    nbins: int,
    width: int,
) -> np.ndarray:
    if name not in fields:
        raise RuntimeError(f"missing {name!r} needed to populate the climatology cache")
    values = np.asarray(fields[name], dtype=dtype)
    if values.shape != (nbins, width):
        raise ValueError(f"{name} shape {values.shape} does not match expected {(nbins, width)}")
    return values


def _validate_batch_slice(start: int, stop: int, npoints: int) -> None:
    if not 0 <= start < stop <= npoints:
        raise ValueError(f"invalid point batch {start}:{stop} for {npoints} points")


def _zarr() -> Any:
    try:
        import zarr
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise ImportError("automatic climatology caching requires the zarr dependency") from exc
    return zarr
