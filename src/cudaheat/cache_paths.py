"""Central, Git-ignored locations for CUDAHeat-derived local products."""

from __future__ import annotations

import os
from pathlib import Path


_CACHE_ENVIRONMENT_VARIABLE = "CUDAHEAT_CACHE_DIR"
_DEFAULT_CACHE_DIRECTORY = ".cudaheat-cache"


def derived_cache_root() -> Path:
    """Return the central cache directory, optionally overridden for deployments."""
    configured = os.environ.get(_CACHE_ENVIRONMENT_VARIABLE)
    if configured:
        return Path(configured).expanduser().resolve()
    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "pyproject.toml").is_file():
        return package_root / _DEFAULT_CACHE_DIRECTORY
    return Path.cwd() / _DEFAULT_CACHE_DIRECTORY


def optimized_netcdf_path(source_path: str | Path, source_sha256: str) -> Path:
    """Return the central optimized-NetCDF path for one raw source revision."""
    source = Path(source_path)
    suffix = source.suffix or ".nc"
    return _source_cache_directory(source_sha256) / f"{source.stem}.cudaheat{suffix}"


def climatology_zarr_path(
    source_path: str | Path, source_sha256: str, climatology_hash: str
) -> Path:
    """Return the central Zarr path for one source revision and cycle configuration."""
    source = Path(source_path)
    stem = source.stem.removesuffix(".cudaheat")
    return _source_cache_directory(source_sha256) / f"{stem}.cudaheat.clim-{climatology_hash}.zarr"


def _source_cache_directory(source_sha256: str) -> Path:
    if len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a full SHA-256 hexadecimal digest")
    return derived_cache_root() / source_sha256
