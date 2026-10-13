"""Standalone utilities for preparing CUDAHeat benchmark inputs."""

from .extend_netcdf_backward import extend_backward
from .timing import log_elapsed_time

__all__ = ["extend_backward", "log_elapsed_time"]
