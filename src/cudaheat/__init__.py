"""CUDAHeat: GPU-native preprocessing for hourly marine heatwave analysis."""

from .core import CUDAHeat, BatchOutput, EventDetectionResult, PipelineResult
from .exceptions import CUDAHeatError, CUDAUnavailableError, InsufficientVRAMError
from .preprocessing import PreprocessingReport, choose_chunk_shape, inspect_netcdf, optimize_netcdf
from .validation import ValidationResult, validate_cpu_gpu

__all__ = [
    "BatchOutput",
    "CUDAHeat",
    "CUDAHeatError",
    "CUDAUnavailableError",
    "EventDetectionResult",
    "InsufficientVRAMError",
    "PipelineResult",
    "PreprocessingReport",
    "ValidationResult",
    "validate_cpu_gpu",
    "choose_chunk_shape",
    "inspect_netcdf",
    "optimize_netcdf",
]

__version__ = "0.1.0"
