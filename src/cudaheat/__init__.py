"""CUDAHeat: GPU-native preprocessing for hourly marine heatwave analysis."""

from .core import CUDAHeat, BatchOutput, PipelineResult
from .exceptions import CUDAHeatError, CUDAUnavailableError, InsufficientVRAMError
from .validation import ValidationResult, validate_cpu_gpu

__all__ = [
    "BatchOutput",
    "CUDAHeat",
    "CUDAHeatError",
    "CUDAUnavailableError",
    "InsufficientVRAMError",
    "PipelineResult",
    "ValidationResult",
    "validate_cpu_gpu",
]

__version__ = "0.1.0"
