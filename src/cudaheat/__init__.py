"""CUDAHeat: GPU-native preprocessing for hourly marine heatwave analysis."""

from .core import CUDAHeat, BatchOutput, PipelineResult
from .exceptions import CUDAHeatError, CUDAUnavailableError, InsufficientVRAMError

__all__ = [
    "BatchOutput",
    "CUDAHeat",
    "CUDAHeatError",
    "CUDAUnavailableError",
    "InsufficientVRAMError",
    "PipelineResult",
]

__version__ = "0.1.0"
