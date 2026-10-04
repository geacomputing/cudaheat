"""CUDAHeat-specific exceptions."""


class CUDAHeatError(RuntimeError):
    """Base exception for CUDAHeat."""


class CUDAUnavailableError(CUDAHeatError):
    """Raised when CUDA/CuPy cannot provide a selected device."""


class InsufficientVRAMError(CUDAHeatError):
    """Raised when one complete time series cannot be processed on the GPU."""
