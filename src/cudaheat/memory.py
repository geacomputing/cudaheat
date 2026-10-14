"""GPU discovery, conservative VRAM estimation, and batch telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .exceptions import CUDAUnavailableError


def cupy() -> Any:
    """Import CuPy lazily so CPU-only installations can import CUDAHeat."""
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise CUDAUnavailableError(
            "CuPy is required for GPU processing. Install CUDAHeat with its GPU extra."
        ) from exc
    return cp


@dataclass(frozen=True)
class GPUInfo:
    """A runtime snapshot of one CUDA device."""

    device_id: int
    name: str
    total_bytes: int
    free_bytes: int
    used_bytes: int
    compute_capability: tuple[int, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatchEstimate:
    """Transparent, deliberately conservative estimate of one spatial batch."""

    batch_size: int
    bytes_per_point: int
    budget_bytes: int
    available_bytes: int
    safety_fraction: float
    float_arrays: int
    workspace_multiplier: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatchTelemetry:
    """Measured timing and VRAM information for a processed spatial batch."""

    device_id: int
    point_start: int
    point_stop: int
    batch_size: int
    dtype: str
    bytes_per_point: int
    safety_fraction: float
    oom_retries: int
    free_before_bytes: int
    used_before_bytes: int
    free_after_bytes: int
    used_after_bytes: int
    peak_used_bytes: int
    pool_used_bytes: int
    transfer_to_device_seconds: float
    gpu_compute_seconds: float
    transfer_from_device_seconds: float
    host_read_seconds: float = 0.0
    host_prepare_seconds: float = 0.0
    gpu_allocation_seconds: float = 0.0
    gpu_metadata_transfer_seconds: float = 0.0
    result_assembly_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_gpu_info(device_id: int = 0) -> GPUInfo:
    """Return current memory information for ``device_id``."""
    cp = cupy()
    try:
        with cp.cuda.Device(device_id):
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode()
            return GPUInfo(
                device_id=device_id,
                name=str(name),
                total_bytes=int(total_bytes),
                free_bytes=int(free_bytes),
                used_bytes=int(total_bytes - free_bytes),
                compute_capability=(int(props["major"]), int(props["minor"])),
            )
    except Exception as exc:  # CuPy exposes several CUDA-specific exception types.
        raise CUDAUnavailableError(f"CUDA device {device_id} is unavailable: {exc}") from exc


class VRAMEstimator:
    """Estimate a safe point-only batch size without ever partitioning time.

    The estimate reserves float arrays for SST, climatology, threshold, anomaly,
    excess, and two reduction/percentile work arrays. It also reserves the
    four hourly-cycle fields and float64 smoothing prefix sums (even for a
    float32 pipeline). It additionally reserves a boolean mask and applies a
    workspace multiplier. Actual allocations are recorded in
    :class:`BatchTelemetry` and OOM recovery remains authoritative.
    """

    def __init__(
        self,
        ntime: int,
        itemsize: int,
        safety_fraction: float = 0.30,
        float_arrays: int = 7,
        workspace_multiplier: float = 1.20,
        cycle_length: int = 366 * 24,
    ) -> None:
        if ntime < 1:
            raise ValueError("ntime must be positive")
        if itemsize not in (4, 8):
            raise ValueError("itemsize must correspond to float32 or float64")
        if not 0 <= safety_fraction < 1:
            raise ValueError("vram_safety must be in [0, 1)")
        if float_arrays < 1 or workspace_multiplier < 1 or cycle_length < 1:
            raise ValueError("memory estimate parameters must be positive and conservative")
        self.ntime = ntime
        self.itemsize = itemsize
        self.safety_fraction = safety_fraction
        self.float_arrays = float_arrays
        self.workspace_multiplier = workspace_multiplier
        self.cycle_length = cycle_length

    @property
    def bytes_per_point(self) -> int:
        time_arrays = self.ntime * (self.itemsize * self.float_arrays + 1)
        # Four retained raw/smoothed cycle fields, a padded float workspace,
        # its validity mask, and two float64 prefix sums used for stable
        # rolling means. This matters for short inputs where a fixed annual
        # cycle otherwise dominates the allocation.
        cycle_workspace = self.cycle_length * (5 * self.itemsize + 1 + 2 * 8)
        raw = time_arrays + cycle_workspace
        return int(raw * self.workspace_multiplier)

    def estimate(self, free_bytes: int, npoints: int) -> BatchEstimate:
        if free_bytes < 0 or npoints < 1:
            raise ValueError("free_bytes must be non-negative and npoints must be positive")
        budget = int(free_bytes * (1 - self.safety_fraction))
        batch_size = min(npoints, budget // self.bytes_per_point)
        return BatchEstimate(
            batch_size=int(batch_size),
            bytes_per_point=self.bytes_per_point,
            budget_bytes=budget,
            available_bytes=int(free_bytes),
            safety_fraction=self.safety_fraction,
            float_arrays=self.float_arrays,
            workspace_multiplier=self.workspace_multiplier,
        )


def reduced_batch_size(batch_size: int) -> int:
    """Return the next point-only OOM retry size, never changing time."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return max(1, batch_size // 2)


_STRUCTURAL_BATCH_SIZES = (512, 1024, 1536, 2048)


def guess_batch_size(estimate: BatchEstimate, *, npoints: int) -> int:
    """Choose one fixed, chunk-aligned batch size without executing CUDA work.

    Optimized CUDAHeat inputs use 512-point spatial chunks, so the only
    structural choices are 512, 1024, 1536, and 2048 points.  The safe VRAM
    ceiling is deliberately divided by four for allocator and workspace
    headroom, then the nearest permitted candidate is used.  This is an
    immediate heuristic, not a hardware-specific benchmark; OOM recovery
    remains authoritative.
    """
    if npoints < 1:
        raise ValueError("npoints must be positive")
    ceiling = min(estimate.batch_size, npoints)
    if ceiling < 1:
        raise ValueError("the VRAM estimate cannot fit one spatial point")
    target = ceiling / 4
    candidates = tuple(size for size in _STRUCTURAL_BATCH_SIZES if size <= ceiling)
    if not candidates:
        # A constrained GPU can still make progress, but cannot use one of
        # the standard chunk-aligned sizes.
        return ceiling
    return min(candidates, key=lambda size: (abs(size - target), -size))


def release_gpu_memory(device_id: int) -> None:
    """Release reusable CuPy allocations after an OOM or completed batch."""
    cp = cupy()
    with cp.cuda.Device(device_id):
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
