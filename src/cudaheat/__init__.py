"""CUDAHeat: GPU-native preprocessing for hourly marine heatwave analysis."""

from .core import CUDAHeat, BatchOutput, EventDetectionResult, PipelineResult
from .exceptions import CUDAHeatError, CUDAUnavailableError, InsufficientVRAMError
from .enrichment import EnrichmentRule, RuleContext, RuleValidationError, discover_rules, enrich
from .preprocessing import (
    PreparedNetCDF,
    PreprocessingReport,
    choose_chunk_shape,
    inspect_netcdf,
    optimize_netcdf,
    prepare_netcdf_input,
)
from .postgres import (
    DuplicateRunError,
    PostgresConfig,
    create_postgres_schema,
    enrich_stored_run,
    load_postgres_config,
    persist_detection_result,
)
from .validation import ValidationResult, validate_cpu_gpu

__all__ = [
    "BatchOutput",
    "CUDAHeat",
    "CUDAHeatError",
    "CUDAUnavailableError",
    "EnrichmentRule",
    "DuplicateRunError",
    "EventDetectionResult",
    "InsufficientVRAMError",
    "PipelineResult",
    "PreparedNetCDF",
    "PreprocessingReport",
    "PostgresConfig",
    "RuleContext",
    "RuleValidationError",
    "ValidationResult",
    "validate_cpu_gpu",
    "choose_chunk_shape",
    "inspect_netcdf",
    "optimize_netcdf",
    "prepare_netcdf_input",
    "create_postgres_schema",
    "discover_rules",
    "enrich",
    "enrich_stored_run",
    "load_postgres_config",
    "persist_detection_result",
]

__version__ = "0.1.0"
