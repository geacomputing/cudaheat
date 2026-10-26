"""Discovery and lightweight contract checks for self-contained Python rules."""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any


class RuleValidationError(ValueError):
    """Raised when a Python enrichment rule does not meet the shared contract."""


@dataclass(frozen=True)
class EnrichmentRule:
    """A discovered rule module together with immutable file provenance."""

    source_path: Path
    metadata: Mapping[str, Any]
    calculate: Callable[[Any], Mapping[str, Any]]
    sha256: str

    @property
    def id(self) -> str:
        return str(self.metadata["id"])

    @property
    def version(self) -> str:
        return str(self.metadata["version"])


def discover_rules(rule_directory: Path) -> tuple[EnrichmentRule, ...]:
    """Load each ``*.py`` rule file, rejecting malformed metadata clearly."""
    paths = tuple(sorted(path for path in rule_directory.glob("*.py") if not path.name.startswith("_")))
    if not paths:
        raise RuleValidationError(f"no Python enrichment rules found in {rule_directory}")
    rules: list[EnrichmentRule] = []
    identifiers: set[tuple[str, str]] = set()
    for path in paths:
        module = _load_module(path)
        metadata = getattr(module, "METADATA", None)
        calculate = getattr(module, "calculate", None)
        _validate_rule(path, metadata, calculate)
        identifier = (str(metadata["id"]), str(metadata["version"]))
        if identifier in identifiers:
            raise RuleValidationError(
                f"duplicate enrichment rule id/version {identifier[0]}@{identifier[1]}: {path}"
            )
        identifiers.add(identifier)
        rules.append(
            EnrichmentRule(
                source_path=path,
                metadata=metadata,
                calculate=calculate,
                sha256=sha256(path.read_bytes()).hexdigest(),
            )
        )
    return tuple(rules)


def _load_module(path: Path) -> ModuleType:
    module_name = "cudaheat_enrichment_rule_" + sha256(str(path).encode("utf-8")).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuleValidationError(f"could not load enrichment rule module: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuleValidationError(f"could not import enrichment rule {path}: {exc}") from exc
    return module


def _validate_rule(path: Path, metadata: Any, calculate: Any) -> None:
    if not isinstance(metadata, Mapping):
        raise RuleValidationError(f"invalid enrichment rule {path}: METADATA must be a dictionary")
    required = {"id", "version", "name", "description", "category", "output", "formula_latex", "postgres"}
    missing = sorted(required - set(metadata))
    if missing:
        raise RuleValidationError(f"invalid enrichment rule {path}: METADATA lacks {missing}")
    if not isinstance(metadata["id"], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", metadata["id"]):
        raise RuleValidationError(f"invalid enrichment rule {path}: metadata.id is invalid")
    if not isinstance(metadata["version"], str) or not re.fullmatch(r"\d+\.\d+\.\d+", metadata["version"]):
        raise RuleValidationError(f"invalid enrichment rule {path}: metadata.version must be semantic")
    if not all(isinstance(metadata[name], str) and metadata[name] for name in ("name", "description", "category", "formula_latex")):
        raise RuleValidationError(f"invalid enrichment rule {path}: descriptive metadata must be non-empty strings")
    output = metadata["output"]
    if not isinstance(output, Mapping) or {"primary", "unit", "range"} - set(output):
        raise RuleValidationError(
            f"invalid enrichment rule {path}: metadata.output needs primary, unit, and range"
        )
    if not isinstance(output["primary"], str) or not isinstance(output["unit"], str):
        raise RuleValidationError(f"invalid enrichment rule {path}: metadata.output is malformed")
    if not isinstance(output["range"], (list, tuple)) or len(output["range"]) != 2:
        raise RuleValidationError(f"invalid enrichment rule {path}: metadata.output.range needs two values")
    postgres = metadata["postgres"]
    if not isinstance(postgres, Mapping) or {"select_sql", "write_sql"} - set(postgres):
        raise RuleValidationError(f"invalid enrichment rule {path}: metadata.postgres is malformed")
    select_sql = str(postgres["select_sql"]).lstrip().upper()
    write_sql = str(postgres["write_sql"]).lstrip().upper()
    if not select_sql.startswith("SELECT") or "%(RUN_ID)S" not in select_sql:
        raise RuleValidationError(f"invalid enrichment rule {path}: postgres.select_sql is not a run selector")
    for placeholder in ("%(AFTER_GRID_POINT_INDEX)S", "%(AFTER_EVENT_ID)S", "%(BATCH_SIZE)S"):
        if placeholder not in select_sql:
            raise RuleValidationError(f"invalid enrichment rule {path}: select_sql lacks {placeholder.lower()}")
    if not write_sql.startswith("INSERT") or "ON CONFLICT" not in write_sql or "DO NOTHING" not in write_sql:
        raise RuleValidationError(f"invalid enrichment rule {path}: write_sql must preserve version history")
    if not callable(calculate):
        raise RuleValidationError(f"invalid enrichment rule {path}: calculate(context) is required")
