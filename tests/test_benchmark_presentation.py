from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    benchmark_dir = Path(__file__).parents[1] / "benchmarks"
    sys.path.insert(0, str(benchmark_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "benchmark_full_pipeline_presentation",
            benchmark_dir / "benchmark_full_pipeline_presentation.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(benchmark_dir))


def test_timestamped_presentation_run_directories_are_never_reused(tmp_path):
    module = _load_module()
    root = tmp_path / "presentation"
    first = module._timestamped_run_directory(root)
    first.mkdir()
    second = module._timestamped_run_directory(root)

    assert first != second
    assert first.name.startswith("presentation_")
    assert second.name.startswith("presentation_")
