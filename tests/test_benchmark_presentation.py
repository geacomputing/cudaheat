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


def test_gpu_campaign_rows_drop_cpu_performance_metrics():
    module = _load_module()

    row = module._gpu_row(
        {
            "gpu_compute_seconds": 1.0,
            "cpu_compute_seconds": 4.0,
            "compute_speedup": 4.0,
            "pipeline_speedup": 2.0,
            "cpu_dtype": "float64",
        }
    )

    assert row == {"gpu_compute_seconds": 1.0}


def test_gpu_campaign_keeps_artifacts_in_one_run_directory(tmp_path):
    module = _load_module()
    root = tmp_path / "run"

    layout = module._create_layout(root)

    assert root.is_dir()
    assert set(layout.values()) == {root}
