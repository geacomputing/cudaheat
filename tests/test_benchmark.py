from __future__ import annotations

import csv
from dataclasses import replace
import json

from cudaheat.benchmark import BenchmarkRecord, CUDAHeatBenchmark, summarize_records
from cudaheat.benchmark.presentation import _full_batch_rows
from cudaheat.benchmark.report import create_run_directory, write_markdown_report


def _record(repeat: int, cpu: float = 4.0, gpu: float = 1.0) -> BenchmarkRecord:
    return BenchmarkRecord(
        repeat=repeat,
        ntime=10,
        npoints=20,
        dtype="float32",
        device_id=0,
        cpu_compute_seconds=cpu,
        gpu_compute_seconds=gpu,
        transfer_to_device_seconds=0.5,
        transfer_from_device_seconds=0.1,
        gpu_end_to_end_seconds=1.6,
        batch_size=20,
        free_vram_bytes=100,
        peak_used_vram_bytes=50,
        safety_fraction=0.30,
    )


def test_benchmark_record_separates_compute_and_end_to_end_metrics():
    record = _record(0)
    assert record.compute_speedup == 4.0
    assert record.end_to_end_speedup == 2.5
    assert record.values_processed == 200
    assert record.gpu_compute_values_per_second == 200.0


def test_pipeline_speedup_and_automatic_batch_rows_are_measured_not_duplicated():
    first = replace(_record(0), pipeline_wall_seconds=2.0, batch_size=10, requested_batch_size=None)
    second = replace(_record(1), pipeline_wall_seconds=2.0, batch_size=11, requested_batch_size=None)
    manual = replace(_record(2), pipeline_wall_seconds=2.0, batch_size=12, requested_batch_size=12)

    assert first.pipeline_speedup == 2.0
    rows = _full_batch_rows([first, second, manual])
    assert len([row for row in rows if row["automatic"]]) == 1
    assert len(rows) == 2


def test_benchmark_summary_and_exports_retain_each_repeat(tmp_path):
    benchmark = CUDAHeatBenchmark(object())
    benchmark.records = [_record(0), _record(1, cpu=6.0, gpu=2.0)]
    benchmark.metadata = {"dataset_path": "synthetic.nc"}
    summary = summarize_records(benchmark.records)
    assert summary["points=20;dtype=float32;batch=automatic"]["cpu_compute_seconds"].count == 2

    json_path = benchmark.export_json(tmp_path / "benchmark.json")
    csv_path = benchmark.export_csv(tmp_path / "benchmark.csv")
    payload = json.loads(json_path.read_text())
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(payload["records"]) == 2
    assert len(rows) == 2


def test_timestamped_report_includes_measured_before_after_summary(tmp_path):
    run = create_run_directory(tmp_path)
    float32 = replace(
        _record(0),
        pipeline_wall_seconds=2.0,
        requested_batch_size=None,
        peak_used_vram_bytes=2**30,
    )
    float64 = replace(float32, dtype="float64", gpu_compute_seconds=1.5, gpu_end_to_end_seconds=2.1)
    validation = {
        "gpu32_seed_0": {
            "mask": {
                "agreement_percent": 100.0,
                "disagreements": 0,
                "total_elements": 200,
                "false_positives": 0,
                "false_negatives": 0,
            },
            "float_fields": {
                name: {"max_absolute_error": 0.0, "mean_absolute_error": 0.0, "rmse": 0.0}
                for name in ("climatology", "threshold", "anomaly", "excess")
            },
        }
    }
    metadata = {
        "dataset_path": "synthetic.nc", "variable": "sst", "ntime": 10, "npoints": 20,
        "cpu_dtype": "float64", "gpu": {"name": "test GPU", "total_bytes": 2**30},
        "baseline": (2000, 2001), "percentile": 0.9, "smooth_window_hours": 3,
    }
    target = write_markdown_report(
        run,
        metadata=metadata,
        records=[float32, float64],
        validation=validation,
        before={"transfer_to_device_seconds": 1.0, "gpu_compute_seconds": 2.0, "gpu_end_to_end_seconds": 3.0},
    )

    text = target.read_text()
    assert "Before vs after transfer-path optimisation" in text
    assert "Float32 versus float64" in text
    assert "GPU float64" in text
