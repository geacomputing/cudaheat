from __future__ import annotations

import csv
import json

from cudaheat.benchmark import BenchmarkRecord, CUDAHeatBenchmark, summarize_records


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
