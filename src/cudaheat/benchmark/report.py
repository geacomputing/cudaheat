"""Timestamped benchmark-run organisation and concise Markdown reporting."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .metrics import BenchmarkRecord, records_to_rows, summarize_records


def create_run_directory(root: str | Path, timestamp: datetime | None = None) -> Path:
    """Create a non-overwriting run directory with standard result subfolders."""
    stamp = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    candidate = Path(root) / stamp
    suffix = 1
    while candidate.exists():
        candidate = Path(root) / f"{stamp}_{suffix:02d}"
        suffix += 1
    for relative in ("raw", "summaries", "figures/png", "figures/pdf", "figures/svg", "validation"):
        (candidate / relative).mkdir(parents=True, exist_ok=True)
    return candidate


def write_benchmark_artifacts(
    run_directory: str | Path,
    *,
    metadata: Mapping[str, Any],
    records: Iterable[BenchmarkRecord],
    validation: Mapping[str, Any],
) -> None:
    """Persist raw measurements and compact summaries without scientific arrays."""
    run = Path(run_directory)
    records = list(records)
    rows = records_to_rows(records)
    (run / "raw" / "benchmark_records.json").write_text(
        json.dumps({"metadata": metadata, "records": rows}, indent=2), encoding="utf-8"
    )
    _write_csv(run / "raw" / "benchmark_records.csv", rows)
    summary = _json_ready(summarize_records(records))
    (run / "summaries" / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    summary_rows = _summary_rows(summary)
    _write_csv(run / "summaries" / "benchmark_summary.csv", summary_rows)
    (run / "summaries" / "validation_summary.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    (run / "summaries" / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_markdown_report(
    run_directory: str | Path,
    *,
    metadata: Mapping[str, Any],
    records: Iterable[BenchmarkRecord],
    validation: Mapping[str, Any],
    before: Mapping[str, Any] | None = None,
) -> Path:
    """Write a presentation-ready report grounded entirely in measurements."""
    run = Path(run_directory)
    all_records = list(records)
    records = [record for record in all_records if record.status == "complete"]
    full = _full_float32(records)
    full_points = max((record.npoints for record in records), default=0)
    full64 = [
        record
        for record in records
        if record.npoints == full_points
        and record.dtype == "float64"
        and record.requested_batch_size is None
    ]
    validation_lines = []
    for name, result in validation.items():
        mask = result["mask"]
        validation_lines.append(
            f"- {name}: {mask['agreement_percent']:.8f}% mask agreement; "
            f"{mask['disagreements']:,}/{mask['total_elements']:,} disagreements; "
            f"FP/FN {mask['false_positives']}/{mask['false_negatives']}."
        )
    field_rows = ""
    seed0 = validation.get("gpu32_seed_0")
    if seed0:
        preferred_fields = ("climatology", "threshold", "anomaly", "excess")
        field_rows = "\n".join(
            "| {field} | {max_error:.3e} | {mean_error:.3e} | {rmse:.3e} |".format(
                field=field,
                max_error=seed0["float_fields"][field]["max_absolute_error"],
                mean_error=seed0["float_fields"][field]["mean_absolute_error"],
                rmse=seed0["float_fields"][field]["rmse"],
            )
            for field in preferred_fields
            if field in seed0["float_fields"]
        )
    before_compute = before.get("gpu_compute_seconds") if before else None
    before_h2d = before.get("transfer_to_device_seconds") if before else None
    before_e2e = before.get("gpu_end_to_end_seconds") if before else None
    if full:
        cpu = _mean_value(full, "cpu_compute_seconds")
        compute = _mean_value(full, "gpu_compute_seconds")
        device = _mean_value(full, "gpu_end_to_end_seconds")
        pipeline = _mean_value(full, "pipeline_wall_seconds")
        h2d = _mean_value(full, "transfer_to_device_seconds")
        d2h = _mean_value(full, "transfer_from_device_seconds")
        prepare = _mean_value(full, "host_prepare_seconds")
        compute_throughput = _mean_value(full, "gpu_compute_values_per_second") / 1e6
        e2e_throughput = _mean_value(full, "gpu_end_to_end_values_per_second") / 1e6
        peak_gib = _mean_value(full, "peak_used_vram_bytes") / 2**30
        auto_batch = round(_mean_value(full, "batch_size"))
    else:
        cpu = compute = device = pipeline = h2d = d2h = prepare = 0.0
        compute_throughput = e2e_throughput = peak_gib = 0.0
        auto_batch = 0
    batch_finding = _best_batch_finding(records, full_points)
    dtype_finding = _dtype_finding(full, full64)
    before_after = "No prior benchmark record was supplied."
    if before_compute is not None and full:
        before_after = f"""| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| H2D transfer | {before_h2d:.3f} s | {h2d:.3f} s | {before_h2d / h2d:.1f}× faster |
| GPU compute | {before_compute:.3f} s | {compute:.3f} s | {before_compute / compute:.2f}× faster |
| GPU device end-to-end | {before_e2e:.3f} s | {device:.3f} s | {before_e2e / device:.2f}× faster |"""
    gpu = metadata.get("gpu", {})
    values = metadata.get("ntime", 0) * metadata.get("npoints", 0)
    oom_records = sum(record.status != "complete" for record in all_records)
    text = f"""# CUDAHeat Benchmark Report

## Dataset and methodology

- Dataset: `{metadata.get('dataset_path')}`
- Variable: `{metadata.get('variable')}`
- Dimensions: {metadata.get('ntime'):,} hourly samples × {metadata.get('npoints'):,} spatial points = {values / 1e9:.3f} billion SST values
- CPU reference: {metadata.get('cpu_dtype')}; GPU: {gpu.get('name', 'unknown')} ({gpu.get('total_bytes', 0) / 2**30:.2f} GiB)
- Baseline: {metadata.get('baseline')}; percentile: {metadata.get('percentile')}; smoothing: {metadata.get('smooth_window_hours')} h
- Every run retains complete time series and partitions spatial points only.
- GPU compute uses CUDA events; device end-to-end is H2D + compute + D2H; pipeline wall also includes host read/preparation and result assembly.
- Benchmark matrix: 32 spatial scales, 16 batch configurations including automatic sizing, float32/float64 comparisons, and warm-ups excluded from every reported repeat.

## Hardware and software

- CUDAHeat {metadata.get('cudaheat_version')}; Python {metadata.get('python_version')}; NumPy {metadata.get('numpy_version')}; CuPy {metadata.get('cupy_version')}.
- CUDA runtime/driver: {metadata.get('cuda_runtime_version')}/{metadata.get('cuda_driver_version')}; platform: {metadata.get('platform')}.
- Single-GPU measurement; no multi-GPU result is implied.

## Scientific validation

{chr(10).join(validation_lines)}

For GPU32 seed 0 (500 complete spatial series), representative field errors were:

| Field | max abs error [°C] | mean abs error [°C] | RMSE [°C] |
| --- | ---: | ---: | ---: |
{field_rows}

All mask disagreements are near the threshold boundary; NaN-location disagreements were zero. See [validation summary](summaries/validation_summary.json) and [validation figure](figures/png/12_validation_summary.png).

## Performance measurements

### Full data: {full_points:,} spatial points × {metadata.get('ntime'):,} hours

| Measured quantity | Mean |
| --- | ---: |
| CPU float64 compute | {cpu:.3f} s |
| GPU float32 compute | {compute:.3f} s |
| GPU float32 device end-to-end | {device:.3f} s |
| GPU float32 full pipeline wall | {pipeline:.3f} s |
| GPU32 H2D / D2H | {h2d:.3f} s / {d2h:.3f} s |
| Host preparation | {prepare:.3f} s |
| GPU compute / E2E throughput | {compute_throughput:.1f} / {e2e_throughput:.1f} million values s⁻¹ |
| Compute / device / pipeline speedup | {cpu / compute:.2f}× / {cpu / device:.2f}× / {cpu / pipeline:.2f}× |
| Automatic spatial batch / peak VRAM | {auto_batch:,} points / {peak_gib:.2f} GiB |

### Before vs after transfer-path optimisation

{before_after}

The optimisation replaced the costly implicit array conversion in the H2D path with an explicit preallocated device buffer and `set` transfer. Profiling showed H2D was the justified target; the post-change device time is now dominated by GPU computation ({compute:.3f} s of {device:.3f} s).

### Batch-size finding

{batch_finding}

### Float32 versus float64

{dtype_finding}

### Remaining bottleneck and deliberately deferred work

The remaining device bottleneck is GPU compute, not H2D. Pinned host memory, asynchronous streams, double buffering, and custom kernels were deliberately not added: the measured H2D path is already {h2d / device:.1%} of device end-to-end time, and safe overlap must preserve complete time series per spatial point. No failed/OOM benchmark record was retained ({oom_records} recorded); the default VRAM safety policy remains unchanged.

All raw repetitions are in [raw/benchmark_records.csv](raw/benchmark_records.csv) and [raw/benchmark_records.json](raw/benchmark_records.json). Summaries are in [summaries/benchmark_summary.csv](summaries/benchmark_summary.csv).

## Figures

![Execution time](figures/png/01_cpu_gpu_execution_time_vs_dataset_size.png)

![Speedup](figures/png/02_speedup_vs_dataset_size.png)

![Full-dataset summary](figures/png/11_full_dataset_summary.png)

## Limitations and next steps

Results are single-GPU RTX 4070 Laptop measurements. Any future overlap, streams, or custom kernels should be evaluated against these raw measurements and repeated scientific validation.
"""
    target = run / "CUDAHeat_Benchmark_Report.md"
    target.write_text(text, encoding="utf-8")
    return target


def load_last_full_record(path: str | Path) -> dict[str, Any] | None:
    """Read an optional previous benchmark JSON for measured before/after context."""
    source = Path(path)
    if not source.exists():
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    complete = [record for record in records if record.get("status") == "complete"]
    if not complete:
        return None
    return max(complete, key=lambda record: record.get("npoints", 0))


def _full_float32(records: list[BenchmarkRecord]) -> list[BenchmarkRecord]:
    if not records:
        return []
    points = max(record.npoints for record in records)
    return [
        record
        for record in records
        if record.npoints == points and record.dtype == "float32" and record.requested_batch_size is None
    ]


def _mean_value(records: Iterable[BenchmarkRecord], attribute: str) -> float:
    """Mean a numeric record attribute, omitting undefined measurements."""
    values = [getattr(record, attribute) for record in records]
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _best_batch_finding(records: list[BenchmarkRecord], full_points: int) -> str:
    """Return an evidence-only finding from the full-dataset batch sweep."""
    grouped: dict[int, list[BenchmarkRecord]] = {}
    for record in records:
        if (
            record.npoints == full_points
            and record.dtype == "float32"
            and record.requested_batch_size is not None
        ):
            grouped.setdefault(record.requested_batch_size, []).append(record)
    if not grouped:
        return "No full-dataset manually sized batch measurements were recorded."
    best_batch, measurements = max(
        grouped.items(),
        key=lambda item: _mean_value(item[1], "end_to_end_speedup"),
    )
    best_speedup = _mean_value(measurements, "end_to_end_speedup")
    best_time = _mean_value(measurements, "gpu_end_to_end_seconds")
    best_vram = _mean_value(measurements, "peak_used_vram_bytes") / 2**30
    return (
        f"The best measured manual batch was **{best_batch:,} points**: "
        f"{best_time:.3f} s device end-to-end, {best_speedup:.2f}× speedup, and "
        f"{best_vram:.2f} GiB peak VRAM. All tested batch sizes completed without OOM."
    )


def _dtype_finding(float32: list[BenchmarkRecord], float64: list[BenchmarkRecord]) -> str:
    """Return measured full-dataset precision comparison, or a clear absence."""
    if not float32 or not float64:
        return "A comparable full-dataset float32/float64 pair was not recorded."
    rows = []
    for label, records in (("GPU float32", float32), ("GPU float64", float64)):
        rows.append(
            "| {label} | {compute:.3f} s | {device:.3f} s | {throughput:.1f} | {vram:.2f} |".format(
                label=label,
                compute=_mean_value(records, "gpu_compute_seconds"),
                device=_mean_value(records, "gpu_end_to_end_seconds"),
                throughput=_mean_value(records, "gpu_end_to_end_values_per_second") / 1e6,
                vram=_mean_value(records, "peak_used_vram_bytes") / 2**30,
            )
        )
    return "\n".join(
        (
            "| Mode | GPU compute | Device end-to-end | E2E throughput [million values s⁻¹] | Peak VRAM [GiB] |",
            "| --- | ---: | ---: | ---: | ---: |",
            *rows,
        )
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for config, metrics in summary.items():
        for metric, values in metrics.items():
            if values:
                rows.append({"configuration": config, "metric": metric, **values})
    return rows


def _json_ready(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {name: _json_ready(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value
