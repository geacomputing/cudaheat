"""Run CUDAHeat validation, dense benchmarks, figures, and a Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

from cudaheat import CUDAHeat
from cudaheat.benchmark import CUDAHeatBenchmark, presentation
from cudaheat.benchmark.report import (
    create_run_directory,
    load_last_full_record,
    write_benchmark_artifacts,
    write_markdown_report,
)


DENSE_POINTS = (
    10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1250,
    1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000, 4500, 5000, 6000,
    7000, 8000, 9000, 10000, 12000, 14000, 16000,
)
BATCH_SIZES = (64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2176, 2304, 2560, 3072)
DTYPE_POINTS = (500, 1000, 2000, 5000, 10000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--baseline", type=int, nargs=2, metavar=("START", "END"), required=True)
    parser.add_argument("--percentile", type=float, default=0.90)
    parser.add_argument("--smooth-window-hours", type=int, default=120)
    parser.add_argument("--output-root", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--quick", action="store_true", help="Run a smaller smoke-test campaign.")
    parser.add_argument("--skip-full-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = tuple(args.baseline)
    run = create_run_directory(args.output_root)
    print(f"Results: {run}")
    before = load_last_full_record(args.output_root / "benchmark.json")
    with CUDAHeat(args.dataset, variable=args.variable, dtype="float32") as hw:
        validation = _run_validation(hw, run, args, baseline)
        benchmark = CUDAHeatBenchmark(hw)
        _run_benchmarks(benchmark, hw.npoints, args, baseline)
        metadata = {**benchmark.metadata, "gpu_name": hw.gpu_info.name, "run_directory": str(run)}
        write_benchmark_artifacts(
            run, metadata=metadata, records=benchmark.records, validation=validation
        )
        _write_figures(run, benchmark.records, metadata, validation)
        write_markdown_report(
            run,
            metadata=metadata,
            records=benchmark.records,
            validation=validation,
            before=before,
        )
    print(f"Report: {run / 'CUDAHeat_Benchmark_Report.md'}")


def _run_validation(hw: CUDAHeat, run: Path, args: argparse.Namespace, baseline: tuple[int, int]) -> dict:
    cases = [("gpu32_seed_0", "float32", 500, 0), ("gpu32_seed_42", "float32", 500, 42), ("gpu32_seed_123", "float32", 500, 123), ("gpu64_seed_0", "float64", 500, 0)]
    if not args.skip_full_validation:
        cases.append(("gpu32_seed_0_1000_points", "float32", 1000, 0))
    if args.quick:
        cases = [("gpu32_seed_0", "float32", 50, 0), ("gpu64_seed_0", "float64", 50, 0)]
    results = {}
    for name, dtype, points, seed in cases:
        print(f"Validation: {name} ({points} points)")
        result = hw.validate(
            points=points,
            random_seed=seed,
            gpu_dtype=dtype,
            baseline=baseline,
            percentile=args.percentile,
            smooth_window_hours=args.smooth_window_hours,
        )
        results[name] = result.to_dict()
        result.to_json(run / "validation" / f"{name}.json")
    return results


def _run_benchmarks(
    benchmark: CUDAHeatBenchmark,
    npoints: int,
    args: argparse.Namespace,
    baseline: tuple[int, int],
) -> None:
    dense = tuple(point for point in DENSE_POINTS if point < npoints) + (npoints,)
    if args.quick:
        dense = tuple(point for point in (10, 100, 500) if point <= npoints)
    small = tuple(point for point in dense if point <= 2000)
    large = tuple(point for point in dense if point > 2000)
    common = dict(
        cpu_dtype="float64",
        baseline=baseline,
        percentile=args.percentile,
        smooth_window_hours=args.smooth_window_hours,
    )
    if small:
        print(f"Dense scaling: {len(small)} small/medium sizes × 5 repeats")
        benchmark.run(repeats=5 if not args.quick else 2, warmup=1, point_counts=small, batch_sizes=(None,), gpu_dtypes=("float32",), **common)
    if large:
        print(f"Dense scaling: {len(large)} large sizes × 3 repeats")
        benchmark.run(repeats=3 if not args.quick else 2, warmup=1, point_counts=large, batch_sizes=(None,), gpu_dtypes=("float32",), **common)
    batch_sizes = (None,) + BATCH_SIZES
    if args.quick:
        batch_sizes = (None, 128, 512, 1024)
    batch_points = min(npoints, 500) if args.quick else npoints
    print(f"Batch-size scaling: {len(batch_sizes)} configurations")
    benchmark.run(repeats=3 if not args.quick else 2, warmup=1, point_counts=(batch_points,), batch_sizes=batch_sizes, gpu_dtypes=("float32",), **common)
    dtype_points = tuple(point for point in DTYPE_POINTS if point < npoints) + (npoints,)
    if args.quick:
        dtype_points = tuple(point for point in (500,) if point <= npoints)
    print(f"Float32/float64 comparison: {len(dtype_points)} sizes")
    benchmark.run(repeats=3 if not args.quick else 2, warmup=1, point_counts=dtype_points, batch_sizes=(None,), gpu_dtypes=("float32", "float64"), **common)


def _write_figures(run: Path, records, metadata: dict, validation: dict) -> None:
    figures = (
        ("01_cpu_gpu_execution_time_vs_dataset_size", presentation.plot_execution_time, (records, metadata)),
        ("02_speedup_vs_dataset_size", presentation.plot_speedup, (records, metadata)),
        ("03_gpu_time_breakdown_vs_dataset_size", presentation.plot_time_breakdown, (records,)),
        ("04_gpu_throughput_vs_dataset_size", presentation.plot_throughput, (records,)),
        ("05_speedup_vs_batch_size", presentation.plot_speedup_batch_size, (records,)),
        ("06_vram_vs_batch_size", presentation.plot_vram_batch_size, (records,)),
        ("07_transfer_vs_compute_vs_batch_size", presentation.plot_transfer_compute_batch_size, (records,)),
        ("08_float32_vs_float64_performance", presentation.plot_dtype_performance, (records,)),
        ("09_repeated_run_variability", presentation.plot_repeated_variability, (records,)),
        ("10_speedup_heatmap", presentation.plot_speedup_heatmap, (records,)),
        ("11_full_dataset_summary", presentation.plot_full_dataset_summary, (records, metadata)),
    )
    for name, plotter, parameters in figures:
        figure, _ = plotter(*parameters)
        presentation.save_figure_set(figure, run, name)
    figure, _ = presentation.plot_validation_summary(validation["gpu32_seed_0"])
    presentation.save_figure_set(figure, run, "12_validation_summary")


if __name__ == "__main__":
    main()
