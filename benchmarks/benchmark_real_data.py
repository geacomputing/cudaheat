"""Benchmark CUDAHeat on a NetCDF SST dataset using complete time series."""

from __future__ import annotations

import argparse
from pathlib import Path

from cudaheat import CUDAHeat
from cudaheat.benchmark import CUDAHeatBenchmark
from cudaheat.benchmark import plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--baseline", type=int, nargs=2, metavar=("START", "END"))
    parser.add_argument("--percentile", type=float, default=0.90)
    parser.add_argument("--smooth-window-hours", type=int, default=120)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--gpu-dtypes", choices=("float32", "float64"), nargs="+")
    parser.add_argument("--points", type=int, nargs="+")
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = tuple(args.baseline) if args.baseline else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with CUDAHeat(args.dataset, variable=args.variable, dtype=args.dtype) as hw:
        default_points = [10, 50, 100, 250, 500, 1000, 2000, 5000, 10000, hw.npoints]
        point_counts = args.points or [count for count in default_points if count <= hw.npoints]
        benchmark = CUDAHeatBenchmark(hw)
        benchmark.run(
            repeats=args.repeats,
            warmup=args.warmup,
            point_counts=point_counts,
            batch_sizes=args.batch_sizes or (None,),
            gpu_dtypes=args.gpu_dtypes or (args.dtype,),
            cpu_dtype="float64",
            baseline=baseline,
            percentile=args.percentile,
            smooth_window_hours=args.smooth_window_hours,
        )
        benchmark.export_csv(args.output_dir / "benchmark.csv")
        benchmark.export_json(args.output_dir / "benchmark.json")
        plotters = {
            "execution_time.png": plots.plot_cpu_vs_gpu_time,
            "speedup_dataset_size.png": plots.plot_speedup_vs_dataset_size,
            "speedup_batch_size.png": plots.plot_speedup_vs_batch_size,
            "vram_batch_size.png": plots.plot_vram_vs_batch_size,
            "transfer_vs_compute.png": plots.plot_transfer_vs_compute,
            "throughput_dataset_size.png": plots.plot_throughput_vs_dataset_size,
            "dtype_comparison.png": plots.plot_dtype_comparison,
            "repeated_runs.png": plots.plot_repeated_run_boxplot,
            "speedup_heatmap.png": plots.plot_speedup_heatmap,
        }
        for name, plotter in plotters.items():
            figure, _ = plotter(benchmark.records)
            figure.savefig(args.output_dir / name, bbox_inches="tight")
            figure.clear()
    print(f"CSV: {args.output_dir / 'benchmark.csv'}")
    print(f"JSON: {args.output_dir / 'benchmark.json'}")


if __name__ == "__main__":
    main()
