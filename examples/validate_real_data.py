"""Validate CPU float64 reference fields against CUDAHeat GPU fields."""

from __future__ import annotations

import argparse
from pathlib import Path

from cudaheat import CUDAHeat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--baseline", type=int, nargs=2, metavar=("START", "END"))
    parser.add_argument("--percentile", type=float, default=0.90)
    parser.add_argument("--smooth-window-hours", type=int, default=120)
    parser.add_argument("--points", type=int, default=50)
    parser.add_argument("--point-indices", type=int, nargs="*")
    parser.add_argument("--full-dataset", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("validation-results/validation.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = tuple(args.baseline) if args.baseline else None
    with CUDAHeat(args.dataset, variable=args.variable, dtype=args.gpu_dtype) as hw:
        result = hw.validate(
            points=args.points,
            point_indices=args.point_indices,
            full_dataset=args.full_dataset,
            random_seed=args.seed,
            gpu_dtype=args.gpu_dtype,
            baseline=baseline,
            percentile=args.percentile,
            smooth_window_hours=args.smooth_window_hours,
            validation_batch_size=args.validation_batch_size,
        )
    result.to_json(args.output)
    print(result.summary())
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
