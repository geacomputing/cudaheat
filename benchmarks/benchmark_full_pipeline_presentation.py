"""Timestamped CUDAHeat benchmark with slide-ready PNG artifacts.

This campaign reuses the established full-pipeline measurement matrix, while
retaining every raw measurement and rendering the advanced ``checkgraph``
figures as 16:9 PNGs (opaque and transparent variants) at 300 dpi.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import benchmark_full_pipeline_beta as beta


BENCHMARK_NAME = "CUDAHeat Full-Pipeline Presentation Benchmark"
BENCHMARK_ID = "full_pipeline_presentation"
DEFAULT_OUTPUT_ROOT = Path("benchmark_results") / "full_pipeline_presentation"
DEFAULT_TITLE = "CUDAHeat Full-Pipeline Benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=beta.DEFAULT_DATASET)
    parser.add_argument("--variable", default=beta.DEFAULT_VARIABLE)
    parser.add_argument("--baseline", type=int, nargs=2, metavar=("START", "END"), default=beta.DEFAULT_BASELINE)
    parser.add_argument("--percentile", type=float, default=0.90)
    parser.add_argument("--smooth-window-hours", type=int, default=120)
    parser.add_argument("--output-root", "--output-dir", dest="output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Presentation title placed on every figure.")
    parser.add_argument(
        "--titles-json",
        type=Path,
        help="Optional JSON object mapping figure basenames to replacement presentation titles.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--validation-points", type=int, default=512)
    parser.add_argument("--lazy-spatial-chunk", type=int, default=256)
    parser.add_argument("--dask-chunk-sizes", type=int, nargs="+")
    parser.add_argument("--chunk-repeats", type=int, default=2)
    parser.add_argument("--vram-safety", type=float, default=beta.VRAM_SAFETY)
    parser.add_argument("--gpu-total-vram-gib", type=float, default=beta.GPU_TOTAL_VRAM_GIB)
    parser.add_argument("--vram-safety-values", type=float, nargs="+", metavar="FRACTION")
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--quick", action="store_true", help="Run the compact smoke-test matrix.")
    parser.add_argument("--skip-lazy-eager", action="store_true")
    parser.add_argument("--skip-vram-safety-sweep", action="store_true")
    parser.add_argument(
        "--render-only",
        type=Path,
        metavar="RUN_DIRECTORY",
        help="Regenerate PNG figures from a retained timestamped campaign.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure_titles = _load_figure_titles(args.titles_json)
    _install_presentation_hooks(args.title, figure_titles)
    if args.render_only is not None:
        beta._render_existing_campaign(args.render_only)
        return

    beta._validate_campaign_args(args)
    args.output_dir = _timestamped_run_directory(args.output_root)
    args.overwrite = False
    beta._run_campaign(args)
    print(f"Presentation benchmark artifacts: {args.output_dir}")


def _timestamped_run_directory(root: Path) -> Path:
    """Return a never-reused run directory next to the requested root name."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = root.with_name(f"{root.name}_{stamp}")
    suffix = 2
    while candidate.exists():
        candidate = root.with_name(f"{root.name}_{stamp}_{suffix:02d}")
        suffix += 1
    return candidate


def _install_presentation_hooks(title: str, figure_titles: Mapping[str, str]) -> None:
    """Keep beta's measured data contract, replacing only derivative outputs."""
    beta.BENCHMARK_NAME = BENCHMARK_NAME
    beta.BENCHMARK_ID = BENCHMARK_ID

    def render_figures(layout: Mapping[str, Path], *_: Any) -> None:
        _render_advanced_figures(
            layout["tables"], layout["root"] / "figures" / "presentation", title, figure_titles
        )

    def write_report(layout: Mapping[str, Path], metadata: Mapping[str, Any], *_: Any) -> None:
        _write_presentation_report(layout["root"], metadata, title)

    beta._write_figures = render_figures
    beta._write_report = write_report


def _render_advanced_figures(
    tables: Path, output: Path, title: str, figure_titles: Mapping[str, str]
) -> None:
    """Run the maintained advanced chart suite against this campaign's tables."""
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1] / "quickAndDirty" / "checkgraph.py"
    environment = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "CUDAHEAT_TABLE_FOLDER": str(tables.resolve()),
        "CUDAHEAT_PLOT_OUTPUT": str(output.resolve()),
        "CUDAHEAT_PLOT_TITLE": title,
        "CUDAHEAT_PLOT_TITLES": json.dumps(figure_titles),
        "CUDAHEAT_FIGURE_DPI": "300",
    }
    subprocess.run([sys.executable, str(script)], check=True, env=environment)


def _load_figure_titles(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(name, str) and isinstance(title, str) for name, title in payload.items()
    ):
        raise ValueError("titles JSON must be an object mapping figure basenames to strings")
    return payload


def _write_presentation_report(root: Path, metadata: Mapping[str, Any], title: str) -> None:
    report = f"""# {title}

This is a timestamped, non-overwriting CUDAHeat benchmark campaign.

- Dataset: `{metadata['dataset_path']}`
- Scale: {metadata['campaign_npoints']:,} points × {metadata['ntime']:,} time samples
- Raw measurements: `raw/` (CSV and JSON)
- Derived tables: `tables/` (CSV and JSON)
- Figures: `figures/presentation/` (300 dpi PNG plus transparent PNG variants)

The measurement matrix is the standard CUDAHeat full-pipeline benchmark. The
rendering layer uses the advanced chart collection from `quickAndDirty/checkgraph.py`.
"""
    (root / "README.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
