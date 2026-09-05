"""
MIP — Full End-to-End Pipeline Runner
======================================
Executes ingestion, modelling, analysis, and dashboard rebuild in order.
Supports multi-market via --market flag (default: IN).

Usage:
    uv run python run_pipeline.py                       # full India run with CV
    uv run python run_pipeline.py --no-cv               # skip CV (~2 min)
    uv run python run_pipeline.py --skip-ingestion      # models only
    uv run python run_pipeline.py --market IN --no-cv   # explicit market
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.markets import get_adapter, list_markets

PROJECT_ROOT = Path(__file__).resolve().parent


def _build_cmd(step: dict) -> list[str]:
    """Build the subprocess command from a pipeline step definition."""
    if step.get("script"):
        return [sys.executable, step["script"]]
    return [sys.executable, "-m", step["module"]]


def main():
    available = ", ".join(list_markets())
    parser = argparse.ArgumentParser(description="MIP pipeline runner")
    parser.add_argument("--market", default="IN", help=f"Market code ({available})")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip ingestion step")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    args = parser.parse_args()

    adapter = get_adapter(args.market)
    steps = adapter.get_pipeline_steps()
    meta = adapter.meta

    active_steps = [
        s for s in steps
        if not (s.get("skip_flag") == "skip_ingestion" and args.skip_ingestion)
    ]

    print(f"\nMarket: {meta.name} ({meta.code}) — {meta.currency}")
    print(f"Pipeline steps: {len(active_steps)}")

    t0 = time.time()

    for i, step in enumerate(active_steps, 1):
        cmd = _build_cmd(step)
        if args.no_cv and step.get("cv_flag"):
            cmd.append("--no-cv")

        header = f"STEP {i}/{len(active_steps)} -- {step['name']}"
        print(f"\n{'=' * 60}\n{header}\n{'=' * 60}")
        print(f"  Command: {' '.join(cmd)}\n")

        step_t0 = time.time()
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        elapsed = time.time() - step_t0

        if result.returncode != 0:
            print(f"\n{'!' * 60}")
            print(f"FAILED: {step['name']} (exit code {result.returncode})")
            print(f"  Time spent: {elapsed:.1f}s")
            print(f"{'!' * 60}\n\nPipeline stopped. Fix the error above and re-run.")
            sys.exit(1)

        print(f"  Done in {elapsed:.1f}s")

    total = time.time() - t0
    minutes, seconds = int(total // 60), total % 60

    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE [{meta.code}] — {minutes}m {seconds:.0f}s total")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
