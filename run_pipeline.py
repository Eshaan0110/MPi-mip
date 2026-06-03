"""
MIP Phase 1 -- Full End-to-End Pipeline Runner
================================================
Executes ingestion, modelling, analysis, and dashboard rebuild in order.

Usage:
    uv run python run_pipeline.py                 # full run with CV (~20 min)
    uv run python run_pipeline.py --no-cv         # skip CV (~2 min)
    uv run python run_pipeline.py --skip-ingestion # models only (data already fresh)
    uv run python run_pipeline.py --skip-ingestion --no-cv  # fastest possible
"""

import argparse
import subprocess
import sys
import time


STEPS = [
    {
        "name": "Ingestion",
        "cmd": [sys.executable, "-m", "src.ingestion"],
        "skip_flag": "skip_ingestion",
    },
    {
        "name": "Aggregate Model (CC + DC Outstanding)",
        "cmd": [sys.executable, "-m", "src.modelling.aggregate_model"],
        "cv_flag": True,
    },
    {
        "name": "Bank-Level Model (20 CC + 20 DC)",
        "cmd": [sys.executable, "-m", "src.modelling.bank_model"],
        "cv_flag": True,
    },
    {
        "name": "Transaction Volume Models (CC / DC / UPI)",
        "cmd": [sys.executable, "-m", "src.modelling.txn_volume_model"],
        "cv_flag": True,
    },
    {
        "name": "UPI Displacement Analysis",
        "cmd": [sys.executable, "-m", "src.modelling.upi_analysis"],
    },
    {
        "name": "Rebuild Dashboard",
        "cmd": [sys.executable, "scripts/rebuild_dashboard.py"],
    },
]


def main():
    parser = argparse.ArgumentParser(description="MIP full pipeline runner")
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip ingestion step (data already fresh)",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation in all modelling steps (~2 min vs ~20 min)",
    )
    args = parser.parse_args()

    total_steps = sum(
        1 for s in STEPS
        if not (s.get("skip_flag") and getattr(args, s["skip_flag"], False))
    )

    t0 = time.time()
    step_num = 0

    for step in STEPS:
        # Check skip flag
        skip_flag = step.get("skip_flag")
        if skip_flag and getattr(args, skip_flag, False):
            continue

        step_num += 1
        cmd = list(step["cmd"])

        # Append --no-cv if applicable
        if args.no_cv and step.get("cv_flag"):
            cmd.append("--no-cv")

        header = f"STEP {step_num}/{total_steps} -- {step['name']}"
        print()
        print("=" * 60)
        print(header)
        print("=" * 60)
        print(f"  Command: {' '.join(cmd)}")
        print()

        step_t0 = time.time()
        result = subprocess.run(cmd, cwd=str(__file__).rsplit("\\", 1)[0] or ".")

        step_elapsed = time.time() - step_t0

        if result.returncode != 0:
            print()
            print("!" * 60)
            print(f"FAILED: {step['name']} (exit code {result.returncode})")
            print(f"  Time spent: {step_elapsed:.1f}s")
            print("!" * 60)
            print()
            print("Pipeline stopped. Fix the error above and re-run.")
            sys.exit(1)

        print(f"  Done in {step_elapsed:.1f}s")

    total = time.time() - t0
    minutes = int(total // 60)
    seconds = total % 60

    print()
    print("=" * 60)
    print(f"PIPELINE COMPLETE -- {minutes}m {seconds:.0f}s total")
    print("=" * 60)
    print()
    print("Open dashboard.html in your browser to view results.")


if __name__ == "__main__":
    main()
