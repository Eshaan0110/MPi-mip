"""Run all ingestion pipelines in dependency order.

Usage:
    python -m src.ingestion                  # run all
    python -m src.ingestion --only rbi       # run one pipeline

Available pipelines:
    rbi        — RBI PSI (cards outstanding + txn volumes)   → data/raw/rbi_psi/
    npci       — NPCI UPI product stats                      → data/raw/npci_upi/
    bankwise   — RBI bank-wise card outstanding               → data/raw/rbi_bankwise/
    cpi        — MoSPI CPI inflation                          → data/raw/mospi_cpi/
    repo_rate  — RBI repo rate (Table 43)                    → data/raw/rbi_repo_rate/
    p2p_upi    — NPCI UPI P2M ecosystem stats                → data/raw/npci_p2m/
"""

import argparse
import sys
from loguru import logger
from src.config import load_settings
from src.ingestion import (
    run_rbi_ingestion,
    run_npci_ingestion,
    run_bankwise_ingestion,
    run_cpi_ingestion,
    run_repo_rate_ingestion,
    run_p2p_upi_ingestion,
)

PIPELINES = {
    "rbi":       run_rbi_ingestion,
    "npci":      run_npci_ingestion,
    "bankwise":  run_bankwise_ingestion,
    "cpi":       run_cpi_ingestion,
    "repo_rate": run_repo_rate_ingestion,
    "p2p_upi":   run_p2p_upi_ingestion,
}

SOURCE_DIRS = {
    "rbi":       "data/raw/rbi_psi/",
    "npci":      "data/raw/npci_upi/",
    "bankwise":  "data/raw/rbi_bankwise/",
    "cpi":       "data/raw/mospi_cpi/",
    "repo_rate": "data/raw/rbi_repo_rate/",
    "p2p_upi":   "data/raw/npci_p2m/",
}


def main():
    parser = argparse.ArgumentParser(
        description="MIP ingestion pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {name:<12} ← {src}" for name, src in SOURCE_DIRS.items()
        ),
    )
    parser.add_argument(
        "--only",
        choices=list(PIPELINES),
        help="Run a single pipeline instead of all",
    )
    args = parser.parse_args()

    settings = load_settings()
    targets = [args.only] if args.only else list(PIPELINES)

    failed = []
    skipped = []
    for name in targets:
        logger.info(f"═══ Running {name} ingestion ═══")
        try:
            PIPELINES[name](settings)
            logger.success(f"✓ {name} complete")
        except FileNotFoundError as e:
            logger.warning(f"⚠ {name} skipped — source file not found in {SOURCE_DIRS[name]}\n  {e}")
            skipped.append(name)
        except Exception as e:
            logger.error(f"✗ {name} failed: {e}")
            failed.append(name)

    if skipped:
        logger.warning(f"Skipped (missing source files): {skipped}")
    if failed:
        logger.error(f"Failed pipelines: {failed}")
        sys.exit(1)
    else:
        logger.success("All available ingestion pipelines complete.")


if __name__ == "__main__":
    main()
