"""Run all ingestion pipelines in dependency order.

Usage:
    python -m src.ingestion             # run all
    python -m src.ingestion --only rbi  # run one (rbi | npci | bankwise | cpi | repo_rate | p2p_upi)
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

def main():
    parser = argparse.ArgumentParser(description="MIP ingestion pipeline runner")
    parser.add_argument(
        "--only",
        choices=list(PIPELINES),
        help="Run a single pipeline instead of all",
    )
    args = parser.parse_args()

    settings = load_settings()
    targets = [args.only] if args.only else list(PIPELINES)

    failed = []
    for name in targets:
        logger.info(f"═══ Running {name} ingestion ═══")
        try:
            PIPELINES[name](settings)
            logger.success(f"✓ {name} complete")
        except FileNotFoundError as e:
            logger.warning(f"⚠ {name} skipped — source file not found: {e}")
        except Exception as e:
            logger.error(f"✗ {name} failed: {e}")
            failed.append(name)

    if failed:
        logger.error(f"Failed pipelines: {failed}")
        sys.exit(1)
    else:
        logger.success("All ingestion pipelines complete.")

if __name__ == "__main__":
    main()
