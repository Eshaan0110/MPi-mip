"""Run both ingestion pipelines.

Usage:
    uv run python -m src.ingestion
"""

from src.ingestion.npci import run_npci_ingestion
from src.ingestion.rbi import run_rbi_ingestion


def main() -> None:
    print("=" * 60)
    print("RBI Payment System Indicators")
    print("=" * 60)
    run_rbi_ingestion()

    print()
    print("=" * 60)
    print("NPCI UPI Product Statistics")
    print("=" * 60)
    run_npci_ingestion()


if __name__ == "__main__":
    main()
