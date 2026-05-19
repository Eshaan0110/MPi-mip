"""Data ingestion: RBI Payment System Indicators and NPCI UPI Statistics."""

from src.ingestion.npci import run_npci_ingestion
from src.ingestion.rbi import run_rbi_ingestion

__all__ = ["run_rbi_ingestion", "run_npci_ingestion"]
