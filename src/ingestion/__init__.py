"""Ingestion subpackage — data loaders for MIP Phase 1.

Exposed entry points:
  run_rbi_ingestion()      — RBI PSI cards outstanding + transaction volumes
  run_npci_ingestion()     — NPCI UPI product statistics
  run_bankwise_ingestion() — RBI bank-wise card statistics (ground-up model inputs)
  run_cpi_ingestion()      — MoSPI CPI inflation series
  run_repo_rate_ingestion()— RBI repo rate monthly series
  run_p2p_upi_ingestion()  — NPCI UPI P2P/P2M ecosystem statistics
"""

from src.ingestion.rbi import run_rbi_ingestion
from src.ingestion.npci import run_npci_ingestion
from src.ingestion.bankwise import run_bankwise_ingestion
from src.ingestion.cpi import run_cpi_ingestion
from src.ingestion.repo_rate import run_repo_rate_ingestion
from src.ingestion.p2p_upi import run_p2p_upi_ingestion

__all__ = [
    "run_rbi_ingestion",
    "run_npci_ingestion",
    "run_bankwise_ingestion",
    "run_cpi_ingestion",
    "run_repo_rate_ingestion",
    "run_p2p_upi_ingestion",
]
