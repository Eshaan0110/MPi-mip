"""NPCI UPI P2P/P2M Ecosystem Statistics — parse monthly files, combine, save.

NPCI publishes one file per month in data/raw/P2P_P2M_UPI/. Each file has
a single data row with Total, P2P, and P2M transaction volumes and values.

Date formats in the cell vary wildly across years (datetime objects, "Jan'24",
"Aug-25", full month names, encoding corruption). The filename is always
authoritative: Ecosystem-Statistics-UPI-P2p-and-p2m-transactions-YYYY-Mon.xlsx

Output (monthly series, sorted ascending):
  date, upi_p2p_p2m_total_vol_mn, upi_p2p_p2m_total_val_cr,
  upi_p2p_vol_mn, upi_p2p_val_cr, upi_p2m_vol_mn, upi_p2m_val_cr
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import openpyxl
import pandas as pd
from loguru import logger

from src.config import Settings, load_settings
from src.ingestion.validation import check_data_quality

_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "june": 6, "july": 7,
}

_FILE_RE = re.compile(
    r"(\d{4})-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
    re.IGNORECASE,
)


def _date_from_filename(filepath: Path) -> pd.Timestamp | None:
    """Extract month-start date from filename. Returns None if no match."""
    m = _FILE_RE.search(filepath.stem)
    if not m:
        return None
    year = int(m.group(1))
    month = _MONTH_ABBR[m.group(2).lower()]
    return pd.Timestamp(year=year, month=month, day=1)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if s in ("", "-", "NA", "N/A"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_p2p_file(filepath: Path) -> dict | None:
    """Parse one monthly P2P/P2M file. Returns a record dict or None on failure."""
    date = _date_from_filename(filepath)
    if date is None:
        logger.warning(f"  Cannot extract date from filename: {filepath.name} — skipping.")
        return None

    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    except Exception as e:
        logger.warning(f"  Failed to open {filepath.name}: {e} — skipping.")
        return None

    # Data is always in row index 3 (row 4 in Excel):
    #   row 0: title, row 1: column headers, row 2: sub-headers, row 3: data
    if len(rows) < 4:
        logger.warning(f"  {filepath.name}: fewer than 4 rows — skipping.")
        return None

    data = rows[3]
    if len(data) < 7:
        logger.warning(f"  {filepath.name}: data row has only {len(data)} columns — skipping.")
        return None

    return {
        "date":                      date,
        "upi_p2p_p2m_total_vol_mn":  _safe_float(data[1]),
        "upi_p2p_p2m_total_val_cr":  _safe_float(data[2]),
        "upi_p2p_vol_mn":            _safe_float(data[3]),
        "upi_p2p_val_cr":            _safe_float(data[4]),
        "upi_p2m_vol_mn":            _safe_float(data[5]),
        "upi_p2m_val_cr":            _safe_float(data[6]),
    }


def run_p2p_upi_ingestion(settings: Settings | None = None) -> pd.DataFrame:
    """Entry point: parse all monthly P2P/P2M files, validate, save."""
    if settings is None:
        settings = load_settings()

    p2p_dir = settings.paths.raw_dir / "P2P_P2M_UPI"
    processed_dir = settings.paths.processed_dir

    if not p2p_dir.exists():
        raise FileNotFoundError(
            f"P2P_P2M_UPI directory not found at {p2p_dir}.\n"
            "Create data/raw/P2P_P2M_UPI/ and place the monthly NPCI P2P/P2M files there.\n"
            "Source: https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics"
        )

    files = sorted(p2p_dir.glob("Ecosystem-Statistics-UPI-P2p-and-p2m-transactions-*.xlsx"))
    if not files:
        raise FileNotFoundError(
            f"No P2P/P2M files matching the expected pattern in {p2p_dir}."
        )

    logger.info(f"Found {len(files)} P2P/P2M monthly files")

    records: list[dict] = []
    for fp in files:
        rec = _parse_p2p_file(fp)
        if rec is not None:
            records.append(rec)

    if not records:
        raise ValueError("No records parsed from any P2P/P2M file.")

    combined = (
        pd.DataFrame(records)
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    logger.info(
        f"Combined {len(combined)} months | "
        f"{combined['date'].min():%b %Y} → {combined['date'].max():%b %Y}"
    )

    check_data_quality(
        combined,
        date_col="date",
        max_null_pct=settings.validation.max_null_pct,
        max_date_gap_days=settings.validation.max_date_gap_days,
        min_rows=24,
    )

    csv_path = processed_dir / "upi_p2p_p2m.csv"
    parquet_path = processed_dir / "upi_p2p_p2m.parquet"
    combined.to_csv(csv_path, index=False)
    combined.to_parquet(parquet_path, index=False)
    logger.info(f"Saved {csv_path.name} and {parquet_path.name}")

    return combined


if __name__ == "__main__":
    df = run_p2p_upi_ingestion()
    print(f"\nShape: {df.shape}")
    print(df.tail(6).to_string(index=False))
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
