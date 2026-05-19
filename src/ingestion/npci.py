"""NPCI UPI Product Statistics — parse yearly Excel files, combine, save.

NPCI publishes one file per year. We glob all matching files, parse each,
stack into a single sorted/deduplicated series, validate, and save.
"""

from pathlib import Path

import openpyxl
import pandas as pd
from loguru import logger

from src.config import Settings, load_settings
from src.ingestion.validation import check_data_quality


def _safe_float(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s in ("", "-", "NA", "N/A"):
        return None
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


def parse_npci_excel(filepath: Path) -> pd.DataFrame:
    """Parse a single NPCI yearly UPI Excel.

    Expected layout (consistent across NPCI yearly files):
      Row 0 = header
      Row 1+ = data
      Columns: Month | No. of Banks live on UPI | Volume (Mn) | Value (Cr)
    """
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    records: list[dict] = []
    for row in rows[1:]:
        if not row or not row[0]:
            continue
        try:
            parsed_date = pd.to_datetime(row[0], format="%B-%Y")
        except (ValueError, TypeError):
            continue

        records.append(
            {
                "date": parsed_date,
                "upi_banks_live": _safe_float(row[1] if len(row) > 1 else None),
                "upi_volume_mn":  _safe_float(row[2] if len(row) > 2 else None),
                "upi_value_cr":   _safe_float(row[3] if len(row) > 3 else None),
            }
        )

    return pd.DataFrame(records)


def run_npci_ingestion(settings: Settings | None = None) -> pd.DataFrame:
    """Entry point: combine all yearly NPCI files, validate, save."""
    if settings is None:
        settings = load_settings()

    cfg = settings.npci_upi
    raw_dir = settings.paths.raw_dir
    processed_dir = settings.paths.processed_dir

    files = sorted(raw_dir.glob(cfg.file_pattern))
    if not files:
        raise FileNotFoundError(
            f"No NPCI UPI files matching '{cfg.file_pattern}' in {raw_dir}\n"
            f"Download from: "
            f"https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics"
        )

    logger.info(f"Found {len(files)} NPCI yearly files")

    frames: list[pd.DataFrame] = []
    for f in files:
        logger.info(f"Parsing {f.name}")
        frames.append(parse_npci_excel(f))

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )

    logger.info(
        f"Combined {len(combined)} months | "
        f"{combined['date'].min().strftime('%b %Y')} -> "
        f"{combined['date'].max().strftime('%b %Y')}"
    )

    # UPI data starts ~2016, so we use a lower min_rows than RBI
    check_data_quality(
        combined,
        max_null_pct=settings.validation.max_null_pct,
        max_date_gap_days=settings.validation.max_date_gap_days,
        min_rows=24,
    )

    csv_path = processed_dir / "npci_upi.csv"
    parquet_path = processed_dir / "npci_upi.parquet"
    combined.to_csv(csv_path, index=False)
    combined.to_parquet(parquet_path, index=False)
    logger.info(f"Saved {csv_path.name} and {parquet_path.name}")

    return combined


if __name__ == "__main__":
    df = run_npci_ingestion()
    print("\nLast 5 months:")
    print(df.tail().to_string(index=False))
