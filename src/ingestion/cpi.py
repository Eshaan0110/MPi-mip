"""MoSPI CPI (Consumer Price Index) — parse, normalise, save.

Source file: CPI.xlsx (pre-filtered to General-Overall, All India Combined,
2012 base year). 186 rows spanning January 2011 to December 2025.

Columns in source:
  baseyear, year, month_code, month, state, sector, group, subgroup, index, inflation, status

Output (monthly series, sorted ascending):
  date, cpi_index, cpi_inflation_pct

Note on base year linking:
  MoSPI launched a new 2024=100 series in early 2026. This file contains
  the 2012=100 series only (Jan 2011 – Dec 2025). When adding post-2025
  months, the two series must be linked using an overlap period.
  See: MIP_DataCollection_Guide.docx §2 for the splice methodology.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd
from loguru import logger

from src.config import Settings, load_settings
from src.ingestion.validation import SchemaValidationError, check_data_quality

# Month-name → month-number for parsing
_MONTH_MAP: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_cpi_file(filepath: Path) -> pd.DataFrame:
    """Parse the CPI Excel file into a clean monthly DataFrame."""
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    if not rows:
        raise SchemaValidationError(f"{filepath.name}: file is empty.")

    # Validate header
    header = [str(c).strip().lower() if c is not None else "" for c in rows[0]]
    required = {"year", "month_code", "month", "index", "inflation"}
    missing = required - set(header)
    if missing:
        raise SchemaValidationError(
            f"{filepath.name}: missing expected columns {missing}. "
            f"Found: {header}"
        )

    year_idx       = header.index("year")
    month_name_idx = header.index("month")
    index_idx      = header.index("index")
    inflation_idx  = header.index("inflation")

    records: list[dict] = []
    skipped = 0

    for row in rows[1:]:
        if all(v is None for v in row):
            continue

        try:
            year = int(row[year_idx])
        except (ValueError, TypeError):
            skipped += 1
            continue

        month_raw = str(row[month_name_idx]).strip().lower() if row[month_name_idx] else ""
        month_num = _MONTH_MAP.get(month_raw)
        if month_num is None:
            skipped += 1
            continue

        date = pd.Timestamp(year=year, month=month_num, day=1)

        def _to_float(v) -> float | None:
            if v is None:
                return None
            try:
                return float(str(v).strip())
            except (ValueError, TypeError):
                return None

        records.append({
            "date": date,
            "cpi_index": _to_float(row[index_idx]),
            "cpi_inflation_pct": _to_float(row[inflation_idx]),
        })

    if skipped:
        logger.debug(f"  Skipped {skipped} non-data rows")

    df = (
        pd.DataFrame(records)
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    logger.info(
        f"  Parsed {len(df)} months | "
        f"{df['date'].min():%b %Y} → {df['date'].max():%b %Y}"
    )
    return df


def run_cpi_ingestion(settings: Settings | None = None) -> pd.DataFrame:
    """Entry point: parse CPI file, validate, save."""
    if settings is None:
        settings = load_settings()

    raw_dir = settings.paths.mospi_cpi_dir
    processed_dir = settings.paths.processed_dir

    candidates = sorted(raw_dir.glob(settings.mospi_cpi.file_pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No CPI file matching '{settings.mospi_cpi.file_pattern}' in {raw_dir}.\n"
            "Copy CPI.xlsx into data/raw/mospi_cpi/ and retry.\n"
            "Source: https://cpi.mospi.gov.in → Time Series Data"
        )
    filepath = candidates[-1]
    logger.info(f"Parsing CPI file: {filepath.name}")

    df = _parse_cpi_file(filepath)

    check_data_quality(
        df,
        date_col="date",
        max_null_pct=10.0,
        max_date_gap_days=35,
        min_rows=100,
    )

    # Warn if inflation column is mostly empty (older data has no inflation field)
    null_inf = df["cpi_inflation_pct"].isna().mean()
    if null_inf > 0.3:
        logger.warning(
            f"cpi_inflation_pct is {null_inf:.0%} null — "
            "the model should use cpi_index directly and compute inflation internally."
        )

    csv_path = processed_dir / "cpi.csv"
    parquet_path = processed_dir / "cpi.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved {csv_path.name} and {parquet_path.name}")

    return df


if __name__ == "__main__":
    df = run_cpi_ingestion()
    print(f"\nShape: {df.shape}")
    print(df.tail(6).to_string(index=False))
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
