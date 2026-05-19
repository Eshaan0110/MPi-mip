"""RBI Payment System Indicators — parse, validate, store.

Replaces the screening's hardcoded column-index approach with header-name
matching driven entirely by config/settings.toml.

Pipeline:
  1. Locate the most recent matching Excel file in data/raw/.
  2. Compute its SHA256 hash; log whether this is a new file or a re-run.
  3. Resolve expected column names to indices via header-text patterns.
  4. Parse data rows, normalising numeric values.
  5. Run quality checks (min rows, null %, date gaps).
  6. Save Parquet + CSV outputs and record the hash for next run.
"""

from pathlib import Path

import openpyxl
import pandas as pd
from loguru import logger

from src.config import Settings, load_settings
from src.ingestion.validation import (
    check_data_quality,
    detect_freshness,
    record_hash,
    resolve_columns,
)


def _safe_float(value) -> float | None:
    """Convert a cell value to float, or None for missing / invalid input.

    Handles comma-formatted numbers ('21,703.44') and common placeholder
    strings ('-', 'NA', 'N/A').
    """
    if value is None:
        return None
    s = str(value).strip()
    if s in ("", "-", "NA", "N/A"):
        return None
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


def _find_excel_file(raw_dir: Path, pattern: str) -> Path:
    """Find the most recent RBI PSI Excel matching `pattern`.

    Sorts matches by modification time and returns the newest. Logs all
    matches so it's clear which file the pipeline chose.
    """
    matches = sorted(
        raw_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No RBI PSI file matching '{pattern}' in {raw_dir}\n"
            f"Download from: RBI DBIE -> Statistics -> Financial Sector -> "
            f"Payment Systems"
        )
    if len(matches) > 1:
        logger.info(
            f"Multiple files match '{pattern}': "
            f"{[p.name for p in matches]}. Using most recent: {matches[0].name}"
        )
    return matches[0]


def parse_psi_excel(filepath: Path, settings: Settings) -> pd.DataFrame:
    """Parse the RBI PSI Excel into a normalised monthly DataFrame.

    Uses header-name matching (config-driven) — no hardcoded column indices.
    """
    cfg = settings.rbi_psi
    logger.info(f"Parsing {filepath.name}")

    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    # Resolve column indices from header rows
    header_rows = [rows[i] for i in cfg.header_rows if i < len(rows)]
    column_map = resolve_columns(header_rows, cfg.columns)

    # Parse data rows
    date_col_idx = column_map["date"]
    records: list[dict] = []
    skipped = 0

    for row in rows[cfg.data_start_row:]:
        if len(row) <= date_col_idx:
            skipped += 1
            continue

        date_val = row[date_col_idx]
        if date_val is None or not isinstance(date_val, str):
            skipped += 1
            continue

        try:
            parsed_date = pd.to_datetime(date_val, format="%b-%Y")
        except (ValueError, TypeError):
            skipped += 1
            continue

        record: dict = {"date": parsed_date}
        for col_name, col_idx in column_map.items():
            if col_name == "date":
                continue
            cell = row[col_idx] if col_idx < len(row) else None
            record[col_name] = _safe_float(cell)
        records.append(record)

    if skipped:
        logger.debug(f"Skipped {skipped} non-data rows (titles, footnotes)")

    df = (
        pd.DataFrame(records)
        .sort_values("date")
        .reset_index(drop=True)
    )

    logger.info(
        f"Parsed {len(df)} rows | "
        f"{df['date'].min().strftime('%b %Y')} -> "
        f"{df['date'].max().strftime('%b %Y')}"
    )
    return df


def run_rbi_ingestion(settings: Settings | None = None) -> pd.DataFrame:
    """Entry point: locate file, check freshness, parse, validate, save."""
    if settings is None:
        settings = load_settings()

    cfg = settings.rbi_psi
    raw_dir = settings.paths.raw_dir
    processed_dir = settings.paths.processed_dir

    filepath = _find_excel_file(raw_dir, cfg.file_pattern)

    # Freshness check — distinguishes new files from re-runs
    hash_record = processed_dir / ".rbi_psi.sha256"
    is_new, current_hash = detect_freshness(filepath, hash_record)
    if is_new:
        logger.info(f"NEW file detected (hash {current_hash[:12]}...)")
    else:
        logger.info("REPROCESSING existing file (hash unchanged)")

    # Parse
    df = parse_psi_excel(filepath, settings)

    # Validate
    check_data_quality(
        df,
        max_null_pct=settings.validation.max_null_pct,
        max_date_gap_days=settings.validation.max_date_gap_days,
        min_rows=settings.validation.min_rows,
    )

    # Save
    csv_path = processed_dir / "rbi_psi_cards.csv"
    parquet_path = processed_dir / "rbi_psi_cards.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved {csv_path.name} and {parquet_path.name}")

    record_hash(hash_record, current_hash)

    return df


if __name__ == "__main__":
    df = run_rbi_ingestion()
    print("\nLast 5 months:")
    print(df.tail().to_string(index=False))
    print("\nNull counts:")
    print(df.isnull().sum())
