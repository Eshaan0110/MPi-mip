"""RBI Payment System Indicators — parse both formats, stitch, validate, store.

RBI publishes PSI in two layouts that together span Apr 2004 to present:
  - Old format (sheet "Old Format"): Apr 2004 - Oct 2019
  - New format (sheet "New Format"): Nov 2019 - present

This module finds all PSI files in data/raw/, auto-detects each one's format
by sheet name, resolves columns via header patterns (config-driven, not fixed
positions), parses each into a common schema, and stitches them into one
continuous monthly series.

Pipeline:
  1. Find all files matching the PSI glob.
  2. Compute a combined SHA256; log new-data vs re-run.
  3. For each file: detect format, resolve columns, parse rows.
  4. Concatenate, sort by date, drop any duplicate months (prefer new format).
  5. Run quality checks; save Parquet + CSV; record the hash.
"""

from pathlib import Path

import openpyxl
import pandas as pd
from loguru import logger

from src.config import PsiFormatConfig, Settings, load_settings
from src.ingestion.validation import (
    SchemaValidationError,
    check_data_quality,
    combined_hash,
    detect_freshness,
    record_hash,
    resolve_psi_columns,
)


def _safe_float(value) -> float | None:
    """Convert a cell to float, or None for missing / invalid input.

    Handles comma-formatted numbers ('21,703.44') and placeholders ('-', 'NA').
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


def _detect_format(
    sheet_name: str,
    formats: dict[str, PsiFormatConfig],
) -> tuple[str | None, PsiFormatConfig | None]:
    """Return the (name, config) of the format whose sheet_match is in the sheet name."""
    sn = sheet_name.lower()
    for fmt_name, fmt in formats.items():
        if fmt.sheet_match.lower() in sn:
            return fmt_name, fmt
    return None, None


def parse_psi_file(filepath: Path, settings: Settings) -> pd.DataFrame:
    """Parse one PSI file, auto-detecting old vs new format by sheet name."""
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active

    fmt_name, fmt = _detect_format(ws.title, settings.rbi_psi.formats)
    if fmt is None:
        expected = [f.sheet_match for f in settings.rbi_psi.formats.values()]
        raise SchemaValidationError(
            f"{filepath.name}: sheet '{ws.title}' matches no known format "
            f"(expected sheet name containing one of {expected})."
        )

    logger.info(f"{filepath.name}: detected '{fmt_name}' format (sheet '{ws.title}')")

    rows = list(ws.iter_rows(values_only=True))

    expected_cols = {name: spec.model_dump() for name, spec in fmt.columns.items()}
    column_map = resolve_psi_columns(rows, fmt.label_row, fmt.unit_row, expected_cols)

    date_idx = fmt.date_col
    records: list[dict] = []
    skipped = 0

    for row in rows[fmt.data_start_row:]:
        if len(row) <= date_idx:
            skipped += 1
            continue
        date_val = row[date_idx]
        if date_val is None or not isinstance(date_val, str):
            skipped += 1
            continue
        try:
            parsed_date = pd.to_datetime(date_val, format=settings.rbi_psi.date_format)
        except (ValueError, TypeError):
            skipped += 1
            continue

        record: dict = {"date": parsed_date, "source_format": fmt_name}
        for col_name, col_idx in column_map.items():
            cell = row[col_idx] if col_idx < len(row) else None
            record[col_name] = _safe_float(cell)
        records.append(record)

    df = pd.DataFrame(records)
    if skipped:
        logger.debug(f"  skipped {skipped} non-data rows (titles, footnotes)")
    logger.info(
        f"  parsed {len(df)} rows | "
        f"{df['date'].min():%b %Y} -> {df['date'].max():%b %Y}"
    )
    return df


def run_rbi_ingestion(settings: Settings | None = None) -> pd.DataFrame:
    """Entry point: find PSI files, parse both formats, stitch, validate, save."""
    if settings is None:
        settings = load_settings()

    raw_dir = settings.paths.rbi_psi_dir
    processed_dir = settings.paths.processed_dir

    files = sorted(raw_dir.glob(settings.rbi_psi.file_pattern))
    if not files:
        raise FileNotFoundError(
            f"No PSI files matching '{settings.rbi_psi.file_pattern}' in {raw_dir}\n"
            f"Download from RBI DBIE → Statistics → Financial Sector → Payment Systems\n"
            f"and save to data/raw/rbi_psi/."
        )

    logger.info(f"Found {len(files)} PSI file(s): {[f.name for f in files]}")

    # Freshness across all input files
    hash_record = processed_dir / ".rbi_psi.sha256"
    current_hash = combined_hash(files)
    if detect_freshness(current_hash, hash_record):
        logger.info(f"NEW data detected (combined hash {current_hash[:12]}...)")
    else:
        logger.info("REPROCESSING existing files (combined hash unchanged)")

    # Parse each file
    frames = [parse_psi_file(fp, settings) for fp in files]

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    # Log the stitch boundary if more than one format is present
    if combined["source_format"].nunique() > 1:
        new_start = combined.loc[combined["source_format"] == "new", "date"].min()
        logger.info(f"Stitched old + new formats at {new_start:%b %Y}")

    check_data_quality(
        combined,
        max_null_pct=settings.validation.max_null_pct,
        max_date_gap_days=settings.validation.max_date_gap_days,
        min_rows=settings.validation.min_rows,
    )

    csv_path = processed_dir / "rbi_psi_cards.csv"
    parquet_path = processed_dir / "rbi_psi_cards.parquet"
    combined.to_csv(csv_path, index=False)
    combined.to_parquet(parquet_path, index=False)
    logger.info(
        f"Saved {len(combined)} months "
        f"({combined['date'].min():%b %Y} -> {combined['date'].max():%b %Y}) "
        f"to {csv_path.name} and {parquet_path.name}"
    )

    record_hash(hash_record, current_hash)
    return combined


if __name__ == "__main__":
    df = run_rbi_ingestion()
    print("\nFirst 3 months:")
    print(df.head(3).to_string(index=False))
    print("\nLast 3 months:")
    print(df.tail(3).to_string(index=False))
    print("\nRows per source format:")
    print(df["source_format"].value_counts().to_string())
    print("\nNull counts:")
    print(df.isnull().sum().to_string())
