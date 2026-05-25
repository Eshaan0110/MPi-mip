"""RBI Bank-wise ATM/POS/Card Statistics — parse all sources, melt, save.

Three source types that together span Apr 2012 to May 2025:

  SOURCE A — Year folders (data/raw/rbi_bankwise/YYYY/)
    Format: .xls files, one file covers several months (each month = one sheet).
    Sheets named e.g. 'April 2012', 'May 2012', 'Aug.2012', 'Oct-12' etc.
    'Var ...' sheets are variance sheets — skipped.
    Column layout (0-indexed):
      col 1  = bank name
      col 9  = CC outstanding (individual cards)
      col 10 = CC ATM transactions
      col 11 = CC POS transactions
      col 12 = CC ATM value (Rs Million)
      col 13 = CC POS value (Rs Million)
      col 14 = DC outstanding
      col 15 = DC ATM transactions
      col 16 = DC POS transactions
      col 17 = DC ATM value (Rs Million)
      col 18 = DC POS value (Rs Million)
    Data rows start after category/header rows. Skip rows where col 1 is a
    section header (Scheduled Commercial Banks, Public Sector Banks, Total,
    Grand Total, etc.) — identified by absence of a numeric value in col 9 or 14.

  SOURCE B — RBI_Data_Debit_Credit_1.xlsx (sheets '1' through '41')
    Format: .xlsx, one sheet per month (Jan 2022 – May 2025).
    Column layout (0-indexed):
      col 2  = bank name
      col 9  = CC outstanding
      col 10 = CC ATM transactions
      col 11 = CC POS transactions
      col 14 = DC outstanding
      col 15 = DC ATM transactions
      col 16 = DC POS transactions
    Date extracted from sheet title row (row 0, col 2):
      e.g. 'ATM, Acceptance Infrastructure and Card Statistics - January 2022'

  SOURCE C — Summary sheets in RBI_Data_Debit_Credit_1.xlsx
    'Summary CC (2)' and 'Summary DC (2)' — wide pivot tables already built.
    Used only to fill any months not covered by Source B sheets.

Pipeline:
  1. Parse all year-folder xls files (Source A).
  2. Parse numbered sheets from the main xlsx (Source B).
  3. Parse summary sheets (Source C) as fallback.
  4. Combine all three; deduplicate on (bank_name_raw, date, card_type).
     Priority: B > A > C (individual monthly sheets > summaries).
  5. Standardise bank names; flag low-coverage banks.
  6. Save bankwise_cards_cc.parquet/.csv and bankwise_cards_dc.parquet/.csv.
"""

from __future__ import annotations

import datetime
import hashlib
import re
from pathlib import Path

import openpyxl
import pandas as pd
from loguru import logger

try:
    import xlrd
    _HAS_XLRD = True
except ImportError:
    _HAS_XLRD = False
    logger.warning("xlrd not installed — year-folder .xls files will be skipped. Run: pip install xlrd")

from src.config import Settings, load_settings
from src.ingestion.validation import SchemaValidationError, check_data_quality

# ── Source B / C column indices (xlsx, 0-indexed) ─────────────────────────
_XLSX_BANK_COL   = 2
_XLSX_CC_OUT     = 9
_XLSX_CC_ATM_VOL = 10
_XLSX_CC_POS_VOL = 11
_XLSX_DC_OUT     = 14
_XLSX_DC_ATM_VOL = 15
_XLSX_DC_POS_VOL = 16

# ── Source A column indices (xls, 0-indexed) ──────────────────────────────
_XLS_BANK_COL   = 1
_XLS_CC_OUT     = 9
_XLS_CC_ATM_VOL = 10
_XLS_CC_POS_VOL = 11
_XLS_DC_OUT     = 14
_XLS_DC_ATM_VOL = 15
_XLS_DC_POS_VOL = 16

# Rows with these keywords in the bank name are section headers / totals — skip
_SKIP_KEYWORDS = {
    "total", "grand total", "scheduled commercial", "public sector",
    "private sector", "foreign bank", "nationalised", "state bank group",
    "other public", "old private", "new private", "small finance",
    "payments bank", "regional rural", "co-operative", "bank name",
    "sr. no", "sr no",
}

_MIN_MONTHS_FOR_MODELLING = 48  # Rahul: top 20 only; <48 months = out of scope for individual modelling

# Month name → number for sheet-name parsing
_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_float(v) -> float | None:
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s in ("", "-", "NA", "N/A", "nan", "0.0"):
        try:
            f = float(s.replace(",",""))
            return f if f != 0.0 else None
        except Exception:
            return None
    try:
        f = float(s)
        return f if f != 0.0 else None
    except (ValueError, TypeError):
        return None


def _safe_float_zero(v) -> float | None:
    """Like _safe_float but keeps zeros (for outstanding card counts)."""
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s in ("", "-", "NA", "N/A", "nan"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _is_skip_row(bank_name: str) -> bool:
    low = bank_name.lower().strip()
    if not low:
        return True
    return any(kw in low for kw in _SKIP_KEYWORDS)


def _parse_sheet_name_date(sheet_name: str) -> pd.Timestamp | None:
    """Parse a month/year from inconsistent sheet names like:
    'April 2012', 'Aug.2012', 'Sept12', 'Oct-12', 'Jan 2023'
    """
    s = sheet_name.strip().lower()
    s = re.sub(r"[.\-_]", " ", s)  # normalise separators
    s = re.sub(r"\s+", " ", s)

    # Try: word year (e.g. 'april 2012', 'aug 2012')
    m = re.match(r"([a-z]+)\s+(\d{4})", s)
    if m:
        mon = _MONTH_MAP.get(m.group(1)[:4])
        if mon:
            return pd.Timestamp(year=int(m.group(2)), month=mon, day=1)

    # Try: word 2-digit year (e.g. 'sept12', 'oct 12')
    m = re.match(r"([a-z]+)\s*(\d{2})$", s)
    if m:
        mon = _MONTH_MAP.get(m.group(1)[:4])
        yr2 = int(m.group(2))
        if mon:
            year = 2000 + yr2 if yr2 < 50 else 1900 + yr2
            return pd.Timestamp(year=year, month=mon, day=1)

    return None


def _parse_title_date(title: str) -> pd.Timestamp | None:
    """Parse date from xlsx sheet title row:
    'ATM, Acceptance Infrastructure and Card Statistics - January 2022'
    """
    m = re.search(r"([A-Za-z]+)\s+(\d{4})", title)
    if m:
        mon = _MONTH_MAP.get(m.group(1).lower())
        if mon:
            return pd.Timestamp(year=int(m.group(2)), month=mon, day=1)
    return None


# ── Format detection and XLS parsing ──────────────────────────────────────
#
# RBI changed the bankwise file layout several times. Four known variants:
#
# FMT_A (2011–2013): date in sheet name (e.g. 'May 2012', 'Aug.2012')
#   bank=col1, CC_out=col9, CC_ATM=col10, CC_POS=col11
#   DC_out=col14, DC_ATM=col15, DC_POS=col16
#
# FMT_B (2014): date in row0 col0 ("ATM & Card Statistics for March, 2014")
#   sheets named Sheet1/Sheet2/Sheet3 (one per month per file sometimes)
#   bank=col1, CC_out=col6, CC_ATM=col7, CC_POS=col8
#   DC_out=col11, DC_ATM=col12, DC_POS=col13
#
# FMT_C (2015–2020 xls): date in row1 col1 ("ATM & Card Statistics for Oct, 2016")
#   sheet usually named 'Sheet1' or 'ATM'
#   bank=col2 (col1 = Sr.No.), CC_out=col7, CC_ATM=col8, CC_POS=col9
#   DC_out=col12, DC_ATM=col13, DC_POS=col14
#
# FMT_D (2019–2021 xlsx): date in row1 col1, sheet named "Month - YYYY"
#   bank=col2, CC_out=col7, CC_ATM=col8, CC_POS=col9
#   DC_out=col12, DC_ATM=col13, DC_POS=col14

_FORMATS = {
    "A": {"bank": 1, "cc_out": 9,  "cc_atm": 10, "cc_pos": 11,
           "dc_out": 14, "dc_atm": 15, "dc_pos": 16, "data_start": 7},
    "B": {"bank": 1, "cc_out": 6,  "cc_atm": 7,  "cc_pos": 8,
           "dc_out": 11, "dc_atm": 12, "dc_pos": 13, "data_start": 5},
    "C": {"bank": 2, "cc_out": 7,  "cc_atm": 8,  "cc_pos": 9,
           "dc_out": 12, "dc_atm": 13, "dc_pos": 14, "data_start": 6},
    "D": {"bank": 2, "cc_out": 7,  "cc_atm": 8,  "cc_pos": 9,
           "dc_out": 12, "dc_atm": 13, "dc_pos": 14, "data_start": 6},
}


def _extract_date_from_title(title: str) -> pd.Timestamp | None:
    """Extract date from title strings like:
    'ATM & Card Statistics for March, 2014'
    'ATM & Card Statistics for October, 2016'
    'ATM, Acceptance Infrastructure and Card Statistics - January 2022'
    'ATM & Card Statistics for December - 2019'
    """
    # Normalise separators
    s = title.strip()
    # Try month-name + 4-digit year
    m = re.search(r"([A-Za-z]+)[,\s\-]+(\d{4})", s)
    if m:
        mon = _MONTH_MAP.get(m.group(1).lower())
        if mon:
            return pd.Timestamp(year=int(m.group(2)), month=mon, day=1)
    return None


def _detect_xls_format(ws) -> tuple[str, pd.Timestamp | None]:
    """Detect format variant and extract date from an xls worksheet."""
    nrows = ws.nrows

    def cell(r, c):
        if r < nrows and c < ws.ncols:
            v = ws.cell_value(r, c)
            return str(v).strip() if v else ""
        return ""

    # Check row 0 col 0 for FMT_B title
    r0c0 = cell(0, 0)
    if "card statistics" in r0c0.lower() or "atm" in r0c0.lower():
        date = _extract_date_from_title(r0c0)
        if date:
            return "B", date

    # Check row 1 col 0 or col 1 for FMT_C/D title
    for col in [0, 1]:
        r1 = cell(1, col)
        if "card statistics" in r1.lower() or "atm" in r1.lower():
            date = _extract_date_from_title(r1)
            if date:
                return "C", date

    # FMT_A: date comes from sheet name, format already known
    return "A", None


def _detect_xlsx_format(rows: list) -> tuple[str, pd.Timestamp | None]:
    """Detect format and date from an xlsx worksheet rows list."""
    def cell(r, c):
        if r < len(rows) and c < len(rows[r]) and rows[r][c]:
            return str(rows[r][c]).strip()
        return ""

    for col in [1, 2]:
        r1 = cell(1, col)
        if "card statistics" in r1.lower() or "atm" in r1.lower():
            date = _extract_date_from_title(r1)
            if date:
                return "D", date

    return "D", None


def _parse_rows_with_format(
    get_val,       # callable(row_idx, col_idx) -> raw value
    nrows: int,
    fmt: dict,
    date: pd.Timestamp,
    source: str,
) -> list[dict]:
    """Generic row parser given a format spec and value accessor."""
    records = []
    for i in range(fmt["data_start"], nrows):
        bank_raw = str(get_val(i, fmt["bank"]) or "").strip()
        if not bank_raw or bank_raw == "None" or _is_skip_row(bank_raw):
            continue

        cc_out = _safe_float_zero(get_val(i, fmt["cc_out"]))
        dc_out = _safe_float_zero(get_val(i, fmt["dc_out"]))
        if cc_out is None and dc_out is None:
            continue

        records.append({
            "date":           date,
            "bank_name_raw":  bank_raw,
            "bank_category":  "",
            "cc_outstanding": cc_out,
            "cc_atm_vol":     _safe_float(get_val(i, fmt["cc_atm"])),
            "cc_pos_vol":     _safe_float(get_val(i, fmt["cc_pos"])),
            "dc_outstanding": dc_out,
            "dc_atm_vol":     _safe_float(get_val(i, fmt["dc_atm"])),
            "dc_pos_vol":     _safe_float(get_val(i, fmt["dc_pos"])),
            "source":         source,
        })
    return records


def _parse_xls_sheet(ws, sheet_name: str) -> list[dict]:
    """Parse one xls sheet, auto-detecting format."""
    fmt_key, date = _detect_xls_format(ws)

    # FMT_A: date comes from sheet name
    if fmt_key == "A" or date is None:
        date = _parse_sheet_name_date(sheet_name)
        fmt_key = "A"
        if date is None:
            return []

    fmt = _FORMATS[fmt_key]

    def get_val(r, c):
        if r < ws.nrows and c < ws.ncols:
            return ws.cell_value(r, c)
        return None

    return _parse_rows_with_format(get_val, ws.nrows, fmt, date, "xls_monthly")


def _parse_xlsx_sheet_generic(rows: list) -> list[dict]:
    """Parse one xlsx sheet (year-folder or numbered), auto-detecting format."""
    fmt_key, date = _detect_xlsx_format(rows)

    # Also try title spanning multiple cells
    if date is None:
        for col in range(5):
            v = rows[1][col] if len(rows) > 1 and col < len(rows[1]) else None
            if v:
                date = _extract_date_from_title(str(v))
                if date:
                    break

    if date is None:
        return []

    fmt = _FORMATS[fmt_key]

    def get_val(r, c):
        if r < len(rows) and c < len(rows[r]):
            return rows[r][c]
        return None

    return _parse_rows_with_format(get_val, len(rows), fmt, date, "xlsx_monthly")



def parse_year_folder(year_dir: Path) -> list[dict]:
    """Parse all .xls/.xlsx files in one year folder."""
    if not _HAS_XLRD:
        return []

    records = []
    all_files = (
        sorted(year_dir.glob("*.xls")) +
        sorted(year_dir.glob("*.XLS")) +
        sorted(year_dir.glob("*.xlsx")) +
        sorted(year_dir.glob("*.XLSX"))
    )

    for filepath in all_files:
        file_records = []
        try:
            if filepath.suffix.lower() == ".xls":
                wb = xlrd.open_workbook(str(filepath))
                for sheet_name in wb.sheet_names():
                    if sheet_name.lower().strip().startswith("var"):
                        continue
                    ws = wb.sheet_by_name(sheet_name)
                    file_records.extend(_parse_xls_sheet(ws, sheet_name))
            else:
                wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
                for sheet_name in wb.sheetnames:
                    if sheet_name.lower().strip().startswith("var"):
                        continue
                    ws = wb[sheet_name]
                    rows = list(ws.iter_rows(values_only=True))
                    file_records.extend(_parse_xlsx_sheet_generic(rows))
        except Exception as e:
            logger.warning(f"  Could not parse {filepath.name}: {e}")
            continue

        records.extend(file_records)
        if file_records:
            logger.debug(f"  {filepath.name}: +{len(file_records)} records")
        else:
            logger.debug(f"  {filepath.name}: 0 records parsed")

    return records


# ── Source B: numbered sheets in main xlsx ────────────────────────────────

def _parse_xlsx_numbered_sheet(ws, rows: list) -> list[dict] | None:
    """Parse one numbered monthly sheet from the main xlsx file."""
    if not rows:
        return None

    # Date from title row (row 1, col 2) — read enough columns to get full string
    title = ""
    for col_idx in [2, 3, 4]:  # title may span merged cells
        v = rows[1][col_idx] if len(rows) > 1 and col_idx < len(rows[1]) and rows[1][col_idx] else ""
        title += str(v)
        if re.search(r"[A-Za-z]+\s+\d{4}", title):
            break
    date = _parse_title_date(title)
    if date is None:
        return None

    records = []
    last_category = ""

    for row in rows[6:]:  # data starts after 6 header rows
        if all(v is None for v in row):
            continue

        # Category tracking (col 2 has section headers and bank names)
        bank_cell = row[_XLSX_BANK_COL] if _XLSX_BANK_COL < len(row) else None
        if bank_cell is None:
            continue

        bank_raw = str(bank_cell).strip()
        if not bank_raw or bank_raw == "None":
            continue

        # Detect category headers (no numeric data in CC/DC cols)
        cc_out = _safe_float_zero(row[_XLSX_CC_OUT] if _XLSX_CC_OUT < len(row) else None)
        dc_out = _safe_float_zero(row[_XLSX_DC_OUT] if _XLSX_DC_OUT < len(row) else None)

        if _is_skip_row(bank_raw):
            if not any(kw in bank_raw.lower() for kw in ("total", "grand")):
                last_category = bank_raw  # it's a category header
            continue

        if cc_out is None and dc_out is None:
            last_category = bank_raw
            continue

        records.append({
            "date":           date,
            "bank_name_raw":  bank_raw,
            "bank_category":  last_category,
            "cc_outstanding": cc_out,
            "cc_atm_vol":     _safe_float(row[_XLSX_CC_ATM_VOL] if _XLSX_CC_ATM_VOL < len(row) else None),
            "cc_pos_vol":     _safe_float(row[_XLSX_CC_POS_VOL] if _XLSX_CC_POS_VOL < len(row) else None),
            "dc_outstanding": dc_out,
            "dc_atm_vol":     _safe_float(row[_XLSX_DC_ATM_VOL] if _XLSX_DC_ATM_VOL < len(row) else None),
            "dc_pos_vol":     _safe_float(row[_XLSX_DC_POS_VOL] if _XLSX_DC_POS_VOL < len(row) else None),
            "source":         "xlsx_monthly",
        })

    return records


# ── Source C: summary pivot sheets ────────────────────────────────────────

def _parse_summary_sheet(ws, card_type: str) -> list[dict]:
    """Parse a wide summary sheet into long records."""
    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 4:
        return []

    # Row 2 = date headers (datetime objects), starting col 2
    header_row = rows[2]
    date_cols: list[tuple[int, pd.Timestamp]] = []
    for j in range(2, len(header_row)):
        cell = header_row[j]
        if cell is None or isinstance(cell, (int, str)):
            continue
        try:
            ts = pd.Timestamp(cell).replace(day=1)
            date_cols.append((j, ts))
        except Exception:
            continue

    if not date_cols:
        return []

    records = []
    last_category = ""

    for row in rows[3:]:
        if all(v is None for v in row):
            continue

        cat_cell = row[0] if len(row) > 0 else None
        if cat_cell and str(cat_cell).strip():
            last_category = str(cat_cell).strip()

        bank_cell = row[1] if len(row) > 1 else None
        if not bank_cell or not str(bank_cell).strip():
            continue

        bank_raw = str(bank_cell).strip()
        if _is_skip_row(bank_raw):
            continue

        for col_idx, date in date_cols:
            val = _safe_float_zero(row[col_idx] if col_idx < len(row) else None)
            rec = {
                "date":          date,
                "bank_name_raw": bank_raw,
                "bank_category": last_category,
                "source":        "xlsx_summary",
            }
            if card_type == "cc":
                rec.update({"cc_outstanding": val, "cc_atm_vol": None,
                             "cc_pos_vol": None, "dc_outstanding": None,
                             "dc_atm_vol": None, "dc_pos_vol": None})
            else:
                rec.update({"dc_outstanding": val, "dc_atm_vol": None,
                             "dc_pos_vol": None, "cc_outstanding": None,
                             "cc_atm_vol": None, "cc_pos_vol": None})
            records.append(rec)

    return records


# ── Name standardisation ──────────────────────────────────────────────────

def _standardise_bank_names(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Map raw bank names to canonical names from settings.toml issuers list.

    Special cases handled:
    - 'SBI' raw → 'State Bank Of India' canonical
    - State Bank associates (merged into SBI 2017) → kept separate historically
    """
    # Build lookup: raw_substring (upper) → canonical name
    # Earlier entries in the list take priority over later ones
    canonical_map: list[tuple[str, str]] = []
    for _sector, names in settings.issuers.items():
        for name in names:
            canonical_map.append((name.upper(), name))

    # Explicit overrides for known raw-name variants
    _OVERRIDES = {
        "SBI": "State Bank Of India",
    }

    def _find_canonical(raw: str) -> str:
        raw_stripped = raw.strip()
        # Check explicit overrides first (exact match)
        if raw_stripped.upper() in _OVERRIDES:
            return _OVERRIDES[raw_stripped.upper()]
        # Substring match against canonical list
        raw_up = raw_stripped.upper()
        for key_up, canonical in canonical_map:
            if key_up in raw_up:
                return canonical
        return raw_stripped.title()

    df["bank_name"] = df["bank_name_raw"].apply(_find_canonical)
    return df


def _flag_low_coverage(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    coverage = (
        df[df[target_col].notna()]
        .groupby("bank_name").size()
        .rename("non_null_months")
    )
    df = df.merge(coverage.reset_index(), on="bank_name", how="left")
    df["low_coverage"] = df["non_null_months"] < _MIN_MONTHS_FOR_MODELLING
    return df.drop(columns=["non_null_months"])


def _file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main entry point ───────────────────────────────────────────────────────

def run_bankwise_ingestion(settings: Settings | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse all bankwise sources, combine, validate, save.

    Returns (df_cc, df_dc) as long-format DataFrames with columns:
        date, bank_name, bank_name_raw, bank_category,
        cc_outstanding / dc_outstanding,
        cc_atm_vol, cc_pos_vol / dc_atm_vol, dc_pos_vol,
        low_coverage, source
    """
    if settings is None:
        settings = load_settings()

    raw_dir = settings.paths.rbi_bankwise_dir
    processed_dir = settings.paths.processed_dir

    all_records: list[dict] = []

    # ── SOURCE A: year folders ─────────────────────────────────────────────
    year_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if year_dirs:
        logger.info(f"Source A: {len(year_dirs)} year folders ({year_dirs[0].name}–{year_dirs[-1].name})")
        for year_dir in year_dirs:
            recs = parse_year_folder(year_dir)
            all_records.extend(recs)
            logger.info(f"  {year_dir.name}: {len(recs)} records")
    else:
        logger.warning("No year folders found in rbi_bankwise/ — Source A skipped.")

    # ── SOURCE B: numbered sheets in main xlsx ─────────────────────────────
    xlsx_candidates = sorted(raw_dir.glob(settings.rbi_bankwise.file_pattern))
    if xlsx_candidates:
        filepath = xlsx_candidates[-1]
        logger.info(f"Source B: {filepath.name} (numbered monthly sheets)")
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)

        numbered_sheets = [s for s in wb.sheetnames if s.isdigit()]
        logger.info(f"  Found {len(numbered_sheets)} numbered sheets")

        b_records = 0
        for sheet_name in numbered_sheets:
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))
            recs = _parse_xlsx_sheet_generic(rows)
            if recs:
                all_records.extend(recs)
                b_records += len(recs)

        logger.info(f"  Source B: {b_records} records parsed")

        # ── SOURCE C: summary sheets as fallback ───────────────────────────
        logger.info("Source C: summary pivot sheets (fallback)")
        for sheet_name, card_type in [("Summary CC (2)", "cc"), ("Summary DC (2)", "dc")]:
            if sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                recs = _parse_summary_sheet(ws, card_type)
                all_records.extend(recs)
                logger.info(f"  {sheet_name}: {len(recs)} records")
    else:
        logger.warning(f"No xlsx file matching '{settings.rbi_bankwise.file_pattern}' found — Sources B & C skipped.")

    if not all_records:
        raise SchemaValidationError("No bankwise records parsed from any source.")

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])

    logger.info(f"Total raw records before dedup: {len(df):,}")

    # ── Deduplicate: xlsx_monthly > xls_monthly > xlsx_summary ─────────────
    source_priority = {"xlsx_monthly": 0, "xls_monthly": 1, "xlsx_summary": 2}
    df["_priority"] = df["source"].map(source_priority)
    df = (
        df.sort_values(["bank_name_raw", "date", "_priority"])
        .drop_duplicates(subset=["bank_name_raw", "date"], keep="first")
        .drop(columns=["_priority"])
        .reset_index(drop=True)
    )

    logger.info(f"After dedup: {len(df):,} records")

    # ── Standardise names ──────────────────────────────────────────────────
    df = _standardise_bank_names(df, settings)

    # ── Split into CC and DC ───────────────────────────────────────────────
    cc_cols = ["date", "bank_name", "bank_name_raw", "bank_category",
               "cc_outstanding", "cc_atm_vol", "cc_pos_vol", "source"]
    dc_cols = ["date", "bank_name", "bank_name_raw", "bank_category",
               "dc_outstanding", "dc_atm_vol", "dc_pos_vol", "source"]

    df_cc = df[df["cc_outstanding"].notna()][cc_cols].copy()
    df_dc = df[df["dc_outstanding"].notna()][dc_cols].copy()

    df_cc = _flag_low_coverage(df_cc, "cc_outstanding")
    df_dc = _flag_low_coverage(df_dc, "dc_outstanding")

    df_cc = df_cc.sort_values(["bank_name", "date"]).reset_index(drop=True)
    df_dc = df_dc.sort_values(["bank_name", "date"]).reset_index(drop=True)

    # ── Quality checks ─────────────────────────────────────────────────────
    for label, df_sub, target in [("CC", df_cc, "cc_outstanding"), ("DC", df_dc, "dc_outstanding")]:
        model_ready = df_sub[~df_sub["low_coverage"]]["bank_name"].nunique()
        total_banks = df_sub["bank_name"].nunique()
        logger.info(
            f"{label}: {total_banks} banks | {model_ready} model-ready | "
            f"{df_sub['date'].min():%b %Y} → {df_sub['date'].max():%b %Y} | "
            f"sources: {df_sub['source'].value_counts().to_dict()}"
        )

    # ── Save ───────────────────────────────────────────────────────────────
    for label, df_sub, stem in [("CC", df_cc, "bankwise_cards_cc"), ("DC", df_dc, "bankwise_cards_dc")]:
        csv_path = processed_dir / f"{stem}.csv"
        parquet_path = processed_dir / f"{stem}.parquet"
        df_sub.to_csv(csv_path, index=False)
        df_sub.to_parquet(parquet_path, index=False)
        logger.info(f"Saved {stem}: {len(df_sub):,} rows")

    return df_cc, df_dc


if __name__ == "__main__":
    df_cc, df_dc = run_bankwise_ingestion()
    print(f"\nCC: {len(df_cc):,} rows | {df_cc['bank_name'].nunique()} banks | "
          f"{df_cc['date'].min():%b %Y} → {df_cc['date'].max():%b %Y}")
    print(f"DC: {len(df_dc):,} rows | {df_dc['bank_name'].nunique()} banks | "
          f"{df_dc['date'].min():%b %Y} → {df_dc['date'].max():%b %Y}")
    print(f"\nSource breakdown CC:\n{df_cc['source'].value_counts()}")
    print(f"\nTop 5 CC banks (latest month):")
    latest = df_cc[df_cc.date == df_cc.date.max()].nlargest(5, "cc_outstanding")
    print(latest[["bank_name", "cc_outstanding"]].to_string(index=False))