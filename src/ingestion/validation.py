"""Schema and data-quality validation for ingested files.

Responsibilities:
  1. Resolve PSI columns by (section + label + unit) patterns, handling
     RBI's merged headers and the duplicate "Credit Cards" label that
     appears under both the transactions and the cards-outstanding sections.
  2. Run quality checks (min rows, null %, date gaps).
  3. Compute SHA256 hashes to distinguish new files from re-runs.
"""

import hashlib
import re
from pathlib import Path

import pandas as pd
from loguru import logger


class SchemaValidationError(Exception):
    """Raised when an ingested file fails schema or quality validation."""


# A top-level section header looks like "5 Cards" or "8 Cards Outstanding":
# an integer, whitespace, then text. Sub-items like "5.1 ..." have a dot in
# the number and are deliberately excluded.
_SECTION_RE = re.compile(r"^\s*\d+\s+\S")


def _forward_fill(row: tuple) -> list:
    """Replace blank cells with the nearest non-blank value to the left.

    Excel stores a merged cell's value only in its top-left cell, leaving the
    spanned cells empty. RBI merges the parent label across a volume/value
    pair, so the "value" column's own label cell is blank. Forward-filling
    lets that column inherit the volume column's label.
    """
    out: list = []
    last = None
    for v in row:
        if v is not None and str(v).strip():
            last = v
        out.append(last)
    return out


def _is_section_header(label) -> bool:
    if label is None:
        return False
    return bool(_SECTION_RE.match(str(label)))


def _section_per_column(label_row: tuple) -> list:
    """For each column, the nearest section header at or to its left."""
    sections: list = []
    current = None
    for v in label_row:
        if _is_section_header(v):
            current = v
        sections.append(current)
    return sections


def resolve_psi_columns(
    rows: list,
    label_row_idx: int,
    unit_row_idx: int,
    expected: dict[str, dict],
) -> dict[str, int]:
    """Resolve PSI target columns using section + column + unit patterns.

    Args:
        rows:          all rows from the sheet (tuples of cell values)
        label_row_idx: row holding the item labels
        unit_row_idx:  row holding the units (Volume / Value)
        expected:      target_name -> {section, section_not, column, unit}
                       where each value is a list of case-insensitive
                       substrings. `section`, `column`, `unit` must ALL match;
                       `section_not` must NOT appear in the section.

    Returns:
        target_name -> resolved column index.

    Raises:
        SchemaValidationError if any target cannot be located.
    """
    label_row = rows[label_row_idx]
    unit_row = rows[unit_row_idx]

    labels = _forward_fill(label_row)
    units = _forward_fill(unit_row)
    sections = _section_per_column(label_row)

    n = max(len(labels), len(units), len(sections))

    def cell(seq: list, i: int) -> str:
        return str(seq[i]).lower() if i < len(seq) and seq[i] is not None else ""

    resolved: dict[str, int] = {}
    missing: list[tuple[str, dict]] = []

    for name, spec in expected.items():
        sec_inc = [p.lower() for p in spec.get("section", [])]
        sec_exc = [p.lower() for p in spec.get("section_not", [])]
        col_inc = [p.lower() for p in spec.get("column", [])]
        uni_inc = [p.lower() for p in spec.get("unit", [])]

        found = None
        for j in range(n):
            sec = cell(sections, j)
            lab = cell(labels, j)
            uni = cell(units, j)
            if (
                all(p in sec for p in sec_inc)
                and not any(p in sec for p in sec_exc)
                and all(p in lab for p in col_inc)
                and all(p in uni for p in uni_inc)
            ):
                found = j
                break

        if found is None:
            missing.append((name, spec))
        else:
            resolved[name] = found
            logger.debug(
                f"Resolved '{name}' -> col {found} "
                f"[section='{cell(sections, found)[:28]}', "
                f"label='{cell(labels, found)[:28]}', "
                f"unit='{cell(units, found)[:18]}']"
            )

    if missing:
        details = "\n  ".join(f"'{n}' (spec: {s})" for n, s in missing)
        raise SchemaValidationError(
            f"Could not locate {len(missing)} PSI column(s):\n  {details}\n"
            f"Inspect the file's headers and update config/settings.toml."
        )

    return resolved


def check_data_quality(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    max_null_pct: float = 20.0,
    max_date_gap_days: int = 35,
    min_rows: int = 60,
) -> None:
    """Run quality checks on a parsed DataFrame.

    Hard failures raise SchemaValidationError. Soft issues log warnings.
    """
    if len(df) < min_rows:
        raise SchemaValidationError(
            f"Parsed only {len(df)} rows; expected at least {min_rows}. "
            f"Parsing likely failed or the source file is incomplete."
        )

    null_pct = (df.isnull().mean() * 100).to_dict()
    for col, pct in null_pct.items():
        if pct > max_null_pct:
            logger.warning(
                f"High null rate in '{col}': {pct:.1f}% "
                f"(threshold {max_null_pct}%)"
            )

    if date_col in df.columns and len(df) > 1:
        sorted_dates = df[date_col].sort_values().reset_index(drop=True)
        gaps = sorted_dates.diff().dt.days.dropna()
        if not gaps.empty and gaps.max() > max_date_gap_days:
            big = [
                (
                    sorted_dates[i].strftime("%Y-%m"),
                    sorted_dates[i + 1].strftime("%Y-%m"),
                    int(gaps.iloc[i]),
                )
                for i in range(len(gaps))
                if gaps.iloc[i] > max_date_gap_days
            ]
            logger.warning(f"Date gaps exceed {max_date_gap_days} days: {big}")


def file_sha256(filepath: Path) -> str:
    """Compute the SHA256 hash of a single file."""
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def combined_hash(filepaths: list[Path]) -> str:
    """Compute one SHA256 over several files (sorted, for determinism)."""
    h = hashlib.sha256()
    for fp in sorted(filepaths):
        with fp.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


def detect_freshness(
    current_hash: str,
    hash_record_path: Path,
) -> bool:
    """Return True if current_hash differs from the stored hash (or none stored)."""
    if not hash_record_path.exists():
        return True
    return current_hash != hash_record_path.read_text().strip()


def record_hash(hash_record_path: Path, hash_value: str) -> None:
    """Persist a hash for next-run freshness comparison."""
    hash_record_path.parent.mkdir(parents=True, exist_ok=True)
    hash_record_path.write_text(hash_value)
