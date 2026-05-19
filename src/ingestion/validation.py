"""Schema and data-quality validation for ingested files.

Three responsibilities:
  1. Resolve column names to indices via header-text matching.
  2. Run quality checks (min rows, null %, date gaps).
  3. Compute SHA256 hashes to distinguish new files from re-runs.
"""

import hashlib
from pathlib import Path

import pandas as pd
from loguru import logger


class SchemaValidationError(Exception):
    """Raised when an ingested file fails schema or quality validation."""


def find_column_index(
    header_rows: list[tuple],
    patterns: list[str],
) -> int | None:
    """Return the column index whose combined header text contains ALL patterns.

    RBI uses multi-row merged headers, so we concatenate the cell values from
    each row in `header_rows` for a given column, then test patterns
    (case-insensitive substring) against that combined string.

    Args:
        header_rows: list of row tuples — each tuple holds one row's cells
        patterns: substrings that must all appear in the combined header text

    Returns:
        Column index, or None if no column matches.
    """
    if not header_rows:
        return None

    n_cols = max(len(r) for r in header_rows)
    patterns_lower = [p.lower() for p in patterns]

    for col_idx in range(n_cols):
        parts: list[str] = []
        for row in header_rows:
            if col_idx < len(row) and row[col_idx] is not None:
                parts.append(str(row[col_idx]).lower())
        combined = " ".join(parts)

        if all(p in combined for p in patterns_lower):
            return col_idx

    return None


def resolve_columns(
    header_rows: list[tuple],
    expected: dict[str, list[str]],
) -> dict[str, int]:
    """Resolve every expected column to its index, raising loudly on any miss.

    Args:
        header_rows: header rows from the source file
        expected:    mapping of column name -> list of substring patterns

    Returns:
        Mapping of column name -> resolved column index.

    Raises:
        SchemaValidationError: if any expected column cannot be located.
    """
    resolved: dict[str, int] = {}
    missing: list[tuple[str, list[str]]] = []

    for col_name, patterns in expected.items():
        idx = find_column_index(header_rows, patterns)
        if idx is None:
            missing.append((col_name, patterns))
        else:
            resolved[col_name] = idx
            logger.debug(f"Resolved '{col_name}' -> column index {idx}")

    if missing:
        details = "\n  ".join(
            f"'{name}' (patterns: {pats})" for name, pats in missing
        )
        raise SchemaValidationError(
            f"Could not locate {len(missing)} expected column(s):\n  {details}\n"
            f"Inspect the file's actual headers and update "
            f"config/settings.toml [rbi_psi.columns]."
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

    Hard failures raise SchemaValidationError.
    Soft issues are logged as warnings.
    """
    if len(df) < min_rows:
        raise SchemaValidationError(
            f"Parsed only {len(df)} rows; expected at least {min_rows}. "
            f"Parsing likely failed or the source file is incomplete."
        )

    # Null percentage per column
    null_pct = (df.isnull().mean() * 100).to_dict()
    for col, pct in null_pct.items():
        if pct > max_null_pct:
            logger.warning(
                f"High null rate in '{col}': {pct:.1f}% "
                f"(threshold {max_null_pct}%)"
            )

    # Date gaps
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
            logger.warning(
                f"Date gaps exceed {max_date_gap_days} days: {big}"
            )


def file_sha256(filepath: Path) -> str:
    """Compute the SHA256 hash of a file."""
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_freshness(
    filepath: Path,
    hash_record_path: Path,
) -> tuple[bool, str]:
    """Compare the current file hash to the previously recorded one.

    Returns:
        (is_new, current_hash) — is_new is True if the hash differs from
        the stored value, or if no prior hash exists.
    """
    current = file_sha256(filepath)
    if not hash_record_path.exists():
        return True, current
    previous = hash_record_path.read_text().strip()
    return (current != previous), current


def record_hash(hash_record_path: Path, hash_value: str) -> None:
    """Persist a file hash for next-run freshness comparison."""
    hash_record_path.parent.mkdir(parents=True, exist_ok=True)
    hash_record_path.write_text(hash_value)
