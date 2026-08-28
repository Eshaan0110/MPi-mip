"""
MIP — Data Versioning (P3.4)
=============================
Snapshot data/processed after each pipeline run.
Each snapshot is a timestamped copy of all parquet files,
stored in data/vintages/YYYY-MM-DD/.

Required for:
  - Vintage scoring (P3.1): score forecasts against first-release data
  - P4 scorecard: reproducible runs

Usage:
    from src.modelling.data_versioning import snapshot_processed, list_vintages, load_vintage
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED = _PROJECT_ROOT / "data" / "processed"
_VINTAGES = _PROJECT_ROOT / "data" / "vintages"


def snapshot_processed(label: str | None = None) -> Path:
    """Copy all parquet/csv files from data/processed/ into a dated vintage folder.

    Args:
        label: optional label (defaults to today's date YYYY-MM-DD)

    Returns:
        Path to the vintage snapshot directory
    """
    label = label or date.today().isoformat()
    dest = _VINTAGES / label
    dest.mkdir(parents=True, exist_ok=True)

    count = 0
    for f in sorted(_PROCESSED.iterdir()):
        if f.is_file() and f.suffix in (".parquet", ".csv", ".json"):
            shutil.copy2(f, dest / f.name)
            count += 1
        elif f.is_dir():
            sub_dest = dest / f.name
            if sub_dest.exists():
                shutil.rmtree(sub_dest)
            shutil.copytree(f, sub_dest)
            count += 1

    logger.info(f"Snapshot '{label}': {count} items → {dest}")
    return dest


def list_vintages() -> list[str]:
    """Return sorted list of vintage labels (newest first)."""
    if not _VINTAGES.exists():
        return []
    return sorted(
        [d.name for d in _VINTAGES.iterdir() if d.is_dir()],
        reverse=True,
    )


def load_vintage(label: str, stem: str) -> "pd.DataFrame":
    """Load a specific parquet file from a vintage snapshot."""
    import pandas as pd

    path = _VINTAGES / label / f"{stem}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No vintage '{label}' or file '{stem}.parquet' in it")
    return pd.read_parquet(path)


def get_latest_vintage() -> str | None:
    """Return the most recent vintage label, or None."""
    vintages = list_vintages()
    return vintages[0] if vintages else None
