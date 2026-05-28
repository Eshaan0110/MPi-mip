"""
data_loader.py
--------------
Loads and prepares all datasets for the NeuralProphet experiment.
Uses the same config/settings.toml and src/config.py as the main pipeline.

Data source: RBI bankwise ATM/Card statistics sheets (Source B).
Each numbered sheet (1-41) and X1-X4 represents one month.
Row ~70 contains the 'Total' row: CC outstanding (col 9), DC outstanding (col 14).
"""

import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path so we can import src.config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings

_settings = load_settings()

RBI_BANKWISE_DIR = _settings.paths.rbi_bankwise_dir
RBI_REPO_DIR     = _settings.paths.rbi_repo_dir
MOSPI_CPI_DIR    = _settings.paths.mospi_cpi_dir

CC_OUTSTANDING_COL = 9
DC_OUTSTANDING_COL = 14


def _find_file(directory: Path, pattern: str) -> Path:
    """Return the most recently modified file matching pattern in directory."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    return matches[-1]


def _extract_date_from_sheet(raw: pd.DataFrame) -> pd.Timestamp | None:
    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    for ri in range(3):
        for ci in range(min(raw.shape[1], 6)):
            v = str(raw.iloc[ri, ci])
            m = re.search(rf"({months})[\s\-](\d{{4}})", v, re.IGNORECASE)
            if m:
                return pd.to_datetime(f"01 {m.group(1)} {m.group(2)}", format="%d %B %Y")
    return None


def _extract_totals_from_sheet(raw: pd.DataFrame) -> tuple[float, float]:
    for ri in range(len(raw)):
        cell = str(raw.iloc[ri, 2]).strip().lower()
        if cell == "total":
            cc = pd.to_numeric(raw.iloc[ri, CC_OUTSTANDING_COL], errors="coerce")
            dc = pd.to_numeric(raw.iloc[ri, DC_OUTSTANDING_COL], errors="coerce")
            return (float(cc) if not np.isnan(cc) else np.nan,
                    float(dc) if not np.isnan(dc) else np.nan)
    return np.nan, np.nan


def load_cards_outstanding() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (cc_df, dc_df) — each with columns [ds, y] ready for NeuralProphet.
    """
    rbi_file = _find_file(RBI_BANKWISE_DIR, _settings.rbi_bankwise.file_pattern)

    xl = pd.ExcelFile(rbi_file)
    target_sheets = [s for s in xl.sheet_names if s.isdigit() or s.startswith("X")]

    records = []
    for sh in target_sheets:
        try:
            raw  = pd.read_excel(rbi_file, sheet_name=sh, header=None)
            date = _extract_date_from_sheet(raw)
            if date is None:
                continue
            cc, dc = _extract_totals_from_sheet(raw)
            if not np.isnan(cc) and not np.isnan(dc):
                records.append({"ds": date, "cc": cc, "dc": dc})
        except Exception as e:
            print(f"  Warning: sheet {sh} skipped — {e}")

    df = (pd.DataFrame(records)
            .sort_values("ds")
            .drop_duplicates("ds")
            .reset_index(drop=True))
    df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp()

    cc_df = df[["ds", "cc"]].rename(columns={"cc": "y"})
    dc_df = df[["ds", "dc"]].rename(columns={"dc": "y"})
    return cc_df, dc_df


def load_repo_rate(date_index: pd.DatetimeIndex) -> pd.Series:
    repo_file = _find_file(RBI_REPO_DIR, _settings.rbi_repo.file_pattern)
    raw = pd.read_excel(repo_file, skiprows=4, usecols=[1, 3], header=None)
    raw.columns = ["date", "repo_rate"]
    raw["date"]      = pd.to_datetime(raw["date"], errors="coerce")
    raw["repo_rate"] = pd.to_numeric(raw["repo_rate"], errors="coerce")
    raw = raw.dropna(subset=["date", "repo_rate"]).sort_values("date")

    full_idx = pd.date_range(raw["date"].min(), date_index.max(), freq="MS")
    repo = raw.set_index("date")["repo_rate"].reindex(full_idx, method="ffill")
    repo = repo.reindex(date_index, method="ffill").bfill()
    repo.name = "repo_rate"
    return repo


def load_cpi(date_index: pd.DatetimeIndex) -> pd.Series:
    cpi_file = _find_file(MOSPI_CPI_DIR, _settings.mospi_cpi.file_pattern)
    raw = pd.read_excel(cpi_file)
    raw = raw[raw["subgroup"] == "General-Overall"].copy()
    raw["ds"] = pd.to_datetime(
        raw["year"].astype(str) + "-" + raw["month_code"].astype(str).str.zfill(2) + "-01"
    )
    raw = raw.sort_values("ds").drop_duplicates("ds")
    cpi = raw.set_index("ds")["index"].astype(float)
    cpi = cpi.reindex(date_index, method="ffill").bfill()
    cpi.name = "cpi"
    return cpi


def add_structural_events(df: pd.DataFrame, card_type: str) -> pd.DataFrame:
    df = df.copy()
    ds = pd.to_datetime(df["ds"])

    df["covid_shock"] = ((ds.dt.year == 2020) & (ds.dt.month.isin([4, 5]))).astype(float)

    if card_type == "cc":
        df["rbi_tightening_2023"] = (ds >= "2023-11-01").astype(float)

    if card_type == "dc":
        df["pmjdy_launch"]   = (ds >= "2014-08-01").astype(float)
        df["demonetisation"] = (ds >= "2016-11-01").astype(float)
        df["upi_inflection"] = (ds >= "2022-01-01").astype(float)

    return df


if __name__ == "__main__":
    cc_df, dc_df = load_cards_outstanding()
    print(f"CC: {len(cc_df)} months | {cc_df['ds'].min().date()} → {cc_df['ds'].max().date()}")
    print(f"DC: {len(dc_df)} months | {dc_df['ds'].min().date()} → {dc_df['ds'].max().date()}")