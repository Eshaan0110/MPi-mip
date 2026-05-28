"""
data_loader.py
--------------
Loads and prepares all datasets for the NeuralProphet experiment.

Data source: RBI bankwise ATM/Card statistics sheets.
Each numbered sheet (1-41) and X1-X4 represents one month.
Row ~70 contains the 'Total' row with aggregate CC outstanding (col 9) and DC outstanding (col 14).
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(os.environ.get("MIP_DATA_DIR", "/mnt/project"))
RBI_FILE  = DATA_DIR / "RBI_Data_Debit_Credit_1.xlsx"
CPI_FILE  = DATA_DIR / "CPI.xlsx"
REPO_FILE = DATA_DIR / "RepoRate2007.XLSX"

# Column indices for the individual monthly sheets
CC_OUTSTANDING_COL = 9   # "No. of outstanding cards as at end of month" — credit
DC_OUTSTANDING_COL = 14  # Same — debit


def _extract_date_from_sheet(raw: pd.DataFrame) -> pd.Timestamp | None:
    """Scan first 3 rows for a month-year pattern in the sheet title."""
    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    for ri in range(3):
        for ci in range(min(raw.shape[1], 6)):
            v = str(raw.iloc[ri, ci])
            m = re.search(rf"({months})[\s\-](\d{{4}})", v, re.IGNORECASE)
            if m:
                return pd.to_datetime(f"01 {m.group(1)} {m.group(2)}", format="%d %B %Y")
    return None


def _extract_totals_from_sheet(raw: pd.DataFrame) -> tuple[float, float]:
    """Find the 'Total' row and return (cc_outstanding, dc_outstanding)."""
    for ri in range(len(raw)):
        cell = str(raw.iloc[ri, 2]).strip().lower()
        if cell == "total":
            cc = pd.to_numeric(raw.iloc[ri, CC_OUTSTANDING_COL], errors="coerce")
            dc = pd.to_numeric(raw.iloc[ri, DC_OUTSTANDING_COL], errors="coerce")
            return float(cc) if not np.isnan(cc) else np.nan, float(dc) if not np.isnan(dc) else np.nan
    return np.nan, np.nan


def load_cards_outstanding() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads all monthly sheets from the RBI bankwise file.
    Returns (cc_df, dc_df) each with columns [ds, y] ready for NeuralProphet.
    """
    xl = pd.ExcelFile(RBI_FILE)
    target_sheets = [s for s in xl.sheet_names if s.isdigit() or s.startswith("X")]

    records = []
    for sh in target_sheets:
        try:
            raw = pd.read_excel(RBI_FILE, sheet_name=sh, header=None)
            date = _extract_date_from_sheet(raw)
            if date is None:
                continue
            cc, dc = _extract_totals_from_sheet(raw)
            if not np.isnan(cc) and not np.isnan(dc):
                records.append({"ds": date, "cc": cc, "dc": dc, "sheet": sh})
        except Exception as e:
            print(f"  Warning: sheet {sh} skipped — {e}")

    df = pd.DataFrame(records).sort_values("ds").drop_duplicates("ds").reset_index(drop=True)
    df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp()

    cc_df = df[["ds", "cc"]].rename(columns={"cc": "y"})
    dc_df = df[["ds", "dc"]].rename(columns={"dc": "y"})
    return cc_df, dc_df


def load_repo_rate(date_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fills RBI repo rate event dates into a monthly series."""
    raw = pd.read_excel(REPO_FILE, skiprows=4, usecols=[1, 3], header=None)
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
    """Loads MoSPI CPI (All India Combined, base 2012) aligned to date_index."""
    raw = pd.read_excel(CPI_FILE)
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
    """
    Appends structural event dummy columns to df (must have 'ds' column).
    card_type: 'cc' or 'dc'
    """
    df = df.copy()
    ds = pd.to_datetime(df["ds"])

    # Shared: COVID pulse (April + May 2020 only)
    df["covid_shock"] = ((ds.dt.year == 2020) & (ds.dt.month.isin([4, 5]))).astype(float)

    if card_type == "cc":
        # Step-down: RBI tightening Nov 2023
        df["rbi_tightening_2023"] = (ds >= "2023-11-01").astype(float)

    if card_type == "dc":
        # Step-up: PMJDY launch Aug 2014
        df["pmjdy_launch"] = (ds >= "2014-08-01").astype(float)
        # Step-up: Demonetisation Nov 2016
        df["demonetisation"] = (ds >= "2016-11-01").astype(float)
        # Step-change: UPI inflection Jan 2022
        df["upi_inflection"] = (ds >= "2022-01-01").astype(float)

    return df


if __name__ == "__main__":
    print("Loading cards outstanding from bankwise sheets...")
    cc_df, dc_df = load_cards_outstanding()
    print(f"\nCC outstanding: {len(cc_df)} months | {cc_df['ds'].min().date()} → {cc_df['ds'].max().date()}")
    print(f"  Range: {cc_df['y'].min()/1e6:.1f}M → {cc_df['y'].max()/1e6:.1f}M cards")
    print(f"\nDC outstanding: {len(dc_df)} months | {dc_df['ds'].min().date()} → {dc_df['ds'].max().date()}")
    print(f"  Range: {dc_df['y'].min()/1e6:.1f}M → {dc_df['y'].max()/1e6:.1f}M cards")

    print("\nTesting macro data...")
    idx = cc_df["ds"]
    repo = load_repo_rate(pd.DatetimeIndex(idx))
    cpi  = load_cpi(pd.DatetimeIndex(idx))
    print(f"  Repo rate: {repo.min()}% → {repo.max()}%  (latest: {repo.iloc[-1]}%)")
    print(f"  CPI index: {cpi.min():.1f} → {cpi.max():.1f}  (latest: {cpi.iloc[-1]:.1f})")
