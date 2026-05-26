"""
MIP Modelling — Bank-Level Data Preparation
=============================================
Loads bankwise parquet files, cleans and deduplicates bank names,
selects the top N issuers, and prepares Prophet-ready DataFrames
for each individual bank and the residual bucket.

Responsibilities:
  1. Load bankwise CC and DC parquets.
  2. Apply canonical name aliases (resolves HDFC/Hdfc, Citi/Citibank etc.)
  3. Merge duplicate bank series (after aliasing, sum outstanding per date).
  4. Filter out numeric artifact rows (ingestion residue from summary sheets).
  5. Select top N issuers by average outstanding (model-ready banks only).
  6. Build residual bucket: all remaining banks aggregated.
  7. Return per-bank Prophet DataFrames and the residual DataFrame.
  8. Log coverage: what % of PSI total do the modelled banks represent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from src.modelling.bank_config import (
    TOP_N_ISSUERS,
    MIN_MONTHS,
    BANK_NAME_ALIASES,
    TERMINATED_BANKS,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED    = _PROJECT_ROOT / "data" / "processed"


# ── Loaders ────────────────────────────────────────────────────────────────

def _load_bankwise(card_type: str) -> pd.DataFrame:
    """Load bankwise parquet for 'cc' or 'dc'."""
    stem = f"bankwise_cards_{card_type}"
    path = _PROCESSED / f"{stem}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.ingestion --only bankwise` first."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    return df


# ── Cleaning ───────────────────────────────────────────────────────────────

def _is_numeric_name(name: str) -> bool:
    """Returns True for names like '116001', '25634.0' — ingestion artifacts."""
    return bool(pd.Series([name]).str.match(r"^\d+\.?\d*$").iloc[0])


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Replace known duplicate raw names with their canonical form."""
    df = df.copy()
    df["bank_name"] = df["bank_name"].replace(BANK_NAME_ALIASES)
    return df


def _clean_bankwise(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Full cleaning pipeline: alias → filter artifacts → dedup → sort."""
    df = _apply_aliases(df)

    # Drop numeric artifact rows
    mask_numeric = df["bank_name"].apply(_is_numeric_name)
    n_dropped = mask_numeric.sum()
    if n_dropped > 0:
        logger.info(f"  Dropped {n_dropped} numeric artifact rows")
    df = df[~mask_numeric].copy()

    # After aliasing, some (date, bank_name) pairs may be duplicates
    # (e.g. HDFC Bank + Hdfc Bank Ltd. both mapped to HDFC Bank for same month).
    # Sum outstanding — we want the union of all sub-entities.
    df = (
        df.groupby(["date", "bank_name"], as_index=False)
        .agg({
            target_col:       "sum",
            "bank_category":  "first",
            "source":         "first",
        })
    )

    df = df.sort_values(["bank_name", "date"]).reset_index(drop=True)
    return df


# ── Bank selection ─────────────────────────────────────────────────────────

def _select_top_banks(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = TOP_N_ISSUERS,
    min_months: int = MIN_MONTHS,
) -> tuple[list[str], list[str]]:
    """Return (top_banks, residual_banks) lists.

    Top banks: model-ready (>=min_months non-null) ranked by average outstanding.
    Residual banks: everything else.
    """
    coverage = (
        df[df[target_col].notna()]
        .groupby("bank_name")
        .agg(
            months=  (target_col, "count"),
            avg_out= (target_col, "mean"),
        )
        .sort_values("avg_out", ascending=False)
    )

    model_ready = coverage[coverage["months"] >= min_months]
    top_banks   = model_ready.head(top_n).index.tolist()

    # All banks not in top N go to residual
    all_banks      = df["bank_name"].unique().tolist()
    residual_banks = [b for b in all_banks if b not in top_banks]

    logger.info(
        f"  Model-ready banks (>={min_months}m): {len(model_ready)} | "
        f"Top {top_n} selected | Residual bucket: {len(residual_banks)} banks"
    )
    return top_banks, residual_banks


# ── Coverage reporting ─────────────────────────────────────────────────────

def _log_coverage(
    df: pd.DataFrame,
    top_banks: list[str],
    target_col: str,
    psi_latest: float | None = None,
) -> float:
    """Log what % of total outstanding the top banks represent.

    Returns coverage fraction (0–1).
    """
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]

    top_sum   = latest[latest["bank_name"].isin(top_banks)][target_col].sum()
    total_sum = latest[target_col].sum()

    coverage_pct = top_sum / total_sum * 100 if total_sum > 0 else 0

    logger.info(
        f"  Coverage ({latest_date:%b %Y}): top {len(top_banks)} banks = "
        f"{top_sum:,.0f} of {total_sum:,.0f} ({coverage_pct:.1f}% of bankwise total)"
    )

    if psi_latest:
        psi_coverage = (top_sum / psi_latest) * 100
        logger.info(f"  vs PSI total: {psi_coverage:.1f}% of PSI aggregate")

    return coverage_pct / 100


# ── Prophet DataFrame builder ──────────────────────────────────────────────

def build_bank_prophet_df(
    df: pd.DataFrame,
    bank_name: str,
    target_col: str,
) -> pd.DataFrame | None:
    """Build a Prophet-ready (ds, y) DataFrame for one bank.

    Returns None if the bank has insufficient data after filtering.
    """
    bank_df = df[df["bank_name"] == bank_name][[
        "date", target_col
    ]].rename(columns={"date": "ds", target_col: "y"}).copy()

    # Drop rows where y is 0 or null (zero often means no data for that month)
    bank_df = bank_df[bank_df["y"].notna() & (bank_df["y"] > 0)].copy()
    bank_df = bank_df.sort_values("ds").reset_index(drop=True)

    if len(bank_df) < MIN_MONTHS:
        logger.warning(
            f"  {bank_name}: only {len(bank_df)} valid rows — skipping individual model"
        )
        return None

    # Flag terminated banks
    if bank_name in TERMINATED_BANKS:
        logger.info(
            f"  {bank_name}: terminated bank ({TERMINATED_BANKS[bank_name]}). "
            f"Model fits to available history only; forecast extrapolates trend."
        )

    logger.debug(
        f"  {bank_name}: {len(bank_df)} months | "
        f"{bank_df['ds'].min():%b %Y} → {bank_df['ds'].max():%b %Y}"
    )
    return bank_df


def build_residual_prophet_df(
    df: pd.DataFrame,
    residual_banks: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Aggregate all residual banks into one combined series for the residual model."""
    residual_df = (
        df[df["bank_name"].isin(residual_banks)]
        .groupby("date")[target_col]
        .sum()
        .reset_index()
        .rename(columns={"date": "ds", target_col: "y"})
        .sort_values("ds")
    )

    # Zero months mean no data — treat as null
    residual_df.loc[residual_df["y"] == 0, "y"] = None
    residual_df = residual_df[residual_df["y"].notna()].copy()

    logger.info(
        f"  Residual bucket: {len(residual_banks)} banks | "
        f"{len(residual_df)} months of combined data | "
        f"{residual_df['ds'].min():%b %Y} → {residual_df['ds'].max():%b %Y}"
    )
    return residual_df


# ── Main loader ────────────────────────────────────────────────────────────

def load_bank_data(card_type: str) -> dict:
    """Full pipeline: load → clean → select → build Prophet DFs.

    Args:
        card_type: 'cc' or 'dc'

    Returns dict with keys:
        df           — cleaned full bankwise DataFrame
        target_col   — name of the outstanding column
        top_banks    — list of top N bank names
        residual_banks — list of remaining bank names
        bank_dfs     — dict: bank_name → Prophet DataFrame (or None if skipped)
        residual_df  — Prophet DataFrame for residual bucket
        coverage_pct — fraction of bankwise total covered by top banks
    """
    assert card_type in ("cc", "dc"), "card_type must be 'cc' or 'dc'"
    target_col = f"{card_type}_outstanding"

    logger.info(f"\nLoading bankwise {card_type.upper()} data...")
    raw = _load_bankwise(card_type)
    logger.info(f"  Raw: {len(raw):,} rows | {raw['bank_name'].nunique()} banks")

    df = _clean_bankwise(raw, target_col)
    logger.info(f"  After cleaning: {len(df):,} rows | {df['bank_name'].nunique()} banks")

    top_banks, residual_banks = _select_top_banks(df, target_col)
    coverage_pct = _log_coverage(df, top_banks, target_col)

    # Build individual Prophet DFs
    bank_dfs: dict[str, pd.DataFrame | None] = {}
    for bank in top_banks:
        bank_dfs[bank] = build_bank_prophet_df(df, bank, target_col)

    # Build residual DF
    residual_df = build_residual_prophet_df(df, residual_banks, target_col)

    return {
        "df":             df,
        "target_col":     target_col,
        "top_banks":      top_banks,
        "residual_banks": residual_banks,
        "bank_dfs":       bank_dfs,
        "residual_df":    residual_df,
        "coverage_pct":   coverage_pct,
    }


if __name__ == "__main__":
    for ct in ["cc", "dc"]:
        result = load_bank_data(ct)
        print(f"\n{ct.upper()} — Top {len(result['top_banks'])} banks:")
        for b in result["top_banks"]:
            df = result["bank_dfs"].get(b)
            if df is not None:
                print(f"  {b:<40} {len(df):3d} months | {df['ds'].min():%b %Y} → {df['ds'].max():%b %Y}")
            else:
                print(f"  {b:<40} SKIPPED (insufficient data)")