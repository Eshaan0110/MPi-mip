"""
MIP Modelling -- Bank-Level Data Preparation
=============================================
Loads bankwise parquet files, cleans, selects specified banks,
applies per-bank training cutoffs, log1p transform, merger dummies,
and prepares Prophet-ready DataFrames.

Enhancement (Jun 2026):
  - Explicit bank lists (CC_BANK_LIST, DC_BANK_LIST) replace auto-selection
  - Per-bank start dates from BANK_START_DATES (stable regime, not longest history)
  - log1p(y) transform for variance stabilisation (USE_LOG_TRANSFORM flag)
  - Merger step dummies from BANK_MERGER_EVENTS
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.modelling.bank_config import (
    TOP_N_ISSUERS,
    MIN_MONTHS,
    BANK_NAME_ALIASES,
    TERMINATED_BANKS,
    CC_LIVE_BANK_TRAIN_START,
    DC_LIVE_BANK_TRAIN_START,
    BANK_START_DATES,
    BANK_MERGER_EVENTS,
    USE_LOG_TRANSFORM,
    CC_BANK_LIST,
    DC_BANK_LIST,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED    = _PROJECT_ROOT / "data" / "processed"


# ── Loaders ────────────────────────────────────────────────────────────────

def _load_bankwise(card_type: str) -> pd.DataFrame:
    stem = f"bankwise_cards_{card_type}"
    path = _PROCESSED / f"{stem}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.ingestion --only bankwise` first."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    return df


def _load_psi_series(card_type: str) -> pd.Series:
    psi_path = _PROCESSED / "rbi_psi_cards.parquet"
    if not psi_path.exists():
        raise FileNotFoundError(f"{psi_path} not found.")
    psi = pd.read_parquet(psi_path)
    psi["date"] = pd.to_datetime(psi["date"]).dt.to_period("M").dt.to_timestamp()
    psi_col = "credit_cards_outstanding_lakh" if card_type == "cc" else "debit_cards_outstanding_lakh"
    psi_series = psi.set_index("date")[psi_col].dropna()
    return (psi_series * 1e5).rename("psi_cards")


# ── Cleaning ───────────────────────────────────────────────────────────────

def _is_numeric_name(name: str) -> bool:
    return bool(pd.Series([name]).str.match(r"^\d+\.?\d*$").iloc[0])


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bank_name"] = df["bank_name"].replace(BANK_NAME_ALIASES)
    return df


def _clean_bankwise(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = _apply_aliases(df)
    mask_numeric = df["bank_name"].apply(_is_numeric_name)
    n_dropped = mask_numeric.sum()
    if n_dropped > 0:
        logger.info(f"  Dropped {n_dropped} numeric artifact rows")
    df = df[~mask_numeric].copy()

    df = (
        df.groupby(["date", "bank_name"], as_index=False)
        .agg({target_col: "sum", "bank_category": "first", "source": "first"})
    )
    df = df.sort_values(["bank_name", "date"]).reset_index(drop=True)
    return df


# ── Bank selection ─────────────────────────────────────────────────────────

def _select_banks(
    df: pd.DataFrame,
    target_col: str,
    card_type: str,
) -> tuple[list[str], list[str]]:
    """Return (modelled_banks, residual_banks) using explicit bank lists."""
    explicit_list = CC_BANK_LIST if card_type == "cc" else DC_BANK_LIST
    available = set(df["bank_name"].unique())

    modelled = [b for b in explicit_list if b in available]
    missing = [b for b in explicit_list if b not in available]
    if missing:
        logger.warning(f"  Banks in config but not in data: {missing}")

    residual = [b for b in available if b not in modelled]

    logger.info(
        f"  Explicit bank list ({card_type.upper()}): {len(modelled)} modelled | "
        f"{len(residual)} in residual"
    )
    return modelled, residual


# ── Coverage ───────────────────────────────────────────────────────────────

def _log_coverage(
    df: pd.DataFrame, top_banks: list[str], target_col: str,
    psi_latest: float | None = None,
) -> float:
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]
    top_sum = latest[latest["bank_name"].isin(top_banks)][target_col].sum()
    total_sum = latest[target_col].sum()
    coverage_pct = top_sum / total_sum * 100 if total_sum > 0 else 0
    logger.info(
        f"  Coverage ({latest_date:%b %Y}): modelled banks = "
        f"{top_sum:,.0f} of {total_sum:,.0f} ({coverage_pct:.1f}% of bankwise total)"
    )
    return coverage_pct / 100


# ── Prophet DataFrame builder ──────────────────────────────────────────────

def build_bank_prophet_df(
    df: pd.DataFrame,
    bank_name: str,
    target_col: str,
    card_type: str = "cc",
) -> pd.DataFrame | None:
    """Build a Prophet-ready (ds, y) DataFrame for one bank.

    Applies:
      1. Per-bank start date from BANK_START_DATES (falls back to card-type default)
      2. log1p(y) transform if USE_LOG_TRANSFORM is True
      3. Merger step dummy column if bank has an entry in BANK_MERGER_EVENTS

    Returns None if insufficient data after filtering.
    """
    bank_df = df[df["bank_name"] == bank_name][[
        "date", target_col
    ]].rename(columns={"date": "ds", target_col: "y"}).copy()

    bank_df = bank_df[bank_df["y"].notna() & (bank_df["y"] > 0)].copy()
    bank_df = bank_df.sort_values("ds").reset_index(drop=True)

    # Per-bank start date
    if bank_name not in TERMINATED_BANKS:
        key = (bank_name, card_type)
        if key in BANK_START_DATES:
            cutoff = BANK_START_DATES[key]
        elif card_type == "cc":
            cutoff = CC_LIVE_BANK_TRAIN_START
        else:
            cutoff = DC_LIVE_BANK_TRAIN_START

        n_before = len(bank_df)
        bank_df = bank_df[bank_df["ds"] >= cutoff].copy()
        n_after = len(bank_df)
        if n_before > n_after:
            logger.debug(
                f"  {bank_name}: start={cutoff:%b %Y} ({card_type.upper()}) "
                f"({n_before} -> {n_after} months)"
            )

    if len(bank_df) < MIN_MONTHS:
        logger.warning(
            f"  {bank_name}: only {len(bank_df)} months after cutoff -- skipping"
        )
        return None

    if bank_name in TERMINATED_BANKS:
        reason = TERMINATED_BANKS[bank_name]["reason"]
        logger.info(f"  {bank_name}: terminated ({reason}). Forecast clipped at exit.")

    # log1p transform
    if USE_LOG_TRANSFORM:
        bank_df["y"] = np.log1p(bank_df["y"])

    # Merger step dummy
    merger_key = (bank_name, card_type)
    if merger_key in BANK_MERGER_EVENTS:
        event = BANK_MERGER_EVENTS[merger_key]
        merger_date = pd.Timestamp(event["date"])
        col_name = f"merger_{event['label']}"
        bank_df[col_name] = (bank_df["ds"] >= merger_date).astype(float)
        logger.debug(
            f"  {bank_name}: merger dummy '{col_name}' at {merger_date:%b %Y} "
            f"(sum={bank_df[col_name].sum():.0f})"
        )

    logger.debug(
        f"  {bank_name}: {len(bank_df)} months | "
        f"{bank_df['ds'].min():%b %Y} -> {bank_df['ds'].max():%b %Y}"
        f"{' (log1p)' if USE_LOG_TRANSFORM else ''}"
    )
    return bank_df


# ── Residual ───────────────────────────────────────────────────────────────

def build_residual_prophet_df(
    df: pd.DataFrame,
    top_banks: list[str],
    target_col: str,
    card_type: str,
) -> pd.DataFrame:
    """Residual = PSI total - sum(modelled banks). log1p applied if flag set."""
    top_sum = (
        df[df["bank_name"].isin(top_banks)]
        .groupby("date")[target_col].sum()
        .rename("top_sum")
    )
    psi_cards = _load_psi_series(card_type)
    aligned = pd.concat([psi_cards, top_sum], axis=1, join="inner")
    aligned["residual"] = aligned["psi_cards"] - aligned["top_sum"]

    n_total = len(aligned)
    aligned = aligned[(aligned["top_sum"] > 0) & (aligned["residual"] > 0)]
    n_dropped = n_total - len(aligned)
    if n_dropped > 0:
        logger.warning(
            f"  [{card_type.upper()}] Dropped {n_dropped} months from residual "
            f"(zero top_sum or negative residual)"
        )

    residual_df = (
        aligned.reset_index()
        .rename(columns={"date": "ds", "residual": "y"})
        [["ds", "y"]]
        .sort_values("ds")
        .reset_index(drop=True)
    )

    if residual_df.empty:
        raise RuntimeError(f"[{card_type.upper()}] Residual series empty after alignment.")

    if USE_LOG_TRANSFORM:
        residual_df["y"] = np.log1p(residual_df["y"])

    last_date = residual_df["ds"].iloc[-1]
    logger.info(
        f"  [{card_type.upper()}] Residual: {len(residual_df)} months | "
        f"{residual_df['ds'].min():%b %Y} -> {last_date:%b %Y}"
        f"{' (log1p)' if USE_LOG_TRANSFORM else ''}"
    )
    return residual_df


# ── Main loader ────────────────────────────────────────────────────────────

def load_bank_data(card_type: str) -> dict:
    """Full pipeline: load -> clean -> select -> build Prophet DFs."""
    assert card_type in ("cc", "dc")
    target_col = f"{card_type}_outstanding"

    logger.info(f"\nLoading bankwise {card_type.upper()} data...")
    raw = _load_bankwise(card_type)
    logger.info(f"  Raw: {len(raw):,} rows | {raw['bank_name'].nunique()} banks")

    df = _clean_bankwise(raw, target_col)
    logger.info(f"  After cleaning: {len(df):,} rows | {df['bank_name'].nunique()} banks")

    top_banks, residual_banks = _select_banks(df, target_col, card_type)
    coverage_pct = _log_coverage(df, top_banks, target_col)

    bank_dfs: dict[str, pd.DataFrame | None] = {}
    for bank in top_banks:
        bank_dfs[bank] = build_bank_prophet_df(df, bank, target_col, card_type)

    residual_df = build_residual_prophet_df(df, top_banks, target_col, card_type)

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
        print(f"\n{ct.upper()} -- {len(result['top_banks'])} banks:")
        for b in result["top_banks"]:
            bdf = result["bank_dfs"].get(b)
            if bdf is not None:
                extra = [c for c in bdf.columns if c.startswith("merger_")]
                extras = f" + {extra}" if extra else ""
                print(f"  {b:<35} {len(bdf):3d} months | {bdf['ds'].min():%b %Y} -> {bdf['ds'].max():%b %Y}{extras}")
            else:
                print(f"  {b:<35} SKIPPED")
