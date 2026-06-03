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
  6. Build residual bucket as PSI total minus the sum of top banks
     (NOT as the sum of remaining bankwise banks — see build_residual_prophet_df).
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
    CC_LIVE_BANK_TRAIN_START,
    DC_LIVE_BANK_TRAIN_START,
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


def _load_psi_series(card_type: str) -> pd.Series:
    """Load PSI cards outstanding for the given card type.

    Returns a pd.Series indexed by month-start date, in RAW CARD COUNTS
    (converted from lakh by multiplying by 1e5 to match bankwise units).
    """
    psi_path = _PROCESSED / "rbi_psi_cards.parquet"
    if not psi_path.exists():
        raise FileNotFoundError(
            f"{psi_path} not found. Run `python -m src.ingestion --only rbi` first. "
            f"The residual bucket requires PSI to compute true coverage."
        )
    psi = pd.read_parquet(psi_path)
    psi["date"] = pd.to_datetime(psi["date"]).dt.to_period("M").dt.to_timestamp()
    psi_col = "credit_cards_outstanding_lakh" if card_type == "cc" else "debit_cards_outstanding_lakh"
    if psi_col not in psi.columns:
        raise KeyError(
            f"Expected column '{psi_col}' not found in {psi_path}. "
            f"Available columns: {list(psi.columns)}"
        )
    psi_series = psi.set_index("date")[psi_col].dropna()
    # lakh → raw cards to match bankwise units
    return (psi_series * 1e5).rename("psi_cards")


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
    Residual banks: everything else (retained only for backward-compatible
    coverage logging — the actual residual model fits to PSI − top_sum, not
    to these banks summed).
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

    Returns coverage fraction (0–1) of top banks vs the bankwise total.
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


# ── Per-bank training cutoff overrides ────────────────────────────────────
# Some CC banks have pre-2017 data that belongs to an incompatible regime
# (e.g. HDFC and ICICI had different growth dynamics pre-demonetisation;
# Union Bank's pre-merger series is incompatible with the post-triple-merger
# entity). These banks revert to the DC-style 2017-01-01 cutoff even for CC.
_CC_CUTOFF_OVERRIDES: dict[str, pd.Timestamp] = {
    "HDFC Bank":          DC_LIVE_BANK_TRAIN_START,   # pre-2017 growth regime incompatible
    "ICICI Bank":         DC_LIVE_BANK_TRAIN_START,   # pre-demonetisation trajectory differs
    "Union Bank of India": DC_LIVE_BANK_TRAIN_START,  # pre-triple-merger entity doesn't exist
}


# ── Prophet DataFrame builder ──────────────────────────────────────────────

def build_bank_prophet_df(
    df: pd.DataFrame,
    bank_name: str,
    target_col: str,
    card_type: str = "cc",
) -> pd.DataFrame | None:
    """Build a Prophet-ready (ds, y) DataFrame for one bank.

    For live banks (not in TERMINATED_BANKS), training data is truncated
    to a card-type-specific start date:
      - CC: 2013-01-01 default, with per-bank overrides for banks whose
        pre-2017 data is an incompatible regime (see _CC_CUTOFF_OVERRIDES)
      - DC: 2017-01-01 (pre-2017 distorted by PMJDY mass issuance)
    Terminated banks keep their full history.

    Returns None if the bank has insufficient data after filtering.
    """
    bank_df = df[df["bank_name"] == bank_name][[
        "date", target_col
    ]].rename(columns={"date": "ds", target_col: "y"}).copy()

    # Drop rows where y is 0 or null (zero often means no data for that month)
    bank_df = bank_df[bank_df["y"].notna() & (bank_df["y"] > 0)].copy()
    bank_df = bank_df.sort_values("ds").reset_index(drop=True)

    # Truncate live banks to the appropriate cutoff
    if bank_name not in TERMINATED_BANKS:
        if card_type == "cc" and bank_name in _CC_CUTOFF_OVERRIDES:
            cutoff = _CC_CUTOFF_OVERRIDES[bank_name]
        elif card_type == "cc":
            cutoff = CC_LIVE_BANK_TRAIN_START
        else:
            cutoff = DC_LIVE_BANK_TRAIN_START
        n_before = len(bank_df)
        bank_df = bank_df[bank_df["ds"] >= cutoff].copy()
        n_after = len(bank_df)
        if n_before > n_after:
            logger.debug(
                f"  {bank_name}: truncated to post-{cutoff:%b %Y} "
                f"({card_type.upper()} cutoff) ({n_before} -> {n_after} months)"
            )

    if len(bank_df) < MIN_MONTHS:
        logger.warning(
            f"  {bank_name}: only {len(bank_df)} valid rows — skipping individual model"
        )
        return None

    if bank_name in TERMINATED_BANKS:
        reason = TERMINATED_BANKS[bank_name]["reason"]
        logger.info(
            f"  {bank_name}: terminated bank ({reason}). "
            f"Model fits to available history only; forecast clipped at exit date."
        )

    logger.debug(
        f"  {bank_name}: {len(bank_df)} months | "
        f"{bank_df['ds'].min():%b %Y} → {bank_df['ds'].max():%b %Y}"
    )
    return bank_df


def build_residual_prophet_df(
    df: pd.DataFrame,
    top_banks: list[str],
    target_col: str,
    card_type: str,
) -> pd.DataFrame:
    """Residual bucket = PSI total − sum(top banks).

    This is the correct residual definition. Summing the "other 67 banks"
    inside bankwise misses everything bankwise itself doesn't capture —
    FinTech CC issuers (Slice, OneCard), banking-as-a-service co-brands,
    and any banks that don't appear in RBI's bankwise reporting at all.
    Defining residual against PSI guarantees the ground-up aggregate
    matches PSI in-sample by construction.

    Args:
        df:          cleaned bankwise DataFrame (full series across all banks)
        top_banks:   names of the top-N banks being modelled individually
        target_col:  e.g. 'cc_outstanding' (raw card count in bankwise)
        card_type:   'cc' or 'dc'
    """
    # Sum the top banks at each date (in raw card counts)
    top_sum = (
        df[df["bank_name"].isin(top_banks)]
        .groupby("date")[target_col]
        .sum()
        .rename("top_sum")
    )

    # PSI total in raw cards (lakh × 1e5)
    psi_cards = _load_psi_series(card_type)

    # Residual = PSI − top_banks (aligned on overlapping dates)
    aligned = pd.concat([psi_cards, top_sum], axis=1, join="inner")
    aligned["residual"] = aligned["psi_cards"] - aligned["top_sum"]

    # Drop any pre-period where top_sum is zero (bankwise didn't report yet)
    # or where residual is negative (top reports higher than PSI for that
    # month — data quality artifact, e.g. around the May-2025 sheet-41 anomaly
    # where bankwise drops 20% but PSI doesn't).
    n_total = len(aligned)
    aligned = aligned[(aligned["top_sum"] > 0) & (aligned["residual"] > 0)]
    n_dropped = n_total - len(aligned)
    if n_dropped > 0:
        logger.warning(
            f"  [{card_type.upper()}] Dropped {n_dropped} months from residual fit "
            f"(zero top_sum or negative residual — data quality)"
        )

    residual_df = (
        aligned.reset_index()
        .rename(columns={"date": "ds", "residual": "y"})
        [["ds", "y"]]
        .sort_values("ds")
        .reset_index(drop=True)
    )

    if residual_df.empty:
        raise RuntimeError(
            f"[{card_type.upper()}] Residual series is empty after alignment. "
            f"Check PSI date range and bankwise coverage."
        )

    last_residual = residual_df["y"].iloc[-1]
    last_psi = aligned["psi_cards"].iloc[-1]
    last_date = residual_df["ds"].iloc[-1]

    logger.info(
        f"  [{card_type.upper()}] Residual bucket (PSI − top {len(top_banks)} banks): "
        f"{len(residual_df)} months | "
        f"{residual_df['ds'].min():%b %Y} → {last_date:%b %Y}"
    )
    logger.info(
        f"  [{card_type.upper()}] Residual size: "
        f"{last_residual/1e6:.1f}M cards "
        f"({last_residual/last_psi*100:.1f}% of PSI) as of {last_date:%b %Y}"
    )
    return residual_df


# ── Main loader ────────────────────────────────────────────────────────────

def load_bank_data(card_type: str) -> dict:
    """Full pipeline: load → clean → select → build Prophet DFs.

    Args:
        card_type: 'cc' or 'dc'

    Returns dict with keys:
        df             — cleaned full bankwise DataFrame
        target_col     — name of the outstanding column
        top_banks      — list of top N bank names
        residual_banks — list of remaining bank names (for logging only)
        bank_dfs       — dict: bank_name → Prophet DataFrame (or None if skipped)
        residual_df    — Prophet DataFrame for residual bucket (PSI − top banks)
        coverage_pct   — fraction of bankwise total covered by top banks
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
        bank_dfs[bank] = build_bank_prophet_df(df, bank, target_col, card_type)

    # Build residual DF: PSI − top_banks (NOT the sum of other bankwise banks)
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
        print(f"\n{ct.upper()} — Top {len(result['top_banks'])} banks:")
        for b in result["top_banks"]:
            df = result["bank_dfs"].get(b)
            if df is not None:
                print(f"  {b:<40} {len(df):3d} months | {df['ds'].min():%b %Y} → {df['ds'].max():%b %Y}")
            else:
                print(f"  {b:<40} SKIPPED (insufficient data)")