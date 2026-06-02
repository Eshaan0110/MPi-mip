"""
MIP Modelling — Data Preparation
==================================
Builds the Prophet-ready training DataFrame from processed parquet files.

Responsibilities:
  1. Load and align all processed datasets on a monthly date index.
  2. Apply lags specified in model_config.RegressorSpec.
  3. Fill nulls using the method specified per regressor.
  4. Build dummy columns for structural events (pulse and step types).
  5. Build the forward-forecast future DataFrame (12 months ahead)
     with projected regressor values (Approach 1 — flat/trend extrapolation).

All logic is deterministic and stateless. No Prophet calls here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.modelling.model_config import (
    CC_CONFIG,
    DC_CONFIG,
    STRUCTURAL_EVENTS,
    FORECAST_CONFIG,
    RegressorSpec,
)


# ── Paths ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED    = _PROJECT_ROOT / "data" / "processed"


# ── Loaders ────────────────────────────────────────────────────────────────

def _load(stem: str) -> pd.DataFrame:
    path = _PROCESSED / f"{stem}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.ingestion` first."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Load every processed dataset. Returns a dict keyed by stem name."""
    stems = ["rbi_psi_cards", "npci_upi", "upi_p2p_p2m", "cpi", "repo_rate"]
    data = {}
    for stem in stems:
        try:
            data[stem] = _load(stem)
            logger.info(f"Loaded {stem}: {len(data[stem])} rows")
        except FileNotFoundError as e:
            logger.warning(str(e))
    return data


# ── Master alignment ───────────────────────────────────────────────────────

def build_master(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all datasets onto a single monthly date index (month-start)."""
    psi = data["rbi_psi_cards"].copy()

    master = psi[[
        "date",
        "credit_cards_outstanding_lakh",
        "debit_cards_outstanding_lakh",
        "credit_card_vol_lakh",
        "debit_card_vol_lakh",
        "credit_card_pos_vol_lakh",
        "debit_card_pos_vol_lakh",
        "pos_terminals_lakh",
        "upi_qr_lakh",
        "bharat_qr_lakh",
    ]].copy()

    # Normalise to month-start
    master["date"] = master["date"].dt.to_period("M").dt.to_timestamp()

    # Merge regressors
    if "npci_upi" in data:
        npci = data["npci_upi"][["date", "upi_volume_mn", "upi_value_cr"]].copy()
        npci["date"] = npci["date"].dt.to_period("M").dt.to_timestamp()
        master = master.merge(npci, on="date", how="left")

    if "upi_p2p_p2m" in data:
        p2m = data["upi_p2p_p2m"][["date", "upi_p2m_vol_mn", "upi_p2p_vol_mn"]].copy()
        p2m["date"] = p2m["date"].dt.to_period("M").dt.to_timestamp()
        master = master.merge(p2m, on="date", how="left")

    if "repo_rate" in data:
        repo = data["repo_rate"][["date", "repo_rate"]].copy()
        repo["date"] = repo["date"].dt.to_period("M").dt.to_timestamp()
        master = master.merge(repo, on="date", how="left")
        master["repo_rate"] = master["repo_rate"].ffill()  # rate persists until next change

    if "cpi" in data:
        cpi = data["cpi"][["date", "cpi_index", "cpi_inflation_pct"]].copy()
        cpi["date"] = cpi["date"].dt.to_period("M").dt.to_timestamp()
        master = master.merge(cpi, on="date", how="left")

    master = master.sort_values("date").reset_index(drop=True)
    logger.info(
        f"Master DataFrame: {len(master)} rows × {len(master.columns)} cols | "
        f"{master['date'].min():%b %Y} → {master['date'].max():%b %Y}"
    )
    return master


# ── Null filling ───────────────────────────────────────────────────────────

def _apply_fill(series: pd.Series, method: str) -> pd.Series:
    """Apply the fill strategy specified in RegressorSpec."""
    if method == "zero":
        return series.fillna(0.0)
    elif method == "ffill":
        return series.ffill().bfill()  # bfill handles leading nulls
    elif method == "bfill":
        return series.bfill().ffill()
    elif method == "linear":
        return series.interpolate(method="linear").ffill().bfill()
    else:
        raise ValueError(f"Unknown fill method: {method!r}. Use zero|ffill|bfill|linear.")


# ── Structural event columns ───────────────────────────────────────────────

def build_event_columns(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Add structural event dummy columns for the given model ('cc' or 'dc')."""
    df = df.copy()

    for event_name, spec in STRUCTURAL_EVENTS.items():
        if model_key not in spec["models"]:
            continue

        col = f"event_{event_name}"

        if spec["type"] == "dummy_pulse":
            # 1 only for the specific month(s)
            pulse_dates = [pd.Timestamp(d) for d in spec["dates"]]
            df[col] = df["date"].isin(pulse_dates).astype(float)

        elif spec["type"] == "dummy_step":
            # 0 before date, 1 from date onwards
            step_date = pd.Timestamp(spec["date"])
            df[col] = (df["date"] >= step_date).astype(float)

        elif spec["type"] == "changepoint":
            # Changepoints go into Prophet's changepoints list, not as regressors
            # No column needed here — handled in model builder
            continue

        logger.debug(
            f"  Event column '{col}': "
            f"{spec['type']}, sum={df[col].sum():.0f}"
        )

    return df


# ── Training DataFrame builder ─────────────────────────────────────────────

def build_training_df(
    master: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Build a Prophet-ready training DataFrame for one model (CC or DC).

    Prophet expects columns: ds (datetime), y (target), plus any regressors.

    Args:
        master:  Full merged master DataFrame from build_master().
        config:  CC_CONFIG or DC_CONFIG from model_config.

    Returns:
        DataFrame with ds, y, and all regressor columns, filtered to rows
        where y is non-null.
    """
    df = master.copy()
    target_col = config["target_col"]
    # Derive model_key for structural event filtering.
    # Explicit key in config takes precedence; otherwise infer from name.
    if "model_key" in config:
        model_key = config["model_key"]
    elif "credit" in config["name"]:
        model_key = "cc"
    elif "debit" in config["name"]:
        model_key = "dc"
    elif "upi" in config["name"]:
        model_key = "upi"
    else:
        model_key = "cc"  # fallback

    # Add structural event dummy columns
    df = build_event_columns(df, model_key)

    # Apply lag + fill for each regressor
    regressors: list[RegressorSpec] = config["regressors"]
    for spec in regressors:
        raw_col = spec.col
        if raw_col not in df.columns:
            logger.warning(f"Regressor '{raw_col}' not found in master — skipping.")
            continue

        # Fill nulls first, then lag
        filled = _apply_fill(df[raw_col], spec.fill_method)

        if spec.lag > 0:
            lagged_col = f"{raw_col}_lag{spec.lag}"
            df[lagged_col] = filled.shift(spec.lag)
            # Backfill any leading nulls introduced by the lag
            df[lagged_col] = _apply_fill(df[lagged_col], "bfill")
        else:
            df[raw_col] = filled

    # Rename to Prophet convention
    df = df.rename(columns={"date": "ds", target_col: "y"})

    # Filter to rows where target is non-null
    df_train = df[df["y"].notna()].copy()

    # Apply optional training_start cutoff (e.g. to exclude GFC period for CC)
    training_start = config.get("training_start")
    if training_start:
        df_train = df_train[df_train["ds"] >= pd.Timestamp(training_start)].copy()
        logger.info(f"  training_start applied: {training_start} → {len(df_train)} rows")

    # Report column inventory
    event_cols = [c for c in df_train.columns if c.startswith("event_")]
    regressor_final_cols = []
    for spec in regressors:
        col = f"{spec.col}_lag{spec.lag}" if spec.lag > 0 else spec.col
        if col in df_train.columns:
            regressor_final_cols.append(col)

    logger.info(
        f"Training DF ({config['name']}): {len(df_train)} rows | "
        f"ds={df_train['ds'].min():%b %Y} → {df_train['ds'].max():%b %Y} | "
        f"regressors: {regressor_final_cols} | events: {event_cols}"
    )

    null_check = df_train[["y"] + regressor_final_cols + event_cols].isnull().sum()
    if null_check.any():
        logger.warning(f"Nulls remain in training DF after filling:\n{null_check[null_check > 0]}")

    return df_train


# ── Future DataFrame builder ───────────────────────────────────────────────

def build_future_df(
    train_df: pd.DataFrame,
    config: dict,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Build the future DataFrame for Prophet's predict() call.

    For the 12-month forecast window, regressor values are projected using
    Approach 1 (Rahul spec):
      - repo_rate:  flat at last known value (current RBI rate)
      - cpi_index:  trailing 12-month average
      - upi_qr:     trailing 6-month linear trend extrapolation
      - upi_volume: trailing 6-month linear trend extrapolation
      - debit_card_vol: trailing 6-month linear trend extrapolation
      - credit_card_vol: trailing 6-month linear trend extrapolation
      - event dummies: pulse events = 0 (shocks don't repeat), step events = last value

    All assumptions are logged explicitly so they can be reviewed.
    """
    from prophet import Prophet

    periods = FORECAST_CONFIG["periods"]
    freq    = FORECAST_CONFIG["freq"]

    # Build date range: historical + forecast
    last_date = train_df["ds"].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods,
        freq=freq,
    )
    all_dates = pd.concat([
        train_df[["ds"]],
        pd.DataFrame({"ds": future_dates}),
    ], ignore_index=True)

    future = all_dates.copy()
    model_key = "cc" if "credit" in config["name"] else "dc"

    # Add event columns
    future_with_events = future.rename(columns={"ds": "date"})
    future_with_events = build_event_columns(future_with_events, model_key)
    future = future_with_events.rename(columns={"date": "ds"})

    # Pulse events = 0 in forecast (shocks don't repeat by assumption)
    for col in [c for c in future.columns if c.startswith("event_")]:
        future.loc[future["ds"] > last_date, col] = 0.0

    # Project each regressor forward
    regressors: list[RegressorSpec] = config["regressors"]
    m = master.copy()
    m["date"] = m["date"].dt.to_period("M").dt.to_timestamp()

    for spec in regressors:
        raw_col = spec.col
        if raw_col not in m.columns:
            continue

        filled = _apply_fill(m[raw_col], spec.fill_method)
        if spec.lag > 0:
            filled = filled.shift(spec.lag).pipe(_apply_fill, "bfill")

        final_col = f"{raw_col}_lag{spec.lag}" if spec.lag > 0 else raw_col

        # Get historical values aligned to future dates
        hist = pd.DataFrame({"ds": m["date"].dt.to_period("M").dt.to_timestamp(), final_col: filled.values})
        future = future.merge(hist, on="ds", how="left")

        # Project forward: use last 6 months to extrapolate linearly
        hist_vals = filled.dropna().values
        if len(hist_vals) >= 6:
            last6 = hist_vals[-6:]
            slope = np.polyfit(range(6), last6, 1)[0]
            last_val = hist_vals[-1]
        else:
            slope = 0
            last_val = hist_vals[-1] if len(hist_vals) > 0 else 0

        # Special cases: repo = flat, cpi = 12m avg
        if "repo_rate" in raw_col:
            proj_val = float(hist_vals[-1]) if len(hist_vals) > 0 else 6.5
            slope = 0
            logger.info(f"Forward proj {final_col}: flat at {proj_val:.2f}% (current RBI rate)")
        elif "cpi" in raw_col:
            proj_val = float(np.mean(hist_vals[-12:])) if len(hist_vals) >= 12 else float(np.mean(hist_vals))
            slope = 0
            logger.info(f"Forward proj {final_col}: flat at trailing 12m avg = {proj_val:.1f}")
        else:
            proj_val = last_val
            logger.info(f"Forward proj {final_col}: trend extrapolation, slope={slope:.2f}/month")

        # Fill forecast months
        mask = future["ds"] > last_date
        for i, idx in enumerate(future[mask].index, start=1):
            future.loc[idx, final_col] = proj_val + slope * i

    logger.info(
        f"Future DF: {len(future)} rows ({len(train_df)} hist + {periods} forecast) | "
        f"cols: {list(future.columns)}"
    )
    return future


if __name__ == "__main__":
    data   = load_all()
    master = build_master(data)
    cc_df  = build_training_df(master, CC_CONFIG)
    dc_df  = build_training_df(master, DC_CONFIG)
    print(f"\nCC training: {len(cc_df)} rows, cols: {list(cc_df.columns)}")
    print(f"DC training: {len(dc_df)} rows, cols: {list(dc_df.columns)}")
    print("\nCC tail:")
    print(cc_df.tail(3)[["ds","y"]].to_string(index=False))
    print("\nDC tail:")
    print(dc_df.tail(3)[["ds","y"]].to_string(index=False))