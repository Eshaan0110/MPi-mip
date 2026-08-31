"""
MIP — Vintage Scoring (P3.1)
==============================
RBI revises historical data — the pipeline overwrites in place.
This module scores forecasts against *first-release* data (the vintage
that existed when the forecast was made), measuring real-time skill —
which is what decisions are actually made on.

Workflow:
  1. Before each pipeline run, snapshot_processed() saves the current data
  2. After generating forecasts, save_forecast_vintage() records predictions
  3. Later, score_vintage() compares forecast vs first-release actuals

Usage:
    from src.modelling.vintage_scoring import (
        save_forecast_vintage,
        score_vintage,
        score_all_vintages,
    )
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_VINTAGES = _PROJECT_ROOT / "data" / "vintages"
_FORECAST_VINTAGES = _PROJECT_ROOT / "data" / "forecast_vintages"


def save_forecast_vintage(
    forecast_df: pd.DataFrame,
    model_name: str,
    label: str | None = None,
    metadata: dict | None = None,
) -> Path:
    """Save a forecast with its vintage label for later scoring.

    Args:
        forecast_df: DataFrame with at least 'date' and 'forecast' columns
        model_name: e.g. 'forecast_cc', 'forecast_dc'
        label: vintage label (defaults to today's date)
        metadata: optional dict of model params, weights, etc.

    Returns:
        Path to saved forecast vintage file
    """
    label = label or date.today().isoformat()
    dest = _FORECAST_VINTAGES / label
    dest.mkdir(parents=True, exist_ok=True)

    forecast_df.to_parquet(dest / f"{model_name}.parquet", index=False)

    if metadata:
        meta_path = dest / f"{model_name}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Forecast vintage '{label}/{model_name}': {len(forecast_df)} rows")
    return dest


_TARGET_TO_SOURCE = {
    "credit_cards_outstanding_lakh": "rbi_psi_cards",
    "debit_cards_outstanding_lakh": "rbi_psi_cards",
    "cc_txn_vol_lakh": "rbi_psi_cards",
    "dc_txn_vol_lakh": "rbi_psi_cards",
    "upi_volume_mn": "npci_upi",
    "upi_value_cr": "npci_upi",
}


def _load_first_release_actuals(
    forecast_dates: list[pd.Timestamp],
    target_col: str,
    source_file: str | None = None,
) -> pd.Series | None:
    """Find the earliest vintage that contains each forecast date's actual value.

    For each forecast date, we want the first-release value — the number
    that was first published, before any RBI revisions.

    Args:
        forecast_dates: dates to look up
        target_col: column name for actuals
        source_file: parquet stem in vintage snapshot (auto-detected from target_col)
    """
    vintages = sorted(
        [d.name for d in _VINTAGES.iterdir() if d.is_dir()]
    ) if _VINTAGES.exists() else []

    if not vintages:
        return None

    if source_file is None:
        source_file = _TARGET_TO_SOURCE.get(target_col, "rbi_psi_cards")

    first_release = {}

    for fc_date in forecast_dates:
        if fc_date in first_release:
            continue
        for vintage_label in vintages:
            src_path = _VINTAGES / vintage_label / f"{source_file}.parquet"
            if not src_path.exists():
                continue
            try:
                df = pd.read_parquet(src_path)
                df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
                row = df[df["date"] == fc_date]
                if not row.empty and target_col in row.columns:
                    val = row[target_col].iloc[0]
                    if pd.notna(val):
                        first_release[fc_date] = val
                        break
            except Exception:
                continue

    if not first_release:
        return None

    return pd.Series(first_release, name="first_release_actual")


def score_vintage(
    model_name: str,
    vintage_label: str,
    target_col: str,
) -> dict | None:
    """Score a forecast vintage against first-release actuals.

    Args:
        model_name: e.g. 'forecast_cc'
        vintage_label: the date label of the forecast vintage
        target_col: column name in rbi_psi_cards for actuals

    Returns:
        dict with mape, mae, n_scored, details, or None if insufficient data
    """
    fc_path = _FORECAST_VINTAGES / vintage_label / f"{model_name}.parquet"
    if not fc_path.exists():
        logger.warning(f"No forecast vintage: {vintage_label}/{model_name}")
        return None

    fc_df = pd.read_parquet(fc_path)
    fc_df["date"] = pd.to_datetime(fc_df["date"]).dt.to_period("M").dt.to_timestamp()

    forecast_dates = fc_df["date"].tolist()
    actuals = _load_first_release_actuals(forecast_dates, target_col)

    if actuals is None or len(actuals) == 0:
        logger.info(f"No first-release actuals yet for {vintage_label}/{model_name}")
        return None

    merged = fc_df.set_index("date")[["forecast"]].join(actuals.rename("actual"))
    merged = merged.dropna()

    if merged.empty:
        return None

    errors = np.abs((merged["actual"] - merged["forecast"]) / merged["actual"])
    mape = errors.mean() * 100
    mae = np.abs(merged["actual"] - merged["forecast"]).mean()

    result = {
        "model": model_name,
        "vintage": vintage_label,
        "n_scored": len(merged),
        "mape": round(mape, 2),
        "mae": round(mae, 2),
        "details": [
            {
                "date": str(idx),
                "forecast": round(row["forecast"], 1),
                "actual": round(row["actual"], 1),
                "ape": round(abs(row["actual"] - row["forecast"]) / row["actual"] * 100, 2),
            }
            for idx, row in merged.iterrows()
        ],
    }

    logger.info(
        f"Vintage score {vintage_label}/{model_name}: "
        f"MAPE={mape:.2f}% on {len(merged)} months (first-release)"
    )
    return result


def score_all_vintages(
    model_name: str,
    target_col: str,
) -> list[dict]:
    """Score all available forecast vintages for a model."""
    if not _FORECAST_VINTAGES.exists():
        return []

    results = []
    for d in sorted(_FORECAST_VINTAGES.iterdir()):
        if not d.is_dir():
            continue
        r = score_vintage(model_name, d.name, target_col)
        if r is not None:
            results.append(r)

    return results
