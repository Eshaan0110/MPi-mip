"""
FIX F10: True Out-of-Sample Holdout Test
=========================================
Runs a proper OOS test: train on all data through Dec 2024,
forecast Jan–Jun 2025, measure MAPE on those 6 unseen months.

This is the DEFINITIVE accuracy number. CV MAPE is for tuning.
OOS MAPE is for reporting. Run ONCE after all tuning is done.

Usage:
    uv run python experiments/axiom_audit/fix_f10_oos_holdout.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modelling.data_prep import load_all, build_master, build_training_df
from src.modelling.model_config import CC_CONFIG, DC_CONFIG, STRUCTURAL_EVENTS
from src.modelling.aggregate_model import build_prophet_model


HOLDOUT_CUTOFF = pd.Timestamp("2024-12-01")
OOS_END = pd.Timestamp("2025-06-01")

OUTPUT_DIR = Path(__file__).parent / "oos_results"


def run_oos_test(config: dict, master: pd.DataFrame, label: str) -> dict:
    """Train on data through HOLDOUT_CUTOFF, forecast to OOS_END, measure MAPE."""
    train_df = build_training_df(master, config)

    # Split
    train_only = train_df[train_df["ds"] <= HOLDOUT_CUTOFF].copy()
    oos = train_df[(train_df["ds"] > HOLDOUT_CUTOFF) & (train_df["ds"] <= OOS_END)].copy()

    if oos.empty:
        logger.warning(f"{label}: no OOS data after {HOLDOUT_CUTOFF:%b %Y}")
        return {}

    logger.info(f"\n{'='*50}")
    logger.info(f"OOS TEST: {label}")
    logger.info(f"Train: {len(train_only)} rows ({train_only['ds'].min():%b %Y} → {train_only['ds'].max():%b %Y})")
    logger.info(f"OOS: {len(oos)} rows ({oos['ds'].min():%b %Y} → {oos['ds'].max():%b %Y})")

    # Fit on train_only
    model = build_prophet_model(config, train_only)

    # Predict on OOS dates (need to build future df with regressor values)
    future = train_df[train_df["ds"] <= OOS_END].copy()  # full df with actual regressor values
    forecast = model.predict(future)

    # Extract OOS predictions
    fc_oos = forecast[forecast["ds"] > HOLDOUT_CUTOFF][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    actual_oos = oos[["ds", "y"]].copy()

    merged = fc_oos.merge(actual_oos, on="ds")

    # MAPE
    merged["ape"] = (merged["yhat"] - merged["y"]).abs() / merged["y"].abs() * 100
    mape = merged["ape"].mean()
    max_ape = merged["ape"].max()

    # CI coverage: what % of actuals fall within 90% CI?
    merged["in_ci"] = (merged["y"] >= merged["yhat_lower"]) & (merged["y"] <= merged["yhat_upper"])
    ci_coverage = merged["in_ci"].mean() * 100

    logger.info(f"\nOOS RESULTS ({label}):")
    logger.info(f"  MAPE: {mape:.2f}%")
    logger.info(f"  Max APE: {max_ape:.2f}%")
    logger.info(f"  90% CI coverage: {ci_coverage:.0f}% ({merged['in_ci'].sum()}/{len(merged)} months)")
    logger.info(f"\n  Month-by-month:")
    for _, row in merged.iterrows():
        status = "OK" if row["in_ci"] else "MISS"
        logger.info(
            f"    {row['ds']:%b %Y}: actual={row['y']:.2f} pred={row['yhat']:.2f} "
            f"APE={row['ape']:.1f}% [{row['yhat_lower']:.2f}, {row['yhat_upper']:.2f}] {status}"
        )

    return {
        "model": label,
        "oos_mape": round(mape, 2),
        "oos_max_ape": round(max_ape, 2),
        "ci_coverage_pct": round(ci_coverage, 1),
        "n_months": len(merged),
        "train_end": str(HOLDOUT_CUTOFF.date()),
        "oos_end": str(OOS_END.date()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_all()
    master = build_master(data)

    results = []
    for label, config in [("CC Outstanding", CC_CONFIG), ("DC Outstanding", DC_CONFIG)]:
        try:
            r = run_oos_test(config, master, label)
            if r:
                results.append(r)
        except Exception as e:
            logger.error(f"{label} OOS test failed: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_DIR / "oos_holdout_results.csv", index=False)
        print(f"\n{'='*50}")
        print("OOS HOLDOUT SUMMARY")
        print(f"{'='*50}")
        print(df.to_string(index=False))
        print(f"\nResults saved to {OUTPUT_DIR / 'oos_holdout_results.csv'}")


if __name__ == "__main__":
    main()
