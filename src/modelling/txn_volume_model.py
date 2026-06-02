"""
MIP Modelling -- Transaction Volume Forecasts
==============================================
Forecasts monthly transaction volumes for:
  - Credit card total transactions (lakh)
  - Debit card total transactions (lakh)
  - UPI total transactions (million)

Same Prophet framework as the cards outstanding models. Uses the configs
defined in model_config.py (CC_VOL_CONFIG, DC_VOL_CONFIG, UPI_VOL_CONFIG).

Run:
    uv run python -m src.modelling.txn_volume_model            # all three
    uv run python -m src.modelling.txn_volume_model --no-cv    # skip CV

Outputs to data/processed/:
    forecast_cc_vol.parquet, forecast_cc_vol.csv
    forecast_dc_vol.parquet, forecast_dc_vol.csv
    forecast_upi_vol.parquet, forecast_upi_vol.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.modelling.model_config import (
    CC_VOL_CONFIG,
    DC_VOL_CONFIG,
    UPI_VOL_CONFIG,
    CV_CONFIG,
    FORECAST_CONFIG,
    STRUCTURAL_EVENTS,
    RegressorSpec,
)
from src.modelling.data_prep import (
    load_all,
    build_master,
    build_training_df,
    build_future_df,
)

from src.utils.run_logger import RunLogger

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED    = _PROJECT_ROOT / "data" / "processed"


def _build_prophet(config: dict, train_df: pd.DataFrame):
    """Fit a Prophet model from config. Returns fitted model."""
    from prophet import Prophet

    model_key = config["output_stem"]

    changepoint_dates = list(config.get("extra_changepoints", []))
    for event_name, spec in STRUCTURAL_EVENTS.items():
        if spec["type"] == "changepoint" and any(
            k in config.get("structural_events", []) for k in [event_name]
        ):
            changepoint_dates.append(
                spec.get("date", spec.get("dates", [""])[0])
            )

    prophet_kwargs = dict(config["prophet_kwargs"])
    if changepoint_dates:
        valid_cps = [
            pd.Timestamp(d)
            for d in changepoint_dates
            if train_df["ds"].min() < pd.Timestamp(d) < train_df["ds"].max()
        ]
        if valid_cps:
            prophet_kwargs["changepoints"] = valid_cps
            logger.info(f"  Changepoints: {[d.strftime('%b %Y') for d in valid_cps]}")

    m = Prophet(**prophet_kwargs)

    regressors: list[RegressorSpec] = config["regressors"]
    for spec in regressors:
        final_col = f"{spec.col}_lag{spec.lag}" if spec.lag > 0 else spec.col
        if final_col not in train_df.columns:
            logger.warning(f"  Regressor '{final_col}' missing -- skipping")
            continue
        m.add_regressor(final_col, standardize=spec.standardize, mode=spec.mode)
        logger.info(f"  Regressor: {final_col} (mode={spec.mode})")

    event_cols = [c for c in train_df.columns if c.startswith("event_")]
    for col in event_cols:
        m.add_regressor(col, standardize=False, mode="additive")

    m.fit(train_df)
    logger.info(f"  Fitted on {len(train_df)} rows")
    return m


def _run_cv(model, config: dict) -> pd.DataFrame:
    """Run rolling CV. Returns metrics DataFrame."""
    from prophet.diagnostics import cross_validation, performance_metrics

    try:
        cv_df = cross_validation(
            model,
            initial=CV_CONFIG["initial"],
            period=CV_CONFIG["period"],
            horizon=CV_CONFIG["horizon"],
            parallel=CV_CONFIG.get("parallel", "processes"),
            disable_tqdm=True,
        )
        metrics = performance_metrics(cv_df)
        mape = metrics["mape"] * 100
        logger.info(
            f"  CV MAPE: mean={mape.mean():.2f}% | "
            f"range=[{mape.min():.2f}%, {mape.max():.2f}%] | "
            f"{len(metrics)} windows"
        )
        stem = config["output_stem"]
        cv_df.to_parquet(_PROCESSED / f"{stem}_cv_raw.parquet", index=False)
        metrics.to_csv(_PROCESSED / f"{stem}_cv_metrics.csv", index=False)
        return metrics
    except Exception as e:
        logger.error(f"  CV failed: {e}")
        return pd.DataFrame()


def _run_forecast(model, config: dict, train_df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Generate 12-month forecast. Returns forecast-only rows."""
    future_df = build_future_df(train_df, config, master)
    forecast = model.predict(future_df)

    last_hist = train_df["ds"].max()
    fc = forecast[forecast["ds"] > last_hist][[
        "ds", "yhat", "yhat_lower", "yhat_upper", "trend", "yearly",
    ]].copy()
    fc.columns = [
        "date", "forecast", "forecast_lower", "forecast_upper",
        "trend_component", "seasonality_component",
    ]

    # Clip negative forecasts to zero (transaction volumes can't be negative)
    for col in ["forecast", "forecast_lower", "forecast_upper"]:
        fc[col] = fc[col].clip(lower=0)

    stem = config["output_stem"]
    fc.to_parquet(_PROCESSED / f"{stem}.parquet", index=False)
    fc.to_csv(_PROCESSED / f"{stem}.csv", index=False)

    # Full historical fit + forecast
    full = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()
    full.columns = ["date", "yhat", "yhat_lower", "yhat_upper", "trend"]
    full["actual"] = train_df.set_index("ds")["y"].reindex(full["date"]).values
    full.to_parquet(_PROCESSED / f"{stem}_full.parquet", index=False)
    full.to_csv(_PROCESSED / f"{stem}_full.csv", index=False)

    logger.info(f"  12-month forecast ({config['name']}):")
    for _, row in fc.iterrows():
        logger.info(
            f"    {row['date']:%b %Y}: {row['forecast']:.1f} "
            f"[{row['forecast_lower']:.1f}, {row['forecast_upper']:.1f}]"
        )
    return fc


def run_txn_volume_models(run_cv: bool = True) -> dict:
    """Run all three transaction volume models."""
    logger.info("=== Transaction Volume Models ===")

    data = load_all()
    master = build_master(data)

    # For UPI, the target is in npci_upi, already merged into master.
    # For CC/DC vol, the targets are in rbi_psi_cards, already in master.

    configs = [
        ("cc_vol", CC_VOL_CONFIG),
        ("dc_vol", DC_VOL_CONFIG),
        ("upi_vol", UPI_VOL_CONFIG),
    ]

    results = {}
    for key, config in configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Model: {config['name']}")
        logger.info(f"{'='*50}")

        train_df = build_training_df(master, config)

        model = _build_prophet(config, train_df)

        cv_metrics = pd.DataFrame()
        if run_cv:
            cv_metrics = _run_cv(model, config)

        forecast = _run_forecast(model, config, train_df, master)

        results[key] = {
            "model": model,
            "train_df": train_df,
            "forecast": forecast,
            "cv_metrics": cv_metrics,
        }

    # Auto-log
    try:
        for key, res in results.items():
            config = {"cc_vol": CC_VOL_CONFIG, "dc_vol": DC_VOL_CONFIG, "upi_vol": UPI_VOL_CONFIG}[key]
            log = RunLogger(f"txn_{key}")
            log.add("Target", config["target_col"])
            log.add("Training rows", len(res["train_df"]))
            if not res["cv_metrics"].empty:
                mape = res["cv_metrics"]["mape"] * 100
                log.add("CV MAPE mean", f"{mape.mean():.2f}%")
            fc = res["forecast"]
            log.add("Feb 2027 forecast", f"{fc['forecast'].iloc[-1]:.1f}")
            log.add("90% CI", f"[{fc['forecast_lower'].iloc[-1]:.1f}, {fc['forecast_upper'].iloc[-1]:.1f}]")
            log.add_section("Regressors", [f"{r.col} (lag={r.lag})" for r in config["regressors"]] or ["None (trend + seasonality only)"])
            log.save()
    except Exception:
        pass

    logger.info("\n=== Transaction Volume Models Complete ===")
    return results


if __name__ == "__main__":
    import sys
    run_cv = "--no-cv" not in sys.argv
    results = run_txn_volume_models(run_cv=run_cv)

    print("\n" + "=" * 55)
    print("TRANSACTION VOLUME FORECAST SUMMARY")
    print("=" * 55)
    for key, res in results.items():
        config_name = res["train_df"].columns  # just for the label
        fc = res["forecast"]
        print(f"\n{key.upper()}:")
        print(f"  Training rows: {len(res['train_df'])}")
        if not res["cv_metrics"].empty:
            mape = res["cv_metrics"]["mape"] * 100
            print(f"  CV MAPE: mean={mape.mean():.2f}% | range=[{mape.min():.2f}%, {mape.max():.2f}%]")
        print(f"  Feb 2027 forecast: {fc['forecast'].iloc[-1]:.1f}")
        print(f"  90% CI: [{fc['forecast_lower'].iloc[-1]:.1f}, {fc['forecast_upper'].iloc[-1]:.1f}]")
