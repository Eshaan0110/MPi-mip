"""
MIP Modelling — Aggregate Prophet Model Builder
================================================
Builds, validates, and saves the aggregate India-level Prophet models
for credit cards outstanding and debit cards outstanding.

This module handles:
  1. Model construction from config (CC_CONFIG / DC_CONFIG).
  2. Adding regressors and structural event dummies.
  3. Rolling cross-validation with MAPE reporting across all windows.
  4. 12-month forward forecast with 90% confidence intervals.
  5. Saving forecast outputs as parquet + CSV.
  6. Structured coefficient/component logging for the ever-learning model.

Run directly:
    uv run python -m src.modelling.aggregate_model

Or import:
    from src.modelling.aggregate_model import run_aggregate_model
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.modelling.model_config import (
    CC_CONFIG,
    DC_CONFIG,
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

warnings.filterwarnings("ignore")  # suppress Stan/Prophet deprecation noise

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED    = _PROJECT_ROOT / "data" / "processed"
_PROCESSED.mkdir(parents=True, exist_ok=True)


# ── Model builder ──────────────────────────────────────────────────────────

def build_prophet_model(config: dict, train_df: pd.DataFrame):
    """Instantiate and fit a Prophet model from config.

    Args:
        config:   CC_CONFIG or DC_CONFIG.
        train_df: Prophet-ready DataFrame (ds, y, regressor columns).

    Returns:
        Fitted Prophet model.
    """
    from prophet import Prophet

    model_key = "cc" if "credit" in config["name"] else "dc"

    # Explicit changepoints: automatic + from config + structural events
    changepoint_dates = list(config.get("extra_changepoints", []))
    for event_name, spec in STRUCTURAL_EVENTS.items():
        if spec["type"] == "changepoint" and model_key in spec["models"]:
            changepoint_dates.append(spec["date"])

    prophet_kwargs = dict(config["prophet_kwargs"])
    if changepoint_dates:
        # Filter to dates within the training window
        valid_cps = [
            pd.Timestamp(d) for d in changepoint_dates
            if train_df["ds"].min() < pd.Timestamp(d) < train_df["ds"].max()
        ]
        if valid_cps:
            prophet_kwargs["changepoints"] = valid_cps
            logger.info(f"  Explicit changepoints: {[d.strftime('%b %Y') for d in valid_cps]}")

    m = Prophet(**prophet_kwargs)

    # Add regressors
    regressors: list[RegressorSpec] = config["regressors"]
    for spec in regressors:
        final_col = f"{spec.col}_lag{spec.lag}" if spec.lag > 0 else spec.col
        if final_col not in train_df.columns:
            logger.warning(f"  Regressor column '{final_col}' missing — skipping.")
            continue
        m.add_regressor(
            final_col,
            standardize=spec.standardize,
            mode=spec.mode,
        )
        logger.info(f"  Regressor added: {final_col} (lag={spec.lag}, mode={spec.mode})")

    # Add structural event dummies (pulse and step types only)
    event_cols = [c for c in train_df.columns if c.startswith("event_")]
    for col in event_cols:
        m.add_regressor(col, standardize=False, mode="additive")
        logger.info(f"  Event dummy added: {col}")

    # Fit
    logger.info(f"  Fitting model on {len(train_df)} rows...")
    m.fit(train_df)
    logger.success(f"  Model fitted: {config['name']}")

    return m


# ── Cross-validation ───────────────────────────────────────────────────────

def run_cross_validation(model, config: dict) -> pd.DataFrame:
    """Run rolling cross-validation and return performance metrics.

    Reports MAPE across all CV windows — not just best-case.
    """
    from prophet.diagnostics import cross_validation, performance_metrics

    logger.info(
        f"  Running cross-validation: "
        f"initial={CV_CONFIG['initial']}, "
        f"period={CV_CONFIG['period']}, "
        f"horizon={CV_CONFIG['horizon']}"
    )

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

        mape_mean = metrics["mape"].mean() * 100
        mape_min  = metrics["mape"].min()  * 100
        mape_max  = metrics["mape"].max()  * 100

        logger.info(
            f"  CV MAPE — mean: {mape_mean:.2f}% | "
            f"range: [{mape_min:.2f}%, {mape_max:.2f}%] | "
            f"windows: {len(metrics)}"
        )

        # Save CV results
        stem = config["output_stem"]
        cv_df.to_parquet(_PROCESSED / f"{stem}_cv_raw.parquet", index=False)
        metrics.to_csv(_PROCESSED / f"{stem}_cv_metrics.csv", index=False)
        logger.info(f"  CV results saved to {stem}_cv_raw.parquet and {stem}_cv_metrics.csv")

        return metrics

    except Exception as e:
        logger.error(f"  Cross-validation failed: {e}")
        return pd.DataFrame()


# ── Coefficient logging ────────────────────────────────────────────────────

def log_model_coefficients(model, config: dict) -> pd.DataFrame:
    """Extract and log key model coefficients for the ever-learning model spec."""
    params = model.params

    records = []

    # Regressor coefficients (beta params)
    if hasattr(model, "extra_regressors") and model.extra_regressors:
        for reg_name in model.extra_regressors:
            beta = params.get("beta", np.array([[]]))
            reg_idx = list(model.extra_regressors.keys()).index(reg_name)
            if beta.shape[1] > reg_idx:
                coef = float(np.mean(beta[:, reg_idx]))
                records.append({
                    "model":      config["name"],
                    "component":  reg_name,
                    "type":       "regressor_beta",
                    "mean_coeff": coef,
                })

    # Trend changepoint magnitudes
    if "delta" in params:
        delta = np.mean(params["delta"], axis=0)
        for i, d in enumerate(delta):
            if abs(d) > 0.01:   # only log meaningful changepoints
                records.append({
                    "model":      config["name"],
                    "component":  f"changepoint_{i}",
                    "type":       "trend_delta",
                    "mean_coeff": float(d),
                })

    coeff_df = pd.DataFrame(records)
    if not coeff_df.empty:
        logger.info(f"  Model coefficients ({config['name']}):")
        for _, row in coeff_df.iterrows():
            logger.info(f"    {row['component']}: {row['mean_coeff']:+.4f}")

    # Save
    stem = config["output_stem"]
    coeff_df.to_csv(_PROCESSED / f"{stem}_coefficients.csv", index=False)
    return coeff_df


# ── Forecast ───────────────────────────────────────────────────────────────

def run_forecast(
    model,
    config: dict,
    train_df: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Generate 12-month forward forecast with 90% confidence intervals."""
    future_df = build_future_df(train_df, config, master)
    forecast  = model.predict(future_df)

    # Extract forecast-only rows
    last_hist = train_df["ds"].max()
    fc = forecast[forecast["ds"] > last_hist][[
        "ds", "yhat", "yhat_lower", "yhat_upper",
        "trend", "yearly",
    ]].copy()

    fc.columns = [
        "date", "forecast_lakh", "forecast_lower_lakh", "forecast_upper_lakh",
        "trend_component", "seasonality_component",
    ]

    logger.info(f"  12-month forecast ({config['name']}):")
    for _, row in fc.iterrows():
        logger.info(
            f"    {row['date']:%b %Y}: {row['forecast_lakh']:.1f} lakh "
            f"[{row['forecast_lower_lakh']:.1f}, {row['forecast_upper_lakh']:.1f}]"
        )

    # Save
    stem = config["output_stem"]
    fc.to_parquet(_PROCESSED / f"{stem}.parquet", index=False)
    fc.to_csv(_PROCESSED / f"{stem}.csv", index=False)

    # Also save full historical + forecast for dashboard
    full = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()
    full.columns = ["date", "yhat_lakh", "yhat_lower_lakh", "yhat_upper_lakh", "trend_lakh"]
    full["actual_lakh"] = train_df.set_index("ds")["y"].reindex(full["date"]).values
    full.to_parquet(_PROCESSED / f"{stem}_full.parquet", index=False)
    full.to_csv(_PROCESSED / f"{stem}_full.csv", index=False)

    logger.info(f"  Forecast saved to {stem}.parquet, {stem}.csv, {stem}_full.*")
    return fc


# ── Main entry point ───────────────────────────────────────────────────────

def run_aggregate_model(
    run_cc: bool = True,
    run_dc: bool = True,
    run_cv: bool = True,
) -> dict:
    """Run aggregate Prophet models for CC and/or DC outstanding.

    Args:
        run_cc: Run credit card model.
        run_dc: Run debit card model.
        run_cv: Run cross-validation (adds ~5-10 min).

    Returns:
        Dict with keys 'cc' and/or 'dc', each containing:
            model, forecast, cv_metrics, coefficients
    """
    logger.info("═══ MIP Aggregate Model Run ═══")

    data   = load_all()
    master = build_master(data)
    results = {}

    configs = []
    if run_cc:
        configs.append(("cc", CC_CONFIG))
    if run_dc:
        configs.append(("dc", DC_CONFIG))

    for key, config in configs:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Model: {config['name']}")
        logger.info(f"{'─'*50}")

        train_df = build_training_df(master, config)

        model = build_prophet_model(config, train_df)

        cv_metrics = pd.DataFrame()
        if run_cv:
            cv_metrics = run_cross_validation(model, config)

        coefficients = log_model_coefficients(model, config)
        forecast     = run_forecast(model, config, train_df, master)

        results[key] = {
            "model":        model,
            "train_df":     train_df,
            "forecast":     forecast,
            "cv_metrics":   cv_metrics,
            "coefficients": coefficients,
        }

    logger.success("\n═══ Aggregate model run complete ═══")
    return results


if __name__ == "__main__":
    import sys
    run_cv = "--no-cv" not in sys.argv
    results = run_aggregate_model(run_cc=True, run_dc=True, run_cv=run_cv)

    for key, res in results.items():
        print(f"\n{'='*50}")
        print(f"{key.upper()} MODEL SUMMARY")
        print(f"{'='*50}")
        print(f"Training rows: {len(res['train_df'])}")

        if not res["cv_metrics"].empty:
            mape = res["cv_metrics"]["mape"] * 100
            print(f"CV MAPE: mean={mape.mean():.2f}% | range=[{mape.min():.2f}%, {mape.max():.2f}%]")

        print(f"\n12-month forecast:")
        print(res["forecast"][["date","forecast_lakh","forecast_lower_lakh","forecast_upper_lakh"]].to_string(index=False))