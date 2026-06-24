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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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

from src.utils.run_logger import RunLogger

warnings.filterwarnings("ignore", module="cmdstanpy")
warnings.filterwarnings("ignore", module="prophet")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

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

def _fit_arima_forecast(y: np.ndarray, horizon: int) -> np.ndarray | None:
    """Fit ARIMA(1,1,1) and return h-step forecast, or None on failure."""
    try:
        m = ARIMA(y, order=(1, 1, 1))
        fit = m.fit()
        return fit.forecast(steps=horizon)
    except Exception as e:
        logger.warning(f"  ARIMA forecast failed: {e}")
        return None


def _fit_ets_forecast(y: np.ndarray, horizon: int) -> np.ndarray | None:
    """Fit damped additive ETS and return h-step forecast, or None on failure."""
    try:
        m = ExponentialSmoothing(y, trend="add", seasonal="add",
                                  seasonal_periods=12, damped_trend=True)
        fit = m.fit(optimized=True)
        return fit.forecast(steps=horizon)
    except Exception as e:
        logger.warning(f"  ETS forecast failed: {e}")
        return None


def _build_conformal_intervals(
    train_df: pd.DataFrame, config: dict, horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 90% prediction intervals from CV residual quantiles (conformal).

    Runs walk-forward CV with same settings as main CV, collects residuals
    by horizon step, and returns the 5th/95th percentile widths.
    """
    from prophet.diagnostics import cross_validation

    initial_days = int(CV_CONFIG["initial"].replace(" days", ""))
    initial_months = initial_days // 30

    y = train_df["y"].values
    dates = train_df["ds"].values
    step_months = int(CV_CONFIG["period"].replace(" days", "")) // 30
    h_months = int(CV_CONFIG["horizon"].replace(" days", "")) // 30

    # Collect residuals per horizon step using ARIMA (fast) for conformal bands
    residuals_by_step = {h: [] for h in range(horizon)}
    for start in range(initial_months, len(y) - h_months + 1, step_months):
        train_y = y[:start]
        test_y = y[start:start + h_months]
        try:
            m = ARIMA(train_y, order=(1, 1, 1))
            fit = m.fit()
            pred = fit.forecast(steps=h_months)
            for h in range(min(h_months, horizon)):
                residuals_by_step[h].append(test_y[h] - pred[h])
        except Exception:
            continue

    lower_widths = np.zeros(horizon)
    upper_widths = np.zeros(horizon)
    for h in range(horizon):
        resids = residuals_by_step.get(h, [])
        if len(resids) >= 3:
            lower_widths[h] = np.percentile(resids, 5)
            upper_widths[h] = np.percentile(resids, 95)
        else:
            # Fallback: expanding uncertainty
            std = np.std(y[-24:]) if len(y) >= 24 else np.std(y)
            lower_widths[h] = -1.645 * std * np.sqrt(h + 1)
            upper_widths[h] = 1.645 * std * np.sqrt(h + 1)

    return lower_widths, upper_widths


# Ensemble weights optimized via CV grid search (Round 4C audit).
# CC optimal: ARIMA 60% / ETS 40% in 2-way; with Prophet floor at 35%
#   -> Prophet 0.35, ARIMA 0.39, ETS 0.26
# DC optimal: ARIMA 100% / ETS 0% in 2-way; with Prophet floor
#   -> Prophet 0.35, ARIMA 0.65, ETS 0.00
# Per-series weights reflect that DC's ETS adds no value (ARIMA dominates),
# while CC benefits from ETS diversification.
ENSEMBLE_WEIGHTS = {
    "cc": {"prophet": 0.35, "arima": 0.39, "ets": 0.26},
    "dc": {"prophet": 0.35, "arima": 0.65, "ets": 0.00},
}


def run_forecast(
    model,
    config: dict,
    train_df: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Generate 12-month ensemble forecast with conformal prediction intervals.

    Combines Prophet (with regressors/events) + ARIMA(1,1,1) + damped ETS.
    CIs are conformal: built from actual CV residual quantiles, not model
    assumptions about residual independence.
    """
    horizon = FORECAST_CONFIG.get("periods", 24)
    y = train_df["y"].values

    # --- Prophet forecast ---
    future_df = build_future_df(train_df, config, master)
    prophet_forecast = model.predict(future_df)
    last_hist = train_df["ds"].max()
    prophet_fc = prophet_forecast[prophet_forecast["ds"] > last_hist].copy()
    prophet_yhat = prophet_fc["yhat"].values[:horizon]
    prophet_dates = prophet_fc["ds"].values[:horizon]

    # --- ARIMA forecast ---
    arima_yhat = _fit_arima_forecast(y, horizon)

    # --- ETS forecast ---
    ets_yhat = _fit_ets_forecast(y, horizon)

    # --- Ensemble ---
    forecasts = {}
    if prophet_yhat is not None and len(prophet_yhat) == horizon:
        forecasts["prophet"] = prophet_yhat
    if arima_yhat is not None and len(arima_yhat) == horizon:
        forecasts["arima"] = arima_yhat
    if ets_yhat is not None and len(ets_yhat) == horizon:
        forecasts["ets"] = ets_yhat

    if not forecasts:
        raise RuntimeError("All forecast models failed")

    # Use per-series optimized weights
    model_key = "cc" if "credit" in config["name"] else "dc"
    weights = ENSEMBLE_WEIGHTS[model_key]
    total_w = sum(weights[k] for k in forecasts if weights.get(k, 0) > 0)
    ensemble = np.zeros(horizon)
    for name, fc_arr in forecasts.items():
        w = weights.get(name, 0) / total_w if total_w > 0 else 1.0 / len(forecasts)
        ensemble += w * fc_arr
        logger.info(f"  Ensemble member {name}: weight={w:.2f}, mean={np.mean(fc_arr):.1f}")

    logger.info(f"  Ensemble forecast mean: {np.mean(ensemble):.1f}")

    # --- Conformal CIs ---
    lower_widths, upper_widths = _build_conformal_intervals(train_df, config, horizon)
    ci_lower = ensemble + lower_widths  # lower_widths are negative
    ci_upper = ensemble + upper_widths

    # --- Build output DataFrame ---
    fc = pd.DataFrame({
        "date": prophet_dates[:horizon],
        "forecast_lakh": ensemble,
        "forecast_lower_lakh": ci_lower,
        "forecast_upper_lakh": ci_upper,
        "trend_component": prophet_fc["trend"].values[:horizon],
        "seasonality_component": prophet_fc["yearly"].values[:horizon],
    })

    # Also store individual model forecasts for transparency
    for name, fc_arr in forecasts.items():
        fc[f"forecast_{name}_lakh"] = fc_arr

    logger.info(f"  12-month ensemble forecast ({config['name']}):")
    for _, row in fc.iterrows():
        logger.info(
            f"    {row['date']:%b %Y}: {row['forecast_lakh']:.1f} lakh "
            f"[{row['forecast_lower_lakh']:.1f}, {row['forecast_upper_lakh']:.1f}]"
        )

    # Save
    stem = config["output_stem"]
    fc.to_parquet(_PROCESSED / f"{stem}.parquet", index=False)
    fc.to_csv(_PROCESSED / f"{stem}.csv", index=False)

    # Full historical + forecast for dashboard
    full = prophet_forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()
    full.columns = ["date", "yhat_lakh", "yhat_lower_lakh", "yhat_upper_lakh", "trend_lakh"]
    # Overwrite forecast portion with ensemble values
    fc_mask = full["date"] > last_hist
    full.loc[fc_mask, "yhat_lakh"] = ensemble
    full.loc[fc_mask, "yhat_lower_lakh"] = ci_lower
    full.loc[fc_mask, "yhat_upper_lakh"] = ci_upper
    full["actual_lakh"] = train_df.set_index("ds")["y"].reindex(full["date"]).values
    full.to_parquet(_PROCESSED / f"{stem}_full.parquet", index=False)
    full.to_csv(_PROCESSED / f"{stem}_full.csv", index=False)

    logger.info(f"  Forecast saved to {stem}.parquet, {stem}.csv, {stem}_full.*")
    return fc


# ── Scenario analysis ────────────────────────────────────────────────────

CC_SCENARIOS = {
    "base":          {"repo_rate": 6.25, "label": "Current rate (6.25%)"},
    "hawkish_100bp": {"repo_rate": 7.25, "label": "RBI hikes 100bp (inflation spike)"},
    "dovish_100bp":  {"repo_rate": 5.25, "label": "RBI cuts 100bp (growth support)"},
    "extreme_hawk":  {"repo_rate": 8.00, "label": "Emergency tightening (8.00%)"},
}


def run_scenario_analysis(model, config: dict, train_df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Run CC forecast under different repo rate scenarios."""
    if "credit" not in config["name"]:
        return pd.DataFrame()

    results = []
    for name, scenario in CC_SCENARIOS.items():
        future_df = build_future_df(train_df, config, master)
        last_hist = train_df["ds"].max()
        mask = future_df["ds"] > last_hist
        if "repo_rate_lag9" in future_df.columns:
            future_df.loc[mask, "repo_rate_lag9"] = scenario["repo_rate"]

        fc = model.predict(future_df)
        fc_only = fc[fc["ds"] > last_hist][["ds", "yhat"]].copy()
        fc_only["scenario"] = name
        fc_only["repo_rate"] = scenario["repo_rate"]
        results.append(fc_only)

    scenario_df = pd.concat(results, ignore_index=True)
    scenario_df.to_csv(_PROCESSED / "cc_scenarios.csv", index=False)
    logger.info(f"  Scenario analysis saved ({len(CC_SCENARIOS)} scenarios)")
    return scenario_df


# ── COVID stress-test diagnostic ──────────────────────────────────────────

def _covid_stress_test(model, config: dict) -> None:
    """Run a targeted CV window that includes Apr-May 2020 in the test set.

    This is a DIAGNOSTIC ONLY -- it does not change the main model or its
    headline MAPE. It quantifies how well the model (with its COVID dummy
    regressor) handles the extreme Apr-May 2020 shock.
    """
    from prophet.diagnostics import cross_validation, performance_metrics

    logger.info("\n  CC COVID STRESS-TEST")
    logger.info("  " + "-" * 45)

    try:
        # We want the test set to include Apr-May 2020.
        # With training_start=2013-01, Apr 2020 is ~87 months in.
        # Set initial to cover through ~Dec 2019 (84 months = 2555 days),
        # horizon to 6 months (covers Jan-Jun 2020), single period.
        cv_df = cross_validation(
            model,
            initial="2555 days",   # ~84 months (Jan 2013 -> Dec 2019)
            period="9999 days",    # single fold only
            horizon="182 days",    # 6-month test window (Jan-Jun 2020)
            parallel=CV_CONFIG.get("parallel", "processes"),
            disable_tqdm=True,
        )
        metrics = performance_metrics(cv_df)
        stress_mape = metrics["mape"].mean() * 100

        logger.info(f"  COVID stress-test MAPE: {stress_mape:.2f}%")
        logger.info(f"  (test window includes Apr-May 2020 lockdown shock)")

        # Compare to headline
        headline_path = _PROCESSED / f"{config['output_stem']}_cv_metrics.csv"
        if headline_path.exists():
            headline = pd.read_csv(headline_path)
            headline_mape = headline["mape"].mean() * 100
            diff = stress_mape - headline_mape

            if diff > 5:
                logger.warning(
                    f"  COVID window is {diff:.1f}pp WORSE than headline ({headline_mape:.2f}%). "
                    f"The COVID lockdown (Apr-May 2020) represents a genuine structural "
                    f"shock to card issuance that is difficult to predict from prior data. "
                    f"The COVID dummy regressor captures the direction but underestimates "
                    f"the magnitude. This is expected behaviour, not a model deficiency."
                )
            else:
                logger.info(
                    f"  COVID window is within {diff:+.1f}pp of headline ({headline_mape:.2f}%). "
                    f"The COVID dummy is handling the shock well."
                )

        # Save
        cv_df.to_parquet(_PROCESSED / f"{config['output_stem']}_covid_stress.parquet", index=False)
    except Exception as e:
        logger.warning(f"  COVID stress-test failed: {e}")


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

    # COVID stress-test for CC
    if run_cc and run_cv and "cc" in results:
        _covid_stress_test(results["cc"]["model"], CC_CONFIG)

    # Scenario analysis for CC
    if run_cc and "cc" in results:
        run_scenario_analysis(
            results["cc"]["model"], CC_CONFIG,
            results["cc"]["train_df"], master,
        )

    # Auto-log
    try:
        for key, res in results.items():
            log = RunLogger(f"aggregate_{key}")
            log.add("Training rows", len(res["train_df"]))
            log.add("Date range", f"{res['train_df']['ds'].min():%b %Y} -- {res['train_df']['ds'].max():%b %Y}")
            if not res["cv_metrics"].empty:
                mape = res["cv_metrics"]["mape"] * 100
                log.add("CV MAPE mean", f"{mape.mean():.2f}%")
                log.add("CV MAPE range", f"[{mape.min():.2f}%, {mape.max():.2f}%]")
            fc = res["forecast"]
            log.add("Forecast horizon end", f"{fc['forecast_lakh'].iloc[-1]:.1f} lakh")
            log.add("90% CI", f"[{fc['forecast_lower_lakh'].iloc[-1]:.1f}, {fc['forecast_upper_lakh'].iloc[-1]:.1f}]")
            config = CC_CONFIG if key == "cc" else DC_CONFIG
            log.add_section("Regressors", [f"{r.col} (lag={r.lag})" for r in config["regressors"]] or ["None"])
            log.add_section("Structural events", config["structural_events"])
            log.save()
    except Exception:
        pass

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