"""
Axiom Audit — Master Fix Runner
================================
Applies all 12 fixes and re-runs the pipeline to measure improvement.

This does NOT modify the original source files. Instead, it monkey-patches
the relevant configs/functions at runtime and writes results to
experiments/axiom_audit/results/.

Usage:
    uv run python experiments/axiom_audit/run_all_fixes.py
    uv run python experiments/axiom_audit/run_all_fixes.py --quick   # skip CV

Estimated runtime: ~15 min with CV, ~3 min without.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QUICK = "--quick" in sys.argv


def apply_fix_f11():
    """Fix step dummies being zeroed in forecast."""
    from src.modelling import data_prep
    from src.modelling.model_config import STRUCTURAL_EVENTS

    _original = data_prep.build_future_df

    def patched_build_future_df(train_df, config, master):
        future = _original(train_df, config, master)
        last_date = train_df["ds"].max()

        pulse_events = {
            f"event_{name}" for name, spec in STRUCTURAL_EVENTS.items()
            if spec["type"] == "dummy_pulse"
        }
        step_events = {
            f"event_{name}" for name, spec in STRUCTURAL_EVENTS.items()
            if spec["type"] == "dummy_step"
        }

        for col in [c for c in future.columns if c.startswith("event_")]:
            if col in pulse_events:
                future.loc[future["ds"] > last_date, col] = 0.0
            elif col in step_events:
                future.loc[future["ds"] > last_date, col] = 1.0

        return future

    data_prep.build_future_df = patched_build_future_df
    logger.info("[F11] Step dummies now persist in forecast period")


def apply_fix_f04():
    """Reduce DC vol changepoints from 25 to 10."""
    from src.modelling.model_config import DC_VOL_CONFIG
    DC_VOL_CONFIG["prophet_kwargs"]["n_changepoints"] = 10
    logger.info("[F04] DC vol n_changepoints reduced to 10")


def apply_fix_f05():
    """Improve residual model specification."""
    import src.modelling.bank_config as bc
    bc.RESIDUAL_PROPHET_CONFIG["changepoint_prior_scale"] = 0.05
    logger.info("[F05] Residual changepoint_prior_scale increased to 0.05")


def apply_fix_f01():
    """Replace fake ETS CIs with simulation-based intervals."""
    from experiments.axiom_audit.fix_f01_ets_ci import ETSWrapperFixed
    import src.modelling.bank_model as bm

    _original_fit_ets = bm._fit_ets_model

    def patched_fit_ets(bank_df, bank_name, card_type):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from src.modelling.bank_config import USE_LOG_TRANSFORM

        y = bank_df["y"].values
        n = len(y)
        model = ExponentialSmoothing(
            y, trend="add", seasonal="add", seasonal_periods=12,
            initialization_method="heuristic",
        )
        fit = model.fit(optimized=True)
        logger.info(f"    ETS (Holt-Winters) fitted: {n} months, AIC={fit.aic:.0f}")
        return ETSWrapperFixed(fit, bank_df, bank_name, card_type)

    bm._fit_ets_model = patched_fit_ets
    logger.info("[F01] ETS CIs now use simulation-based prediction intervals")


def apply_fix_f07():
    """Clip forward regressors at zero."""
    from src.modelling import data_prep

    _original = data_prep.build_future_df

    def patched_clip(train_df, config, master):
        future = _original(train_df, config, master)

        # Clip all regressor columns to >= 0 in forecast period
        last_date = train_df["ds"].max()
        mask = future["ds"] > last_date
        reg_cols = [c for c in future.columns if c not in ["ds", "y"] and not c.startswith("event_")]
        for col in reg_cols:
            if col in future.columns:
                future.loc[mask, col] = future.loc[mask, col].clip(lower=0)

        return future

    data_prep.build_future_df = patched_clip
    logger.info("[F07] Forward regressors clipped at zero")


def run_patched_aggregate():
    """Run aggregate models with all patches applied."""
    from src.modelling.aggregate_model import run_aggregate_model

    logger.info("\n" + "=" * 60)
    logger.info("RUNNING PATCHED AGGREGATE MODELS")
    logger.info("=" * 60)

    results = run_aggregate_model(run_cc=True, run_dc=True, run_cv=not QUICK)

    summary = {}
    for key, res in results.items():
        s = {"training_rows": len(res["train_df"])}
        if not res["cv_metrics"].empty:
            mape = res["cv_metrics"]["mape"] * 100
            s["cv_mape_mean"] = round(mape.mean(), 2)
            s["cv_mape_range"] = [round(mape.min(), 2), round(mape.max(), 2)]
        summary[key] = s

    return summary


def run_patched_banks():
    """Run bank models with all patches applied."""
    from src.modelling.bank_model import run_all_bank_models

    logger.info("\n" + "=" * 60)
    logger.info("RUNNING PATCHED BANK MODELS")
    logger.info("=" * 60)

    results = run_all_bank_models(run_cv=not QUICK)

    summary = {}
    for ct, res in results.items():
        cv = res["cv_results"]
        valid = [r for r in cv if r.get("mape_median") is not None]
        if valid:
            medians = [r["mape_median"] for r in valid]
            summary[ct] = {
                "bank_cv_median": round(pd.Series(medians).median(), 2),
                "bank_cv_range": [round(min(medians), 2), round(max(medians), 2)],
                "n_banks": len(valid),
            }
        summary[f"{ct}_coverage"] = res.get("coverage_pct")

    return summary


def main():
    logger.info("=" * 60)
    logger.info("AXIOM AUDIT — APPLYING ALL FIXES")
    logger.info("=" * 60)

    # Apply patches (order matters: F11 and F07 both patch build_future_df)
    apply_fix_f11()
    apply_fix_f04()
    apply_fix_f05()
    apply_fix_f01()
    # F07 must come after F11 since both patch build_future_df
    # (F07's patch wraps whatever build_future_df currently is)
    apply_fix_f07()

    # Run patched models
    agg_summary = run_patched_aggregate()
    bank_summary = run_patched_banks()

    # Combined results
    all_results = {
        "aggregate": agg_summary,
        "banks": bank_summary,
        "fixes_applied": ["F01", "F04", "F05", "F07", "F11"],
        "fixes_not_applied_here": {
            "F02": "ETS CV — run fix_f02_ets_cv.py separately",
            "F03": "Scale verification — manual audit",
            "F06": "Dynamic caps — run fix_f06_dynamic_caps.py separately",
            "F08": "Accuracy tiering — UI change",
            "F09": "Paytm reclass — config change",
            "F10": "OOS holdout — run fix_f10_oos_holdout.py separately",
            "F12": "Drift reconciliation — depends on F05 results",
        },
    }

    # Save
    with open(RESULTS_DIR / "patched_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Aggregate: {json.dumps(agg_summary, indent=2, default=str)}")
    logger.info(f"Banks: {json.dumps(bank_summary, indent=2, default=str)}")
    logger.info(f"\nFull results saved to {RESULTS_DIR / 'patched_results.json'}")


if __name__ == "__main__":
    main()
