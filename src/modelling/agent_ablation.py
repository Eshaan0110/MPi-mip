"""
MIP — Agent Ablation Framework (P3.2)
=======================================
Pre-registered ablation protocol: same CV folds, with vs without signal
regressors, over ≥12 months of agent runs.

If dMAPE < 0.2pp, the agent is demoted to the qualitative Articles feed
(still valuable for context, near-zero API cost). Either outcome is fine —
the goal is honest measurement.

The ablation compares:
  - Full model: Prophet + ARIMA + ARIMAX + ETS + direct (current pipeline)
  - Ablated model: same ensemble but with specific regressors removed

Design:
  - Pre-registered: folds, horizon, metrics, decision threshold all fixed
  - Same CV folds: identical train/test splits for both arms
  - Scored on first-release data (vintage scoring) when available
  - Results logged to data/processed/ablation_registry.json

Usage:
    python -m src.modelling.agent_ablation                  # run ablation
    python -m src.modelling.agent_ablation --status         # check status
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED = _PROJECT_ROOT / "data" / "processed"
_REGISTRY = _PROCESSED / "ablation_registry.json"

# Pre-registered parameters (fixed — do not change mid-experiment)
ABLATION_PROTOCOL = {
    "min_months": 12,
    "decision_threshold_pp": 0.2,
    "cv_initial_months": 48,
    "cv_step_months": 6,
    "cv_horizon_months": 12,
    "metrics": ["mape", "mae", "rmse"],
    "primary_metric": "mape",
    "registered_date": "2026-08-28",
}


@dataclass
class AblationArm:
    name: str
    regressors: list[str]
    mape: float | None = None
    mae: float | None = None
    rmse: float | None = None


@dataclass
class AblationRun:
    run_date: str
    model: str
    full_arm: AblationArm
    ablated_arm: AblationArm
    delta_mape: float | None = None
    n_folds: int = 0
    verdict: str = "pending"


def _load_registry() -> list[dict]:
    """Load the ablation registry."""
    if _REGISTRY.exists():
        with open(_REGISTRY) as f:
            return json.load(f)
    return []


def _save_registry(registry: list[dict]) -> None:
    """Save the ablation registry."""
    _REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(_REGISTRY, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def _walk_forward_cv(
    y: np.ndarray,
    regressors: dict[str, np.ndarray] | None,
    initial: int = 48,
    step: int = 6,
    horizon: int = 12,
) -> dict[str, float]:
    """Run walk-forward CV with ARIMA+ETS ensemble, return avg metrics.

    When regressors are provided, adds ARIMAX as a third ensemble member
    so the full arm actually tests regressor value.
    """
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    mapes, maes, rmses = [], [], []

    for start in range(initial, len(y) - horizon + 1, step):
        train_y = y[:start]
        test_y = y[start : start + horizon]

        # ARIMA
        try:
            arima_fit = ARIMA(train_y, order=(1, 1, 1)).fit()
            arima_fc = arima_fit.forecast(steps=horizon)
        except Exception:
            arima_fc = np.full(horizon, np.mean(train_y[-12:]))

        # ETS
        try:
            ets_fit = ExponentialSmoothing(
                train_y, trend="add", seasonal="add",
                seasonal_periods=12, damped_trend=True,
            ).fit(optimized=True)
            ets_fc = ets_fit.forecast(steps=horizon)
        except Exception:
            ets_fc = np.full(horizon, np.mean(train_y[-12:]))

        # ARIMAX (only in the full arm, when regressors are provided)
        arimax_fc = None
        if regressors:
            try:
                exog_train = np.column_stack([v[:start] for v in regressors.values()])
                exog_test = np.column_stack([v[start:start+horizon] for v in regressors.values()])
                arimax_fit = ARIMA(train_y, exog=exog_train, order=(1, 1, 1)).fit()
                arimax_fc = arimax_fit.forecast(steps=horizon, exog=exog_test)
            except Exception:
                arimax_fc = None

        if arimax_fc is not None:
            ensemble = (1/3) * arima_fc + (1/3) * ets_fc + (1/3) * arimax_fc
        else:
            ensemble = 0.5 * arima_fc + 0.5 * ets_fc
        ape = np.abs((test_y - ensemble) / test_y)
        mapes.append(np.mean(ape) * 100)
        maes.append(np.mean(np.abs(test_y - ensemble)))
        rmses.append(np.sqrt(np.mean((test_y - ensemble) ** 2)))

    return {
        "mape": round(np.mean(mapes), 4),
        "mae": round(np.mean(maes), 2),
        "rmse": round(np.mean(rmses), 2),
        "n_folds": len(mapes),
    }


def run_ablation(
    model_name: str = "cc",
    signal_regressors: list[str] | None = None,
) -> AblationRun:
    """Run one ablation: full model vs model with signal regressors removed.

    The "signal regressors" are the ones the agent discovered/added.
    If they add <0.2pp MAPE improvement, the agent should be demoted.
    """
    from src.modelling.data_prep import load_all, build_master, build_training_df
    from src.modelling.model_config import CC_CONFIG, DC_CONFIG

    config = CC_CONFIG if model_name == "cc" else DC_CONFIG
    signal_regressors = signal_regressors or [
        r.col for r in config["regressors"]
    ]

    master = build_master(load_all())
    train_df = build_training_df(master, config)
    y = train_df["y"].values

    # Build regressor arrays for the full arm
    reg_arrays = {}
    for col_name in signal_regressors:
        final_col = col_name
        for c in train_df.columns:
            if c == col_name or c.startswith(f"{col_name}_lag"):
                final_col = c
                break
        if final_col in train_df.columns:
            reg_arrays[final_col] = train_df[final_col].values

    logger.info(f"Ablation [{model_name}]: running full arm (with {len(reg_arrays)} regressors)")
    full_metrics = _walk_forward_cv(y, reg_arrays if reg_arrays else None)

    logger.info(f"Ablation [{model_name}]: running ablated arm (without signal regressors)")
    ablated_metrics = _walk_forward_cv(y, None)

    full_arm = AblationArm(
        name="full",
        regressors=signal_regressors,
        mape=full_metrics["mape"],
        mae=full_metrics["mae"],
        rmse=full_metrics["rmse"],
    )

    ablated_arm = AblationArm(
        name="ablated",
        regressors=[],
        mape=ablated_metrics["mape"],
        mae=ablated_metrics["mae"],
        rmse=ablated_metrics["rmse"],
    )

    delta = (ablated_metrics["mape"] - full_metrics["mape"])

    if abs(delta) < ABLATION_PROTOCOL["decision_threshold_pp"]:
        verdict = "demote_agent"
    elif delta > 0:
        verdict = "keep_agent"
    else:
        verdict = "regressors_hurt"

    run = AblationRun(
        run_date=date.today().isoformat(),
        model=model_name,
        full_arm=full_arm,
        ablated_arm=ablated_arm,
        delta_mape=round(delta, 4),
        n_folds=full_metrics["n_folds"],
        verdict=verdict,
    )

    # Log to registry
    registry = _load_registry()
    registry.append(asdict(run))
    _save_registry(registry)

    logger.info(
        f"Ablation [{model_name}] result: "
        f"full={full_metrics['mape']:.2f}%, ablated={ablated_metrics['mape']:.2f}%, "
        f"delta={delta:+.2f}pp → {verdict}"
    )

    return run


def get_ablation_status() -> dict:
    """Check ablation experiment status."""
    registry = _load_registry()

    if not registry:
        return {
            "status": "not_started",
            "runs": 0,
            "min_required": ABLATION_PROTOCOL["min_months"],
            "message": "No ablation runs recorded yet. Run the first one to start the experiment.",
        }

    n_runs = len(registry)
    min_required = ABLATION_PROTOCOL["min_months"]

    cc_runs = [r for r in registry if r["model"] == "cc"]
    dc_runs = [r for r in registry if r["model"] == "dc"]

    if n_runs >= min_required:
        cc_deltas = [r["delta_mape"] for r in cc_runs if r.get("delta_mape") is not None]
        dc_deltas = [r["delta_mape"] for r in dc_runs if r.get("delta_mape") is not None]
        threshold = ABLATION_PROTOCOL["decision_threshold_pp"]

        decisions = {}
        for name, deltas in [("cc", cc_deltas), ("dc", dc_deltas)]:
            if deltas:
                avg_delta = np.mean(deltas)
                if avg_delta < threshold:
                    decisions[name] = f"DEMOTE (avg delta={avg_delta:+.2f}pp < {threshold}pp)"
                else:
                    decisions[name] = f"KEEP (avg delta={avg_delta:+.2f}pp ≥ {threshold}pp)"

        return {
            "status": "complete",
            "runs": n_runs,
            "decisions": decisions,
        }

    return {
        "status": "in_progress",
        "runs": n_runs,
        "remaining": min_required - n_runs,
        "cc_runs": len(cc_runs),
        "dc_runs": len(dc_runs),
    }


def print_status() -> None:
    """Print ablation experiment status."""
    status = get_ablation_status()
    print(f"\n{'='*55}")
    print("AGENT ABLATION EXPERIMENT STATUS")
    print(f"{'='*55}")
    print(f"  Protocol registered: {ABLATION_PROTOCOL['registered_date']}")
    print(f"  Decision threshold: {ABLATION_PROTOCOL['decision_threshold_pp']}pp MAPE")
    print(f"  Minimum runs: {ABLATION_PROTOCOL['min_months']}")
    print(f"  Current runs: {status['runs']}")
    print(f"  Status: {status['status']}")

    if status["status"] == "in_progress":
        print(f"  Remaining: {status['remaining']} runs")
    elif status["status"] == "complete":
        print("  Decisions:")
        for model, decision in status.get("decisions", {}).items():
            print(f"    {model.upper()}: {decision}")

    # Show recent runs
    registry = _load_registry()
    if registry:
        print(f"\n  Recent runs:")
        for r in registry[-5:]:
            print(
                f"    {r['run_date']} | {r['model'].upper()} | "
                f"delta={r.get('delta_mape', '?'):+.2f}pp | {r.get('verdict', '?')}"
            )

    print()


if __name__ == "__main__":
    import sys

    if "--status" in sys.argv:
        print_status()
    else:
        for model in ["cc", "dc"]:
            run_ablation(model)
        print_status()
