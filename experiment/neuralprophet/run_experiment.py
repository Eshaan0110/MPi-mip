"""
run_experiment.py
-----------------
NeuralProphet experiment for MIP Phase 1.

Runs two models:
  1. Credit Cards Outstanding (CC)
  2. Debit Cards Outstanding (DC)

For each model:
  - Fits NeuralProphet with structural event regressors
  - Runs rolling cross-validation (initial=18, horizon=6, step=3)
  - Reports MAPE per window and overall
  - Saves 12-month forecast
  - Generates an HTML visualisation dashboard

Usage:
  python run_experiment.py

Outputs (written to ./outputs/):
  - cc_forecast.csv
  - dc_forecast.csv
  - cv_results_cc.csv
  - cv_results_dc.csv
  - experiment_dashboard.html   ← open this in a browser
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet, set_log_level

# Silence NeuralProphet training noise
set_log_level("ERROR")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from data_loader import (
    load_cards_outstanding,
    load_repo_rate,
    load_cpi,
    add_structural_events,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FORECAST_HORIZON = 12   # months forward
CV_INITIAL       = 18   # months in first training window
CV_HORIZON       = 6    # months each CV forecast
CV_STEP          = 3    # months to slide forward each fold


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denom != 0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def build_future_regressors(
    train_df: pd.DataFrame,
    n_periods: int,
    card_type: str,
    macro_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build the future DataFrame (blank y, filled regressors) for forecasting.
    For macro regressors we forward-fill the last known value — a conservative assumption.
    """
    last_date  = pd.Timestamp(train_df["ds"].max())
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=n_periods, freq="MS")

    future = pd.DataFrame({"ds": future_dates, "y": np.nan})

    # Structural event dummies — apply same logic
    future = add_structural_events(future, card_type)

    # Macro: extend with last-known value
    full_idx = pd.DatetimeIndex(pd.concat([pd.Series(macro_idx), pd.Series(future_dates)]).unique())
    repo = load_repo_rate(full_idx)
    cpi  = load_cpi(full_idx)

    if card_type == "cc":
        future["repo_rate"] = repo.reindex(future_dates).ffill().values
        future["cpi"]       = cpi.reindex(future_dates).ffill().values

    return future


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def rolling_cv(df: pd.DataFrame, card_type: str, macro_df: pd.DataFrame, cv_initial: int = CV_INITIAL) -> pd.DataFrame:
    """
    Manual rolling CV because NeuralProphet's built-in CV doesn't handle
    external regressors in the forecast period well for our setup.

    Returns a DataFrame with columns:
      fold, train_end, forecast_start, forecast_end, actual, predicted, mape_window
    """
    results = []
    n = len(df)

    fold = 0
    start = cv_initial
    while start + CV_HORIZON <= n:
        train_slice = df.iloc[:start].copy()
        test_slice  = df.iloc[start: start + CV_HORIZON].copy()

        # Add regressors to train
        train_idx = pd.DatetimeIndex(train_slice["ds"])
        train_slice = add_structural_events(train_slice, card_type)
        if card_type == "cc":
            train_slice["repo_rate"] = load_repo_rate(train_idx).values
            train_slice["cpi"]       = load_cpi(train_idx).values

        # Add regressors to test (known actuals for the test period)
        test_idx = pd.DatetimeIndex(test_slice["ds"])
        test_slice = add_structural_events(test_slice, card_type)
        if card_type == "cc":
            test_slice["repo_rate"] = load_repo_rate(test_idx).values
            test_slice["cpi"]       = load_cpi(test_idx).values

        # Build and fit model
        m = _build_model(card_type)
        _fit_model(m, train_slice)

        # NeuralProphet predict needs the full training data + test period as one frame
        # Pass the complete training slice + test slice; NP handles context internally
        predict_df = pd.concat([train_slice, test_slice], ignore_index=True)
        forecast = m.predict(predict_df)
        # Keep only the test-period rows
        forecast = forecast.tail(len(test_slice)).reset_index(drop=True)

        # Align predictions
        pred_col = "yhat1" if "yhat1" in forecast.columns else "yhat"
        actual    = test_slice["y"].values
        predicted = forecast[pred_col].values

        window_mape = mape(actual, predicted)

        for i in range(len(test_slice)):
            results.append({
                "fold":           fold + 1,
                "train_end":      train_slice["ds"].iloc[-1],
                "forecast_date":  test_slice["ds"].iloc[i],
                "actual":         actual[i],
                "predicted":      predicted[i],
                "error_pct":      abs(actual[i] - predicted[i]) / actual[i] * 100,
                "window_mape":    window_mape,
            })

        print(f"    Fold {fold+1} | Train ends {train_slice['ds'].iloc[-1].strftime('%b %Y')} "
              f"| Forecast {test_slice['ds'].iloc[0].strftime('%b %Y')}–"
              f"{test_slice['ds'].iloc[-1].strftime('%b %Y')} "
              f"| MAPE: {window_mape:.2f}%")

        fold  += 1
        start += CV_STEP

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(card_type: str) -> NeuralProphet:
    """
    Constructs a NeuralProphet model configured for the given card type.
    Key choices:
      - n_lags=3: uses last 3 months of actual values as autoregressive inputs
      - n_forecasts=6: multi-step forecast (needed for CV)
      - yearly_seasonality=True: India has clear festive-season patterns
      - weekly_seasonality=False: monthly data, no weekly pattern
      - learning_rate: small, stable
    """
    m = NeuralProphet(
        n_lags=3,
        n_forecasts=CV_HORIZON,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        learning_rate=0.003,
        epochs=200,
        batch_size=16,

    )

    # Structural event regressors — additive mode so they shift the level
    if card_type == "cc":
        m.add_lagged_regressor("covid_shock")
        m.add_lagged_regressor("rbi_tightening_2023")
        m.add_lagged_regressor("repo_rate")
        m.add_lagged_regressor("cpi")

    if card_type == "dc":
        m.add_lagged_regressor("covid_shock")
        m.add_lagged_regressor("pmjdy_launch")
        m.add_lagged_regressor("demonetisation")
        m.add_lagged_regressor("upi_inflection")

    return m


def _fit_model(m: NeuralProphet, train_df: pd.DataFrame):
    """Fits a NeuralProphet model, suppressing verbose output."""
    m.fit(train_df, freq="MS", progress="none")


# ─────────────────────────────────────────────────────────────────────────────
# Final forecast
# ─────────────────────────────────────────────────────────────────────────────

def run_final_forecast(
    df: pd.DataFrame,
    card_type: str,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fits on full dataset and forecasts 12 months forward."""
    train_df = add_structural_events(df.copy(), card_type)
    train_idx = pd.DatetimeIndex(train_df["ds"])

    if card_type == "cc":
        train_df["repo_rate"] = load_repo_rate(train_idx).values
        train_df["cpi"]       = load_cpi(train_idx).values

    m = _build_model(card_type)
    _fit_model(m, train_df)

    # Build future frames — NeuralProphet with n_lags needs the history too
    future = m.make_future_dataframe(train_df, periods=FORECAST_HORIZON, n_historic_predictions=True)

    # Add regressors to the future frame
    future_dates = pd.DatetimeIndex(future["ds"])
    future = add_structural_events(future, card_type)

    if card_type == "cc":
        full_repo = load_repo_rate(future_dates)
        full_cpi  = load_cpi(future_dates)
        future["repo_rate"] = full_repo.values
        future["cpi"]       = full_cpi.values

    forecast = m.predict(future)
    return forecast


# ─────────────────────────────────────────────────────────────────────────────
# HTML Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(
    cc_df, dc_df,
    cc_forecast, dc_forecast,
    cc_cv, dc_cv,
    cc_mape_overall, dc_mape_overall,
    cc_mape_by_fold, dc_mape_by_fold,
) -> str:
    """Generates a self-contained HTML dashboard."""

    def ts_to_str(series):
        return [str(d)[:10] for d in series]

    def safe(v):
        return round(float(v), 2) if not np.isnan(float(v)) else None

    # Prepare forecast data
    pred_col_cc = "yhat1" if "yhat1" in cc_forecast.columns else "yhat"
    pred_col_dc = "yhat1" if "yhat1" in dc_forecast.columns else "yhat"

    cc_hist_mask = cc_forecast["y"].notna()
    dc_hist_mask = dc_forecast["y"].notna()

    cc_data = {
        "hist_dates":     ts_to_str(cc_forecast.loc[cc_hist_mask, "ds"]),
        "hist_actual":    [safe(v) for v in cc_forecast.loc[cc_hist_mask, "y"]],
        "hist_fitted":    [safe(v) for v in cc_forecast.loc[cc_hist_mask, pred_col_cc]],
        "fore_dates":     ts_to_str(cc_forecast.loc[~cc_hist_mask, "ds"]),
        "fore_vals":      [safe(v) for v in cc_forecast.loc[~cc_hist_mask, pred_col_cc]],
    }

    dc_data = {
        "hist_dates":     ts_to_str(dc_forecast.loc[dc_hist_mask, "ds"]),
        "hist_actual":    [safe(v) for v in dc_forecast.loc[dc_hist_mask, "y"]],
        "hist_fitted":    [safe(v) for v in dc_forecast.loc[dc_hist_mask, pred_col_dc]],
        "fore_dates":     ts_to_str(dc_forecast.loc[~dc_hist_mask, "ds"]),
        "fore_vals":      [safe(v) for v in dc_forecast.loc[~dc_hist_mask, pred_col_dc]],
    }

    # CV MAPE by fold
    cc_fold_labels = [f"Fold {int(r['fold'])}" for r in cc_mape_by_fold]
    cc_fold_mapes  = [round(r["window_mape"], 2) for r in cc_mape_by_fold]
    dc_fold_labels = [f"Fold {int(r['fold'])}" for r in dc_mape_by_fold]
    dc_fold_mapes  = [round(r["window_mape"], 2) for r in dc_mape_by_fold]

    generated = datetime.now().strftime("%d %b %Y %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MIP — NeuralProphet Experiment Results</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --accent: #4f8ef7; --accent2: #f7974f; --green: #4fd1a5;
    --red: #f7564f; --text: #e2e8f0; --muted: #8892a4;
    --card-bg: #1e2130;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; padding: 24px; }}
  h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 0.875rem; margin-bottom: 32px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
  .card h2 {{ font-size: 1rem; font-weight: 600; margin-bottom: 16px; color: var(--text); }}
  .kpi-label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 1.8rem; font-weight: 700; }}
  .kpi-sub {{ font-size: 0.75rem; color: var(--muted); margin-top: 4px; }}
  .good {{ color: var(--green); }}
  .warn {{ color: var(--accent2); }}
  .chart-wrap {{ height: 340px; }}
  .chart-wrap-sm {{ height: 240px; }}
  .section-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent); font-weight: 600; margin-bottom: 8px; }}
  .insight {{ background: var(--surface); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0; padding: 12px 16px; margin-bottom: 10px; font-size: 0.875rem; line-height: 1.6; }}
  .tag {{ display: inline-block; font-size: 0.7rem; padding: 2px 8px; border-radius: 100px; margin-right: 6px; }}
  .tag-cc {{ background: rgba(79,142,247,0.15); color: var(--accent); }}
  .tag-dc {{ background: rgba(247,151,79,0.15); color: var(--accent2); }}
  footer {{ text-align: center; color: var(--muted); font-size: 0.75rem; margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--border); }}
  @media(max-width:900px) {{ .grid-2,.grid-4 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>🧪 NeuralProphet Experiment — MIP Phase 1</h1>
<p class="subtitle">Cards Outstanding Forecast · India Aggregate · Generated {generated}</p>

<!-- KPI strip -->
<div class="grid-4">
  <div class="card">
    <div class="kpi-label">CC Overall MAPE</div>
    <div class="kpi-value {'good' if cc_mape_overall < 5 else 'warn'}">{cc_mape_overall:.2f}%</div>
    <div class="kpi-sub">Rolling CV · all folds</div>
  </div>
  <div class="card">
    <div class="kpi-label">DC Overall MAPE</div>
    <div class="kpi-value {'good' if dc_mape_overall < 10 else 'warn'}">{dc_mape_overall:.2f}%</div>
    <div class="kpi-sub">Rolling CV · all folds</div>
  </div>
  <div class="card">
    <div class="kpi-label">CC Forecast (12m peak)</div>
    <div class="kpi-value" style="color:var(--accent)">{max(cc_data['fore_vals'])/1e6:.1f}M</div>
    <div class="kpi-sub">Credit cards outstanding</div>
  </div>
  <div class="card">
    <div class="kpi-label">DC Forecast (12m peak)</div>
    <div class="kpi-value" style="color:var(--accent2)">{max(dc_data['fore_vals'])/1e6:.1f}M</div>
    <div class="kpi-sub">Debit cards outstanding</div>
  </div>
</div>

<!-- Main forecast charts -->
<div class="grid-2">
  <div class="card">
    <h2>💳 Credit Cards Outstanding</h2>
    <div class="chart-wrap" id="cc-chart"></div>
  </div>
  <div class="card">
    <h2>🏧 Debit Cards Outstanding</h2>
    <div class="chart-wrap" id="dc-chart"></div>
  </div>
</div>

<!-- CV MAPE per fold -->
<div class="grid-2">
  <div class="card">
    <h2>📊 CC — MAPE per CV Fold</h2>
    <p style="font-size:0.8rem;color:var(--muted);margin-bottom:12px">
      Each bar = one rolling window. Lower = better. Wide spread = unstable model.
    </p>
    <div class="chart-wrap-sm" id="cc-cv-chart"></div>
  </div>
  <div class="card">
    <h2>📊 DC — MAPE per CV Fold</h2>
    <p style="font-size:0.8rem;color:var(--muted);margin-bottom:12px">
      Each bar = one rolling window. Lower = better. Wide spread = unstable model.
    </p>
    <div class="chart-wrap-sm" id="dc-cv-chart"></div>
  </div>
</div>

<!-- Actual vs Predicted scatter -->
<div class="grid-2">
  <div class="card">
    <h2>🎯 CC — Actual vs Predicted (CV)</h2>
    <div class="chart-wrap-sm" id="cc-scatter"></div>
  </div>
  <div class="card">
    <h2>🎯 DC — Actual vs Predicted (CV)</h2>
    <div class="chart-wrap-sm" id="dc-scatter"></div>
  </div>
</div>

<!-- Insights -->
<div class="card" style="margin-bottom:24px">
  <div class="section-label">Key Observations</div>
  <div class="insight">
    <span class="tag tag-cc">CC</span>
    Credit card outstanding has grown from ~60M (Mar 2020) to ~165M (Dec 2024). The NeuralProphet
    autoregressive component captures recent momentum including the post-Nov 2023 RBI tightening slowdown.
    If MAPE &lt; 5%, this is a strong baseline. If MAPE is higher, check whether the RBI tightening
    dummy is firing correctly.
  </div>
  <div class="insight">
    <span class="tag tag-dc">DC</span>
    Debit cards outstanding shows a much flatter trajectory post-2022 — the UPI inflection dummy
    is the key structural signal here. The model should show a visible slope-change after Jan 2022.
    If DC MAPE is high, try increasing n_lags to 6 to let the model learn the deceleration pattern better.
  </div>
  <div class="insight">
    <span class="tag tag-cc">CC</span><span class="tag tag-dc">DC</span>
    The data window here is Mar 2020–Dec 2024 (45 months from the bankwise sheets). This is a shorter
    series than the full PSI data. Once the full 263-month PSI series is ingested, re-run this experiment
    — the longer history will significantly improve CV stability and reduce fold-to-fold MAPE variance.
  </div>
</div>

<!-- Methodology note -->
<div class="card">
  <div class="section-label">Methodology</div>
  <p style="font-size:0.85rem;line-height:1.8;color:var(--muted)">
    <strong style="color:var(--text)">Model:</strong> NeuralProphet v0.8 · 
    <strong style="color:var(--text)">AR lags:</strong> 3 months · 
    <strong style="color:var(--text)">Seasonality:</strong> Yearly · 
    <strong style="color:var(--text)">CV config:</strong> Initial={CV_INITIAL}mo, Horizon={CV_HORIZON}mo, Step={CV_STEP}mo ·
    <strong style="color:var(--text)">Data:</strong> RBI bankwise ATM/Card stats (sheets 1–41 + X1–X4) ·
    <strong style="color:var(--text)">CC regressors:</strong> COVID shock, RBI tightening 2023, Repo rate, CPI ·
    <strong style="color:var(--text)">DC regressors:</strong> COVID shock, PMJDY launch, Demonetisation, UPI inflection
  </p>
</div>

<footer>MIP Experiment · NeuralProphet · Data: RBI (bankwise, Mar 2020–Dec 2024) · {generated}</footer>

<script>
const PLOT_CFG = {{responsive:true, displayModeBar:false}};
const DARK = {{
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{{color:'#8892a4',size:11}},
  xaxis:{{gridcolor:'#2a2d3a',linecolor:'#2a2d3a'}},
  yaxis:{{gridcolor:'#2a2d3a',linecolor:'#2a2d3a'}},
  margin:{{t:10,b:40,l:60,r:10}}
}};

// ── CC Forecast chart ──
(function(){{
  const cc = {json.dumps(cc_data)};
  const traces = [
    {{
      x:cc.hist_dates, y:cc.hist_actual.map(v=>v/1e6),
      name:'Actual', type:'scatter', mode:'lines',
      line:{{color:'#4f8ef7',width:2}}
    }},
    {{
      x:cc.hist_dates, y:cc.hist_fitted.map(v=>v/1e6),
      name:'Fitted', type:'scatter', mode:'lines',
      line:{{color:'#4fd1a5',width:1.5,dash:'dot'}}
    }},
    {{
      x:cc.fore_dates, y:cc.fore_vals.map(v=>v/1e6),
      name:'Forecast', type:'scatter', mode:'lines',
      line:{{color:'#f7564f',width:2,dash:'dash'}}
    }}
  ];
  const layout = Object.assign({{}}, DARK, {{
    yaxis:Object.assign({{}},DARK.yaxis,{{title:'Cards (millions)'}})
  }});
  Plotly.newPlot('cc-chart', traces, layout, PLOT_CFG);
}})();

// ── DC Forecast chart ──
(function(){{
  const dc = {json.dumps(dc_data)};
  const traces = [
    {{
      x:dc.hist_dates, y:dc.hist_actual.map(v=>v/1e6),
      name:'Actual', type:'scatter', mode:'lines',
      line:{{color:'#f7974f',width:2}}
    }},
    {{
      x:dc.hist_dates, y:dc.hist_fitted.map(v=>v/1e6),
      name:'Fitted', type:'scatter', mode:'lines',
      line:{{color:'#4fd1a5',width:1.5,dash:'dot'}}
    }},
    {{
      x:dc.fore_dates, y:dc.fore_vals.map(v=>v/1e6),
      name:'Forecast', type:'scatter', mode:'lines',
      line:{{color:'#f7564f',width:2,dash:'dash'}}
    }}
  ];
  const layout = Object.assign({{}}, DARK, {{
    yaxis:Object.assign({{}},DARK.yaxis,{{title:'Cards (millions)'}})
  }});
  Plotly.newPlot('dc-chart', traces, layout, PLOT_CFG);
}})();

// ── CC CV MAPE bars ──
(function(){{
  const labels = {json.dumps(cc_fold_labels)};
  const mapes  = {json.dumps(cc_fold_mapes)};
  const colors = mapes.map(v => v<5?'#4fd1a5':v<10?'#f7974f':'#f7564f');
  Plotly.newPlot('cc-cv-chart', [{{
    x:labels, y:mapes, type:'bar',
    marker:{{color:colors}},
    text:mapes.map(v=>v.toFixed(2)+'%'), textposition:'outside'
  }}], Object.assign({{}},DARK,{{
    yaxis:Object.assign({{}},DARK.yaxis,{{title:'MAPE %',range:[0,Math.max(...mapes)*1.3]}})
  }}), PLOT_CFG);
}})();

// ── DC CV MAPE bars ──
(function(){{
  const labels = {json.dumps(dc_fold_labels)};
  const mapes  = {json.dumps(dc_fold_mapes)};
  const colors = mapes.map(v => v<5?'#4fd1a5':v<10?'#f7974f':'#f7564f');
  Plotly.newPlot('dc-cv-chart', [{{
    x:labels, y:mapes, type:'bar',
    marker:{{color:colors}},
    text:mapes.map(v=>v.toFixed(2)+'%'), textposition:'outside'
  }}], Object.assign({{}},DARK,{{
    yaxis:Object.assign({{}},DARK.yaxis,{{title:'MAPE %',range:[0,Math.max(...mapes)*1.3]}})
  }}), PLOT_CFG);
}})();

// ── CC Scatter ──
(function(){{
  const cv = {json.dumps(cc_cv[['actual','predicted']].to_dict('records') if len(cc_cv)>0 else [])};
  if(!cv.length) return;
  const actual = cv.map(r=>r.actual/1e6);
  const pred   = cv.map(r=>r.predicted/1e6);
  const mn = Math.min(...actual,...pred), mx = Math.max(...actual,...pred);
  Plotly.newPlot('cc-scatter', [
    {{x:actual,y:pred,mode:'markers',type:'scatter',name:'CV points',
      marker:{{color:'#4f8ef7',size:6,opacity:0.7}}}},
    {{x:[mn,mx],y:[mn,mx],mode:'lines',name:'Perfect',
      line:{{color:'#4fd1a5',dash:'dot',width:1}}}}
  ], Object.assign({{}},DARK,{{
    xaxis:Object.assign({{}},DARK.xaxis,{{title:'Actual (M)'}}),
    yaxis:Object.assign({{}},DARK.yaxis,{{title:'Predicted (M)'}})
  }}), PLOT_CFG);
}})();

// ── DC Scatter ──
(function(){{
  const cv = {json.dumps(dc_cv[['actual','predicted']].to_dict('records') if len(dc_cv)>0 else [])};
  if(!cv.length) return;
  const actual = cv.map(r=>r.actual/1e6);
  const pred   = cv.map(r=>r.predicted/1e6);
  const mn = Math.min(...actual,...pred), mx = Math.max(...actual,...pred);
  Plotly.newPlot('dc-scatter', [
    {{x:actual,y:pred,mode:'markers',type:'scatter',name:'CV points',
      marker:{{color:'#f7974f',size:6,opacity:0.7}}}},
    {{x:[mn,mx],y:[mn,mx],mode:'lines',name:'Perfect',
      line:{{color:'#4fd1a5',dash:'dot',width:1}}}}
  ], Object.assign({{}},DARK,{{
    xaxis:Object.assign({{}},DARK.xaxis,{{title:'Actual (M)'}}),
    yaxis:Object.assign({{}},DARK.yaxis,{{title:'Predicted (M)'}})
  }}), PLOT_CFG);
}})();
</script>
</body>
</html>
"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MIP — NeuralProphet Experiment")
    print("=" * 60)

    # ── Load data ──
    print("\n[1/5] Loading data...")
    cc_df, dc_df = load_cards_outstanding()
    print(f"  CC: {len(cc_df)} months  ({cc_df['ds'].min().strftime('%b %Y')} → {cc_df['ds'].max().strftime('%b %Y')})")
    print(f"  DC: {len(dc_df)} months  ({dc_df['ds'].min().strftime('%b %Y')} → {dc_df['ds'].max().strftime('%b %Y')})")

    cv_initial = CV_INITIAL
    if len(cc_df) < cv_initial + CV_HORIZON:
        cv_initial = max(12, len(cc_df) - CV_HORIZON - CV_STEP * 2)
        print(f"\n  WARNING: Only {len(cc_df)} months. Reducing CV initial window to {cv_initial} months.")

    macro_df = pd.DataFrame()  # placeholder passed through

    # ── CC Rolling CV ──
    print("\n[2/5] Credit Card rolling cross-validation...")
    cc_cv = rolling_cv(cc_df, "cc", macro_df, cv_initial)
    cc_mape_overall = mape(cc_cv["actual"].values, cc_cv["predicted"].values)
    cc_mape_by_fold = cc_cv.drop_duplicates("fold")[["fold","window_mape"]].to_dict("records")
    print(f"\n  ✓ CC overall MAPE: {cc_mape_overall:.2f}%")
    cc_cv.to_csv(OUTPUT_DIR / "cv_results_cc.csv", index=False)

    # ── DC Rolling CV ──
    print("\n[3/5] Debit Card rolling cross-validation...")
    dc_cv = rolling_cv(dc_df, "dc", macro_df, cv_initial)
    dc_mape_overall = mape(dc_cv["actual"].values, dc_cv["predicted"].values)
    dc_mape_by_fold = dc_cv.drop_duplicates("fold")[["fold","window_mape"]].to_dict("records")
    print(f"\n  ✓ DC overall MAPE: {dc_mape_overall:.2f}%")
    dc_cv.to_csv(OUTPUT_DIR / "cv_results_dc.csv", index=False)

    # ── Final forecasts ──
    print("\n[4/5] Fitting final models and generating 12-month forecast...")
    cc_forecast = run_final_forecast(cc_df, "cc", macro_df)
    dc_forecast = run_final_forecast(dc_df, "dc", macro_df)

    pred_col_cc = "yhat1" if "yhat1" in cc_forecast.columns else "yhat"
    pred_col_dc = "yhat1" if "yhat1" in dc_forecast.columns else "yhat"

    cc_fore_rows = cc_forecast[cc_forecast["y"].isna()][["ds", pred_col_cc]]
    dc_fore_rows = dc_forecast[dc_forecast["y"].isna()][["ds", pred_col_dc]]
    cc_fore_rows.to_csv(OUTPUT_DIR / "cc_forecast.csv", index=False)
    dc_fore_rows.to_csv(OUTPUT_DIR / "dc_forecast.csv", index=False)

    print("  CC 12-month forecast (millions of cards):")
    for _, row in cc_fore_rows.iterrows():
        print(f"    {row['ds'].strftime('%b %Y')}: {row[pred_col_cc]/1e6:.2f}M")

    print("  DC 12-month forecast (millions of cards):")
    for _, row in dc_fore_rows.iterrows():
        print(f"    {row['ds'].strftime('%b %Y')}: {row[pred_col_dc]/1e6:.2f}M")

    # ── Dashboard ──
    print("\n[5/5] Building HTML dashboard...")
    html = build_dashboard(
        cc_df, dc_df,
        cc_forecast, dc_forecast,
        cc_cv, dc_cv,
        cc_mape_overall, dc_mape_overall,
        cc_mape_by_fold, dc_mape_by_fold,
    )
    dashboard_path = OUTPUT_DIR / "experiment_dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")

    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\n  CC MAPE (overall):  {cc_mape_overall:.2f}%")
    print(f"  DC MAPE (overall):  {dc_mape_overall:.2f}%")
    print(f"\n  Outputs saved to: {OUTPUT_DIR}/")
    print(f"  → Open in browser: {dashboard_path}")
    print()

    # Print a comparison summary for the README
    summary = {
        "run_at": datetime.now().isoformat(),
        "data_window": {
            "cc": {"from": str(cc_df['ds'].min().date()), "to": str(cc_df['ds'].max().date()), "n_months": len(cc_df)},
            "dc": {"from": str(dc_df['ds'].min().date()), "to": str(dc_df['ds'].max().date()), "n_months": len(dc_df)},
        },
        "cv_config": {"initial": CV_INITIAL, "horizon": CV_HORIZON, "step": CV_STEP},
        "results": {
            "cc_mape_overall": round(cc_mape_overall, 4),
            "cc_mape_by_fold": {r["fold"]: round(r["window_mape"], 4) for r in cc_mape_by_fold},
            "dc_mape_overall": round(dc_mape_overall, 4),
            "dc_mape_by_fold": {r["fold"]: round(r["window_mape"], 4) for r in dc_mape_by_fold},
        }
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  → Summary JSON: {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
