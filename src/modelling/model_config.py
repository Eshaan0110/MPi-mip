"""
MIP Modelling Configuration
============================
Single source of truth for all model parameters.
To retune: edit here only — no code changes required.

Structure:
  PROPHET_BASE      — shared Prophet hyperparameters
  CC_CONFIG         — credit card model specification
  DC_CONFIG         — debit card model specification
  CV_CONFIG         — cross-validation settings (Rahul: initial>=48m, horizon=6m, step=6m)
  FORECAST_CONFIG   — forward forecast settings
  STRUCTURAL_EVENTS — changepoints / dummy regressors with confirmed dates
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ── Prophet base hyperparameters ───────────────────────────────────────────
PROPHET_BASE = {
    "yearly_seasonality": True,
    "weekly_seasonality": False,   # monthly data — no weekly pattern
    "daily_seasonality":  False,
    "seasonality_mode":   "multiplicative",   # growth series → multiplicative fits better
    "interval_width":     0.90,               # 90% confidence bands (Rahul spec)
}


# ── Cross-validation settings ──────────────────────────────────────────────
# Rahul: initial window >= 48 months, horizon 6 months, step 6 months
CV_CONFIG = {
    "initial":  "1461 days",   # 48 months ≈ 4 years
    "period":   "182 days",    # 6-month step
    "horizon":  "182 days",    # 6-month horizon
    "parallel": "processes",   # use multiprocessing for speed
}


# ── Forecast settings ──────────────────────────────────────────────────────
FORECAST_CONFIG = {
    "periods": 12,          # 12-month forward forecast (Rahul spec)
    "freq":    "MS",        # month-start frequency
}


# ── Structural events ──────────────────────────────────────────────────────
# Each event gets modelled as a Prophet changepoint OR a dummy regressor.
# "changepoint"  → passed to Prophet's changepoints list (automatic level shift)
# "dummy_pulse"  → 0/1 column, 1 only for the exact month(s) listed (one-off shock)
# "dummy_step"   → 0/1 column, 0 before date, 1 from date onwards (permanent shift)
STRUCTURAL_EVENTS = {
    "pmjdy_launch": {
        "date":     "2014-08-01",
        "type":     "changepoint",
        "models":   ["dc"],        # DC only — mass RuPay debit card issuance
        "notes":    "Jan Dhan Yojana launch. 386M RuPay debit cards issued 2014-2017.",
    },
    "demonetisation": {
        "date":     "2016-11-01",
        "type":     "changepoint",
        "models":   ["cc", "dc"],
        "notes":    "Cash ban. Forced digital adoption. Spike in both CC and DC issuance.",
    },
    "covid_shock": {
        "dates":    ["2020-04-01", "2020-05-01"],   # pulse — only these two months
        "type":     "dummy_pulse",
        "models":   ["cc", "dc"],
        "notes":    "COVID lockdown. New card issuance halted for ~2 months.",
    },
    "upi_inflection": {
        "date":     "2022-01-01",
        "type":     "changepoint",
        "models":   ["dc"],        # DC only — UPI displacing debit at POS
        "notes":    "UPI P2M volumes overtake debit card POS. Primary DC displacement driver.",
    },
    "card_validity_7yr": {
        "date":     "2022-07-01",
        "type":     "dummy_step",
        "models":   ["dc"],        # DC only — reduces attrition rate
        "notes":    "RBI Master Directions Jul 2022. Card validity extended 5→7 years. "
                    "Mechanically slows attrition of DC outstanding.",
    },
    "rbi_credit_tightening": {
        "date":     "2023-11-01",
        "type":     "dummy_step",
        "models":   ["cc"],        # CC only — unsecured lending risk weights raised
        "notes":    "RBI raised risk weights on unsecured consumer credit incl. credit cards. "
                    "Banks slowed CC issuance immediately. Deceleration visible in PSI.",
    },
}


# ── Credit card model specification ───────────────────────────────────────
@dataclass
class RegressorSpec:
    """Specification for one Prophet regressor."""
    col: str               # column name in the training DataFrame
    standardize: bool      # whether Prophet should standardise (True for continuous, False for dummies)
    lag: int               # months to lag before using (0 = same month)
    fill_method: str       # "zero" | "ffill" | "bfill" | "linear"
    mode: str              # "additive" | "multiplicative"
    notes: str = ""


CC_CONFIG = {
    "name": "credit_cards_outstanding_lakh",
    "target_col": "credit_cards_outstanding_lakh",
    "prophet_kwargs": {
        **PROPHET_BASE,
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
    },
    "regressors": [
        # Multicollinearity note: cc_vol, upi_qr, repo, cpi all correlate with time.
        # Including all four causes coefficient sign flips. Keep only the two with
        # the strongest distinct economic mechanisms; let trend+changepoints absorb rest.
        RegressorSpec(
            col="repo_rate",
            standardize=True,
            lag=6,
            fill_method="ffill",
            mode="additive",
            notes="Repo rate lag 6m. Strongest monetary policy signal. Changes "
                  "discontinuously (not collinear with smooth trend). "
                  "Rate tightening → banks slow unsecured issuance ~6m later.",
        ),
        RegressorSpec(
            col="upi_qr_lakh",
            standardize=True,
            lag=0,
            fill_method="zero",
            mode="additive",
            notes="UPI QR code count. Positive regressor for CC — RuPay credit "
                  "on UPI = CC issuance tailwind (Rahul Q2 answer). Zero-filled "
                  "pre-Sep 2020. Adds distinct post-2020 signal not captured by trend.",
        ),
    ],
    # Restrict training start to Jan 2013 — post-GFC recovery complete.
    # Pre-2013 CC series was declining (GFC aftermath); including it inflates
    # CV MAPE because early windows try to forecast recovery from contraction data.
    # The model is for forward forecasting a growth market, not GFC dynamics.
    "training_start": "2013-01-01",
    "structural_events": ["demonetisation", "covid_shock", "rbi_credit_tightening"],
    "extra_changepoints": [],  # GFC changepoint removed (pre training_start)
    "output_stem": "forecast_cc",
}


# ── Debit card model specification ────────────────────────────────────────
DC_CONFIG = {
    "name": "debit_cards_outstanding_lakh",
    "target_col": "debit_cards_outstanding_lakh",
    "prophet_kwargs": {
        **PROPHET_BASE,
        # DC has a clear structural break in 2019 and ongoing UPI displacement.
        # Additive seasonality is more stable when regressors are also additive
        # and the series has multiple hard breaks. Prevents LBFGS optimizer failure.
        "seasonality_mode":        "additive",
        "changepoint_prior_scale": 0.1,
        "seasonality_prior_scale": 10.0,
    },
    "regressors": [
        RegressorSpec(
            col="debit_card_vol_lakh",
            standardize=True,
            lag=0,
            fill_method="linear",
            mode="additive",
            notes="DC transaction volume. Falling usage informs outstanding trajectory.",
        ),
        # NOTE: UPI total volume as a regressor shows wrong sign (+) because the
        # Jan 2022 changepoint is already capturing the UPI displacement effect.
        # Using debit_card_pos_vol_lakh instead — this IS the direct displacement
        # signal (DC POS swipes falling) and has a cleaner relationship.
        RegressorSpec(
            col="debit_card_pos_vol_lakh",
            standardize=True,
            lag=0,
            fill_method="zero",       # zero pre-Nov 2019 (new PSI format only)
            mode="additive",
            notes="DC POS transaction volume. Direct measure of card swipes at merchant "
                  "terminals. Falling as UPI displaces. Cleaner signal than total UPI "
                  "volume for modelling DC outstanding (r=-0.812 with UPI).",
        ),
    ],
    "structural_events": [
        "pmjdy_launch",
        "demonetisation",
        "covid_shock",
        "upi_inflection",
        "card_validity_7yr",
    ],
    "extra_changepoints": ["2019-11-01"],  # PSI definitional change — series break
    "output_stem": "forecast_dc",
}