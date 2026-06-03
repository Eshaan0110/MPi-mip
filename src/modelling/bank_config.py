"""
MIP Modelling — Bank-Level Configuration
==========================================
All parameters for the bank-level ground-up model.
Edit here only — no changes needed in model code.

Architecture:
  - Top N issuers modelled individually with Prophet (trend + seasonality only)
  - Remaining banks aggregated into a residual bucket (simple trend model)
  - Individual forecasts + residual = ground-up India total
  - Cross-check against PSI aggregate to validate coverage and accuracy

Rahul spec:
  - Top 20 issuers per card type
  - <48 months data = out of scope for individual modelling (residual bucket)
  - Approach 2 for regressors: use historical fit only, no forward projection needed
  - Document coverage: what % of total market does modelled issuer data represent
"""

from __future__ import annotations

import pandas as pd

# ── Scope ──────────────────────────────────────────────────────────────────
TOP_N_ISSUERS = 20          # max individual models per card type
MIN_MONTHS    = 48          # minimum months to qualify for individual model

# ── Bank name canonical map ────────────────────────────────────────────────
# Resolves duplicate raw names from different RBI file formats into one
# canonical name. Key = raw bank_name in bankwise parquet, Value = canonical.
# Extend this when new duplicates are found.
BANK_NAME_ALIASES: dict[str, str] = {
    # HDFC appears under two names across format generations
    "Hdfc  Bank Ltd.":                        "HDFC Bank",
    # Citi appears under two names
    "Citi Bank":                              "Citibank",
    # American Express appears under three names
    "American Express Banking Corporation":   "American Express",
    "American Express Bkg. Corp.":            "American Express",
    # HSBC variants
    "Hongkong And Shanghai Bkg Corpn":        "HSBC",
    # SBI associates (merged into SBI in 2017 — keep separate pre-merger,
    # but flag that their series terminates at the merger date)
    "State Bank Of Hyderabad":                "State Bank Of Hyderabad",
    "State Bank Of Bikaner And Jaipur":       "State Bank Of Bikaner And Jaipur",
    "State Bank Of Travancore":               "State Bank Of Travancore",
    # HDFC DC variant
    "Hdfc  Bank Ltd.":                        "HDFC Bank",
    # Union Bank variants
    "Union  Bank Of India":                   "Union Bank Of India",
}

# Banks whose series terminates due to merger/exit — not excluded,
# but documented so the dashboard can annotate them.
# "reason" is for logging; "exit_date" is used to clip forecast output
# to zero after this date (the bank no longer issues cards independently).
TERMINATED_BANKS: dict[str, dict] = {
    "State Bank Of Hyderabad":           {"reason": "Merged into SBI, April 2017",              "exit_date": "2017-04-01"},
    "State Bank Of Bikaner And Jaipur":  {"reason": "Merged into SBI, April 2017",              "exit_date": "2017-04-01"},
    "State Bank Of Travancore":          {"reason": "Merged into SBI, April 2017",              "exit_date": "2017-04-01"},
    "Andhra Bank":                       {"reason": "Merged into Union Bank, April 2020",       "exit_date": "2020-04-01"},
    "Syndicate Bank":                    {"reason": "Merged into Canara Bank, April 2020",      "exit_date": "2020-04-01"},
    "Corporation Bank":                  {"reason": "Merged into Union Bank, April 2020",       "exit_date": "2020-04-01"},
    "United Bank Of India":              {"reason": "Merged into Punjab National Bank, April 2020", "exit_date": "2020-04-01"},
    "Citibank":                          {"reason": "Exited India retail, March 2023",          "exit_date": "2023-03-01"},
    "American Express":                  {"reason": "Restricted by RBI, April 2021",            "exit_date": "2021-04-01"},
}

# ── Prophet config for individual bank models ─────────────────────────────
# Simpler than the aggregate model — Approach 2 (no forward regressors).
# Trend + seasonality + structural events only.
# No regressors: bank-level series are too short and sparse for reliable
# regressor coefficients (Rahul: Approach 2 for bank-level).
BANK_PROPHET_CONFIG = {
    "yearly_seasonality":      True,
    "weekly_seasonality":      False,
    "daily_seasonality":       False,
    "seasonality_mode":        "additive",   # additive safer for shorter series
    "interval_width":          0.90,
    "changepoint_prior_scale": 0.05,         # conservative — avoid overfitting short series
    "seasonality_prior_scale": 5.0,
}

# ── Training windows for live banks (per card type) ───────────────────────
# Terminated banks (Andhra, Corporation, OBC etc.) are NOT truncated --
# their data ended pre-2020 and the full history is needed.
#
# CC: 2013-01-01.  Credit card programs were clean before 2017. PMJDY did
#   not affect CC data (it issued debit cards only). Including 2013-2016
#   adds 4 years of good growth data that improves trend estimation.
#
# DC: 2017-01-01.  Pre-2017 DC data is distorted by PMJDY mass issuance
#   (2014-2016) which produced a one-time structural jump unrelated to
#   organic growth. Including it makes CV folds straddle two regimes.
CC_LIVE_BANK_TRAIN_START = pd.Timestamp("2013-01-01")
DC_LIVE_BANK_TRAIN_START = pd.Timestamp("2017-01-01")

# Legacy alias -- kept for any code that imports the old name
LIVE_BANK_TRAIN_START = DC_LIVE_BANK_TRAIN_START

# ── Structural events applied to bank models ──────────────────────────────
# Same events as aggregate but simpler — changepoints only, no dummy regressors.
# Dummies require enough data on both sides to estimate; shorter bank series
# may not have this. Changepoints are more robust for shorter windows.
BANK_CHANGEPOINTS = [
    "2014-08-01",   # PMJDY — mass DC issuance (DC models only)
    "2016-11-01",   # Demonetisation
    "2020-04-01",   # COVID
    "2022-01-01",   # UPI inflection (DC models only)
]

CC_BANK_CHANGEPOINTS = ["2016-11-01", "2020-04-01"]
DC_BANK_CHANGEPOINTS = ["2014-08-01", "2016-11-01", "2020-04-01", "2022-01-01"]

# ── Residual model config ─────────────────────────────────────────────────
# For banks with <48 months OR outside top 20.
# Pure trend + seasonality — no changepoints (not enough data to anchor them).
RESIDUAL_PROPHET_CONFIG = {
    "yearly_seasonality":      True,
    "weekly_seasonality":      False,
    "daily_seasonality":       False,
    "seasonality_mode":        "additive",
    "interval_width":          0.90,
    "changepoint_prior_scale": 0.01,   # very conservative
    "seasonality_prior_scale": 5.0,
}

# ── Forecast settings ─────────────────────────────────────────────────────
BANK_FORECAST_PERIODS = 12   # months forward
BANK_FORECAST_FREQ    = "MS" # month-start

# ── Cross-validation (lighter than aggregate — shorter series) ────────────
# Initial window 36 months (shorter than aggregate 48m because bank series
# have less data). Horizon and step same as aggregate.
# parallel="threads" on Windows — "processes" causes cmdstanpy file-lock
# races ("Operation not permitted" errors) when multiple subprocesses share
# the same Stan working directory.
BANK_CV_CONFIG = {
    "initial":  "1095 days",   # 36 months
    "period":   "182 days",    # 6-month step
    "horizon":  "182 days",    # 6-month horizon
    "parallel": "threads",
}

# ── Output paths (relative to data/processed/) ────────────────────────────
BANK_OUTPUT_DIR     = "bankwise_forecasts"   # individual bank forecast files go here
GROUNDUP_OUTPUT_DIR = "groundup"             # aggregated ground-up outputs go here