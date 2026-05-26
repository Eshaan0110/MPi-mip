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
TERMINATED_BANKS: dict[str, str] = {
    "State Bank Of Hyderabad":           "Merged into SBI, April 2017",
    "State Bank Of Bikaner And Jaipur":  "Merged into SBI, April 2017",
    "State Bank Of Travancore":          "Merged into SBI, April 2017",
    "Andhra Bank":                       "Merged into Union Bank, April 2020",
    "Syndicate Bank":                    "Merged into Canara Bank, April 2020",
    "Corporation Bank":                  "Merged into Union Bank, April 2020",
    "United Bank Of India":              "Merged into Punjab National Bank, April 2020",
    "Citibank":                          "Exited India retail, March 2023",
    "American Express":                  "Restricted by RBI, April 2021",
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
BANK_CV_CONFIG = {
    "initial":  "1095 days",   # 36 months
    "period":   "182 days",    # 6-month step
    "horizon":  "182 days",    # 6-month horizon
    "parallel": "processes",
}

# ── Output paths (relative to data/processed/) ────────────────────────────
BANK_OUTPUT_DIR     = "bankwise_forecasts"   # individual bank forecast files go here
GROUNDUP_OUTPUT_DIR = "groundup"             # aggregated ground-up outputs go here