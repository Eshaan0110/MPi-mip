"""
MIP Modelling -- Bank-Level Configuration
==========================================
All parameters for the bank-level ground-up model.
Edit here only -- no changes needed in model code.

Architecture:
  - Explicit bank lists for CC (10) and DC (15) individual models
  - Remaining banks aggregated into a residual bucket (PSI - top banks)
  - Individual forecasts + residual = ground-up India total
  - Cross-check against PSI aggregate to validate coverage and accuracy

Enhancement (Jun 2026):
  - Explicit bank lists replace auto-selection by average outstanding
  - Per-bank start dates aligned to stable regimes (post-merger where applicable)
  - log1p(y) transform for variance stabilisation
  - Per-bank changepoint_prior_scale tuning
  - Explicit merger step dummies for absorbed-entity banks
"""

from __future__ import annotations

import pandas as pd

# ── Scope ──────────────────────────────────────────────────────────────────
# Legacy auto-selection parameters (kept for backward compatibility)
TOP_N_ISSUERS = 20
MIN_MONTHS    = 48

# Explicit bank lists -- these override TOP_N_ISSUERS auto-selection.
# CC: 10 banks covering ~91% of India total.
# DC: 15 banks covering ~83% of India total.
CC_BANK_LIST: list[str] = [
    "HDFC Bank",
    "State Bank of India",
    "ICICI Bank",
    "Axis Bank",
    "Kotak Mahindra Bank",
    "IndusInd Bank",
    "Bank of Baroda",
    "Yes Bank",
    "Canara Bank",
    "HSBC",
]

DC_BANK_LIST: list[str] = [
    "State Bank of India",
    "Bank of Baroda",
    "Canara Bank",
    "HDFC Bank",
    "Union Bank of India",
    "Punjab National Bank",
    "Axis Bank",
    "Bank of India",
    "Kotak Mahindra Bank",
    "Indian Bank",
    "Central Bank of India",
    "UCO Bank",
    "ICICI Bank",
    "Indian Overseas Bank",
    "Paytm Payments Bank",
]


# ── Bank name canonical map ────────────────────────────────────────────────
BANK_NAME_ALIASES: dict[str, str] = {
    "Hdfc  Bank Ltd.":                        "HDFC Bank",
    "Citi Bank":                              "Citibank",
    "American Express Banking Corporation":   "American Express",
    "American Express Bkg. Corp.":            "American Express",
    "Hongkong And Shanghai Bkg Corpn":        "HSBC",
    "State Bank Of Hyderabad":                "State Bank Of Hyderabad",
    "State Bank Of Bikaner And Jaipur":       "State Bank Of Bikaner And Jaipur",
    "State Bank Of Travancore":               "State Bank Of Travancore",
    "Union  Bank Of India":                   "Union Bank Of India",
}


# ── Terminated banks ──────────────────────────────────────────────────────
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


# ── Per-bank training start dates ─────────────────────────────────────────
# Key: (bank_name, card_type) -> Timestamp
# Philosophy: use the longest STABLE regime, not the longest history.
# Merger banks start after the merger settles (typically 1-3 months after
# the legal effective date to allow for portfolio integration noise).
# Clean banks use the card-type default (CC=2013, DC=2017).

CC_LIVE_BANK_TRAIN_START = pd.Timestamp("2013-01-01")  # default for CC
DC_LIVE_BANK_TRAIN_START = pd.Timestamp("2017-01-01")  # default for DC
LIVE_BANK_TRAIN_START = DC_LIVE_BANK_TRAIN_START        # legacy alias

BANK_START_DATES: dict[tuple[str, str], pd.Timestamp] = {
    # CC overrides
    ("HDFC Bank",           "cc"): pd.Timestamp("2017-01-01"),  # pre-2017 growth regime differs
    ("State Bank of India", "cc"): pd.Timestamp("2017-04-01"),  # post SBI associate merger
    ("ICICI Bank",          "cc"): pd.Timestamp("2017-01-01"),  # pre-demonetisation trajectory differs
    ("Kotak Mahindra Bank", "cc"): pd.Timestamp("2018-01-01"),  # hypergrowth started 2018; flat before
    ("Bank of Baroda",      "cc"): pd.Timestamp("2019-04-01"),  # post Dena+Vijaya merger
    ("Yes Bank",            "cc"): pd.Timestamp("2020-06-01"),  # post moratorium reconstruction
    ("Canara Bank",         "cc"): pd.Timestamp("2020-04-01"),  # post Syndicate merger
    # Axis, IndusInd, HSBC: use CC default (2013-01-01) -- clean series

    # DC overrides
    ("State Bank of India", "dc"): pd.Timestamp("2017-04-01"),  # post associate merger
    ("Bank of Baroda",      "dc"): pd.Timestamp("2019-04-01"),  # post Dena+Vijaya merger
    ("Canara Bank",         "dc"): pd.Timestamp("2020-04-01"),  # post Syndicate merger
    ("Union Bank of India", "dc"): pd.Timestamp("2020-04-01"),  # post triple merger
    ("Punjab National Bank","dc"): pd.Timestamp("2020-04-01"),  # post OBC+United merger
    ("Indian Bank",         "dc"): pd.Timestamp("2020-04-01"),  # post Allahabad merger
    ("Paytm Payments Bank", "dc"): pd.Timestamp("2018-04-01"),  # launched Apr 2018
    # HDFC, Axis, BoI, Kotak, Central, UCO, ICICI, IOB: use DC default (2017-01-01)
}


# ── Merger step dummies (per bank) ────────────────────────────────────────
# For banks that absorbed other entities, an explicit step dummy at the
# merger date tells Prophet exactly where the portfolio jumped. This is
# more reliable than relying on automatic changepoint detection.
#
# REQUIREMENT: the training window must SPAN the merger date — i.e. there
# must be meaningful pre-merger observations in the training series.
# If the bank's start date (BANK_START_DATES) is on or after the merger
# date, the dummy is all-1s in training and is collinear with the intercept.
# Such dummies waste a regressor slot and can destabilise Stan's optimiser.
#
# Audit (Jun 2026): all previously configured dummies were non-functional
# because every affected bank's start date was set to the merger date itself.
# The dictionary is intentionally left empty. Re-populate if start dates are
# ever moved earlier than the merger date for a given bank.
BANK_MERGER_EVENTS: dict[tuple[str, str], dict] = {
    # Example (re-enable only if start date is set BEFORE the merger date):
    # ("Bank of Baroda", "cc"): {"date": "2019-04-01", "label": "merger_dena_vijaya"},
}


# ── Per-bank Prophet config overrides ─────────────────────────────────────
# Merger banks need higher changepoint_prior_scale to capture the step.
# Stable banks can use lower values for smoother fits.
BANK_PROPHET_OVERRIDES: dict[str, dict] = {
    # Merger banks: more flexible trend
    "Bank of Baroda":       {"changepoint_prior_scale": 0.15},
    "Canara Bank":          {"changepoint_prior_scale": 0.15},
    "Union Bank of India":  {"changepoint_prior_scale": 0.15},
    "Punjab National Bank": {"changepoint_prior_scale": 0.15},
    "Indian Bank":          {"changepoint_prior_scale": 0.15},
    "Yes Bank":             {"changepoint_prior_scale": 0.15},
    # Stable banks: tighter trend
    "HDFC Bank":            {"changepoint_prior_scale": 0.03},
    "State Bank of India":  {"changepoint_prior_scale": 0.03},
    "ICICI Bank":           {"changepoint_prior_scale": 0.03},
    "Axis Bank":            {"changepoint_prior_scale": 0.03},
    "IndusInd Bank":        {"changepoint_prior_scale": 0.03},
}


# ── Base Prophet config ──────────────────────────────────────────────────
BANK_PROPHET_CONFIG = {
    "yearly_seasonality":      True,
    "weekly_seasonality":      False,
    "daily_seasonality":       False,
    "seasonality_mode":        "additive",
    "interval_width":          0.90,
    "changepoint_prior_scale": 0.05,     # default; overridden per bank above
    "seasonality_prior_scale": 5.0,
}


# ── log1p transform ──────────────────────────────────────────────────────
# Applied to all bank models. Stabilises variance for series growing from
# thousands to millions. Prophet assumes constant-variance residuals;
# log-transform makes this assumption hold. Back-transform with expm1()
# after prediction.
USE_LOG_TRANSFORM = True


# ── Structural events (shared changepoints) ──────────────────────────────
BANK_CHANGEPOINTS = [
    "2014-08-01",   # PMJDY
    "2016-11-01",   # Demonetisation
    "2020-04-01",   # COVID
    "2022-01-01",   # UPI inflection
]

CC_BANK_CHANGEPOINTS = ["2016-11-01", "2020-04-01"]
DC_BANK_CHANGEPOINTS = ["2014-08-01", "2016-11-01", "2020-04-01", "2022-01-01"]


# ── Residual model config ────────────────────────────────────────────────
RESIDUAL_PROPHET_CONFIG = {
    "yearly_seasonality":      True,
    "weekly_seasonality":      False,
    "daily_seasonality":       False,
    "seasonality_mode":        "additive",
    "interval_width":          0.90,
    "changepoint_prior_scale": 0.01,
    "seasonality_prior_scale": 5.0,
}


# ── Forecast + CV settings ───────────────────────────────────────────────
BANK_FORECAST_PERIODS = 24   # 24 months: Jun 2025 → May 2027
BANK_FORECAST_FREQ    = "MS"

BANK_CV_CONFIG = {
    "initial":  "1095 days",   # 36 months
    "period":   "182 days",    # 6-month step
    "horizon":  "182 days",    # 6-month horizon
    "parallel": "threads",     # Windows: threads avoids cmdstanpy file-lock races
}


# ── Output paths ─────────────────────────────────────────────────────────
BANK_OUTPUT_DIR     = "bankwise_forecasts"
GROUNDUP_OUTPUT_DIR = "groundup"
