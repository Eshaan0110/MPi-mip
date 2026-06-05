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
        "models":   ["cc", "dc", "upi"],
        "notes":    "COVID lockdown. Card issuance + transactions halted ~2 months. "
                    "UPI volume also dipped Apr-May 2020 before surging.",
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
        "models":   ["dc"],        # DC only -- reduces attrition rate
        # Source note (verified May 2026): The 5->7 year card validity extension
        # has NOT been found in the RBI Master Directions on Credit Card and Debit
        # Card Issuance and Conduct Directions, 2022 (RBI/2022-23/92, effective
        # Jul 1, 2022). Those Directions contain no clause on validity period.
        # The step dummy is RETAINED at Jul 2022 because:
        #   (a) The DC series shows a visible slope change around this date.
        #   (b) The fitted coefficient is +0.013 (plausible positive direction).
        #   (c) The RBI Master Directions on conduct DID take effect Jul 1, 2022,
        #       and may have had indirect effects on DC issuance/attrition.
        # The event label is renamed to reflect the actual confirmed event.
        # TODO: locate the specific circular on card validity if it exists.
        "notes":    "RBI Master Directions on Card Issuance and Conduct effective "
                    "Jul 1, 2022 (RBI/2022-23/92). Step dummy captures structural "
                    "change in DC outstanding slope at this date. The specific "
                    "5->7 year validity extension circular has not been located.",
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
        #
        # upi_qr_lakh DROPPED after QR ablation test (Axiom review, May 2026):
        #   - Rolling-CV MAPE with QR = 2.64%, without = 2.58% (QR worsens CV).
        #   - 12-month forecast delta: mean 1.84%, growing to -2.6% by Feb 2027.
        #   - The negative forecast direction contradicts the assumed positive
        #     RuPay-credit-on-UPI channel. Pending Rahul Q2 confirmation on whether
        #     RuPay-credit-on-UPI transactions are already inside credit_outstanding.
        #     Until confirmed, the sign is ambiguous and the regressor is dropped.
        #   - If Rahul Q2 confirms inclusion: re-add with lag=3 and re-run ablation.
        RegressorSpec(
            col="repo_rate",
            standardize=True,
            lag=9,
            fill_method="ffill",
            mode="additive",
            notes="Repo rate lag 9m. Lag selected by nested rolling CV (18 outer folds, "
                  "Jan 2013-Feb 2026): lag-9 chosen in 12/18 folds (67%); honest outer "
                  "MAPE 2.21% +/- 1.28%. Literature prior was 6m (direct cost-of-funds "
                  "pass-through), but CC APR is fixed so the channel runs through bank "
                  "risk appetite -- CV result takes precedence.",
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


# ── Transaction volume model specifications ───────────────────────────────
# These model card transaction volumes (lakh transactions/month), not
# cards outstanding. Same Prophet framework, different targets and regressors.

CC_VOL_CONFIG = {
    "name": "credit_card_vol_lakh",
    "target_col": "credit_card_vol_lakh",
    "prophet_kwargs": {
        **PROPHET_BASE,
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
    },
    "regressors": [
        RegressorSpec(
            col="credit_cards_outstanding_lakh",
            standardize=True,
            lag=0,
            fill_method="linear",
            mode="multiplicative",
            notes="More cards = more transactions. Multiplicative because the "
                  "effect is proportional to the base volume level.",
        ),
    ],
    "training_start": "2013-01-01",
    "structural_events": ["demonetisation", "covid_shock"],
    "extra_changepoints": [],
    "output_stem": "forecast_cc_vol",
}

DC_VOL_CONFIG = {
    "name": "debit_card_vol_lakh",
    "target_col": "debit_card_vol_lakh",
    "prophet_kwargs": {
        **PROPHET_BASE,
        "seasonality_mode": "additive",
        "changepoint_prior_scale": 0.1,
        "seasonality_prior_scale": 10.0,
    },
    # No UPI regressor -- UPI volume is so dominant that including it as a
    # linear regressor drives the forecast to zero. The UPI displacement
    # effect is better captured by the Jan-2022 changepoint (slope break)
    # and the Nov-2016 demonetisation changepoint (digitisation inflection).
    # The DC vol series peaked in 2019 and has been declining; Prophet's
    # piecewise trend handles this without an explicit regressor.
    "regressors": [],
    # Training starts Jan 2022 (UPI displacement regime only). Including
    # pre-2022 data contaminates the forecast: Prophet averages across a
    # 15-year growth regime and a 3-year decline regime, producing a posterior
    # trend slope with enormous variance. The Nov-2019 window produced a
    # 90% CI of [14, 846] at Feb 2027 -- a 35x range, indefensible for
    # production use.
    #
    # Post-Jan-2022 gives ~50 months of pure decline data. Shorter than
    # ideal for Prophet seasonality (< 4 full annual cycles), but the
    # trade-off is a dramatically tighter CI because the model is not
    # averaging across contradictory regimes. Seasonality prior is
    # loosened to compensate for the short window.
    "training_start": "2022-01-01",
    "structural_events": [],
    "extra_changepoints": [],
    "output_stem": "forecast_dc_vol",
}

UPI_VOL_CONFIG = {
    "name": "upi_volume_mn",
    "target_col": "upi_volume_mn",
    "prophet_kwargs": {
        **PROPHET_BASE,
        "seasonality_mode": "multiplicative",
        "changepoint_prior_scale": 0.15,     # aggressive — UPI has multiple regime changes
        "seasonality_prior_scale": 10.0,
    },
    "regressors": [],  # pure trend + seasonality — no meaningful exogenous regressors
    "training_start": None,  # use all available data (Apr 2016+)
    "structural_events": ["covid_shock"],
    "extra_changepoints": [
        "2018-01-01",   # Google Pay launch accelerated adoption
        "2020-10-01",   # post-COVID digital payments inflection
    ],
    "output_stem": "forecast_upi_vol",
}


# ── Data quality flags ─────────────────────────────────────────────────────
# Known suspect data points that must be cross-checked against RBI source
# before final model fitting. Do not remove entries here without confirming
# against the published PSI Excel file.
DATA_QUALITY_FLAGS = {
    "jun_2025_cc_dip": {
        "date":   "2025-06-01",
        "series": "credit_cards_outstanding_lakh",
        "value":  1109.69,          # lakh
        "issue":  "Single month-on-month decline (-0.80 lakh) in a window of "
                  "consistent +4 to +10 lakh growth. Jul 2025 rebounded +8.4 lakh "
                  "(double the normal pace), consistent with a one-month RBI "
                  "reporting correction where a prior revision pulled Jun down "
                  "and Jul caught up. Confirmed as a PSI reporting blip, not a "
                  "market event. No dummy required — Prophet's robust fitting "
                  "absorbs single-point outliers without contaminating the trend.",
        "status": "CONFIRMED — reporting blip, no action needed",
    },
}