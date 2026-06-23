"""
FIX F5: Improve residual bucket model specification
====================================================
Current: changepoint_prior_scale=0.01 (extremely rigid), no changepoints.
Problem: The residual absorbs ALL structural changes from non-modelled banks
(mergers, exits, new entrants) but the rigid trend can't capture them.

Fix: Add merger dates as changepoints, increase flexibility.
"""
# Proposed replacement for RESIDUAL_PROPHET_CONFIG in bank_config.py:

RESIDUAL_PROPHET_CONFIG_FIXED = {
    "yearly_seasonality":      True,
    "weekly_seasonality":      False,
    "daily_seasonality":       False,
    "seasonality_mode":        "additive",
    "interval_width":          0.90,
    "changepoint_prior_scale": 0.05,   # was 0.01 — too rigid for a bucket with mergers
    "seasonality_prior_scale": 5.0,
    # Add explicit changepoints for events that affect the residual bucket
    "changepoints": [
        "2017-04-01",  # SBI associate mergers (3 banks leave residual → into SBI)
        "2020-04-01",  # Mega merger round (4 banks merge)
        "2023-03-01",  # Citibank exit
    ],
}

# Note: These changepoints need to be filtered to the training window
# in the _run_residual_model() function, same as bank models.
