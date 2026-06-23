"""
FIX F4: Reduce changepoints for DC Volume model (only 50 months of data)
=========================================================================
Prophet defaults to 25 changepoints. With 50 months of training data,
that's one changepoint every 2 months — massive overfitting risk.

Fix: Set n_changepoints=10 for DC_VOL_CONFIG.
"""
# Patch for model_config.py DC_VOL_CONFIG:
#
# BEFORE:
#   "prophet_kwargs": {
#       **PROPHET_BASE,
#       "seasonality_mode": "additive",
#       "changepoint_prior_scale": 0.1,
#       "seasonality_prior_scale": 10.0,
#   },
#
# AFTER:
#   "prophet_kwargs": {
#       **PROPHET_BASE,
#       "seasonality_mode": "additive",
#       "changepoint_prior_scale": 0.1,
#       "seasonality_prior_scale": 10.0,
#       "n_changepoints": 10,   # reduced from default 25 — only 50 months of data
#   },
