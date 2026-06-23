"""
FIX F7: Clip forward regressor projections at zero
===================================================
Linear extrapolation of declining series (e.g., DC POS volume) can
produce negative values within the 24-month forecast horizon.
Negative regressor inputs produce unpredictable Prophet behavior.

Patch for data_prep.py build_future_df(), line ~352.
"""
# BEFORE:
#   future.loc[idx, final_col] = proj_val + slope * i
#
# AFTER:
#   projected = proj_val + slope * i
#   future.loc[idx, final_col] = max(projected, 0.0)
#
# Additionally, for declining series, use multiplicative decay instead:
#
#   if slope < 0 and proj_val > 0:
#       # Multiplicative decay: can't go below 0
#       monthly_rate = slope / proj_val  # negative
#       decay = max(1 + monthly_rate, 0.5)  # floor at 50% monthly retention
#       future.loc[idx, final_col] = proj_val * (decay ** i)
#   else:
#       future.loc[idx, final_col] = max(proj_val + slope * i, 0.0)
