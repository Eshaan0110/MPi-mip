"""
FIX F11: Step dummies zeroed in forecast period
================================================
The current code zeros ALL event_* columns in the forecast period,
but step dummies (rbi_credit_tightening, card_validity_7yr) should
persist as 1.0 in the forecast since they represent permanent shifts.

This fix patches data_prep.py's build_future_df to only zero pulse
dummies, keeping step dummies at their last training value.
"""
# This file documents the fix. The actual change is in data_prep.py.
#
# BEFORE (line 303-304):
#   for col in [c for c in future.columns if c.startswith("event_")]:
#       future.loc[future["ds"] > last_date, col] = 0.0
#
# AFTER:
#   from src.modelling.model_config import STRUCTURAL_EVENTS
#   pulse_events = {
#       f"event_{name}" for name, spec in STRUCTURAL_EVENTS.items()
#       if spec["type"] == "dummy_pulse"
#   }
#   for col in [c for c in future.columns if c.startswith("event_")]:
#       if col in pulse_events:
#           future.loc[future["ds"] > last_date, col] = 0.0
#       # Step dummies keep their last training value (1.0)
