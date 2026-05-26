"""MIP Modelling subpackage.

Entry points:
  run_aggregate_model()    — fits CC and DC aggregate Prophet models, runs CV, saves forecasts
  run_bank_model(ct)       — fits top-20 bank-level models for 'cc' or 'dc', saves ground-up
  run_all_bank_models()    — runs both CC and DC bank-level models
"""
from src.modelling.aggregate_model import run_aggregate_model
from src.modelling.bank_model import run_bank_model, run_all_bank_models

__all__ = ["run_aggregate_model", "run_bank_model", "run_all_bank_models"]