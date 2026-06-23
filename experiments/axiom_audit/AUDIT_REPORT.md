# MIP Phase 1 — Axiom Technical Audit Report

**Date:** June 2026
**Reviewer:** Axiom Analytics (Internal QA)
**Scope:** Full ML pipeline audit — methodology, accuracy, configuration, data handling

---

## Executive Summary

The MIP forecasting pipeline is **competent for a v1 delivery** but contains **12 material issues** ranging from methodological gaps to silent accuracy leaks. Estimated aggregate improvement if all fixes are applied: **2–5pp MAPE reduction on bank-level median, tighter CIs across all models.**

---

## FINDING 1: ETS Confidence Intervals Are Fabricated

**Severity:** CRITICAL
**Location:** `src/modelling/bank_model.py:205-208`

```python
# Simple CI: ±10% for near-term, widening to ±20% at 24 months
spread = np.linspace(0.05, 0.20, len(future))
yhat_lower = yhat * (1 - spread)
yhat_upper = yhat * (1 + spread)
```

**Problem:** The ETS wrapper generates confidence intervals by applying a fixed ±5%→±20% linear spread around the point forecast. This is not a statistical confidence interval — it's cosmetic. The actual prediction intervals from Holt-Winters are available via `forecast()` with `simulate` or the `prediction_intervals` parameter.

**Impact:** 8 bank models (3 CC + 5 DC) display fake CIs. If a stakeholder makes a risk decision based on these bands, it's based on fiction.

**Fix:** Use statsmodels' built-in simulation-based prediction intervals.

---

## FINDING 2: ETS Models Skip Cross-Validation

**Severity:** HIGH
**Location:** `src/modelling/bank_model.py:648-655`

```python
elif run_cv and is_ets:
    logger.info(f"  CV skipped (ETS model — CV was validated in model_comparison.py)")
```

**Problem:** ETS banks don't get CV'd during production runs. The MAPE numbers in the CV summary CSV come from a one-off comparison script, not from the production pipeline. If data changes (new months added), the production pipeline has no way to detect ETS degradation.

**Impact:** You're flying blind on 8 models. If ETS performance degrades after a data update, you won't know until someone eyeballs the forecasts.

**Fix:** Implement rolling CV for ETS (manual walk-forward since statsmodels doesn't have Prophet's built-in CV).

---

## FINDING 3: Bank-Level CV Uses Prophet's Log-Space, Reports Original-Scale — But ETS Gets Neither

**Severity:** MEDIUM
**Location:** `src/modelling/bank_model.py:225-258` vs `_fit_ets_model`

**Problem:** Prophet banks get proper log→original scale MAPE conversion. ETS banks have no CV at all (Finding 2), so there's no MAPE to convert. But the ETS MAPE numbers cited in `bank_config.py` comments (e.g., "ETS 5.97% vs Prophet 9.03%") come from `model_comparison.py` — and that script may or may not have applied the same log1p→expm1 back-transform.

**Impact:** The ETS vs Prophet comparison numbers may be on different scales. If `model_comparison.py` compared log-scale ETS MAPE to original-scale Prophet MAPE, the "ETS wins" conclusion could be inverted.

**Fix:** Verify model_comparison.py applies identical back-transform. Re-run comparison with explicit scale matching.

---

## FINDING 4: DC Volume Model Has Only 50 Training Months

**Severity:** MEDIUM
**Location:** `model_config.py:259-274` — DC_VOL_CONFIG training_start = "2022-01-01"

**Problem:** Prophet needs ≥4 annual cycles to estimate yearly seasonality reliably. With training starting Jan 2022, you have ~50 months = 4.2 cycles. This is borderline. The 19.51% CV MAPE reflects this — it's the worst aggregate model.

The decision to restrict to Jan 2022 is defensible (mixing growth+decline regimes blows up CIs), but 50 months with Prophet's default 25 changepoints is overfitting risk.

**Fix:** Reduce `n_changepoints` to 10-15 for this model. Or consider ETS/ARIMA which handle short series better.

---

## FINDING 5: Residual Bucket Model Is Under-Specified

**Severity:** MEDIUM
**Location:** `bank_config.py:239-247` — RESIDUAL_PROPHET_CONFIG

**Problem:** The residual (PSI minus top banks) is modelled with `changepoint_prior_scale=0.01` (extremely rigid trend) and no changepoints. But the residual absorbs ALL structural changes from non-modelled banks: mergers (Andhra→Union, Corporation→Union, United→PNB, Syndicate→Canara), exits (Citibank, AmEx), and new entrants. These events create real level shifts in the residual that a rigid trend cannot capture.

**Impact:** Residual model likely under-fits merger events, biasing the ground-up total. The CC cross-check shows +2.5% divergence by Feb 2026 — part of this is likely residual mis-specification.

**Fix:** Add merger dates as changepoints to the residual model, increase changepoint_prior_scale to 0.05.

---

## FINDING 6: Logistic Growth Caps Are Static

**Severity:** MEDIUM
**Location:** `bank_config.py:177-186`

**Problem:** Caps are set to ~1.15x–1.4x the last observed value (as of Jun 2026). But these are hardcoded. As actuals approach the cap, the forecast flattens prematurely. If growth accelerates (e.g., HDFC–IDFC merger synergies), the cap becomes a ceiling that biases forecasts downward. If growth decelerates more than expected, the cap is irrelevant but doesn't hurt.

**Impact:** One-directional risk — caps can only hurt (flatten too early), never help.

**Fix:** Make caps dynamic: `cap = max(last_actual * 1.3, trailing_12m_growth_rate * 24 + last_actual)`. Or set caps at market-level theoretical maximums (addressable population × penetration ceiling).

---

## FINDING 7: Forward Regressor Projection Is Naive

**Severity:** LOW-MEDIUM
**Location:** `data_prep.py:326-352`

**Problem:** Future regressor values are projected by 6-month linear extrapolation. For repo_rate this is fine (flat assumption is standard). But for `debit_card_vol_lakh` and `debit_card_pos_vol_lakh` (DC model regressors), linear extrapolation of a declining series can go negative within the 24-month horizon. The code doesn't clip these projections.

**Impact:** If DC POS volume is declining at ~5%/month and you extrapolate 24 months linearly, you get negative projected values. Prophet then fits to negative regressor inputs, producing unpredictable behavior.

**Fix:** Clip forward projections at zero. Or use a multiplicative decay instead of linear extrapolation for declining series.

---

## FINDING 8: Kotak CC MAPE of 20.24% and BoB CC MAPE of 21.87% Are Shipped

**Severity:** HIGH (Business)
**Location:** CV summaries

**Problem:** Two CC bank models have >20% mean MAPE. At this error level, the point forecast is essentially uninformative. The logistic cap helps prevent runaway over-forecasting, but 20% error means the forecast could be off by ±₹1,000+ crore for these banks.

**Impact:** If Axiom presents bank-level forecasts to a client, Kotak and BoB CC forecasts will damage credibility. A 20% MAPE on a ₹5,000cr portfolio is a ₹1,000cr error band.

**Fix:** For these banks, consider:
1. Ensemble Prophet + ETS (weighted by inverse MAPE)
2. Recent-bias weighting (upweight last 12 months in CV)
3. If still >15%, flag as "directional only — not for quantitative decisions" in the dashboard

---

## FINDING 9: Paytm Payments Bank DC Has 27.52% MAPE

**Severity:** MEDIUM (same logic as Finding 8)
**Location:** CV summaries

**Problem:** Paytm is a new entrant with extreme volatility. 27.52% MAPE with a max fold at 53.79% is not a useful forecast.

**Fix:** Drop from individual modelling. Let it flow through the residual bucket. Or use a simpler growth model (exponential smoothing with damped trend).

---

## FINDING 10: No Out-of-Sample (OOS) Holdout Test

**Severity:** HIGH (Methodological)
**Location:** Entire pipeline

**Problem:** The pipeline runs rolling CV (expanding window) which is good. But there's no true OOS test — holding out the last N months entirely, training on everything before, and measuring accuracy on unseen data. Rolling CV's "test" folds overlap with later training windows, creating information leakage in parameter tuning decisions.

If someone tuned `changepoint_prior_scale` or `training_start` to minimize CV MAPE, those parameters are fit to the validation set. Without a held-out OOS test, you can't distinguish genuine accuracy from overfit-to-CV.

**Impact:** Reported MAPEs may be 1-3pp optimistic vs true OOS performance.

**Fix:** Implement a proper holdout: train on all data through Dec 2024, forecast Jan–Jun 2025, measure MAPE on those 6 months. This should be run ONCE after all tuning is complete and never used to adjust parameters.

---

## FINDING 11: Pulse Event Dummies Are Set to 0 in Forecast

**Severity:** LOW
**Location:** `data_prep.py:303-304`

```python
# Pulse events = 0 in forecast (shocks don't repeat by assumption)
for col in [c for c in future.columns if c.startswith("event_")]:
    future.loc[future["ds"] > last_date, col] = 0.0
```

**Problem:** This sets ALL event columns to 0 in the forecast period — including step dummies (which should persist as 1). The code comment says "pulse events" but the mask applies to all events including `event_card_validity_7yr` (a step dummy) and `event_rbi_credit_tightening` (a step dummy).

**Impact:** Step dummies like RBI credit tightening (CC) and card validity extension (DC) are being zeroed in the forecast. This means the forecast doesn't carry forward these structural shifts. The forecast effectively "forgets" that RBI tightened in Nov 2023 and reverts to the pre-tightening trend.

**Fix:** Only zero pulse dummies in forecast; keep step dummies at their last training value (1.0).

---

## FINDING 12: Ground-Up Cross-Check Divergence Is Growing

**Severity:** MEDIUM
**Location:** `data/processed/groundup/crosscheck_cc.csv`

**Problem:** CC ground-up vs PSI divergence grows from +0.46% (Jun 2025) to +2.49% (Feb 2026). This is a systematic drift, not random noise. The ground-up total is increasingly over-estimating vs PSI.

Likely causes:
- Residual model under-fitting (Finding 5)
- Bank-level models collectively over-forecasting by ~2.5%
- PSI data revision that bankwise data hasn't caught up with

DC shows the opposite: -1.2% to -2.2% (under-estimation), suggesting DC bank models are collectively too conservative.

**Fix:** After fixing Finding 5 (residual model), re-run cross-check. If drift persists >1.5%, add a reconciliation adjustment (scale ground-up by PSI/ground-up ratio at the last overlap month).

---

## Summary Table

| # | Finding | Severity | Est. MAPE Impact | Effort |
|---|---------|----------|------------------|--------|
| 1 | Fake ETS confidence intervals | CRITICAL | Accuracy: 0, but trust: high | 2h |
| 2 | ETS models skip CV | HIGH | Unknown (blind spot) | 4h |
| 3 | ETS vs Prophet MAPE scale mismatch risk | MEDIUM | Possible 2-5pp if inverted | 1h |
| 4 | DC Vol only 50 months, too many changepoints | MEDIUM | 1-3pp on DC Vol | 1h |
| 5 | Residual model too rigid | MEDIUM | 0.5-1pp on ground-up | 2h |
| 6 | Static logistic caps | MEDIUM | 0-2pp on capped banks | 2h |
| 7 | Forward regressors can go negative | LOW-MED | Rare but catastrophic | 1h |
| 8 | Kotak/BoB CC >20% MAPE shipped | HIGH | N/A (flag, don't ship) | 4h |
| 9 | Paytm DC 27.5% MAPE shipped | MEDIUM | N/A (move to residual) | 1h |
| 10 | No true OOS holdout | HIGH | 1-3pp optimism | 3h |
| 11 | Step dummies zeroed in forecast | LOW | 0.5-1pp on CC/DC agg | 0.5h |
| 12 | Ground-up drift growing | MEDIUM | 2.5pp CC, 2pp DC | 2h |

**Total estimated effort:** ~23 hours
**Priority order:** F11 (quick win) → F1 (trust) → F7 (safety) → F10 (methodology) → F2+F3 (ETS blind spot) → F5 (residual) → F8+F9 (flag/drop) → F4 → F6 → F12
