# Axiom Technical Audit — MIP

## Round 1: ML Pipeline (12 findings)

| File | Description |
|------|-------------|
| `AUDIT_REPORT.md` | Full 12-finding audit with severity, impact, and fixes |
| `fix_f01_ets_ci.py` | **CRITICAL** — Replace fake ETS CIs with simulation-based intervals |
| `fix_f02_ets_cv.py` | Walk-forward CV for ETS models (currently skipped in production) |
| `fix_f04_dc_vol_changepoints.py` | Reduce DC Vol changepoints from 25 to 10 |
| `fix_f05_residual_model.py` | Improve residual model flexibility for merger events |
| `fix_f06_dynamic_caps.py` | Dynamic logistic growth caps instead of hardcoded |
| `fix_f07_regressor_clip.py` | Clip forward regressors at zero (prevent negative inputs) |
| `fix_f08_f09_accuracy_flags.py` | Accuracy tiering + ensemble for low-confidence banks |
| `fix_f10_oos_holdout.py` | True OOS holdout test (train through Dec 2024, test Jan-Jun 2025) |
| `fix_f11_step_dummies.py` | Fix step dummies being zeroed in forecast period |
| `run_all_fixes.py` | Master runner — applies F01/F04/F05/F07/F11 and re-runs pipeline |

## Round 2: Production Readiness (10 findings)

| File | Description |
|------|-------------|
| `AUDIT_ROUND2.md` | Full 10-finding audit: CI/CD, web app, data pipeline |
| `fix_p01_pipeline.yml` | **CRITICAL** — Remove `continue-on-error`, add proper job chaining |

**Applied directly to source** (web app fixes — non-ML, safe to modify):
- `web/src/app/layout.tsx` — P08: `<a>` → `<Link>` for client-side navigation
- `web/src/app/page.tsx` — P04: Error state handling
- `web/src/app/banks/page.tsx` — P04: Error state handling
- `web/src/app/models/page.tsx` — P04: Error state + accuracy color tiering (green/amber/red)
- `web/src/app/status/page.tsx` — P04: Error state handling
- `.github/workflows/monthly_pipeline.yml` — P01: Proper job dependencies, no silent failures

## Results

### OOS Holdout (the real test)

| Model | OOS MAPE | Max APE | 90% CI Coverage |
|-------|----------|---------|-----------------|
| CC Outstanding | 1.58% | 2.41% | 83% (5/6 months) |
| DC Outstanding | 1.02% | 2.06% | 100% (6/6 months) |

### CV Improvements (Round 1 fixes)

- CC aggregate cross-check: 1.44% → 1.0%
- Kotak DC: -3.6pp MAPE
- ETS CIs: fake ±5-20% → real simulation-based (widths 62.7 → 173.7 over 12mo)
- ETS CV: now has walk-forward validation (median 1.26% on test)

## Quick Start

```bash
# Run all Round 1 ML fixes
uv run python experiments/axiom_audit/run_all_fixes.py

# Run OOS holdout test
uv run python experiments/axiom_audit/fix_f10_oos_holdout.py

# Compare dynamic vs static caps
uv run python experiments/axiom_audit/fix_f06_dynamic_caps.py
```

## No Original ML Code Modified

Round 1 ML fixes use monkey-patching at runtime — `src/` ML code is untouched.
Round 2 web/CI fixes are applied directly (non-ML, low risk).
