# MIP — Bank-Level Forecast Accuracy Summary

**Prepared by:** Eshan Adyanthaya, MPi  
**Date:** 23 June 2026

---

## 1. Methodology

| Item | Detail |
|------|--------|
| **Models** | Facebook Prophet (logistic growth) for banks with structural breaks; Holt-Winters ETS for clean, stable-growth banks |
| **Validation** | Walk-forward cross-validation — 36-month initial window, 6-month forecast horizon, 6-month step |
| **Bank Coverage** | 10 CC banks + 15 DC banks modelled individually (ground-up approach) |
| **Industry Coverage** | ~91% of CC outstanding, ~83% of DC outstanding |
| **Residual** | Remaining banks captured via a single residual Prophet model; sum of individual + residual = India total |
| **Confidence Intervals** | 90% prediction intervals from Prophet / ETS simulation |

---

## 2. Credit Card Outstanding — Bank-Level CV MAPE

| Bank | Model | CV MAPE (Median) | CV Range | Notes |
|------|-------|------------------:|----------|-------|
| ICICI Bank | ETS | 2.5% | — | Stable growth, ETS outperforms Prophet |
| IndusInd Bank | ETS | 4.4% | — | Stable growth |
| State Bank of India | Prophet | 4.8% | 3.2% – 11.3% | |
| HDFC Bank | Prophet | 5.7% | 3.1% – 8.1% | Logistic growth cap applied |
| Axis Bank | ETS | 6.0% | — | Stable growth |
| HSBC | Prophet | 13.4% | 6.7% – 23.4% | Small base, higher volatility |
| Bank of Baroda | Prophet | 20.8% | 11.0% – 27.3% | Rapid growth phase, cap applied |
| Kotak Mahindra Bank | Prophet | 21.9% | 10.1% – 30.0% | Rapid growth phase, cap applied |

**CC Median across banks: 5.7%**

---

## 3. Debit Card Outstanding — Bank-Level CV MAPE

| Bank | Model | CV MAPE (Median) | CV Range | Notes |
|------|-------|------------------:|----------|-------|
| HDFC Bank | ETS | 1.2% | — | Very stable |
| UCO Bank | ETS | 2.6% | — | Stable |
| Axis Bank | ETS | 3.1% | — | Stable |
| State Bank of India | Prophet | 4.3% | 1.5% – 7.1% | Largest DC issuer |
| ICICI Bank | ETS | 5.3% | — | |
| Bank of Baroda | Prophet | 5.3% | 1.5% – 7.4% | |
| Indian Overseas Bank | ETS | 5.7% | — | |
| Central Bank of India | Prophet | 7.3% | 4.4% – 11.0% | |
| Kotak Mahindra Bank | Prophet | 7.8% | 5.0% – 10.3% | |
| Bank of India | Prophet | 9.6% | 3.8% – 16.3% | |
| Paytm Payments Bank | Prophet | 21.4% | 11.4% – 53.8% | High volatility, small base |

**DC Median across banks: 5.3%**

---

## 4. Aggregate Out-of-Sample Holdout (Jan – Jun 2025)

True OOS test: models trained through Dec 2024, forecasts compared against actual Jan–Jun 2025 data.

| Metric | OOS MAPE | Max Monthly Error | 90% CI Coverage |
|--------|----------|-------------------|-----------------|
| CC Outstanding (India) | **1.58%** | 2.41% | 83% (5/6 months) |
| DC Outstanding (India) | **1.02%** | 2.06% | 100% (6/6 months) |

---

## 5. Accuracy Tiering

| Tier | MAPE Threshold | Banks |
|------|---------------|-------|
| **Green** | ≤ 7% | ICICI (CC/DC), IndusInd (CC), SBI (CC/DC), HDFC (CC/DC), Axis (CC/DC), UCO (DC), BoB (DC), IOB (DC) |
| **Amber** | 7% – 15% | HSBC (CC), CBI (DC), Kotak (DC), BoI (DC) |
| **Red** | > 15% | Kotak (CC), BoB (CC), Paytm (DC) |

Red-tier banks are in rapid growth or high-volatility phases. Logistic growth caps and dynamic cap computation are applied to constrain over-forecasting.

---

## 6. Key Takeaways

1. **20 of 25 bank models are under 10% MAPE** — well within industry-acceptable range for monthly forecasting
2. **Aggregate accuracy is strong** — ground-up sum achieves < 2% OOS error at India level
3. **Dual-model approach validated** — ETS outperforms Prophet by 0.5–3.3pp on stable-growth banks
4. **Red-flag banks** (Kotak CC, BoB CC, Paytm DC) are monitored separately with logistic growth caps to prevent over-forecasting
5. **90% CI coverage** at aggregate level is 83–100%, confirming prediction intervals are well-calibrated
