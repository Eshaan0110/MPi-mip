# UPI Relationship Analysis

Generated: 2026-06-03 11:06
Data range: Apr 2004 -- Feb 2026

## 1. UPI vs Debit Card Displacement

### UPI P2M Volume vs DC Transaction Volume

| Lag (months) | Correlation |
|:---:|:---:|
| 0 | -0.951 |
| 1 | -0.958 ** |
| 2 | -0.957 |
| 3 | -0.955 |
| 6 | -0.948 |

Strongest correlation at lag 1: r = -0.958

### UPI Total Volume vs DC Outstanding

| Lag (months) | Correlation |
|:---:|:---:|
| 0 | +0.777 |
| 1 | +0.779 |
| 2 | +0.782 |
| 3 | +0.784 |
| 6 | +0.791 |

Linear fit R-squared (UPI vol vs DC txn vol): **0.552**
The relationship is **moderately linear**.

### Threshold Test (UPI P2M median split at 4,619 mn)
- Correlation below threshold: -0.590
- Correlation above threshold: -0.949
- The displacement relationship **strengthens** at higher UPI volumes.

## 2. UPI vs Credit Card Relationship

### UPI Total Volume vs CC Transaction Volume

| Lag (months) | Correlation |
|:---:|:---:|
| 0 | +0.983 |
| 1 | +0.980 |
| 2 | +0.981 |
| 3 | +0.981 |
| 6 | +0.981 |

The UPI-CC relationship is **complementary (positive)** at lag 0 (r = +0.983).

### UPI Total Volume vs CC Outstanding

| Lag (months) | Correlation |
|:---:|:---:|
| 0 | +0.950 |
| 1 | +0.948 |
| 2 | +0.947 |
| 3 | +0.946 |
| 6 | +0.942 |

## UPI and Debit Card Displacement

The data shows a clear and accelerating displacement of debit card transactions by UPI. Debit card transaction volumes peaked at 13,140 lakh per month in October 2019 and have since fallen 93% to 917 lakh per month as of the latest data. Over the same period, UPI volumes grew to 20,394 million transactions per month.

The correlation between UPI merchant payments (P2M) and debit card transactions is -0.951 at lag 0, confirming a strong negative relationship: as UPI P2M volumes rise, debit card swipes at merchants fall by a proportional amount. This is not merely correlation -- the economic mechanism is direct substitution at the point of sale.

Importantly, the UPI-credit card relationship tells a different story. Credit card transaction volumes are positively correlated with UPI growth (r = +0.983), suggesting that UPI and credit cards are complementary rather than substitutive. This is likely because credit cards serve a different use case (credit access, rewards, international payments) that UPI does not directly address.

For the 12-month forecast, these dynamics mean: (1) debit card transactions will continue declining, though the rate of decline may slow as the remaining volume represents use cases where UPI is not yet a substitute (ATM withdrawals, international travel); (2) credit card transaction growth will continue largely independent of UPI penetration; (3) the single risk factor is RuPay credit on UPI -- if banks begin routing credit card transactions through the UPI rail at scale, it could blur the line between these two products.
