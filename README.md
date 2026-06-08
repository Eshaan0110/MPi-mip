# MPi Market Intelligence Platform (MIP)

Internal forecasting platform for India's credit and debit card market.

Forecasts cards outstanding per bank (top 10 CC, top 15 DC), aggregate India totals, and transaction volumes (CC/DC/UPI) with a self-contained interactive dashboard.

## Quick Start

Requires **Python 3.12+** and **[uv](https://astral.sh/uv)**.

```bash
git clone https://github.com/Eshaan0110/MPi-mip.git
cd MPi-mip
uv sync
```

### Run everything (first time)

```bash
uv run python run_pipeline.py
```

This runs all 5 steps in sequence (~5 min with CV, ~30 sec without):
1. Aggregate CC/DC outstanding models
2. Bank-level ground-up models (10 CC + 15 DC banks)
3. Transaction volume models (CC/DC/UPI)
4. UPI displacement analysis
5. Dashboard rebuild

Then open **`dashboard.html`** in any browser. No server needed.

### Run faster (data already fresh)

```bash
uv run python run_pipeline.py --skip-ingestion --no-cv
```

### Ingest new data (when RBI publishes a new month)

Place the new bankwise Excel in `data/raw/rbi_bankwise/`, then:

```bash
uv run python run_pipeline.py
```

The pipeline auto-detects new files and re-runs everything.

## Dashboard

Open `dashboard.html` in any browser. Features:

- **Overview tab** — CC/DC outstanding, transaction volumes, UPI volume charts with forecasts
- **Monthly Forecast table** — pick any month from Jul 2026 to Jul 2027, see all metrics for that month with 90% confidence intervals
- **Bank-Level tab** — select any bank + any month, see forecast, CI, and growth vs last actual. Table shows all banks ranked by forecast for the selected month. Click any row to see the chart.
- **CV MAPE table** — model accuracy summary

## Accuracy (tested on March + April 2026 actuals)

### Credit Cards — Per Bank

| Bank | Accuracy |
|------|----------|
| ICICI Bank | 98–99% |
| Bank of Baroda | 97–99% |
| Canara Bank | 97% |
| Yes Bank | 93–95% |
| Axis Bank | 93% |
| HDFC Bank | 93% |
| SBI | 88% |
| Kotak Mahindra | 85–87% |
| IndusInd | 81–83% |
| **CC Median** | **83%** |

### Debit Cards — Per Bank

| Bank | Accuracy |
|------|----------|
| Axis Bank | 97–99% |
| ICICI Bank | 98–99% |
| Indian Bank | 97–99% |
| Kotak Mahindra | 97–98% |
| HDFC Bank | 96% |
| Bank of India | 92–99% |
| Punjab National Bank | 92% |
| Union Bank | 92% |
| Canara Bank | 88–91% |
| Bank of Baroda | 89–92% |
| Central Bank | 90–91% |
| UCO Bank | 89% |
| Indian Overseas | 87–88% |
| SBI | 85–87% |
| Paytm | 81% |
| **DC Median** | **92%** |

### Aggregate (India Total)

| Model | CV MAPE | Notes |
|-------|---------|-------|
| CC Outstanding | 3.46% | Prophet + Repo Rate lag-9 |
| DC Outstanding | 7.08% | Prophet + DC txn volume regressor |
| CC Bank median | 7.93% | Top 10 banks, original-scale |
| DC Bank median | 7.44% | Top 15 banks, original-scale |

## Project Structure

```
MPi-mip/
+-- config/settings.toml           # paths, column patterns, structural events
+-- data/
|   +-- raw/rbi_bankwise/          # RBI bankwise Excel files (2011-2025)
|   +-- raw/                       # RBI PSI, NPCI UPI Excels
|   +-- processed/                 # cleaned parquets + forecast CSVs
+-- src/
|   +-- ingestion/
|   |   +-- rbi.py                 # RBI PSI parser (old + new format)
|   |   +-- bankwise.py            # RBI bankwise parser (ATMs, PoS, txn vols)
|   |   +-- npci.py                # NPCI UPI parser
|   |   +-- cpi.py, repo_rate.py   # macro data parsers
|   +-- modelling/
|   |   +-- bank_config.py         # all bank lists, start dates, ETS/Prophet flags
|   |   +-- bank_data_prep.py      # builds Prophet-ready DataFrames per bank
|   |   +-- bank_model.py          # fits Prophet or ETS per bank, CV, forecast
|   |   +-- aggregate_model.py     # India-level CC/DC outstanding model
|   |   +-- txn_volume_model.py    # CC/DC/UPI transaction volume models
|   |   +-- model_config.py        # aggregate model configs, structural events
+-- scripts/
|   +-- rebuild_dashboard.py       # regenerates dashboard data
|   +-- bank_oos.py                # out-of-sample accuracy test
|   +-- bankwise_eda_report.py     # per-bank EDA HTML report
|   +-- method_comparison_fair.py  # fixed share vs trend vs ground-up comparison
|   +-- model_comparison.py        # Prophet vs ETS comparison
+-- reports/
|   +-- bankwise_eda_report.docx   # per-bank regressor EDA (for Axiom)
|   +-- bankwise_eda_report.html   # same, HTML version
+-- run_pipeline.py                # one-command full pipeline runner
+-- dashboard.html                 # self-contained interactive dashboard
+-- PIPELINE_DESIGN.md             # Phase 3 production architecture spec
```

## Models

### Bank-Level (Ground-Up)

Each of the 25 banks (10 CC + 15 DC) has its own model, trained only on its **stable regime** — post-merger data only, no pre-merger history that would confuse the trend.

- **Stable banks** (Axis, ICICI, IndusInd CC; HDFC, Axis, ICICI, UCO, Indian Overseas DC): **Holt-Winters ETS** — outperforms Prophet by 0.5–3.3pp on clean additive-trend series.
- **Merger/hypergrowth banks** (BoB, Canara, Union, PNB, Kotak, Yes Bank): **Prophet** with per-bank `changepoint_prior_scale` tuning.
- **Over-forecasting banks** (Kotak CC, BoB CC/DC): **Logistic growth cap** to prevent runaway extrapolation into a normalising market.

Residual bucket = PSI total minus sum of top banks. Ground-up aggregate reconciles against PSI within 0.3% (DC) to 8.1% (CC).

### Aggregate (India Total)

Prophet with structural event dummies (demonetisation, COVID, RBI credit tightening) and validated regressors (Repo Rate lag-9 for CC, DC transaction volume for DC).

## Data Sources

| Source | What | Frequency | Range |
|--------|------|-----------|-------|
| RBI PSI | India total CC/DC outstanding, txn volumes, PoS/ATM counts | Monthly | Apr 2004 – Feb 2026 |
| RBI Bankwise | Per-bank CC/DC outstanding, ATMs, PoS, txn volumes | Monthly | Apr 2011 – May 2025 |
| NPCI | UPI transaction volumes | Monthly | Apr 2016 – Feb 2026 |
| RBI | Repo Rate | Monthly | Jan 2007 – Feb 2026 |
| MOSPI | CPI | Monthly | Jan 2011 – Dec 2025 |

## Key Design Decisions

- **Per-bank start dates** — merger banks train only on post-merger data (stable regime)
- **log1p variance stabilisation** — applied to all bank models, back-transformed before evaluation
- **CV MAPE on original scale** — Prophet CV outputs are expm1-transformed before computing MAPE
- **No external regressors at bank level** — Granger causality tested 15 candidates; none passed the bar for production (documented in `reports/bankwise_eda_report.docx`)
- **Header-name matching** — column detection by pattern, not position; survives RBI Excel layout changes
