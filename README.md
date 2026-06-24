# MPi Market Intelligence Platform (MIP)

Automated forecasting platform for India's credit card, debit card, and digital payments market. Scrapes RBI/NPCI data monthly, runs ensemble ML models, and serves 24-month forecasts through a live web dashboard.

**Live dashboard:** https://web-mocha-kappa-71.vercel.app

## Quick Start

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (Python package manager)
- **Node.js 18+** (for the web dashboard)

### 1. Clone & install Python dependencies

```bash
git clone https://github.com/Eshaan0110/MPi-mip.git
cd MPi-mip
uv sync
```

### 2. Run the forecasting pipeline

```bash
# Full run with cross-validation (~20 min)
uv run python run_pipeline.py

# Skip CV for faster run (~2 min)
uv run python run_pipeline.py --no-cv

# Skip ingestion if data is already fresh
uv run python run_pipeline.py --skip-ingestion --no-cv
```

This runs 5 steps in sequence:
1. **Ingestion** — parse raw RBI/NPCI files into clean Parquet
2. **Aggregate models** — CC/DC outstanding (Prophet + ARIMA + ETS ensemble)
3. **Bank-level models** — ~80 individual bank models (Prophet or ETS)
4. **Transaction volume models** — CC/DC/UPI volumes
5. **Dashboard rebuild** — generate `dashboard.html`

### 3. View results locally

Open `dashboard.html` in any browser — no server needed.

### 4. (Optional) Run the web dashboard locally

```bash
cd web
npm install
cp .env.local.example .env.local
# Edit .env.local with your Supabase credentials
npm run dev
```

Open http://localhost:3000

### 5. (Optional) Sync forecasts to Supabase

```bash
# Set environment variables (or they'll use defaults in the script)
export SUPABASE_URL=https://nwevrclikkiuemttovih.supabase.co
export SUPABASE_SERVICE_KEY=<your-service-key>

uv run python scripts/sync_to_supabase.py
# Preview without writing:
uv run python scripts/sync_to_supabase.py --dry-run
```

---

## Data Sources

| Source | What it provides | Frequency | Range |
|--------|-----------------|-----------|-------|
| **RBI Bankwise** | Per-bank CC/DC outstanding, ATMs, PoS, txn volumes | Monthly | Apr 2011 – present |
| **RBI PSI** | India-total CC/DC outstanding, txn volumes | Monthly | Apr 2004 – present |
| **NPCI UPI** | UPI transaction volumes and value | Monthly | Apr 2016 – present |
| **RBI Repo Rate** | Policy interest rate | Monthly | Jan 2007 – present |

Raw data goes into `data/raw/`. Processed Parquet files go into `data/processed/`.

---

## Models

### Aggregate — India Total (CC & DC Outstanding)

**Architecture:** Weighted ensemble of Prophet + ARIMA(1,1,1) + Damped ETS

| Series | Prophet | ARIMA | ETS | CV MAPE |
|--------|---------|-------|-----|---------|
| CC Outstanding | 35% | 39% | 26% | ~3.5% |
| DC Outstanding | 35% | 65% | 0% | ~5.7% |

**Regressors:**
- CC: `repo_rate` at lag 9 months (RBI rate → bank risk appetite → card issuance)
- DC: `debit_card_vol_lakh` at lag 4 months (transaction volume → issuance) + `debit_card_pos_vol_lakh` (UPI displacement signal)

**Structural events coded:**
- Demonetisation (Nov 2016) — changepoint
- COVID lockdown (Apr–May 2020) — pulse dummy
- PMJDY / Jan Dhan (Aug 2014) — changepoint, DC only
- UPI inflection (Jan 2022) — changepoint, DC only
- RBI credit tightening (Nov 2023) — step dummy, CC only

**Confidence intervals:** Conformal prediction intervals from walk-forward CV residual quantiles (5th/95th percentile). Distribution-free — no normality assumption.

### Bank-Level (~80 models)

Each bank × card type gets its own model:
- **Large/complex banks** → Prophet with logistic growth caps
- **Small/stable banks** → Holt-Winters ETS

Bank forecasts are summed and reconciled against the aggregate total (residual adjustment).

**Median CV MAPE:** CC banks ~4–6%, DC banks ~6–9%

### Transaction Volumes

| Model | Regressors | Training Start | CV MAPE |
|-------|-----------|----------------|---------|
| CC Txn Volume | CC outstanding (multiplicative) | 2013 | ~13.6% |
| DC Txn Volume | None (trend only) | 2022 | ~7% |
| UPI Volume | None (trend only) | All data | ~12.3% |

### Cross-Validation

Walk-forward CV: 48-month initial window, 6-month horizon, 6-month step. Model is always tested on data it has never seen.

---

## Project Structure

```
MPi-mip/
├── data/
│   ├── raw/                          # Raw RBI/NPCI files (Excel, JSON)
│   │   ├── rbi_bankwise/             # Bank-level Excel files
│   │   ├── rbi_psi/                  # Payment System Indicators
│   │   └── npci_upi/                 # UPI monthly JSON
│   └── processed/                    # Clean Parquet files + forecasts
│       ├── forecast_cc.parquet       # Aggregate CC forecast
│       ├── forecast_dc.parquet       # Aggregate DC forecast
│       ├── bankwise_forecasts/       # Per-bank forecast Parquet files
│       └── groundup/                 # Bank CV summaries
│
├── src/
│   ├── ingestion/                    # Data parsers (RBI, NPCI, repo rate)
│   ├── modelling/
│   │   ├── model_config.py           # All model configs, events, regressors
│   │   ├── aggregate_model.py        # Ensemble forecasting (Prophet+ARIMA+ETS)
│   │   ├── bank_model.py             # Per-bank Prophet/ETS models
│   │   ├── bank_config.py            # Bank lists, caps, ETS/Prophet flags
│   │   ├── txn_volume_model.py       # CC/DC/UPI transaction volume models
│   │   └── data_prep.py              # Feature engineering, lag application
│   └── scraper/                      # Automated data scrapers
│
├── scripts/
│   ├── sync_to_supabase.py           # Push forecasts to cloud database
│   ├── rebuild_dashboard.py          # Generate dashboard.html data
│   └── gen_accuracy_docx.py          # Bank accuracy report generator
│
├── web/                              # Next.js 14 web dashboard
│   ├── src/app/                      # Pages (dashboard, banks, status, models, about)
│   ├── src/components/               # Charts, KPI cards, nav
│   └── src/lib/                      # Supabase client, types
│
├── experiments/axiom_audit/          # Diagnostic test suites & audit reports
├── .github/workflows/                # CI/CD pipeline (monthly automation)
├── supabase/migrations/              # Database schema (11 tables)
│
├── run_pipeline.py                   # One-command full pipeline runner
├── dashboard.html                    # Self-contained offline dashboard
└── pyproject.toml                    # Python dependencies
```

---

## Web Dashboard

**Live:** https://web-mocha-kappa-71.vercel.app

Built with Next.js 14 + Recharts + Tailwind CSS. Reads from Supabase PostgreSQL.

| Page | What it shows |
|------|---------------|
| **Dashboard** | India-level KPIs, 24-month forecast charts with 90% CI bands, summary table |
| **Bank Explorer** | Per-bank forecasts, CI, ranked table. Toggle CC/DC, pick any bank/month |
| **Data Status** | Scraper run history — success/failure, record counts, error messages |
| **Model Performance** | Every model's CV MAPE, color-coded (green ≤7%, amber ≤15%, red >15%) |
| **About** | Methodology, data sources, accuracy summary, limitations |

### Web dashboard environment variables

Create `web/.env.local`:
```
NEXT_PUBLIC_SUPABASE_URL=https://nwevrclikkiuemttovih.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=<anon-key>
```

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/monthly_pipeline.yml`) runs on the 15th of every month:

```
Job 1: Scrape    → Download latest RBI/NPCI data
Job 2: Train     → Run ingestion + all models
Job 3: Sync      → Push forecasts to Supabase
Job 4: Notify    → Report success/failure summary
```

Can also be triggered manually via `workflow_dispatch`.

**Required GitHub secrets:**
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`

---

## Database (Supabase)

PostgreSQL with 11 tables. Schema in `supabase/migrations/001_initial_schema.sql`.

Key tables:
- `forecasts_aggregate` — India-level forecasts (CC/DC outstanding, txn volumes, UPI)
- `forecasts_bank` — Per-bank forecasts (~80 bank × card type combinations)
- `model_metadata` — CV MAPE, model type, training dates
- `scraper_runs` — Scraper execution log
- `raw_npci_upi` — Raw UPI data

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Ensemble over single model | No single model wins on all series. Weighted combination reduces variance. |
| Conformal CIs over parametric | Residuals aren't Gaussian. Conformal intervals use actual CV errors — no distributional assumptions. |
| DC vol regressor kept despite ablation | Removing it improved MAPE by 1.3pp, but it's the core business driver. Lag fix (0→4) recovered accuracy. |
| CC training starts 2013 | Pre-2013 GFC decline is a different regime — including it pollutes growth-market forecasts. |
| DC vol training starts 2022 | Pre-2022 growth contradicts post-2022 decline. Mixing regimes gives absurdly wide CIs. |
| Per-bank start dates | Merger banks train only on post-merger data (stable regime). |
| Logistic growth caps | Prevents runaway extrapolation for hypergrowth banks. |
| No UPI QR regressor for CC | Ablation showed it worsened CV MAPE. Dropped until domain confirmation. |

---

## Audit

Full AXIOM Round 4 quantitative audit: 12 diagnostic suites, **88/100 final score**.
Report: `experiments/axiom_audit/AUDIT_ROUND4.md`

Tests passed: stationarity (ADF/KPSS), residual diagnostics, Granger causality, regressor ablation, CI calibration, alternative model comparison, lag sensitivity, event sensitivity, scenario/stress testing, weight optimization, horizon drift, data leakage.
