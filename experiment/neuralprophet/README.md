# experiments/neuralprophet/

**Status:** Active experiment — do not merge to main pipeline until CV results reviewed.

---

## What this is

A controlled test of NeuralProphet as an alternative to Prophet for the MIP cards outstanding forecast.

The question being answered: does NeuralProphet's autoregressive component and lagged regressor support
improve CC and DC MAPE over the Prophet baseline (CC: 3.82%, DC: 7.08%)?

---

## Key differences from Prophet being tested

| Feature | Prophet (main pipeline) | NeuralProphet (this experiment) |
|---|---|---|
| Autoregression | No | Yes — AR(3), learns recent momentum |
| Lagged regressors | No — needs future values | Yes — uses past-only values |
| Repo rate handling | Requires future values | Clean via lagged regressor |
| Training speed | Slower (Stan) | Faster (PyTorch) |
| Overfitting risk | Low | Higher on short series |

---

## Files

```
experiments/neuralprophet/
  data_loader.py        ← loads RBI bankwise data, repo rate, CPI; adds structural event dummies
  run_experiment.py     ← main script: rolling CV + 12-month forecast + HTML dashboard
  README.md             ← this file
  outputs/              ← generated on run (not committed)
    cc_forecast.csv
    dc_forecast.csv
    cv_results_cc.csv
    cv_results_dc.csv
    experiment_dashboard.html   ← open in browser to see results
    summary.json
```

---

## How to run

### 1. Install dependencies

```bash
pip install neuralprophet pandas openpyxl plotly
```

NeuralProphet requires PyTorch. It installs automatically as a dependency.

### 2. Set data path (if your data files are not in `/mnt/project`)

```bash
export MIP_DATA_DIR=/path/to/your/data/folder
```

The folder must contain:
- `RBI_Data_Debit_Credit_1.xlsx`
- `CPI.xlsx`
- `RepoRate2007.XLSX`

### 3. Run

```bash
cd experiments/neuralprophet
python run_experiment.py
```

Expected runtime: ~3–8 minutes depending on machine (PyTorch training per CV fold).

### 4. View results

Open `outputs/experiment_dashboard.html` in any browser. No server needed.

---

## CV configuration

| Parameter | Value | Rationale |
|---|---|---|
| initial | 18 months | Minimum for NP to learn AR patterns; bumped up if data allows |
| horizon | 6 months | Same as main pipeline Prophet CV |
| step | 3 months | Tighter step than Prophet (6mo) for more fold coverage on a short series |

---

## Data note

The bankwise sheets (X1–X4, 1–41) cover **Mar 2020 – Dec 2024** (45 months).
This is a shorter window than the full PSI series (263 months, Apr 2004 – Feb 2026).

Once the full PSI ingestion is complete (Task 1), re-run this experiment with the longer series.
A longer history will:
- Reduce fold-to-fold MAPE variance
- Let NeuralProphet learn the pre-2016 demonetisation and PMJDY patterns properly
- Make the CV results more trustworthy

---

## Decision criteria

Migrate NeuralProphet to the main pipeline if:
- CC MAPE improves by > 1 percentage point vs Prophet baseline (3.82%)
- DC MAPE improves by > 2 percentage points vs Prophet baseline (7.08%)
- MAPE is consistent across folds (std dev < 3%)

Keep Prophet if:
- Improvement is marginal (< 1pp) — added complexity not justified
- High fold-to-fold variance — model is unstable on this data length
- DC overfits on the short bankwise series

---

## Structural events modelled

### Credit Cards
- `covid_shock` — pulse dummy, Apr–May 2020
- `rbi_tightening_2023` — step dummy, Nov 2023 onwards
- `repo_rate` — lagged continuous regressor
- `cpi` — lagged continuous regressor

### Debit Cards
- `covid_shock` — pulse dummy, Apr–May 2020
- `pmjdy_launch` — step dummy, Aug 2014 onwards
- `demonetisation` — step dummy, Nov 2016 onwards
- `upi_inflection` — step dummy, Jan 2022 onwards

---

## Author

Eshaan — MIP Phase 1 Intern  
Experiment initiated: May 2026
