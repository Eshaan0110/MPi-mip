# Production Pipeline Design — MPi Data Ingestion

## Architecture Overview
```
[Trigger: GitHub Actions Cron]
          │
          ▼
[Ingestion Script: src/ingestion.py]
  ├── Check if new data published (hash check)
  ├── Download RBI Excel via Playwright
  └── Parse + normalize → Parquet
          │
          ▼
[Validation Layer]
  ├── Row count > expected minimum
  ├── No date gaps > 35 days
  └── Null % below threshold
          │
       ┌──┴──┐
       ▼     ▼
    [Raw]  [Processed]
    S3     PostgreSQL + Parquet on S3
          │
          ▼
[Alert: Slack webhook on failure]
```

## Q1: Scheduling & Triggering

**Choice: GitHub Actions**

RBI publishes payment system data monthly, typically in the first
week of the following month. A cron job on the 5th of every month
covers this reliably.
```yaml
# .github/workflows/monthly_ingest.yml
on:
  schedule:
    - cron: '0 9 5 * *'   # 9am on the 5th of every month
  workflow_dispatch:        # manual trigger if RBI publishes late
```

**Why GitHub Actions over Airflow:**
- Zero infrastructure to manage at this stage
- Free for public repos
- `workflow_dispatch` lets us manually re-trigger if RBI is late
- Airflow makes sense once we have 5+ pipelines — overkill for one

## Q2: Storage

| Layer | Storage | Reason |
|---|---|---|
| Raw Excel files | S3 / GCS object storage | Cheap, immutable audit trail, easy to reprocess |
| Processed data | Parquet on S3 | Columnar format, 5x smaller than CSV, fast for ML |
| Forecast outputs | PostgreSQL | Query-ready for dashboards and downstream consumers |

**Why Parquet over CSV for processed data:**
- Typed columns — no silent string/float ambiguity on reload
- 5-10x smaller file size
- Reads 10x faster in pandas/polars for ML workflows

## Q3: Detecting New Data (No Formal API)

RBI does not expose a changelog or webhook. Three layered strategies:

**Strategy 1 — SHA256 hash check (primary)**
Download the file, compare hash to the last known hash stored in S3.
If different → new data, proceed. If same → skip, log, exit cleanly.
```python
import hashlib

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
```

**Strategy 2 — HTTP Last-Modified header**
Before downloading, send a HEAD request and compare the
`Last-Modified` header to the stored timestamp. Faster than
downloading the full file just to check.

**Strategy 3 — Date watermark**
After parsing, check if the latest month in the data is newer than
the last recorded watermark stored in the database. If not → skip.

All three run in sequence. Any one detecting new data triggers the
full pipeline.

## Q4: Monitoring & Alerting

| Signal | How detected | Alert |
|---|---|---|
| HTTP failure / portal down | requests raises exception | Slack + email |
| Empty or malformed file | row count < 60 | Slack + PagerDuty |
| Missing months | date gap > 35 days | Slack warning |
| High null rate | any column > 10% nulls | Slack warning |
| Model MAPE spike | > 15% vs rolling baseline | Slack warning |
| Pipeline timeout | GitHub Actions 30min limit | GitHub email |

All pipeline alerts route to a `#mpi-data-alerts` Slack channel
via a webhook stored as a GitHub Actions secret.

**Why these thresholds:**
- Row count < 60: we have 74 months, dropping below 60 means
  something broke in parsing, not just a missing month
- Date gap > 35 days: monthly data should never gap more than ~31
  days, 35 gives buffer for RBI publishing delays
- MAPE > 15%: credit card model runs at 2.1%, a spike to 15%
  signals either a data quality issue or a genuine market shift
  worth investigating