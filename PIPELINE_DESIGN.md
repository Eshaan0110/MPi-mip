# Production Pipeline Design — MPi MIP

This document describes how the Phase 1 codebase extends into a production
"live intelligence platform" in Phase 3. Phase 1 itself runs locally with
file-based deliverables (see `README.md`); the architecture below is the
target state once monthly automated refresh is in scope.

## Architecture Overview

```
[Trigger: GitHub Actions Cron]
          |
          v
[Ingestion: src.ingestion]
  - SHA256 freshness check against last-run hash
  - Download RBI Excel via Playwright (auto)
  - Parse + validate -> Parquet
          |
          v
[Validation Layer]
  - Row count >= configured minimum
  - No date gaps > 35 days
  - Null % below per-column threshold
          |
       +--+--+
       v     v
    [Raw]  [Processed]
     S3    PostgreSQL + Parquet on S3
          |
          v
[Alert: Slack webhook on failure]
```

## Q1 — Scheduling and Triggering

**Choice: GitHub Actions cron.**

RBI publishes payment system data monthly, typically in the first week of the
following month. A cron job on the 5th of every month covers this reliably:

```yaml
# .github/workflows/monthly_ingest.yml
on:
  schedule:
    - cron: '0 9 5 * *'   # 09:00 UTC on the 5th of every month
  workflow_dispatch:       # manual trigger if RBI publishes late
```

Why GitHub Actions over Airflow at this stage:
- Zero infrastructure to manage.
- Free for private repos under the standard plan.
- `workflow_dispatch` allows manual re-trigger if RBI is late.
- Airflow is justified once we have 5+ interdependent pipelines.

## Q2 — Storage

| Layer            | Storage                  | Reason                                                  |
|------------------|--------------------------|---------------------------------------------------------|
| Raw Excel files  | S3 / object storage      | Cheap, immutable audit trail, easy to reprocess         |
| Processed data   | Parquet on S3            | Columnar, 5x smaller than CSV, fast for ML workloads    |
| Forecast outputs | PostgreSQL               | Query-ready for dashboards and downstream consumers     |

Why Parquet over CSV for processed data:
- Typed columns — no silent string/float ambiguity on reload.
- 5–10x smaller file size.
- 10x faster reads in pandas / polars for ML workflows.

## Q3 — Detecting New Data Without a Formal API

RBI does not expose a changelog or webhook. Three layered strategies:

**Strategy 1 — SHA256 hash check (primary).**
Download the file, compare its SHA256 to the last recorded hash. If different,
proceed; if identical, log and exit cleanly. Already implemented in
`src/ingestion/validation.py:file_sha256` and `detect_freshness`.

**Strategy 2 — HTTP Last-Modified header.**
Before downloading, issue a HEAD request and compare the `Last-Modified`
header to the stored timestamp. Faster than downloading the full file
just to check.

**Strategy 3 — Date watermark.**
After parsing, check whether the latest month in the data exceeds the last
recorded watermark stored in the database. If not, skip.

All three run in sequence. Any one detecting new data triggers the full
pipeline.

## Q4 — Monitoring and Alerting

| Signal                     | How detected                              | Alert                  |
|----------------------------|-------------------------------------------|------------------------|
| HTTP failure / portal down | Playwright / requests raises exception    | Slack + email          |
| Empty or malformed file    | Row count below `validation.min_rows`     | Slack + PagerDuty      |
| Missing months             | Date gap > `validation.max_date_gap_days` | Slack warning          |
| High null rate             | Any column above `validation.max_null_pct`| Slack warning          |
| Model MAPE spike           | > 15% vs rolling baseline                 | Slack warning          |
| Pipeline timeout           | GitHub Actions 30 min limit               | GitHub email           |

All pipeline alerts route to a `#mpi-data-alerts` Slack channel via a webhook
stored as a GitHub Actions secret.

Threshold rationale:
- Min row count: a sudden drop below the configured minimum signals a parsing
  failure, not a routine missing month.
- Date gap of 35 days: monthly data should never gap more than ~31 days;
  35 days gives buffer for RBI publishing delays.
- MAPE > 15%: the credit card model targets low single-digit error; a spike
  to 15% signals either a data quality issue or a genuine market shift
  worth investigating.
