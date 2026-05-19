# MPi Market Intelligence Platform (MIP)

Internal forecasting platform for India's credit and debit card market.

Phase 1 scope: build the data foundation, forecast cards outstanding
per issuer, and produce a leadership dashboard with structural events
modelled explicitly.

> Status: Phase 1 — Task 1 (data foundation) in progress.

## Project structure

```
mpi-mip/
├── config/
│   └── settings.toml          # paths, column patterns, issuers, structural events
├── data/
│   ├── raw/                   # downloaded Excel files (gitignored)
│   └── processed/             # cleaned Parquet + CSV (gitignored)
├── src/
│   ├── config.py              # loads + validates settings.toml
│   └── ingestion/
│       ├── __main__.py        # runs both pipelines
│       ├── rbi.py             # RBI Payment System Indicators parser
│       ├── npci.py            # NPCI UPI Statistics parser
│       └── validation.py      # column resolution + quality + freshness checks
├── PIPELINE_DESIGN.md
├── probe_rbi_playwright.py
├── pyproject.toml
└── README.md
```

## Setup

Requires Python 3.12+ and [uv](https://astral.sh/uv).

```bash
git clone https://github.com/<your-username>/mpi-mip.git
cd mpi-mip
uv sync
```

## Running the ingestion pipelines

### RBI Payment System Indicators

1. Download the latest PSI Excel from RBI DBIE → Statistics → Financial
   Sector → Payment Systems. Save the file to `data/raw/`.

2. Run:

   ```bash
   uv run python -m src.ingestion.rbi
   ```

   Output: `data/processed/rbi_psi_cards.parquet` and `.csv`.

### NPCI UPI Statistics

1. Download yearly UPI Excel files from
   [npci.org.in](https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics).
   File names follow the pattern `Product-Statistics-UPI-YYYY.xlsx`.
   Save them to `data/raw/`.

2. Run:

   ```bash
   uv run python -m src.ingestion.npci
   ```

   Output: `data/processed/npci_upi.parquet` and `.csv`.

### Run both at once

```bash
uv run python -m src.ingestion
```

## Design principles

- **Header-name matching, not positional indices.** Column lookups happen
  via substring patterns defined in `config/settings.toml`. This survives
  RBI Excel layout changes that would silently break a position-based
  parser.
- **Loud validation.** Missing columns, low row counts, and bad data raise
  `SchemaValidationError` with a clear message. No silent `None` returns.
- **Deterministic file selection.** When multiple files match a pattern,
  the most recent (by modification time) is used and the choice is logged.
- **SHA256 freshness check.** The pipeline distinguishes a newly downloaded
  file from a re-run of the same file. Useful for monthly automation later.
- **Externalised configuration.** Paths, column patterns, issuer lists, and
  structural event dates live in `config/settings.toml`. Code changes are
  not required to tune the pipeline.

## Phase roadmap

- **Phase 1 (current)** — India only, local file-based pipeline plus
  standalone HTML dashboard. Cards outstanding as the primary target,
  per-issuer ground-up modelling, rolling cross-validation, UPI as a
  regressor, structural events as changepoints.
- **Phase 2** — Geographic expansion via country-specific data adapters
  (US, EU, SE Asia).
- **Phase 3** — Live intelligence platform with managed Postgres,
  scheduled monthly refresh, and a hosted dashboard. See
  `PIPELINE_DESIGN.md` for the production architecture.
