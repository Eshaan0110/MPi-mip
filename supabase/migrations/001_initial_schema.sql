-- MIP Phase 2: Initial Database Schema
-- Run this in Supabase SQL Editor to create all tables.

-- ============================================================
-- RAW LAYER (source of truth from scrapers)
-- ============================================================

CREATE TABLE IF NOT EXISTS raw_bankwise (
    id              BIGSERIAL PRIMARY KEY,
    bank_name       TEXT NOT NULL,
    month           DATE NOT NULL,
    cc_outstanding  DOUBLE PRECISION,
    dc_outstanding  DOUBLE PRECISION,
    atm_onsite      INTEGER,
    atm_offsite     INTEGER,
    pos_terminals   INTEGER,
    micro_atm       INTEGER,
    bharat_qr       INTEGER,
    upi_qr          INTEGER,
    cc_pos_vol      DOUBLE PRECISION,
    cc_online_vol   DOUBLE PRECISION,
    cc_others_vol   DOUBLE PRECISION,
    cc_atm_cash_vol DOUBLE PRECISION,
    dc_pos_vol      DOUBLE PRECISION,
    dc_online_vol   DOUBLE PRECISION,
    dc_others_vol   DOUBLE PRECISION,
    dc_atm_cash_vol DOUBLE PRECISION,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (bank_name, month)
);

CREATE TABLE IF NOT EXISTS raw_psi (
    id          BIGSERIAL PRIMARY KEY,
    indicator   TEXT NOT NULL,
    month       DATE NOT NULL,
    value       DOUBLE PRECISION,
    unit        TEXT,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (indicator, month)
);

CREATE TABLE IF NOT EXISTS raw_repo_rate (
    id             BIGSERIAL PRIMARY KEY,
    effective_date DATE NOT NULL UNIQUE,
    rate_pct       DOUBLE PRECISION NOT NULL,
    ingested_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw_npci_upi (
    id          BIGSERIAL PRIMARY KEY,
    month       DATE NOT NULL UNIQUE,
    banks_live  INTEGER,
    volume_mn   DOUBLE PRECISION,
    value_cr    DOUBLE PRECISION,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- PROCESSED LAYER (cleaned, merged, model-ready)
-- ============================================================

CREATE TABLE IF NOT EXISTS processed_bank_series (
    id                  BIGSERIAL PRIMARY KEY,
    bank_name           TEXT NOT NULL,
    card_type           TEXT NOT NULL CHECK (card_type IN ('CC', 'DC')),
    month               DATE NOT NULL,
    y                   DOUBLE PRECISION NOT NULL,
    stable_regime_start DATE,
    growth_type         TEXT,
    repo_rate           DOUBLE PRECISION,
    processed_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (bank_name, card_type, month)
);

CREATE TABLE IF NOT EXISTS processed_aggregate (
    id           BIGSERIAL PRIMARY KEY,
    metric       TEXT NOT NULL,
    month        DATE NOT NULL,
    value        DOUBLE PRECISION NOT NULL,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (metric, month)
);

-- ============================================================
-- SERVING LAYER (forecasts for dashboard)
-- ============================================================

CREATE TABLE IF NOT EXISTS forecasts_bank (
    id              BIGSERIAL PRIMARY KEY,
    bank_name       TEXT NOT NULL,
    card_type       TEXT NOT NULL CHECK (card_type IN ('CC', 'DC')),
    forecast_month  DATE NOT NULL,
    yhat            DOUBLE PRECISION NOT NULL,
    yhat_lower      DOUBLE PRECISION,
    yhat_upper      DOUBLE PRECISION,
    model_type      TEXT,
    trained_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (bank_name, card_type, forecast_month)
);

CREATE TABLE IF NOT EXISTS forecasts_aggregate (
    id              BIGSERIAL PRIMARY KEY,
    metric          TEXT NOT NULL,
    forecast_month  DATE NOT NULL,
    yhat            DOUBLE PRECISION NOT NULL,
    yhat_lower      DOUBLE PRECISION,
    yhat_upper      DOUBLE PRECISION,
    model_type      TEXT,
    trained_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (metric, forecast_month)
);

CREATE TABLE IF NOT EXISTS model_metadata (
    id                  BIGSERIAL PRIMARY KEY,
    bank_name           TEXT,
    card_type           TEXT,
    metric              TEXT,
    model_type          TEXT NOT NULL,
    cv_mape             DOUBLE PRECISION,
    oos_mape            DOUBLE PRECISION,
    params_json         JSONB,
    last_trained        TIMESTAMPTZ DEFAULT NOW(),
    training_duration_s DOUBLE PRECISION,
    UNIQUE NULLS NOT DISTINCT (bank_name, card_type, metric)
);

-- ============================================================
-- OPERATIONS LAYER (audit & monitoring)
-- ============================================================

CREATE TABLE IF NOT EXISTS scraper_runs (
    id               UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    source           TEXT NOT NULL,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at     TIMESTAMPTZ,
    status           TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'success', 'failed', 'partial')),
    files_downloaded INTEGER DEFAULT 0,
    records_written  INTEGER DEFAULT 0,
    error_message    TEXT,
    github_run_id    TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    triggered_by    TEXT NOT NULL DEFAULT 'manual' CHECK (triggered_by IN ('cron', 'manual', 'api')),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'success', 'failed', 'partial')),
    steps_completed INTEGER DEFAULT 0,
    steps_failed    INTEGER DEFAULT 0,
    summary_json    JSONB
);

-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_raw_bankwise_bank_month ON raw_bankwise (bank_name, month);
CREATE INDEX IF NOT EXISTS idx_raw_psi_indicator_month ON raw_psi (indicator, month);
CREATE INDEX IF NOT EXISTS idx_raw_npci_month ON raw_npci_upi (month);
CREATE INDEX IF NOT EXISTS idx_processed_bank ON processed_bank_series (bank_name, card_type, month);
CREATE INDEX IF NOT EXISTS idx_processed_agg ON processed_aggregate (metric, month);
CREATE INDEX IF NOT EXISTS idx_forecasts_bank ON forecasts_bank (bank_name, card_type, forecast_month);
CREATE INDEX IF NOT EXISTS idx_forecasts_bank_future ON forecasts_bank (forecast_month);
CREATE INDEX IF NOT EXISTS idx_forecasts_agg ON forecasts_aggregate (metric, forecast_month);
CREATE INDEX IF NOT EXISTS idx_scraper_runs_source ON scraper_runs (source, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs (status, started_at DESC);

-- ============================================================
-- ROW LEVEL SECURITY (read-only for anon, full for service role)
-- ============================================================

ALTER TABLE raw_bankwise ENABLE ROW LEVEL SECURITY;
ALTER TABLE raw_psi ENABLE ROW LEVEL SECURITY;
ALTER TABLE raw_repo_rate ENABLE ROW LEVEL SECURITY;
ALTER TABLE raw_npci_upi ENABLE ROW LEVEL SECURITY;
ALTER TABLE processed_bank_series ENABLE ROW LEVEL SECURITY;
ALTER TABLE processed_aggregate ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecasts_bank ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecasts_aggregate ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE scraper_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;

-- Anon can read forecast/processed/model data (public dashboard)
CREATE POLICY "anon_read_forecasts_bank" ON forecasts_bank FOR SELECT TO anon USING (true);
CREATE POLICY "anon_read_forecasts_agg" ON forecasts_aggregate FOR SELECT TO anon USING (true);
CREATE POLICY "anon_read_model_meta" ON model_metadata FOR SELECT TO anon USING (true);
CREATE POLICY "anon_read_processed_agg" ON processed_aggregate FOR SELECT TO anon USING (true);
CREATE POLICY "anon_read_processed_bank" ON processed_bank_series FOR SELECT TO anon USING (true);
CREATE POLICY "anon_read_scraper_runs" ON scraper_runs FOR SELECT TO anon USING (true);
CREATE POLICY "anon_read_pipeline_runs" ON pipeline_runs FOR SELECT TO anon USING (true);
CREATE POLICY "anon_read_raw_npci" ON raw_npci_upi FOR SELECT TO anon USING (true);

-- Service role has full access (used by pipeline)
CREATE POLICY "service_all_raw_bankwise" ON raw_bankwise FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_raw_psi" ON raw_psi FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_raw_repo" ON raw_repo_rate FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_raw_npci" ON raw_npci_upi FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_processed_bank" ON processed_bank_series FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_processed_agg" ON processed_aggregate FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_forecasts_bank" ON forecasts_bank FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_forecasts_agg" ON forecasts_aggregate FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_model_meta" ON model_metadata FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_scraper_runs" ON scraper_runs FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_pipeline_runs" ON pipeline_runs FOR ALL TO service_role USING (true) WITH CHECK (true);
