-- MIP Phase 3: Agent Tables
-- Run this in Supabase SQL Editor to create agent-related tables.

-- ============================================================
-- AGENT RUNS — top-level log of each agent execution
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_runs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_type    TEXT NOT NULL DEFAULT 'research',
    status      TEXT NOT NULL DEFAULT 'running',
    started_at  TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    summary     TEXT DEFAULT ''
);

-- ============================================================
-- AGENT FINDINGS — structured signals extracted by the agent
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_findings (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_type       TEXT NOT NULL,
    bank_name         TEXT,
    card_type         TEXT,
    title             TEXT NOT NULL,
    impact_direction  TEXT,
    impact_magnitude  TEXT,
    effective_date    DATE,
    details           TEXT,
    confidence        DOUBLE PRECISION DEFAULT 0,
    source_url        TEXT,
    source_name       TEXT,
    discovered_at     TIMESTAMPTZ DEFAULT NOW(),
    used_in_retrain   BOOLEAN DEFAULT FALSE,
    CONSTRAINT valid_signal_type CHECK (
        signal_type IN ('new_card_launch', 'card_discontinuation',
                        'regulatory_change', 'partnership', 'growth_target',
                        'macro_policy', 'infrastructure_change', 'market_event')
    ),
    CONSTRAINT valid_card_type CHECK (
        card_type IS NULL OR card_type IN ('CC', 'DC', 'both')
    ),
    CONSTRAINT valid_direction CHECK (
        impact_direction IS NULL OR impact_direction IN ('positive', 'negative', 'neutral')
    ),
    CONSTRAINT valid_magnitude CHECK (
        impact_magnitude IS NULL OR impact_magnitude IN ('high', 'medium', 'low')
    )
);

CREATE INDEX IF NOT EXISTS idx_findings_signal_type ON agent_findings(signal_type);
CREATE INDEX IF NOT EXISTS idx_findings_bank ON agent_findings(bank_name);
CREATE INDEX IF NOT EXISTS idx_findings_discovered ON agent_findings(discovered_at DESC);

-- ============================================================
-- AGENT RETRAINS — log of every retrain attempt
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_retrains (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          UUID REFERENCES agent_runs(id),
    metric          TEXT NOT NULL,
    bank_name       TEXT,
    old_cv_mape     DOUBLE PRECISION,
    new_cv_mape     DOUBLE PRECISION,
    promoted        BOOLEAN DEFAULT FALSE,
    regressors_used TEXT[] DEFAULT '{}',
    evaluated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_retrains_metric ON agent_retrains(metric);
CREATE INDEX IF NOT EXISTS idx_retrains_promoted ON agent_retrains(promoted);

-- ============================================================
-- RLS Policies
-- ============================================================

ALTER TABLE agent_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_findings ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_retrains ENABLE ROW LEVEL SECURITY;

-- Anon can read findings (for the dashboard)
CREATE POLICY "anon_read_findings" ON agent_findings
    FOR SELECT USING (true);

CREATE POLICY "anon_read_runs" ON agent_runs
    FOR SELECT USING (true);

CREATE POLICY "anon_read_retrains" ON agent_retrains
    FOR SELECT USING (true);

-- Service role has full access (used by the agent pipeline)
CREATE POLICY "service_full_findings" ON agent_findings
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "service_full_runs" ON agent_runs
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "service_full_retrains" ON agent_retrains
    FOR ALL USING (true) WITH CHECK (true);
