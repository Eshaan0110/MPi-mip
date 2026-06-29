"""
MIP Phase 3 — Autonomous Research & Retrain Agent
==================================================
Researches market intelligence from RBI circulars, bank newsrooms, and
financial news. Extracts structured signals via Claude API, engineers
features for the ensemble models, retrains, evaluates, and (if better)
promotes new forecasts.

Modules:
    researcher  — crawls sources, downloads PDFs/pages
    extractor   — Claude-powered structured extraction
    features    — maps extracted signals → model regressors
    retrainer   — orchestrates ensemble retrain + evaluation
    scheduler   — cron-based autonomous loop
    store       — Supabase persistence for findings
"""
