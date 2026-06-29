"""
Agent Pipeline — Full Autonomous Loop
=======================================
Orchestrates the complete research → extract → feature → retrain cycle.

Usage:
    # Single run
    uv run python -m src.agent.pipeline

    # With environment
    ANTHROPIC_API_KEY=sk-... SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... \
        uv run python -m src.agent.pipeline
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from loguru import logger

from src.agent.researcher import run_research
from src.agent.extractor import extract_from_batch, verify_api_access, ExtractionAborted
from src.agent.features import signals_to_regressors, select_top_regressors
from src.agent.retrainer import retrain_aggregate
from src.agent.store import (
    log_run_start,
    log_run_end,
    save_findings_batch,
)


async def run_agent_pipeline() -> dict:
    """Execute one full agent cycle.

    Returns:
        Summary dict with counts and retrain results.
    """
    run_id = log_run_start("full_pipeline")
    summary = {
        "run_id": run_id,
        "articles_collected": 0,
        "signals_extracted": 0,
        "regressors_generated": 0,
        "retrain_results": {},
    }

    try:
        # ── Pre-flight: verify API access ────────────────────────────
        logger.info("=" * 60)
        logger.info("PRE-FLIGHT: Verifying Claude API access")
        logger.info("=" * 60)

        if not verify_api_access():
            msg = "Claude API access failed — key invalid or credits exhausted. Aborting."
            logger.error(msg)
            log_run_end(run_id, "failed", msg)
            summary["error"] = msg
            return summary

        # ── Step 1: Research ──────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 1: Research — crawling sources")
        logger.info("=" * 60)

        articles = await run_research()
        summary["articles_collected"] = len(articles)

        if not articles:
            logger.warning("No articles found — skipping extraction")
            log_run_end(run_id, "success", "No articles found")
            return summary

        # ── Step 2: Extract ───────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 2: Extract — Claude-powered signal extraction")
        logger.info("=" * 60)

        try:
            signals = extract_from_batch(articles)
        except ExtractionAborted as e:
            msg = f"Extraction aborted — API credits exhausted or key invalid: {e}"
            logger.error(msg)
            log_run_end(run_id, "failed", msg)
            summary["error"] = msg
            return summary
        summary["signals_extracted"] = len(signals)

        if signals:
            findings = []
            for sig in signals:
                findings.append({
                    "signal_type": sig.get("signal_type"),
                    "bank_name": sig.get("bank_name"),
                    "card_type": sig.get("card_type"),
                    "title": sig.get("title", "")[:200],
                    "impact_direction": sig.get("impact_direction"),
                    "impact_magnitude": sig.get("impact_magnitude"),
                    "effective_date": sig.get("effective_date"),
                    "details": sig.get("details", "")[:500],
                    "confidence": sig.get("confidence", 0),
                    "source_url": sig.get("source_url", ""),
                    "source_name": sig.get("source_name", ""),
                    "discovered_at": sig.get("discovered_at",
                                             datetime.now(timezone.utc).isoformat()),
                })
            saved = save_findings_batch(findings)
            logger.info(f"Saved {saved} findings to Supabase")

        if not signals:
            logger.info("No signals extracted — skipping retrain")
            log_run_end(run_id, "success",
                        f"Collected {len(articles)} articles, 0 signals")
            return summary

        # ── Step 3: Feature Engineering ───────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 3: Feature Engineering — signals → regressors")
        logger.info("=" * 60)

        regressors_df = signals_to_regressors(signals)
        summary["regressors_generated"] = len(regressors_df.columns)

        if regressors_df.empty:
            logger.info("No regressors generated — skipping retrain")
            log_run_end(run_id, "success",
                        f"{len(articles)} articles, {len(signals)} signals, 0 regressors")
            return summary

        # ── Step 4: Retrain & Evaluate ────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 4: Retrain — ensemble with new regressors")
        logger.info("=" * 60)

        for metric in ["cc_outstanding", "dc_outstanding"]:
            logger.info(f"\n--- Retraining: {metric} ---")
            result = retrain_aggregate(
                metric=metric,
                extra_regressors=regressors_df,
                run_id=run_id,
            )
            summary["retrain_results"][metric] = result

        # ── Done ──────────────────────────────────────────────────────
        promoted = sum(
            1 for r in summary["retrain_results"].values()
            if r.get("promoted")
        )
        summary_text = (
            f"Articles: {summary['articles_collected']}, "
            f"Signals: {summary['signals_extracted']}, "
            f"Regressors: {summary['regressors_generated']}, "
            f"Models promoted: {promoted}/2"
        )
        logger.info(f"\nPipeline complete: {summary_text}")
        log_run_end(run_id, "success", summary_text)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        log_run_end(run_id, "failed", str(e))
        raise

    return summary


def main():
    """Entry point for command-line execution."""
    import sys
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")

    result = asyncio.run(run_agent_pipeline())

    print("\n" + "=" * 60)
    print("AGENT PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Articles collected:   {result['articles_collected']}")
    print(f"  Signals extracted:    {result['signals_extracted']}")
    print(f"  Regressors generated: {result['regressors_generated']}")
    for metric, res in result.get("retrain_results", {}).items():
        status = "PROMOTED" if res.get("promoted") else "KEPT OLD"
        old = res.get("old_mape", "?")
        new = res.get("new_mape", "?")
        print(f"  {metric}: {old}% → {new}% [{status}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
