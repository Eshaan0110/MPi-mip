"""
Supabase persistence for agent findings and retrain history.

Tables created by migration 002_agent_schema.sql:
    agent_findings     — structured signals extracted from sources
    agent_retrains     — retrain run log with before/after MAPE
    agent_runs         — top-level agent run log
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from supabase import create_client, Client
from loguru import logger


def _get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def log_run_start(run_type: str = "research") -> str:
    """Insert a new agent_runs row, return its id."""
    client = _get_client()
    row = {
        "run_type": run_type,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    result = client.table("agent_runs").insert(row).execute()
    run_id = result.data[0]["id"]
    logger.info(f"Agent run started: {run_id} ({run_type})")
    return run_id


def log_run_end(run_id: str, status: str = "success", summary: str = "") -> None:
    client = _get_client()
    client.table("agent_runs").update({
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
    }).eq("id", run_id).execute()
    logger.info(f"Agent run {run_id} finished: {status}")


def save_finding(finding: dict[str, Any]) -> str:
    """Persist one extracted finding. Returns its id."""
    client = _get_client()
    result = client.table("agent_findings").insert(finding).execute()
    return result.data[0]["id"]


def save_findings_batch(findings: list[dict[str, Any]]) -> int:
    """Persist multiple findings. Returns count saved."""
    if not findings:
        return 0
    client = _get_client()
    result = client.table("agent_findings").insert(findings).execute()
    return len(result.data)


def get_recent_findings(days: int = 30, limit: int = 200) -> list[dict]:
    """Fetch recent findings for feature engineering."""
    client = _get_client()
    cutoff = datetime.now(timezone.utc).isoformat()
    result = (
        client.table("agent_findings")
        .select("*")
        .gte("discovered_at", cutoff)
        .order("discovered_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data


def log_retrain(
    run_id: str,
    metric: str,
    bank_name: str | None,
    old_mape: float,
    new_mape: float,
    promoted: bool,
    regressors_used: list[str],
) -> None:
    """Log a retrain attempt."""
    client = _get_client()
    client.table("agent_retrains").insert({
        "run_id": run_id,
        "metric": metric,
        "bank_name": bank_name,
        "old_cv_mape": old_mape,
        "new_cv_mape": new_mape,
        "promoted": promoted,
        "regressors_used": regressors_used,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }).execute()
    action = "PROMOTED" if promoted else "DISCARDED"
    logger.info(f"Retrain {metric}/{bank_name}: {old_mape:.2f}% → {new_mape:.2f}% [{action}]")
