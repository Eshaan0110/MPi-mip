"""
MIP — Email Alerting (P0.2)
=============================
Pipeline failure/success email notifications via Resend API.

Requires RESEND_API_KEY in environment (GitHub Actions secret).
Recipient list is configured via ALERT_EMAILS env var (comma-separated)
or defaults to the configured list below.

Usage:
    from src.utils.email_alert import send_pipeline_alert
    send_pipeline_alert("monthly_pipeline", status="success", details={...})
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import httpx
from loguru import logger

RESEND_API_URL = "https://api.resend.com/emails"
TEAMS_WEBHOOK_ENV = "TEAMS_WEBHOOK_URL"
DEFAULT_FROM = "MIP Pipeline <onboarding@resend.dev>"

DEFAULT_RECIPIENTS = [
    "eshan.intern@mtl.manipalgroup.info",
]


def _get_recipients() -> list[str]:
    """Get recipient list from env or defaults."""
    env_emails = os.environ.get("ALERT_EMAILS", "")
    if env_emails.strip():
        return [e.strip() for e in env_emails.split(",") if e.strip()]
    return DEFAULT_RECIPIENTS


def _build_subject(pipeline: str, status: str) -> str:
    icon = "✅" if status == "success" else "❌"
    return f"{icon} MIP {pipeline}: {status.upper()}"


def _build_html(
    pipeline: str,
    status: str,
    details: dict | None = None,
    run_url: str | None = None,
) -> str:
    """Build a clean HTML email body."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status_color = "#22c55e" if status == "success" else "#ef4444"

    details_rows = ""
    if details:
        for key, val in details.items():
            details_rows += f"""
            <tr>
                <td style="padding:6px 12px;border-bottom:1px solid #e5e7eb;color:#6b7280;font-size:14px;">{key}</td>
                <td style="padding:6px 12px;border-bottom:1px solid #e5e7eb;font-size:14px;">{val}</td>
            </tr>"""

    run_link = ""
    if run_url:
        run_link = f'<p style="margin-top:16px;"><a href="{run_url}" style="color:#2563eb;">View workflow run</a></p>'

    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:600px;margin:0 auto;">
        <div style="background:#1e293b;padding:20px 24px;border-radius:8px 8px 0 0;">
            <h2 style="color:white;margin:0;font-size:18px;">MIP Pipeline Report</h2>
        </div>
        <div style="border:1px solid #e5e7eb;border-top:none;padding:24px;border-radius:0 0 8px 8px;">
            <div style="display:inline-block;padding:4px 12px;border-radius:4px;background:{status_color};color:white;font-weight:600;font-size:14px;">
                {status.upper()}
            </div>
            <p style="margin-top:12px;color:#374151;font-size:14px;">
                <strong>Pipeline:</strong> {pipeline}<br>
                <strong>Time:</strong> {timestamp}
            </p>
            {"<table style='width:100%;border-collapse:collapse;margin-top:16px;'>" + details_rows + "</table>" if details_rows else ""}
            {run_link}
        </div>
        <p style="color:#9ca3af;font-size:12px;margin-top:12px;text-align:center;">
            MIP — Market Intelligence Platform
        </p>
    </div>
    """


def send_pipeline_alert(
    pipeline: str,
    status: str = "failure",
    details: dict | None = None,
    run_url: str | None = None,
) -> bool:
    """Send a pipeline status email via Resend.

    Args:
        pipeline: pipeline name (e.g. 'monthly_pipeline', 'agent_pipeline')
        status: 'success' or 'failure'
        details: optional dict of key-value pairs to include in the email
        run_url: optional link to the GitHub Actions run

    Returns:
        True if email sent successfully, False otherwise
    """
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        logger.warning("RESEND_API_KEY not set — skipping email alert")
        return False

    recipients = _get_recipients()
    if not recipients:
        logger.warning("No alert recipients configured")
        return False

    subject = _build_subject(pipeline, status)
    html = _build_html(pipeline, status, details, run_url)

    payload = {
        "from": DEFAULT_FROM,
        "to": recipients,
        "subject": subject,
        "html": html,
    }

    try:
        resp = httpx.post(
            RESEND_API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        if resp.status_code in (200, 201):
            logger.info(f"Alert email sent: {subject} -> {recipients}")
            return True
        else:
            logger.error(f"Resend API error {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to send alert email: {e}")
        return False


def send_health_check_alert(
    checks: dict[str, bool],
    details: dict | None = None,
) -> bool:
    """Send a pipeline health check summary email.

    Args:
        checks: dict of check_name -> passed (True/False)
        details: optional extra details
    """
    all_passed = all(checks.values())
    status = "success" if all_passed else "failure"

    check_details = {}
    for name, passed in checks.items():
        check_details[name] = "PASS" if passed else "FAIL"
    if details:
        check_details.update(details)

    return send_pipeline_alert(
        "Health Check",
        status=status,
        details=check_details,
    )


def send_teams_webhook(
    pipeline: str,
    status: str = "failure",
    details: dict | None = None,
    run_url: str | None = None,
) -> bool:
    """Send a pipeline status notification to Microsoft Teams via webhook.

    Requires TEAMS_WEBHOOK_URL env var. Uses Adaptive Card format.
    """
    webhook_url = os.environ.get(TEAMS_WEBHOOK_ENV)
    if not webhook_url:
        logger.debug("TEAMS_WEBHOOK_URL not set — skipping Teams notification")
        return False

    icon = "✅" if status == "success" else "❌"
    color = "good" if status == "success" else "attention"

    facts = []
    if details:
        for key, val in details.items():
            facts.append({"title": key, "value": str(val)})

    actions = []
    if run_url:
        actions.append({
            "type": "Action.OpenUrl",
            "title": "View Workflow Run",
            "url": run_url,
        })

    card = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {
                        "type": "TextBlock",
                        "size": "medium",
                        "weight": "bolder",
                        "text": f"{icon} MIP {pipeline}: {status.upper()}",
                    },
                    {
                        "type": "FactSet",
                        "facts": [
                            {"title": "Pipeline", "value": pipeline},
                            {"title": "Status", "value": status.upper()},
                            {"title": "Time", "value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
                            *facts,
                        ],
                    },
                ],
                "actions": actions,
            },
        }],
    }

    try:
        resp = httpx.post(webhook_url, json=card, timeout=15)
        if resp.status_code in (200, 201, 202):
            logger.info(f"Teams webhook sent: {pipeline} {status}")
            return True
        else:
            logger.error(f"Teams webhook error {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to send Teams webhook: {e}")
        return False


def notify_all(
    pipeline: str,
    status: str = "failure",
    details: dict | None = None,
    run_url: str | None = None,
) -> dict[str, bool]:
    """Send pipeline alert via all configured channels (email + Teams)."""
    return {
        "email": send_pipeline_alert(pipeline, status, details, run_url),
        "teams": send_teams_webhook(pipeline, status, details, run_url),
    }


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        ok = send_pipeline_alert(
            "test_pipeline",
            status="success",
            details={
                "CC MAPE": "3.21%",
                "DC MAPE": "4.56%",
                "Models trained": "5/5",
                "Duration": "8m 32s",
            },
            run_url="https://github.com/Eshaan0110/MPi-mip/actions",
        )
        print(f"Test email {'sent' if ok else 'FAILED'}")
    else:
        print("Usage: python -m src.utils.email_alert --test")
        print("Requires RESEND_API_KEY env var")
