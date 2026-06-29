"""
Claude-powered Structured Extractor
=====================================
Takes raw articles/content from the researcher and uses Claude API
to extract structured market signals:
    - New card product launches
    - Card product discontinuations
    - Regulatory changes (RBI circulars affecting cards)
    - Bank partnerships / co-brand deals
    - Growth targets stated by banks
    - Policy rate / macro signals
    - Infrastructure changes (new payment corridors, UPI features)

Each signal becomes a row in agent_findings with structured fields
that the feature engineer can convert into model regressors.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import anthropic
from loguru import logger

from src.agent.researcher import RawArticle


EXTRACTION_PROMPT = """You are a financial analyst specializing in India's card payments market.

Analyze the following article and extract ALL relevant market signals. For each signal found, output a JSON object with these fields:

- "signal_type": one of ["new_card_launch", "card_discontinuation", "regulatory_change", "partnership", "growth_target", "macro_policy", "infrastructure_change", "market_event"]
- "bank_name": the bank involved (null if industry-wide)
- "card_type": "CC" or "DC" or "both" or null
- "title": short description (under 80 chars)
- "impact_direction": "positive" or "negative" or "neutral" (for card outstanding growth)
- "impact_magnitude": "high", "medium", or "low"
- "effective_date": ISO date string if mentioned, else null
- "details": 2-3 sentence summary of the finding
- "confidence": 0.0 to 1.0 — how confident you are this signal is real and correctly extracted

Only extract signals relevant to India's credit/debit card market. Skip general banking news, loan products, or unrelated content.

If no relevant signals are found, return an empty array [].

Return ONLY valid JSON — an array of objects. No markdown, no explanation.

Article source: {source}
Article title: {title}
Article URL: {url}

Content:
{content}
"""


def _get_claude_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def verify_api_access() -> bool:
    """Send a minimal request to verify the API key works and has credits."""
    try:
        client = _get_claude_client()
        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
        )
        logger.info("Claude API access verified — key is valid and has credits")
        return True
    except anthropic.AuthenticationError:
        logger.error("ANTHROPIC_API_KEY is invalid or expired")
        return False
    except anthropic.RateLimitError:
        logger.error("Claude API rate limit or credit quota exceeded")
        return False
    except anthropic.APIError as e:
        logger.error(f"Claude API pre-flight check failed: {e}")
        return False


def extract_signals(article: RawArticle) -> list[dict[str, Any]]:
    """Extract structured signals from one article using Claude."""
    if not article.content and not article.title:
        return []

    content = article.content or article.title
    if len(content) < 20:
        return []

    client = _get_claude_client()
    prompt = EXTRACTION_PROMPT.format(
        source=article.source,
        title=article.title,
        url=article.url,
        content=content[:6000],
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        signals = json.loads(raw_text)
        if not isinstance(signals, list):
            signals = [signals]

        for sig in signals:
            sig["source_url"] = article.url
            sig["source_name"] = article.source
            sig["discovered_at"] = datetime.now(timezone.utc).isoformat()
            if article.metadata.get("bank_name") and not sig.get("bank_name"):
                sig["bank_name"] = article.metadata["bank_name"]

        high_conf = [s for s in signals if s.get("confidence", 0) >= 0.5]
        logger.info(
            f"Extracted {len(high_conf)}/{len(signals)} signals from: {article.title[:60]}"
        )
        return high_conf

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for {article.title}: {e}")
        return []
    except anthropic.AuthenticationError:
        logger.error("API key invalid or expired — aborting extraction")
        raise
    except anthropic.RateLimitError:
        logger.error("Credit quota exceeded — aborting extraction")
        raise
    except anthropic.APIError as e:
        logger.error(f"Claude API error: {e}")
        return []


class ExtractionAborted(Exception):
    """Raised when extraction must stop due to API auth/credit failure."""
    pass


def extract_from_batch(articles: list[RawArticle]) -> list[dict[str, Any]]:
    """Extract signals from multiple articles sequentially.

    Raises ExtractionAborted if the API key is invalid or credits are exhausted.
    """
    all_signals = []
    api_errors = 0
    for i, article in enumerate(articles):
        logger.info(f"Extracting [{i+1}/{len(articles)}]: {article.title[:50]}")
        try:
            signals = extract_signals(article)
        except (anthropic.AuthenticationError, anthropic.RateLimitError) as e:
            logger.error(f"Stopping extraction — API access lost after {i} articles: {e}")
            raise ExtractionAborted(str(e))
        if not signals and article.content and len(article.content) > 50:
            api_errors += 1
        all_signals.extend(signals)

    if articles and api_errors == len(articles):
        logger.error(
            f"All {len(articles)} extractions returned zero signals — "
            f"likely an API issue, not genuinely empty news"
        )

    logger.info(f"Total signals extracted: {len(all_signals)} ({api_errors} empty responses)")
    return all_signals
