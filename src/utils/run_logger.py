"""
MIP -- Structured Run Logger
=============================
Writes a markdown log file for every model run to logs/runs/.
Never crashes the model run -- all operations are try/except wrapped.

Usage:
    from src.utils.run_logger import RunLogger

    log = RunLogger("aggregate_cc")
    log.add("Training rows", 158)
    log.add("CV MAPE mean", "3.46%")
    log.add_section("Regressors", ["repo_rate (lag=9)"])
    log.save()
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOGS_DIR     = _PROJECT_ROOT / "logs" / "runs"


class RunLogger:
    """Structured markdown logger for model runs."""

    def __init__(self, model_name: str):
        self._name = model_name
        self._ts = datetime.now()
        self._entries: list[str] = []
        self._warnings: list[str] = []

        self._entries.append(f"# Model Run: {model_name}")
        self._entries.append(f"")
        self._entries.append(f"Timestamp: {self._ts:%Y-%m-%d %H:%M:%S}")
        self._entries.append(f"")

    def add(self, key: str, value) -> None:
        """Add a key-value metric."""
        try:
            self._entries.append(f"- **{key}:** {value}")
        except Exception:
            pass

    def add_section(self, title: str, items: list[str]) -> None:
        """Add a titled section with bullet items."""
        try:
            self._entries.append(f"")
            self._entries.append(f"### {title}")
            for item in items:
                self._entries.append(f"- {item}")
        except Exception:
            pass

    def add_warning(self, msg: str) -> None:
        """Add a warning flag."""
        try:
            self._warnings.append(msg)
            self._entries.append(f"- **WARNING:** {msg}")
        except Exception:
            pass

    def add_table(self, headers: list[str], rows: list[list]) -> None:
        """Add a markdown table."""
        try:
            self._entries.append("")
            self._entries.append("| " + " | ".join(str(h) for h in headers) + " |")
            self._entries.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in rows:
                self._entries.append("| " + " | ".join(str(v) for v in row) + " |")
            self._entries.append("")
        except Exception:
            pass

    def save(self) -> Path | None:
        """Write the log to logs/runs/. Returns the path or None on failure."""
        try:
            _LOGS_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"{self._ts:%Y-%m-%d_%H-%M}_{self._name}.md"
            path = _LOGS_DIR / filename
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(self._entries))
            return path
        except Exception:
            return None
