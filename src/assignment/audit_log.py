"""
Assignment 11 — Audit log with redaction and request correlation.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path

from agents.security_boundary import normalize_for_security
from guardrails.output_guardrails import content_filter


def redact_for_log(text: str) -> str:
    """Dùng lại CP2, làm sạch trước khi lưu trong bộ nhớ hoặc export."""
    return content_filter(normalize_for_security(text))["redacted"]


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Open one interaction; request_id separates concurrent requests."""
        key = (user_id, request_id)
        if key in self._open:
            raise ValueError("Request already has an open audit entry")
        self._open[key] = {
            "request_id": request_id or uuid4().hex,
            "user_id": redact_for_log(user_id),
            "timestamp": utc_now_iso(),
            "input": redact_for_log(text),
            "started": time.perf_counter(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Close the matching interaction and append its sanitized outcome."""
        key = (user_id, request_id)
        if key not in self._open:
            raise ValueError("No matching input audit entry")
        entry = self._open.pop(key)
        started = entry.pop("started")
        entry.update(
            completed_at=utc_now_iso(),
            output=redact_for_log(text),
            blocked=bool(blocked),
            layer=layer,
            latency_ms=round(max(0, time.perf_counter() - started) * 1000, 3),
        )
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
