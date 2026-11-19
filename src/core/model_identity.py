"""Verify the provider-reported Gemini version before collecting CP4 evidence."""
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

from core.config import get_red_model, get_red_provider

ROOT = Path(__file__).resolve().parents[2]


class ModelIdentityError(RuntimeError):
    """The requested model cannot be verified from the provider response."""


def verify_red_model_identity(*, client=None, report_path=None, max_retries=0):
    """A small real probe; never equate a request alias with the serving version.

    This checks Gemini only. An unknown/mismatched version fails before attacks.
    The strict exact match is intentional for the lab's exact-model requirement.
    """
    if get_red_provider() != "gemini":
        return None

    requested = get_red_model()
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "requested_model": requested,
        "reported_model_version": None,
        "verified": False,
        "status": "not_checked",
        "attempts": [],
    }
    if client is None:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            http_options=types.HttpOptions(timeout=30000),
        )
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=requested, contents="Reply with OK.", config={"max_output_tokens": 8}
            )
            reported = response.model_version
            report.update(
                reported_model_version=reported,
                response_id=getattr(response, "response_id", None),
                verified=reported == requested,
                status="verified" if reported == requested else "model_mismatch" if reported else "unverifiable",
            )
            report.pop("error_type", None)
            report.pop("http_status", None)
            report["attempts"].append({"attempt": attempt + 1, "status": report["status"]})
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                report["usage_metadata"] = usage.model_dump(mode="json", exclude_none=True)
            break  # Never retry a model mismatch to conceal substitution.
        except Exception as exc:
            # Do not write SDK exceptions or authentication headers to the report.
            status = getattr(exc, "code", None)
            report.update(status="provider_error", error_type=type(exc).__name__, http_status=status)
            report["attempts"].append({"attempt": attempt + 1, "status": "provider_error", "http_status": status})
            if attempt == max_retries or status not in (429, 500, 502, 503, 504):
                break
            delay = 2 ** (attempt + 1)
            print(f"Model identity probe HTTP {status}; retry in {delay}s.", flush=True)
            time.sleep(delay)

    path = Path(report_path or ROOT / "outputs" / "cp4_model_identity.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CP4 identity: requested={requested}, reported={report['reported_model_version']}, status={report['status']}", flush=True)
    if not report["verified"]:
        raise ModelIdentityError(
            f"CP4 stopped before attacks: requested {requested}, provider reported "
            f"{report['reported_model_version']} ({report['status']}). See {path}. "
            "Do not label this as a verified run of the requested model."
        ) from None
    return report
