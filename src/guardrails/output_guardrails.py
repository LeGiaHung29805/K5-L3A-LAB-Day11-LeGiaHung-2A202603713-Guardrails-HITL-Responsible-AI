"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import asyncio
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.utils import chat_with_agent
from core.config import DEMO_SECRETS
from guardrails.input_guardrails import requests_protected_information
from guardrails.text_security import InspectionLimitError, MAX_OUTPUT_CHARS, normalize_text, policy_text, security_views


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

def content_filter(response: str, *, user_input: str | None = None) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    if len(response) > MAX_OUTPUT_CHARS:
        return {"safe": False, "issues": ["size: output exceeds inspection limit"], "redacted": "[REDACTED]"}
    # Do not infer safety from the candidate's wording ("I refuse ... but ...").
    # If a privileged probe reached the model, hold its entire response, including
    # an otherwise innocuous MATCH, a number, or an unknown partial encoding.
    if user_input and requests_protected_information(user_input):
        return {"safe": False, "issues": ["protected_request: response withheld"], "redacted": "[REDACTED]"}
    issues = []
    normalized = normalize_text(response)
    redacted = normalized

    # Ranh giới chữ số ngăn regex che một phần của CCCD hoặc số dài hơn.
    PII_PATTERNS = {
        "phone": r"(?<![\w+])(?:0(?:[ .-]?\d){9,10}|\+84(?:[ .-]?\d){9})(?![ .-]?\d)",
        "email": r"[\w.+%-]+@[\w.-]+\.[a-zA-Z]{2,}",
        "national_id": r"(?<!\d)(?:\d{12}|\d{9})(?!\d)",
        "api_key": r"\b(?:sk-[a-zA-Z0-9_-]+|AIza[a-zA-Z0-9_-]{30,})\b",
        "password": r"\b(?:password|mật\s*khẩu|mat\s*khau)\s*(?:[:=]|\bis\b|là|\bla\b)\s*(?!\[REDACTED\])(?!(?:mở|mo|chọn|chon|open|choose)\s+(?:ứng dụng|ung dung|app|cài đặt|cai dat|settings)\b)[^\s,;]+",
        "formatted_id": r"\b(?:CCCD|CMND|national\s+id)\s*[:=]?\s*\d(?:[ .-]?\d){8,11}(?![ .-]?\d)",
        "internal_host": r"\b(?:[a-zA-Z0-9-]+\.)+internal(?::\d+)?\b",
    }

    # Bắt cả secret đứng riêng, không cần nhãn "password=" hay "API key".
    # Nguồn dữ liệu dùng chung với agent và bộ phát hiện leak của lab.
    if DEMO_SECRETS:
        PII_PATTERNS["protected_value"] = "|".join(
            re.escape(secret)
            for secret in sorted(DEMO_SECRETS, key=len, reverse=True)
        )

    for name, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, normalized, re.IGNORECASE)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

    # A reversible encoding is still disclosure. Withhold the whole message when
    # a decoded/obfuscated view is sensitive: its offsets do not map to the original.
    try:
        views = security_views(response)
    except InspectionLimitError:
        return {"safe": False, "issues": ["inspection_limit: too many encoded views"], "redacted": "[REDACTED]"}
    for view in views:
        folded = policy_text(view)
        compact = re.sub(r"[^a-z0-9]", "", folded)
        # High-confidence literal prefixes only; do not blacklist numeric suffixes
        # such as 123, which are ordinary amounts/transaction references too.
        if any(re.search(r"(?<!\w)" + re.escape(policy_text(secret[:3])) + r"(?!\w)", folded)
               for secret in DEMO_SECRETS if len(secret) >= 6 and not secret[:3].isdigit()):
            issues.append("protected_fragment: lab secret prefix found")
            redacted = "[REDACTED]"
            break
        if any(
            needle and (needle in compact or needle in compact[::-1])
            for secret in DEMO_SECRETS
            for needle in [re.sub(r"[^a-z0-9]", "", policy_text(secret))]
        ):
            if not issues:
                issues.append("protected_value: encoded or obfuscated value found")
            redacted = "[REDACTED]"
            break
        if view != normalized and any(re.search(pattern, view, re.IGNORECASE) for pattern in PII_PATTERNS.values()):
            issues.append("encoded_sensitive: decoded PII or credential found")
            redacted = "[REDACTED]"
            break
        if folded != view.casefold() and any(re.search(pattern, folded, re.IGNORECASE) for pattern in PII_PATTERNS.values()):
            issues.append("obfuscated_sensitive: normalized PII or credential found")
            redacted = "[REDACTED]"
            break

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted if issues else response,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": False, "verdict": "Judge unavailable"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = verdict.strip().upper() == "SAFE"
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True, *, judge_timeout_seconds=10):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        if judge_timeout_seconds <= 0:
            raise ValueError("judge_timeout_seconds must be positive")
        self.judge_timeout_seconds = judge_timeout_seconds

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts or []:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        user_content = getattr(callback_context, "user_content", None)
        user_text = "".join(part.text or "" for part in (getattr(user_content, "parts", None) or []))
        filtered = content_filter(response_text, user_input=user_text)
        final_text = filtered["redacted"]
        if not filtered["safe"]:
            self.redacted_count += 1

        # Judge là tùy chọn; chỉ gửi bản đã che dữ liệu cho bước đánh giá này.
        if self.use_llm_judge:
            try:
                verdict = await asyncio.wait_for(llm_safety_check(final_text), timeout=self.judge_timeout_seconds)
                approved = isinstance(verdict, dict) and verdict.get("safe") is True
            except Exception:
                # No exception body or candidate response is released on failure.
                approved = False
            if not approved:
                self.blocked_count += 1
                final_text = (
                    "Tôi chưa thể cung cấp câu trả lời này một cách an toàn. "
                    "Vui lòng liên hệ nhân viên hỗ trợ VinBank."
                )

        if final_text != response_text:
            llm_response.content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=final_text)],
            )

        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
