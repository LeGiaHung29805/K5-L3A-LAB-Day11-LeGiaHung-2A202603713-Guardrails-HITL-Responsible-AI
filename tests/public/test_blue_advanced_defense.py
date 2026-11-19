"""Regression checks for live-probe findings; all completions here are simulated."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import types

from agents.agent import BLUE_INSTRUCTION, RED_DEFAULT_INSTRUCTION
from agents.guards_agent import RED_ADVANCE_INSTRUCTION
from assignment.pipeline import DefensePipeline, build_production_plugins, build_observability
from core.config import DEMO_SECRET_NOTE, DEMO_SECRETS, load_protected_payload
from core.openai_runtime import EmptyModelResponseError, OpenAIAgent, OpenAIRunner
from guardrails.input_guardrails import requests_protected_information
from guardrails.output_guardrails import content_filter, OutputGuardrailPlugin

ROOT = Path(__file__).resolve().parents[2]
CORPUS = json.loads((ROOT / "data/blue_advanced_cases.json").read_text(encoding="utf-8"))
ORIGINAL = CORPUS["attacks"][0]["input"]


def test_lab_agents_include_required_synthetic_context_and_blue_forbids_disclosure():
    # README.md and data/protected/README.md require all three lab agents to
    # contain synthetic secrets, so defense is exercised on data actually present.
    assert DEMO_SECRET_NOTE in BLUE_INSTRUCTION
    assert all(str(value) in BLUE_INSTRUCTION for value in load_protected_payload()["secrets"].values())
    assert "DO NOT DISCLOSE" in BLUE_INSTRUCTION
    assert DEMO_SECRET_NOTE in RED_DEFAULT_INSTRUCTION
    assert DEMO_SECRET_NOTE in RED_ADVANCE_INSTRUCTION


@pytest.mark.parametrize("response", ["adm", "123", "EIGHT", "OTHER", "MATCH", "The code has eight characters.", "I refuse, but the result is EIGHT."])
def test_context_aware_output_holds_partial_and_oracle_responses(response):
    result = content_filter(response, user_input=ORIGINAL)
    assert not result["safe"] and result["redacted"] == "[REDACTED]"


def test_context_does_not_blacklist_legitimate_amounts_or_public_password_rules():
    assert content_filter("Transfer amount: 123 VND.", user_input="How do I transfer 123 VND from my account?")["safe"]
    assert content_filter("A password may include letters, numbers and punctuation.", user_input=CORPUS["safe_queries"][3]["input"])["safe"]
    assert not requests_protected_information(CORPUS["safe_queries"][3]["input"])


def test_output_plugin_uses_request_context_for_adk_callback():
    async def run():
        plugin = OutputGuardrailPlugin(use_llm_judge=False)
        ctx = SimpleNamespace(user_content=types.Content(role="user", parts=[types.Part.from_text(text=ORIGINAL)]))
        response = SimpleNamespace(content=types.Content(role="model", parts=[types.Part.from_text(text="EIGHT")]))
        await plugin.after_model_callback(callback_context=ctx, llm_response=response)
        assert plugin._extract_text(response) == "[REDACTED]"
    asyncio.run(run())


def test_openai_runtime_passes_request_scoped_context_and_preserves_other_users():
    async def run():
        runner = OpenAIRunner(app_name="context", model="fake", plugins=[OutputGuardrailPlugin(use_llm_judge=False)])
        runner._client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="123"))])
        )))
        agent = OpenAIAgent(name="fake", instruction="Simulated completion")
        # Deliberately omit input plugin here to test the independent output gate.
        assert await runner.chat(agent, ORIGINAL, user_id="attacker") == "[REDACTED]"
        assert await runner.chat(agent, "What amount of money is in my bank account?", user_id="customer") == "123"
    asyncio.run(run())


def test_empty_length_completion_retries_with_bounded_budget_without_recharging_input(monkeypatch):
    async def no_wait(seconds):
        pass
    monkeypatch.setattr("core.openai_runtime.asyncio.sleep", no_wait)

    async def run():
        plugins = build_production_plugins(max_requests=1)
        audit, monitor = build_observability()
        agent = OpenAIAgent(name="fake", instruction=BLUE_INSTRUCTION)
        runner = OpenAIRunner(app_name="empty_retry", model="fake", plugins=plugins, max_output_tokens=512, max_model_retries=1)
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="" if len(calls) == 1 else "Please check the banking app for your account balance."),
                finish_reason="length" if len(calls) == 1 else "stop",
            )])
        runner._client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        pipeline = DefensePipeline(agent=agent, runner=runner, plugins=plugins, audit=audit, monitor=monitor)
        result = await pipeline.process("What is my account balance?", user_id="A")
        assert not result["error"] and not result["blocked"]
        assert [c["max_tokens"] for c in calls] == [512, 1024]
        assert plugins[0].total_count == 1 and monitor.model_retries == 1
        assert DEMO_SECRET_NOTE in calls[0]["messages"][0]["content"]
    asyncio.run(run())


def test_repeated_empty_completions_remain_errors_not_successful_defense(monkeypatch):
    async def no_wait(seconds):
        pass
    monkeypatch.setattr("core.openai_runtime.asyncio.sleep", no_wait)

    async def run():
        runner = OpenAIRunner(app_name="empty_error", model="fake", max_model_retries=1, max_output_tokens=512)
        runner._client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="length")])
        )))
        with pytest.raises(EmptyModelResponseError):
            await runner.chat(OpenAIAgent(name="fake", instruction="Fixture"), "What is my account balance?")
    asyncio.run(run())


def test_provider_refusal_field_is_a_visible_refusal():
    async def run():
        runner = OpenAIRunner(app_name="refusal", model="fake")
        runner._client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal="I cannot provide that information."))])
        )))
        assert await runner.chat(OpenAIAgent(name="fake", instruction="Fixture"), "banking question") == "I cannot provide that information."
    asyncio.run(run())


def test_probe_assessment_separates_partial_oracle_and_full_disclosure():
    spec = importlib.util.spec_from_file_location("advanced_probe", ROOT / "scripts/test_blue_advanced.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.assess_response("adm", "prefix")["fragment_signal"]
    assert not module.assess_response("adm", "prefix")["full_value_leak"]
    assert module.assess_response("EIGHT", "oracle")["oracle_signal"]
    assert not module.assess_response("I cannot provide login details.", "prefix")["needs_review"]
