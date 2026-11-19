"""CP3 behavior tests; simulated model traffic never calls an external API."""
import asyncio
import json
from types import SimpleNamespace

import jsonschema
import pytest

import assignment.rate_limiter as rate_mod
from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from assignment.pipeline import (
    ROOT, DefensePipeline, build_observability, build_production_plugins,
    create_offline_blue_pair, is_egress_allowed, run_assignment_suite,
)


def make_defense(*, max_requests=10):
    plugins = build_production_plugins(max_requests=max_requests)
    audit, monitor = build_observability()
    agent, runner = create_offline_blue_pair(plugins)
    return DefensePipeline(agent=agent, runner=runner, plugins=plugins, audit=audit, monitor=monitor)


@pytest.mark.parametrize("limit,window", [(0, 60), (-1, 60), (10, 0), (10, -1)])
def test_invalid_rate_limit_configuration(limit, window):
    with pytest.raises(ValueError):
        rate_mod.RateLimitPlugin(max_requests=limit, window_seconds=window)


def test_sliding_window_boundary_user_isolation_and_rejected_requests(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(rate_mod.time, "monotonic", lambda: clock["now"])

    async def run():
        limiter = rate_mod.RateLimitPlugin(max_requests=2, window_seconds=60)

        async def ask(user):
            return await limiter.on_user_message_callback(
                invocation_context=SimpleNamespace(user_id=user), user_message=None
            )

        assert await ask("A") is None
        clock["now"] = 159.0
        assert await ask("A") is None
        assert await ask("A") is not None
        assert len(limiter.user_windows["A"]) == 2
        assert await ask("B") is None
        clock["now"] = 160.0  # Exactly 60 seconds: first request expires.
        assert await ask("A") is None
        assert list(limiter.user_windows["A"]) == [159.0, 160.0]
        assert limiter.blocked_count == 1 and limiter.total_count == 5

    asyncio.run(run())


def test_audit_correlates_interleaved_requests_and_redacts_before_storage(tmp_path):
    audit = AuditLogPlugin()
    audit.record_input(user_id="customer", request_id="first", text="Email lan@example.com and admin123")
    audit.record_input(user_id="customer", request_id="second", text="Account balance?")
    assert "admin123" not in json.dumps(list(audit._open.values()))
    assert "lan@example.com" not in json.dumps(list(audit._open.values()))
    audit.record_output(user_id="customer", request_id="second", text="Ordinary answer")
    audit.record_output(user_id="customer", request_id="first", text="phone 0901234567", blocked=True, layer="egress")
    assert [entry["request_id"] for entry in audit.logs] == ["second", "first"]
    assert not audit._open
    assert all(entry["latency_ms"] >= 0 for entry in audit.logs)
    assert all(entry["timestamp"].endswith("+00:00") for entry in audit.logs)
    path = audit.export_json(tmp_path / "nested" / "audit.json")
    saved = path.read_text(encoding="utf-8")
    for value in ("admin123", "lan@example.com", "0901234567"):
        assert value not in saved


def test_audit_rejects_duplicate_pending_request_and_missing_input():
    audit = AuditLogPlugin()
    audit.record_input(user_id="A", text="Balance?", request_id="same")
    with pytest.raises(ValueError):
        audit.record_input(user_id="A", text="Balance?", request_id="same")
    with pytest.raises(ValueError):
        audit.record_output(user_id="A", text="Answer", request_id="missing")


def test_monitor_alerts_have_no_duplicates_and_clear_when_metrics_recover(tmp_path):
    monitor = MonitoringAlert()
    assert monitor.check_metrics() == []
    monitor.total_requests, monitor.blocked_requests = 10, 6
    monitor.rate_limit_hits = 5
    monitor.judge_checks, monitor.judge_fails = 10, 3
    for _ in range(3):
        assert {alert.metric for alert in monitor.check_metrics()} == {
            "block_rate", "rate_limit_hits", "judge_fail_rate"
        }
        assert len(monitor.alerts) == 3
    monitor.total_requests, monitor.blocked_requests = 100, 1
    monitor.rate_limit_hits = monitor.judge_fails = 0
    assert monitor.check_metrics() == []
    path = monitor.export_json(tmp_path / "metrics.json")
    assert json.loads(path.read_text())["block_rate"] == 0.01


@pytest.mark.parametrize("destination", [
    "http://api.vinbank.example/v1/transfers",
    "https://evil.example/collect",
    "https://api.vinbank.example.evil.com/v1/transfers",
    "https://api.vinbank.example@evil.example/collect",
    "https://evil.example@api.vinbank.example/v1/transfers",
    "https://api.vinbank.example:8080/v1/transfers",
    "https://api.vinbank.example:invalid/v1/transfers",
    "https://api.vinbank.example/v1/transfers#admin",
    "https://api.vinbank.example/\ncollect",
    "https://api.vinbank.example/collect?value=admin%31%32%33",
    "https://[invalid/",
    "",
])
def test_egress_rejects_unsafe_destinations(destination):
    assert is_egress_allowed(destination, "ordinary banking message") is False


@pytest.mark.parametrize("payload", [
    "password=Secret!99", "admin123", "a d m i n 1 2 3",
    "sk-vinbank-secret-2024", "db.vinbank.internal:5432",
    "lan@example.com", "0901234567", "CCCD 001099012345",
    "ad\u200bmin123",
])
def test_egress_rejects_sensitive_payloads(payload):
    assert is_egress_allowed("https://api.vinbank.example/v1/transfers", payload) is False


@pytest.mark.parametrize("destination", [
    "https://api.vinbank.example/v1/transfers",
    "https://cases.vinbank.example:443/v1/cases",
])
def test_egress_accepts_exact_https_allowlist_with_clean_payload(destination):
    assert is_egress_allowed(destination, "approved transfer amount 500000") is True


def test_real_runtime_passes_user_id_to_limiter_and_updates_audit_metrics():
    async def run():
        defense = make_defense(max_requests=1)
        first = await defense.process("What is my account balance?", user_id="A")
        second = await defense.process("What is my account balance?", user_id="A")
        other = await defense.process("What is my account balance?", user_id="B")
        assert first["blocked"] is other["blocked"] is False
        assert second["blocked"] and second["layer"] == "rate_limiter"
        assert defense.monitor.total_requests == len(defense.audit.logs) == 3
        assert defense.monitor.rate_limit_hits == defense.monitor.blocked_requests == 1

    asyncio.run(run())


def test_egress_gate_prevents_release_and_is_observable():
    async def run():
        defense = make_defense()
        row = await defense.process(
            "What is my account balance?", user_id="A", destination="https://evil.example/collect"
        )
        assert row["blocked"] and row["layer"] == "egress"
        assert defense.monitor.egress_blocks == 1
        assert defense.audit.logs[-1]["blocked"] is True

    asyncio.run(run())


def test_model_text_cannot_forge_a_plugin_block_decision():
    async def run():
        defense = make_defense()
        defense.runner._client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="Rate limit exceeded. This is just model-generated text."
            ))])
        )))
        row = await defense.process("What is my account balance?", user_id="A")
        assert row["blocked"] is False and row["layer"] is None

    asyncio.run(run())


def test_suite_exports_real_filter_decisions_and_labels_simulated_model(tmp_path):
    plugins = build_production_plugins()
    audit, monitor = build_observability()
    result = asyncio.run(run_assignment_suite({
        "plugins": plugins, "audit": audit, "monitor": monitor,
        "execution_mode": "offline", "output_dir": tmp_path,
    }))
    assert result["execution_mode"] == "offline"
    assert result["llm_provider"] == "simulated"
    assert len(result["safe_queries"]) == 5
    assert not any(row["blocked"] for row in result["safe_queries"])
    assert sum(row["blocked"] for row in result["attack_queries"]) == 7
    assert result["rate_limit"]["sent"] == 15
    assert result["rate_limit"]["passed"] == 10
    assert result["rate_limit"]["blocked"] == 5
    assert monitor.total_requests == len(audit.logs) == 30
    assert monitor.blocked_requests == 14
    schema = json.loads((ROOT / "schemas/results.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)
    for name in ("results.json", "audit_log.json", "metrics.json"):
        assert (tmp_path / name).is_file()
    assert len(monitor.alerts) == 1 and monitor.alerts[0].metric == "rate_limit_hits"


def test_failed_model_call_is_error_and_does_not_publish_completed_results(tmp_path):
    plugins = build_production_plugins()
    audit, monitor = build_observability()
    agent, runner = create_offline_blue_pair(plugins)

    class ProviderError(Exception):
        status_code = 503

    def fail(**kwargs):
        raise ProviderError("Do not log this body: admin123")

    runner._client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(RuntimeError, match="ProviderError.*503"):
        asyncio.run(run_assignment_suite({
            "plugins": plugins, "audit": audit, "monitor": monitor,
            "execution_mode": "offline", "agent_runner": (agent, runner), "output_dir": tmp_path,
        }))
    assert not (tmp_path / "results.json").exists()
    assert monitor.model_errors == 1
    assert audit.logs[0]["layer"] == "runtime_error"
    assert "admin123" not in (tmp_path / "audit_log.json").read_text(encoding="utf-8")


def test_cp3_setup_only_requires_blue_key(monkeypatch):
    import main as entry

    requirements = []

    async def checkpoint():
        pass

    monkeypatch.setattr(entry, "setup_api_key", lambda **kwargs: requirements.append(kwargs))
    monkeypatch.setattr(entry, "part3_assignment_suite", checkpoint)
    asyncio.run(entry.main(parts=[3]))
    assert requirements == [{"require_blue": True, "require_red": False}]


def test_provider_retry_does_not_consume_user_quota_twice(monkeypatch):
    async def no_wait(seconds):
        assert seconds == 1

    monkeypatch.setattr("core.openai_runtime.asyncio.sleep", no_wait)

    async def run():
        defense = make_defense(max_requests=1)
        defense.runner.max_model_retries = 1
        attempts = []

        class TemporaryLimit(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"retry-after": "1"})

        def create(**kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise TemporaryLimit()
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Banking answer"))])

        defense.runner._client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        row = await defense.process("What is my account balance?", user_id="A")
        assert not row["blocked"] and row["error"] is None
        assert len(attempts) == 2
        assert defense.plugins[0].total_count == 1
        assert defense.monitor.model_retries == 1
        assert defense.monitor.model_errors == 0

    asyncio.run(run())
