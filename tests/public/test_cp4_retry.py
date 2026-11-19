"""Recovery must never merge different experiments or overwrite valid evidence."""
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from attacks.attacks import adversarial_prompts


@pytest.fixture
def recovery(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/retry_cp4_errors.py"
    spec = importlib.util.spec_from_file_location("cp4_retry_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_red_provider", lambda: "gemini")
    monkeypatch.setattr(module, "get_red_model", lambda: "gemini-3.5-flash")
    monkeypatch.setattr(module, "verify_red_model_identity", lambda **kwargs: None)
    return module


def evidence():
    def rows(target):
        return [dict(
            id=p["id"], category=p["category"], input=p["input"],
            response="Simulated ordinary reply", response_preview="Simulated ordinary reply",
            leaked=False, blocked=False, blocked_input=False, layer=None, error=None, target=target,
        ) for p in adversarial_prompts]
    return {
        "llm_provider": "gemini", "llm_model": "gemini-3.5-flash",
        "unsafe_attacks": rows("red_default"), "guards_attacks": rows("red_advance"),
    }


@pytest.mark.parametrize("change", ["model", "prompt"])
def test_recovery_rejects_mismatched_experiments(recovery, change):
    data = evidence()
    if change == "model":
        data["llm_model"] = "different-model"
    else:
        data["guards_attacks"][0]["input"] += " modified"
    with pytest.raises(ValueError, match="changed"):
        recovery.validate_saved_run(data)


def test_recovery_only_retries_errors_and_archives_original(recovery, tmp_path, monkeypatch):
    data = evidence()
    for row in data["guards_attacks"]:
        if row["id"] in (4, 5):
            row["layer"] = "nemo_error"
    path = tmp_path / "attack_results.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    calls = []

    async def retry(*args, prompts, target_name, save_json):
        calls.append((target_name, [p["id"] for p in prompts], save_json))
        rows = deepcopy([r for r in data["guards_attacks"] if r["id"] in (4, 5)])
        for row in rows:
            row.update(layer="nemo_input_rail", blocked=True, blocked_input=True)
        return rows

    monkeypatch.setattr(recovery, "create_red_agent_advance", lambda: (None, None))
    monkeypatch.setattr(recovery, "run_attacks", retry)
    assert asyncio.run(recovery.retry_saved_errors(tmp_path)) == 0
    assert calls == [("red_advance", [4, 5], False)]
    archived = list((tmp_path / "cp4_attempts").glob("*/attack_results.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text()) == data
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert [r["response"] for r in saved["unsafe_attacks"]] == [r["response"] for r in data["unsafe_attacks"]]
    assert all(row["layer"] == "nemo_input_rail" for row in saved["guards_attacks"] if row["id"] in (4, 5))


def test_recovery_with_no_errors_does_not_call_models(recovery, tmp_path, monkeypatch):
    (tmp_path / "attack_results.json").write_text(json.dumps(evidence()), encoding="utf-8")

    def unexpected():
        raise AssertionError("No model call should be needed")

    monkeypatch.setattr(recovery, "create_red_agent_default", unexpected)
    monkeypatch.setattr(recovery, "create_red_agent_advance", unexpected)
    assert asyncio.run(recovery.retry_saved_errors(tmp_path)) == 0
    assert not (tmp_path / "cp4_attempts").exists()
