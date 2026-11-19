"""Requested model names must not hide a provider-side version substitution."""
import json
from types import SimpleNamespace

import pytest

import core.model_identity as identity


@pytest.mark.parametrize("reported,status", [
    ("gemini-3.5-flash", "verified"),
    ("gemini-3.6-flash", "model_mismatch"),
    (None, "unverifiable"),
])
def test_identity_checks_provider_response_not_config_label(tmp_path, monkeypatch, reported, status):
    monkeypatch.setattr(identity, "get_red_provider", lambda: "gemini")
    monkeypatch.setattr(identity, "get_red_model", lambda: "gemini-3.5-flash")
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model_version=reported, response_id="test-response", usage_metadata=None)

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    path = tmp_path / "identity.json"
    if status == "verified":
        assert identity.verify_red_model_identity(client=client, report_path=path)["verified"]
    else:
        with pytest.raises(identity.ModelIdentityError):
            identity.verify_red_model_identity(client=client, report_path=path)
    report = json.loads(path.read_text())
    assert report["status"] == status
    assert report["requested_model"] == calls[0]["model"] == "gemini-3.5-flash"
    assert report["reported_model_version"] == reported


def test_provider_error_does_not_become_verification_or_disclose_key(tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "get_red_provider", lambda: "gemini")
    monkeypatch.setattr(identity, "get_red_model", lambda: "gemini-3.5-flash")

    class ProviderError(Exception):
        code = 429

    def fail(**kwargs):
        raise ProviderError("Private credential must not enter report")

    path = tmp_path / "identity.json"
    with pytest.raises(identity.ModelIdentityError):
        identity.verify_red_model_identity(client=SimpleNamespace(models=SimpleNamespace(generate_content=fail)), report_path=path)
    saved = path.read_text()
    assert "Private credential" not in saved
    assert json.loads(saved)["status"] == "provider_error"


def test_transient_probe_failure_retries_same_model_and_records_history(tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "get_red_provider", lambda: "gemini")
    monkeypatch.setattr(identity, "get_red_model", lambda: "gemini-3.6-flash")
    delays, calls = [], []
    monkeypatch.setattr(identity.time, "sleep", delays.append)

    class ProviderError(Exception):
        code = 503

    def generate(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) == 1:
            raise ProviderError()
        return SimpleNamespace(model_version="gemini-3.6-flash", response_id="test", usage_metadata=None)

    report = identity.verify_red_model_identity(
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
        report_path=tmp_path / "identity.json", max_retries=2,
    )
    assert report["verified"] is True
    assert calls == ["gemini-3.6-flash", "gemini-3.6-flash"]
    assert delays == [2]
    assert [attempt["status"] for attempt in report["attempts"]] == ["provider_error", "verified"]


def test_model_substitution_is_never_hidden_by_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "get_red_provider", lambda: "gemini")
    monkeypatch.setattr(identity, "get_red_model", lambda: "gemini-3.6-flash")
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["model"])
        return SimpleNamespace(model_version="gemini-3.8-flash", response_id="test", usage_metadata=None)

    with pytest.raises(identity.ModelIdentityError):
        identity.verify_red_model_identity(
            client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
            report_path=tmp_path / "identity.json", max_retries=2,
        )
    assert calls == ["gemini-3.6-flash"]
