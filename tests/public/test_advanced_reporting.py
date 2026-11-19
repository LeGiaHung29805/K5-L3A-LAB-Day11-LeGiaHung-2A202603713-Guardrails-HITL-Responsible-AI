"""Reporting regressions with synthetic artifacts, never live provider results."""
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("lab_grader", ROOT / "scripts/grade.py")
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)


def make_artifacts(root):
    paths = ("src/agents/agent.py", "src/guardrails/input_guardrails.py", "src/guardrails/output_guardrails.py",
             "src/guardrails/text_security.py", "src/core/openai_runtime.py", "src/assignment/pipeline.py")
    fingerprint = {}
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic policy fixture\n", encoding="utf-8")
        fingerprint[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    corpus = root / "data/blue_advanced_cases.json"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("{}", encoding="utf-8")
    corpus_hash = hashlib.sha256(corpus.read_bytes()).hexdigest()
    out = root / "outputs"
    out.mkdir()
    before = {"complete": False, "corpus_sha256": corpus_hash,
              "summary": {"attacks": 12, "input_blocked": 2, "provider_errors": 1, "empty_responses": 5}}
    after = {"complete": True, "corpus_sha256": corpus_hash, "policy_fingerprint": fingerprint,
             "llm_provider": "simulated", "llm_model": "fake", "execution_mode": "synthetic_fixture",
             "blue_context_contains_demo_values": False,
             "summary": {"attacks": 12, "input_blocked": 12, "model_evaluated_attacks": 0,
                         "released_full_value_leaks": 0, "released_review_signals": 0,
                         "safe_queries": 6, "safe_blocked_or_redacted": 0, "provider_errors": 0, "empty_responses": 0}}
    for phase, data in (("before", before), ("after", after)):
        (out / f"blue_advanced_{phase}.json").write_text(json.dumps(data), encoding="utf-8")


def test_report_keeps_incomplete_baseline_distinct_from_completed_after_run(tmp_path):
    make_artifacts(tmp_path)
    summary = grader.summarize_artifacts(tmp_path)
    advanced = summary["blue_advanced"]
    assert advanced["before_complete"] is False and advanced["after_complete"] is True
    assert advanced["current_policy_matches"] and advanced["current_corpus_matches"]
    path = tmp_path / "outputs/lab_report.md"
    grader.write_lab_report_md(tmp_path, {"artifact_summary": summary, "public_tests": {"skipped": True}}, path)
    text = path.read_text(encoding="utf-8")
    assert "Before complete: **False**; after complete: **True**" in text
    assert "provider errors `1`; empty responses `5`" in text
    assert "attack requests reaching model `0`" in text


def test_report_identifies_stale_policy_instead_of_claiming_current_evidence(tmp_path):
    make_artifacts(tmp_path)
    (tmp_path / "src/agents/agent.py").write_text("# Changed fixture\n", encoding="utf-8")
    advanced = grader.summarize_artifacts(tmp_path)["blue_advanced"]
    assert advanced["current_policy_matches"] is False
    assert advanced["after_complete"] is True  # Completed then; not current now.


def test_report_exposes_failed_cp3_refresh_without_relabeling_old_results(tmp_path):
    make_artifacts(tmp_path)
    out = tmp_path / "outputs"
    (out / "results.json").write_text(json.dumps({"run_id": "old", "generated_at": "earlier"}), encoding="utf-8")
    attempt = out / "cp5_validation/cp3_advanced_20261119T000000Z"
    attempt.mkdir(parents=True)
    (attempt / "metrics.json").write_text(json.dumps({"model_errors": 1}), encoding="utf-8")
    (attempt / "audit_log.json").write_text(json.dumps([{"error": {"type": "RateLimitError", "http_status": 429, "provider_message": "free-models-per-day"}}]), encoding="utf-8")
    summary = grader.summarize_artifacts(tmp_path)
    assert summary["cp3_advanced_refresh"]["published"] is False
    assert summary["cp3_advanced_refresh"]["retained_results_generated_at"] == "earlier"
    assert summary["cp3_advanced_refresh"]["http_status"] == 429
