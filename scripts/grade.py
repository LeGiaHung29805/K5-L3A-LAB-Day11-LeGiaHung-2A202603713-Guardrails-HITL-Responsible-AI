#!/usr/bin/env python3
"""
Machine-readable grader for Assignment 11 packaging + public tests.

Usage:
  python scripts/grade.py --submission-dir . --out outputs/grade_report.json

Also auto-writes outputs/lab_report.md (human-readable summary).
Do not create report files by hand.

Exit codes:
  0 = ran successfully (see report for scores / technical_failure)
  2 = could not start grading (missing paths)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import jsonschema
except ImportError:
    jsonschema = None


def validate_schema(submission: Path) -> dict:
    results = submission / "outputs" / "results.json"
    schema = submission / "schemas" / "results.schema.json"
    if not schema.exists():
        # allow schema from this tooling repo when grading an external zip layout
        schema = Path(__file__).resolve().parents[1] / "schemas" / "results.schema.json"
    if not results.exists():
        return {"ok": False, "error": "missing outputs/results.json", "points": 0}
    if jsonschema is None:
        return {"ok": False, "error": "jsonschema not installed", "points": 0}
    try:
        data = json.loads(results.read_text(encoding="utf-8"))
        sch = json.loads(schema.read_text(encoding="utf-8"))
        jsonschema.validate(instance=data, schema=sch)
        return {"ok": True, "error": None, "points": 10}
    except Exception as e:
        return {"ok": False, "error": str(e), "points": 0}


def required_files(submission: Path) -> dict:
    checks = {
        "audit": (submission / "outputs" / "audit_log.json").exists(),
        "metrics": (submission / "outputs" / "metrics.json").exists(),
        "results": (submission / "outputs" / "results.json").exists(),
        "attack_results": (submission / "outputs" / "attack_results.json").exists(),
    }
    # Core bắt buộc: results + attack_results. audit/metrics khuyến nghị.
    ok = checks["results"] and checks["attack_results"]
    return {
        "ok": ok,
        "details": {k: bool(v) for k, v in checks.items()},
        "required": ["results", "attack_results"],
        "recommended": ["audit", "metrics"],
    }


def run_pytest(submission: Path, path: str) -> dict:
    cmd = [sys.executable, "-m", "pytest", path, "-q", "--tb=no"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(submission),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-2000:],
            "technical_failure": proc.returncode == 2,
        }
    except Exception as e:
        return {
            "returncode": 2,
            "stdout": "",
            "stderr": str(e),
            "technical_failure": True,
        }


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_blocked(items: list | None) -> tuple[int, int]:
    if not isinstance(items, list):
        return 0, 0
    total = len(items)
    blocked = sum(1 for x in items if isinstance(x, dict) and x.get("blocked") is True)
    return blocked, total


def _count_leaks(items: list | None) -> tuple[int, int]:
    if not isinstance(items, list):
        return 0, 0
    total = len(items)
    leaks = sum(1 for x in items if isinstance(x, dict) and x.get("leaked") is True)
    return leaks, total


def summarize_artifacts(submission: Path) -> dict:
    """Pull short stats from results.json / attack_results.json for the report."""
    results = _load_json(submission / "outputs" / "results.json")
    attacks = _load_json(submission / "outputs" / "attack_results.json")

    summary: dict = {
        "framework": None,
        "defense": {},
        "red_team": {},
    }
    if isinstance(results, dict):
        summary["framework"] = results.get("framework")
        safe_b, safe_n = _count_blocked(results.get("safe_queries"))
        atk_b, atk_n = _count_blocked(results.get("attack_queries"))
        edge_b, edge_n = _count_blocked(results.get("edge_cases"))
        rl = results.get("rate_limit") if isinstance(results.get("rate_limit"), dict) else {}
        summary["defense"] = {
            "safe_blocked": safe_b,
            "safe_total": safe_n,
            "attack_blocked": atk_b,
            "attack_total": atk_n,
            "edge_blocked": edge_b,
            "edge_total": edge_n,
            "rate_limit_blocked": rl.get("blocked"),
            "rate_limit_sent": rl.get("sent"),
        }

    if isinstance(attacks, dict):
        unsafe_list = attacks.get("unsafe_attacks")
        guards_list = attacks.get("guards_attacks")
        if not isinstance(unsafe_list, list):
            unsafe_list = []
        if not isinstance(guards_list, list):
            guards_list = []
        summ = attacks.get("summary") if isinstance(attacks.get("summary"), dict) else {}
        u_leaks, u_n = _count_leaks(unsafe_list)
        g_leaks, g_n = _count_leaks(guards_list)
        summary["red_team"] = {
            "llm_provider": attacks.get("llm_provider"),
            "llm_model": attacks.get("llm_model"),
            "unsafe_leaks": summ.get("unsafe_leaked", u_leaks),
            "unsafe_total": u_n,
            "guards_leaks": summ.get("guards_leaked", g_leaks),
            "guards_total": g_n,
        }
    before = _load_json(submission / "outputs/blue_advanced_before.json")
    after = _load_json(submission / "outputs/blue_advanced_after.json")
    if isinstance(after, dict):
        policy_files = (
            "src/agents/agent.py", "src/guardrails/input_guardrails.py",
            "src/guardrails/output_guardrails.py", "src/guardrails/text_security.py",
            "src/core/openai_runtime.py", "src/assignment/pipeline.py",
        )
        fingerprint = after.get("policy_fingerprint") or {}
        policy_matches = all(
            (submission / name).is_file()
            and fingerprint.get(name) == hashlib.sha256((submission / name).read_bytes()).hexdigest()
            for name in policy_files
        )
        corpus = submission / "data/blue_advanced_cases.json"
        corpus_matches = corpus.is_file() and after.get("corpus_sha256") == hashlib.sha256(corpus.read_bytes()).hexdigest()
        baseline = before if isinstance(before, dict) else {}
        summary["blue_advanced"] = {
            "execution_mode": after.get("execution_mode"),
            "provider": after.get("llm_provider"), "model": after.get("llm_model"),
            "before_complete": baseline.get("complete"), "after_complete": after.get("complete"),
            "same_corpus_before_after": bool(baseline.get("corpus_sha256")) and baseline.get("corpus_sha256") == after.get("corpus_sha256"),
            "current_policy_matches": policy_matches, "current_corpus_matches": bool(corpus_matches),
            "blue_context_contains_demo_values": after.get("blue_context_contains_demo_values"),
            "before": baseline.get("summary", {}), "after": after.get("summary", {}),
        }
    local = _load_json(submission / "outputs/cp2_security_evaluation.json")
    if isinstance(local, dict):
        summary["cp2_regression"] = {key: local.get(key) for key in (
            "execution_mode", "scope", "counts", "attack_recall", "safe_false_positive_rate", "all_passed",
        )}
    refreshes = sorted((submission / "outputs/cp5_validation").glob("cp3_advanced_*"))
    if refreshes:
        latest = refreshes[-1]
        refreshed = _load_json(latest / "results.json")
        diagnostics = _load_json(latest / "metrics.json")
        logs = _load_json(latest / "audit_log.json")
        errors = [row["error"] for row in logs if isinstance(row, dict) and row.get("error")] if isinstance(logs, list) else []
        error = errors[-1] if errors else {}
        published = (isinstance(refreshed, dict) and isinstance(results, dict)
                     and refreshed.get("run_id") == results.get("run_id"))
        summary["cp3_advanced_refresh"] = {
            "published": bool(published), "attempt": latest.relative_to(submission).as_posix(),
            "model_errors": diagnostics.get("model_errors") if isinstance(diagnostics, dict) else None,
            "error_type": error.get("type"), "http_status": error.get("http_status"),
            "provider_reason": error.get("provider_message"),
            "retained_results_generated_at": results.get("generated_at") if isinstance(results, dict) else None,
        }
    return summary


def write_lab_report_md(submission: Path, report: dict, out_md: Path) -> Path:
    """Auto-generate a human-readable report — students must not write this by hand."""
    pack = report.get("packaging", {}).get("details", {})
    schema = report.get("results_schema", {})
    public = report.get("public_tests", {})
    art = report.get("artifact_summary", {})
    defense = art.get("defense") or {}
    red = art.get("red_team") or {}

    def yn(ok: bool | None) -> str:
        if ok is True:
            return "OK"
        if ok is False:
            return "MISSING"
        return "—"

    lines = [
        "# Lab 11 — Auto Report",
        "",
        "> File này **tự sinh** bởi `scripts/grade.py`. **Không** viết / sửa tay.",
        "",
        f"- Generated (UTC): `{report.get('generated_at')}`",
        f"- Framework: `{art.get('framework') or '—'}`",
        f"- Technical failure: **{report.get('technical_failure')}**",
        f"- Additional live validation incomplete: **{report.get('additional_validation_incomplete', False)}** (see additional checks below).",
        "",
        "## Packaging",
        "",
        f"| File | Status |",
        f"|------|--------|",
        f"| results.json | {yn(pack.get('results'))} |",
        f"| attack_results.json | {yn(pack.get('attack_results'))} |",
        f"| audit_log.json | {yn(pack.get('audit'))} |",
        f"| metrics.json | {yn(pack.get('metrics'))} |",
        "",
        "## Schema (`results.json`)",
        "",
        f"- Valid: **{schema.get('ok')}**",
        f"- Error: `{schema.get('error')}`",
        "",
        "## Defense snapshot (từ `results.json`)",
        "",
        f"- Safe queries blocked: `{defense.get('safe_blocked')}/{defense.get('safe_total')}`",
        f"- Attack queries blocked: `{defense.get('attack_blocked')}/{defense.get('attack_total')}`",
        f"- Edge cases blocked: `{defense.get('edge_blocked')}/{defense.get('edge_total')}`",
        f"- Rate limit blocked/sent: `{defense.get('rate_limit_blocked')}/{defense.get('rate_limit_sent')}`",
        "",
        "## Red Team snapshot (từ `attack_results.json`)",
        "",
        f"- Provider / model: `{red.get('llm_provider')}` / `{red.get('llm_model')}`",
        f"- Unsafe leaks (Red): `{red.get('unsafe_leaks')}/{red.get('unsafe_total')}`",
        f"- Guards leaks (Red Advance): `{red.get('guards_leaks')}/{red.get('guards_total')}`",
        "",
        "## Public tests",
        "",
    ]
    if public.get("skipped"):
        lines.append("- Skipped (`--skip-public-tests`).")
    else:
        lines.append(f"- Return code: `{public.get('returncode')}`")
        lines.append(f"- Technical failure: `{public.get('technical_failure')}`")
        stdout = (public.get("stdout") or "").strip()
        if stdout:
            lines.extend(["", "```text", stdout[-1500:], "```"])

    advanced = art.get("blue_advanced")
    if isinstance(advanced, dict):
        previous, current = advanced.get("before", {}), advanced.get("after", {})
        lines.extend([
            "", "## Additional Blue adversarial checks", "",
            f"- Provider / model: `{advanced.get('provider')}` / `{advanced.get('model')}`.",
            f"- Before complete: **{advanced.get('before_complete')}**; after complete: **{advanced.get('after_complete')}**.",
            f"- Current policy/corpus match: **{advanced.get('current_policy_matches')} / {advanced.get('current_corpus_matches')}**.",
            f"- Same corpus before/after: **{advanced.get('same_corpus_before_after')}**.",
            f"- Before: input blocks `{previous.get('input_blocked')}/{previous.get('attacks')}`; provider errors `{previous.get('provider_errors')}`; empty responses `{previous.get('empty_responses')}`.",
            f"- After: input blocks `{current.get('input_blocked')}/{current.get('attacks')}`; attack requests reaching model `{current.get('model_evaluated_attacks')}`.",
            f"- After: released full-value leaks `{current.get('released_full_value_leaks')}`; partial/oracle review signals `{current.get('released_review_signals')}`.",
            f"- After: safe queries blocked/redacted `{current.get('safe_blocked_or_redacted')}/{current.get('safe_queries')}`; provider errors `{current.get('provider_errors')}`; empty responses `{current.get('empty_responses')}`.",
            f"- Demo values present in Blue model context after changes: **{advanced.get('blue_context_contains_demo_values')}**.",
            "- Evidence: [before](blue_advanced_before.json), [after](blue_advanced_after.json).",
            "- Errors/empty responses are incomplete evidence, not successful defense. Partial/oracle signals require review. Input rejection means that attack did not reach the model in the after run.",
            "- These independent, known lab probes are not a held-out benchmark or proof of universal safety.",
        ])
    local = art.get("cp2_regression")
    if isinstance(local, dict):
        lines.extend([
            "", "## CP2 local regression", "",
            f"- All passed: **{local.get('all_passed')}**; attack recall `{local.get('attack_recall')}`; safe-query false-positive rate `{local.get('safe_false_positive_rate')}`.",
            "- Execution mode: local deterministic checks, no LLM. Corpus was used to develop the rules, not held out.",
            "- Evidence: [cp2_security_evaluation.json](cp2_security_evaluation.json).",
        ])
    refresh = art.get("cp3_advanced_refresh")
    if isinstance(refresh, dict):
        lines.extend([
            "", "## CP3 live refresh after advanced hardening", "",
            f"- New completed results published: **{refresh.get('published')}**.",
            f"- Model errors in latest attempt: `{refresh.get('model_errors')}`; error `{refresh.get('error_type')}` / HTTP `{refresh.get('http_status')}`.",
            f"- Provider reason: {refresh.get('provider_reason') or 'None recorded'}",
            f"- Root results.json timestamp: `{refresh.get('retained_results_generated_at')}`.",
            f"- Diagnostic attempt: `{refresh.get('attempt')}`.",
            "- If no new results were published, the defense snapshot above is the retained earlier completed run; it is not a completed post-hardening live refresh.",
        ])
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Artifact chấm chính: `outputs/results.json` + `outputs/attack_results.json`.",
            "- Bonus B1/B2 do grader replay quyết định — JSON chỉ là bằng chứng.",
            "- Không nộp `report/*.md` viết tay; dùng file này nếu cần xem tóm tắt.",
            "",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def main():
    parser = argparse.ArgumentParser(description="Grade Assignment 11 submission package")
    parser.add_argument("--submission-dir", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("outputs/grade_report.json"))
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Auto lab report Markdown (default: outputs/lab_report.md next to --out)",
    )
    parser.add_argument("--skip-public-tests", action="store_true")
    parser.add_argument("--no-md", action="store_true", help="Skip writing lab_report.md")
    args = parser.parse_args()

    root = args.submission_dir.resolve()
    if not root.exists():
        print(f"Submission dir not found: {root}", file=sys.stderr)
        sys.exit(2)

    files = required_files(root)
    schema = validate_schema(root)
    artifact_summary = summarize_artifacts(root)

    public = {"skipped": True}
    if not args.skip_public_tests:
        public = run_pytest(root, "tests/public")

    # Machine portion is packaging (files+schema) only here; rubric points for
    # pipeline behavior come from public/hidden pytest + human review.
    technical_failure = (not files["ok"]) or (not schema["ok"]) or public.get("technical_failure", False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "submission_dir": str(root),
        "technical_failure": technical_failure,
        "packaging": files,
        "results_schema": schema,
        "artifact_summary": artifact_summary,
        "additional_validation_incomplete": bool(
            (artifact_summary.get("blue_advanced") and (
                not artifact_summary["blue_advanced"].get("after_complete")
                or not artifact_summary["blue_advanced"].get("current_policy_matches")
                or not artifact_summary["blue_advanced"].get("current_corpus_matches")
            ))
            or (artifact_summary.get("cp3_advanced_refresh") and not artifact_summary["cp3_advanced_refresh"].get("published"))
        ),
        "public_tests": public,
        "human_review_required": [
            "red_team_prompt_quality",
            "unsafe_leak_default_model",
            "bonus_b1_red_leak",
            "bonus_b2_red_advance_leak",
        ],
        "bonus_rubric": {
            "B1_red": {
                "points_max": 5,
                "target": "Red",
                "requires": "unsafe/default leaked=true + grader replay",
            },
            "B2_red_advance": {
                "points_max": 10,
                "target": "Red Advance",
                "requires": "guards/advance leaked=true + grader replay",
            },
            "choose_one": True,
            "note": "Chỉ nhận một trong hai (B1 hoặc B2), không cộng.",
        },
        "notes": (
            "Packaging + schema + public tests. "
            "Base 100: CP2 40 + CP3 40 + CP4 20. "
            "Bonus: chọn một — B1 Red tối đa +5 hoặc B2 Red Advance tối đa +10. "
            "JSON is evidence only — replay decides bonus. "
            "lab_report.md is auto-generated — do not write by hand."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = None
    if not args.no_md:
        md_path = args.md_out or (args.out.parent / "lab_report.md")
        write_lab_report_md(root, report, md_path)

    print(
        json.dumps(
            {
                "out": str(args.out),
                "lab_report_md": str(md_path) if md_path else None,
                "technical_failure": technical_failure,
            },
            indent=2,
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
