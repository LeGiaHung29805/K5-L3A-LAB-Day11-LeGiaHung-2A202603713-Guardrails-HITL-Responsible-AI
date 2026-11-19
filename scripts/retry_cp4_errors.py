"""Retry only CP4 infrastructure failures; preserve all earlier evidence.

Run from the repo root: python scripts/retry_cp4_errors.py
The provider, model and prompt texts must match the saved run.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.agent import create_red_agent_default
from agents.guards_agent import create_red_agent_advance
from attacks.attacks import adversarial_prompts, run_attacks, save_attack_results, write_run_attack_json
from core.config import get_red_model, get_red_provider
from core.model_identity import verify_red_model_identity


def is_error(row):
    return bool(row.get("error")) or row.get("layer") in {"error", "nemo_error"}


def validate_saved_run(data):
    """Never combine evidence from different models or different prompt versions."""
    if (data.get("llm_provider"), data.get("llm_model")) != (get_red_provider(), get_red_model()):
        raise ValueError("Provider/model changed; run --part 4 afresh instead of merging results.")
    expected = {prompt["id"]: prompt["input"] for prompt in adversarial_prompts}
    for name in ("unsafe_attacks", "guards_attacks"):
        rows = data.get(name) or []
        if len(rows) != len(expected) or {row.get("id") for row in rows} != set(expected):
            raise ValueError(f"Incomplete prompt list in {name}; run --part 4 afresh.")
        if any(row.get("input") != expected[row["id"]] for row in rows):
            raise ValueError("Prompt text changed; run --part 4 afresh instead of merging results.")


async def retry_saved_errors(output_dir=None):
    output = Path(output_dir or ROOT / "outputs")
    path = output / "attack_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_saved_run(data)
    if not any(is_error(row) for name in ("unsafe_attacks", "guards_attacks") for row in data[name]):
        print("No CP4 provider/NeMo errors remain; no API calls needed.")
        return 0

    identity = verify_red_model_identity(report_path=output / "cp4_model_identity.json", max_retries=2)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = output / "cp4_attempts" / stamp
    archive.mkdir(parents=True)
    for name in ("attack_results.json", "unsafe_attack_result.json", "guards_attack_result.json"):
        if (output / name).exists():
            shutil.copy2(output / name, archive / name)
    print(f"Preserved previous evidence: {archive}", flush=True)

    targets = [
        ("unsafe_attacks", "red_default", "unsafe_attack_result.json", create_red_agent_default),
        ("guards_attacks", "red_advance", "guards_attack_result.json", create_red_agent_advance),
    ]
    for name, target, filename, factory in targets:
        failed = {row["id"] for row in data[name] if is_error(row)}
        if not failed:
            continue
        print(f"Retry {target}, prompt IDs: {sorted(failed)}", flush=True)
        agent, runner = factory()
        retried = await run_attacks(
            agent, runner, prompts=[p for p in adversarial_prompts if p["id"] in failed],
            target_name=target, save_json=False,
        )
        merged = {row["id"]: row for row in data[name]}
        for row in retried:
            row["notes"] = f"Retried infrastructure failure; previous evidence: cp4_attempts/{stamp}/{filename}"
            merged[row["id"]] = row
        data[name] = [merged[p["id"]] for p in adversarial_prompts]
        write_run_attack_json(data[name], target_name=target, filepath=output / filename)
        save_attack_results(
            unsafe_results=data["unsafe_attacks"], guards_results=data["guards_attacks"],
            filepath=path, model_identity=identity,
        )

    remaining = sum(is_error(row) for name in ("unsafe_attacks", "guards_attacks") for row in data[name])
    print(f"Remaining provider/NeMo errors: {remaining}")
    return 1 if remaining else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(retry_saved_errors()))
