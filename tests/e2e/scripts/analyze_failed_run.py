#!/usr/bin/env python3
"""
Post-run failure analyzer — dream cycle foundation.

Reads a failed run directory and reports:
  - Suspicious numeric values that may indicate constraint violations
  - Assertion failures from summary.json
  - Suggested rubric/config improvements

Usage:
  python3 tests/e2e/scripts/analyze_failed_run.py <run_dir>
  python3 tests/e2e/scripts/analyze_failed_run.py tests/e2e/results/dsp-onboarding/2026-05-13T16-35-59
"""
import json
import pathlib
import re
import sys

SCENARIOS_DIR = pathlib.Path(__file__).parent.parent / "scenarios"
CONSTRAINTS_DIR = pathlib.Path(__file__).parent.parent / "constraints"


def load_run(run_dir: pathlib.Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    output = (run_dir / "output.txt").read_text() if (run_dir / "output.txt").exists() else ""
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines() if l.strip()]
    return {"summary": summary, "output": output, "events": events}


def extract_ms_values(text: str) -> list[tuple[str, float]]:
    """Return (context_snippet, numeric_ms_value) pairs from proposal text."""
    results = []
    for m in re.finditer(r"([^\n]{0,50})(\d+(?:\.\d+)?)\s*ms", text, re.IGNORECASE):
        try:
            results.append((m.group(1).strip(), float(m.group(2))))
        except ValueError:
            pass
    return results


def load_constraint_checks(constraint_id: str) -> list[str]:
    """Load binary checks from the constraint YAML (requires pyyaml)."""
    try:
        import yaml
    except ImportError:
        return []
    for path in CONSTRAINTS_DIR.glob("*.yaml"):
        try:
            content = yaml.safe_load(path.read_text())
            if content.get("id") == constraint_id:
                return content.get("criteria", {}).get("checks", [])
        except Exception:
            continue
    return []


def analyze(run_dir: pathlib.Path) -> None:
    run = load_run(run_dir)
    summary = run["summary"]
    scenario = summary["scenario"]
    output = run["output"]

    print(f"=== Run Analysis: {scenario} / {run_dir.name} ===")
    print(f"pass={summary['pass']}  avg_score={summary.get('avg_verification_score', 0):.3f}  "
          f"j_eff={summary.get('j_eff')}  terminal={summary.get('terminal_kind')}")
    print()

    # Verification scores
    scored = [e for e in run["events"] if e.get("kind") == "VerificationScored"]
    if scored:
        scores = [e["score"] for e in scored]
        print(f"Verification scores ({len(scores)} proposals): {scores}")
        reasons = [e.get("reason", "") for e in scored if e.get("reason") and e["reason"] != "__rubric__"]
        if reasons:
            print(f"Constraint violations: {reasons}")
    print()

    # Failed explorers
    failed = [e for e in run["events"] if e.get("kind") == "ProposalFailed"]
    if failed:
        print(f"⚠  {len(failed)} explorer(s) failed without output:")
        for e in failed:
            print(f"   explorer={e.get('explorer_id', '?')[:8]}  reason={e.get('reason', '?')}")
        print()

    # Load scenario task.json for constraint IDs
    task_json = SCENARIOS_DIR / scenario / "task.json"
    constraint_ids: list[str] = []
    if task_json.exists():
        task = json.loads(task_json.read_text())
        constraint_ids = task.get("constraints", [])

    # Numeric value scan — look for violations
    suspicious = [(ctx, n) for ctx, n in extract_ms_values(output) if n > 100]
    if suspicious:
        print("⚠  Suspicious ms values in output (>100ms — potential T_global violation):")
        for ctx, n in suspicious[:8]:
            print(f"   {n:.0f}ms in: '{ctx}'")
        print()

    # Binary checks from constraint YAML
    for cid in constraint_ids:
        checks = load_constraint_checks(cid)
        if checks:
            print(f"Binary checks for {cid}:")
            for i, check in enumerate(checks, 1):
                print(f"   {i}. {check[:100]}")
            print()

    # Assertion failures
    assertions = summary.get("assertions", {})
    failed_assertions = {k: v for k, v in assertions.items() if not v["pass"]}
    if failed_assertions:
        print("Assertion failures:")
        for k, v in failed_assertions.items():
            print(f"   FAIL {k}: expected={v['expected']} actual={v['actual']}")
        print()

    # Suggestions
    print("Suggestions:")
    if summary.get("j_eff") is None:
        slots = None
        h2ai_toml = SCENARIOS_DIR / scenario / "h2ai.toml"
        if h2ai_toml.exists():
            for line in h2ai_toml.read_text().splitlines():
                if "precision_mode_max_slots" in line:
                    slots = line.strip()
        count = None
        if task_json.exists():
            count = json.loads(task_json.read_text()).get("explorers", {}).get("count")
        print(f"   j_eff=null: ≥2 valid proposals required for ensemble scoring.")
        if count and slots:
            print(f"   explorers.count={count} but {slots} — verify these match.")
    if suspicious:
        print("   Add numeric_checks to constraint YAML for T_global ≤ 100ms bound.")
        print("   Consider raising threshold (e.g. threshold: 0.6) to require ≥3/5 checks.")
    if not failed_assertions and not suspicious and not failed:
        print("   No obvious issues detected — run may have failed due to model quality.")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    analyze(pathlib.Path(sys.argv[1]))
