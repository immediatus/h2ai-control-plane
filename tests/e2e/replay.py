#!/usr/bin/env python3
"""
H2AI scenario replay — regression and improvement analysis.

Starts the server with the scenario-specific config, submits the task,
captures the full SSE event stream, checks assertions, saves artifacts.

Usage:
  python3 tests/e2e/replay.py                              # all scenarios
  python3 tests/e2e/replay.py dsp-onboarding               # single scenario
  python3 tests/e2e/replay.py dsp-onboarding ml-feature-latency
  python3 tests/e2e/replay.py --list                       # list available scenarios
  python3 tests/e2e/replay.py --baseline dsp-onboarding    # direct LLM (no H2AI)

Output per run:
  tests/e2e/results/<scenario>/<timestamp>/
    events.jsonl   — raw SSE event stream (one JSON object per line)
    output.txt     — merged output text from MergeResolved
    summary.json   — signals, assertion results, pass/fail
"""

import argparse
import datetime
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import tomllib
import traceback
import urllib.request

from client import submit_task, stream_events, wait_for_health

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
SCENARIOS_DIR = pathlib.Path(__file__).parent / "scenarios"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"
SERVER_BIN = REPO_ROOT / "target" / "release" / "h2ai-control-plane"
MULTIFAMILY = os.environ.get("H2AI_E2E_MULTIFAMILY", "").strip() == "1"


# ── Scenario loading ──────────────────────────────────────────────────────────

def load_scenarios(names: list[str] | None) -> list[tuple[str, dict]]:
    paths = sorted(SCENARIOS_DIR.glob("*/task.json"))
    if not paths:
        raise RuntimeError(f"no scenarios in {SCENARIOS_DIR}")
    result = []
    for path in paths:
        name = path.parent.name
        if names and name not in names:
            continue
        result.append((name, json.loads(path.read_text())))
    if names:
        found = {name for name, _ in result}
        missing = set(names) - found
        if missing:
            available = [p.parent.name for p in paths]
            raise RuntimeError(f"scenario(s) not found: {missing}. Available: {available}")
    return result


# ── Server lifecycle ──────────────────────────────────────────────────────────

def check_no_server_running() -> None:
    try:
        with urllib.request.urlopen("http://localhost:8080/health", timeout=2):
            raise RuntimeError(
                "server already running on :8080 — unknown config, test isolation violated.\n"
                "  Stop it first:  kill $(lsof -ti:8080)"
            )
    except urllib.error.URLError:
        pass  # nothing listening — good


def start_server(scenario_name: str) -> subprocess.Popen:
    check_no_server_running()
    if not SERVER_BIN.exists():
        raise RuntimeError(f"binary not found: {SERVER_BIN} — run: cargo build --release")
    config_path = SCENARIOS_DIR / scenario_name / "h2ai.toml"
    env = os.environ.copy()
    env["H2AI_CONFIG"] = str(config_path)
    proc = subprocess.Popen(
        [str(SERVER_BIN)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"  server pid={proc.pid}  config={config_path}")
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


# ── H2AI replay ───────────────────────────────────────────────────────────────

def run_scenario(scenario_name: str, task: dict) -> dict:
    payload = {k: v for k, v in task.items() if not k.startswith("_")}
    task_id = submit_task(payload)
    print(f"  task_id: {task_id}")

    timeout_s = task.get("_timeout_s", 1800)
    events_raw: list[dict] = []
    verification_scores: list[float] = []
    srani_events: list[dict] = []
    pruned_constraints: list[str] = []
    j_eff: float | None = None
    merged_output = ""
    terminal_kind = ""
    terminal = False
    # New feature signals
    thinking_loop_event: dict | None = None
    prediction_basis_final: str | None = None
    oracle_calibration_patched: dict | None = None

    for event in stream_events(task_id, timeout_s=timeout_s):
        kind = event.get("kind", "")
        events_raw.append(event)
        suffix = ""

        if kind == "VerificationScored":
            score = event.get("score", 0.0)
            verification_scores.append(score)
            suffix = f"  score={score:.2f}"

        elif kind == "BranchPruned":
            for v in event.get("violated_constraints", []):
                pruned_constraints.append(v.get("constraint_id", ""))

        elif kind == "CorrelatedFabrication":
            srani_events.append(event)
            suffix = f"  CFI={event.get('cfi', 0.0):.3f}"

        elif kind == "ThinkingLoopCompleted":
            thinking_loop_event = event
            suffix = (
                f"  enabled={event.get('enabled')}  iterations={event.get('iterations_run')}"
                f"  coverage={event.get('coverage_score', 0.0):.2f}"
            )

        elif kind == "TaskAttribution":
            prediction_basis_final = event.get("prediction_basis")

        elif kind == "OracleCalibrationPatched":
            oracle_calibration_patched = event
            suffix = (
                f"  pass_rate={event.get('oracle_pass_rate', 0.0):.2f}"
                f"  p_mean {event.get('p_mean_before', 0.0):.3f}→{event.get('p_mean_after', 0.0):.3f}"
            )

        elif kind == "MergeResolved":
            j_eff = event.get("j_eff")
            merged_output = event.get("resolved_output", event.get("output", ""))
            terminal = True
            terminal_kind = kind

        elif kind == "TaskFailed":
            terminal = True
            terminal_kind = kind

        print(f"  event: {kind}{suffix}")
        if terminal:
            break

    avg_score = sum(verification_scores) / len(verification_scores) if verification_scores else 0.0
    print(f"  terminal={terminal_kind}  verified={len(verification_scores)}  avg_score={avg_score:.3f}", end="")
    if j_eff is not None:
        print(f"  j_eff={j_eff:.3f}", end="")
    if thinking_loop_event:
        print(f"  thinking_iters={thinking_loop_event.get('iterations_run')}", end="")
    print()

    return {
        "task_id": task_id,
        "terminal": terminal,
        "terminal_kind": terminal_kind,
        "j_eff": j_eff,
        "verification_scores": verification_scores,
        "avg_verification_score": avg_score,
        "srani_events": srani_events,
        "pruned_constraints": pruned_constraints,
        "merged_output": merged_output,
        "events_raw": events_raw,
        "thinking_loop_event": thinking_loop_event,
        "prediction_basis_final": prediction_basis_final,
        "oracle_calibration_patched": oracle_calibration_patched,
    }


# ── Assertions ────────────────────────────────────────────────────────────────

def check_assertions(result: dict, expected: dict, task_json: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}

    out["terminal"] = {"expected": True, "actual": result["terminal"], "pass": result["terminal"]}

    if "valid_proposals_min" in expected:
        actual = len(result["verification_scores"])
        exp = expected["valid_proposals_min"]
        out["valid_proposals_min"] = {"expected": exp, "actual": actual, "pass": actual >= exp}

    if "j_eff_min" in expected:
        diversity_weight = task_json["pareto_weights"]["diversity"]
        tl = result.get("thinking_loop_event")
        coverage = tl.get("coverage_score", 0.0) if tl else 0.0
        computed_min = diversity_weight * coverage
        actual = result["j_eff"]
        out["j_eff_min"] = {
            "expected": round(computed_min, 4),
            "actual": actual,
            "pass": actual is not None and actual >= computed_min,
            "computed_from": f"diversity={diversity_weight} × coverage={coverage:.3f}",
        }

    if MULTIFAMILY and "should_prune" in expected:
        for item in expected["should_prune"]:
            constraint_id = item.get("violates", "").split("—")[0].strip()
            found = any(constraint_id in c for c in result["pruned_constraints"])
            out[f"prune_{constraint_id}"] = {"expected": True, "actual": found, "pass": found}

    if "thinking_loop_ran" in expected:
        tl = result.get("thinking_loop_event")
        actual = tl is not None and tl.get("enabled", False) and tl.get("iterations_run", 0) >= 1
        out["thinking_loop_ran"] = {"expected": expected["thinking_loop_ran"], "actual": actual, "pass": actual == expected["thinking_loop_ran"]}

    if "thinking_loop_coverage_min" in expected:
        tl = result.get("thinking_loop_event")
        actual = tl.get("coverage_score", 0.0) if tl else 0.0
        exp = expected["thinking_loop_coverage_min"]
        out["thinking_loop_coverage_min"] = {"expected": exp, "actual": actual, "pass": actual >= exp}

    if "oracle_p_patched" in expected:
        actual = result.get("oracle_calibration_patched") is not None
        out["oracle_p_patched"] = {"expected": expected["oracle_p_patched"], "actual": actual, "pass": actual == expected["oracle_p_patched"]}

    return out


# ── Result persistence ────────────────────────────────────────────────────────

def save_results(scenario_name: str, task: dict, result: dict, assertions: dict) -> pathlib.Path:
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = RESULTS_DIR / scenario_name / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "events.jsonl", "w") as f:
        for ev in result["events_raw"]:
            f.write(json.dumps(ev) + "\n")

    if result["merged_output"]:
        (out_dir / "output.txt").write_text(result["merged_output"])

    tl = result.get("thinking_loop_event")
    ocp = result.get("oracle_calibration_patched")
    summary = {
        "scenario": scenario_name,
        "timestamp": ts,
        "task_id": result["task_id"],
        "terminal_kind": result["terminal_kind"],
        "j_eff": result["j_eff"],
        "verification_scores": result["verification_scores"],
        "avg_verification_score": result["avg_verification_score"],
        "srani_events_count": len(result["srani_events"]),
        "srani_cfi": result["srani_events"][0].get("cfi") if result["srani_events"] else None,
        "pruned_constraints": result["pruned_constraints"],
        # Thinking loop signals
        "thinking_loop_enabled": tl.get("enabled") if tl else None,
        "thinking_loop_iterations": tl.get("iterations_run") if tl else None,
        "thinking_loop_coverage": tl.get("coverage_score") if tl else None,
        "thinking_loop_understanding_len": tl.get("shared_understanding_len") if tl else None,
        # ρ EMA / calibration signals
        "prediction_basis_final": result.get("prediction_basis_final"),
        # Oracle p_mean patch
        "oracle_p_patched": ocp is not None,
        "oracle_pass_rate": ocp.get("oracle_pass_rate") if ocp else None,
        "oracle_p_mean_before": ocp.get("p_mean_before") if ocp else None,
        "oracle_p_mean_after": ocp.get("p_mean_after") if ocp else None,
        "assertions": assertions,
        "pass": all(c["pass"] for c in assertions.values()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return out_dir


# ── Baseline mode (direct LLM, no H2AI) ──────────────────────────────────────

def _llm_endpoint_for_scenario(scenario_name: str) -> tuple[str, str]:
    """Return (endpoint_url, model_name) for the scenario's first adapter profile.

    Priority:
      1. H2AI_LLM_ENDPOINT / H2AI_LLM_MODEL env vars
      2. First [[adapter_profiles]] entry in the scenario's h2ai.toml
      3. Hard fallback: host.docker.internal:8080
    """
    if endpoint := os.environ.get("H2AI_LLM_ENDPOINT"):
        model = os.environ.get("H2AI_LLM_MODEL", "local")
        return endpoint, model

    toml_path = SCENARIOS_DIR / scenario_name / "h2ai.toml"
    if toml_path.exists():
        cfg = tomllib.loads(toml_path.read_text())
        for profile in cfg.get("adapter_profiles", []):
            kind = profile.get("kind", {})
            # CloudGeneric and OpenAI-compatible adapters expose "endpoint"
            for adapter_cfg in kind.values():
                if isinstance(adapter_cfg, dict) and "endpoint" in adapter_cfg:
                    base = adapter_cfg["endpoint"].rstrip("/")
                    model = profile.get("name", "local")
                    return f"{base}/chat/completions", model

    return "http://host.docker.internal:8080/v1/chat/completions", "local"


def _llm_call(endpoint: str, model: str, messages: list[dict], max_tokens: int = 8192) -> str:
    payload = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.6}).encode()
    req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except (ConnectionRefusedError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"LLM endpoint unreachable: {endpoint}\n"
            f"  Make sure your LLM server is running, or set:\n"
            f"  H2AI_LLM_ENDPOINT=http://<host>:<port>/v1/chat/completions"
        ) from None


def run_baseline(scenario_name: str, task: dict) -> dict:
    endpoint, model = _llm_endpoint_for_scenario(scenario_name)
    print(f"  LLM endpoint: {endpoint}  model: {model}")

    expected = task.get("_expected", {})
    checks = expected.get("checks", [])
    threshold = expected.get("checks_pass_threshold", len(checks))

    print(f"  Generating baseline answer via LLM…")
    t0 = time.time()
    answer = _llm_call(endpoint, model, [
        {"role": "system", "content": "You are a senior distributed systems engineer. Be concrete and precise."},
        {"role": "user", "content": task["description"] + ("\n\nContext: " + task["context"] if task.get("context") else "")},
    ])
    elapsed = time.time() - t0
    print(f"  Answer: {len(answer)} chars in {elapsed:.0f}s")

    results = []
    for check in checks:
        prompt = (
            f"Evaluate this check against the answer below.\n\n"
            f"CHECK: {check['text']}\n\n"
            f"Reply with PRESENT or MISSING on the first line only.\n\n"
            f"ANSWER:\n{answer[:4000]}"
        )
        resp = _llm_call(endpoint, model, [{"role": "user", "content": prompt}], max_tokens=64)
        verdict = "PRESENT" if "PRESENT" in resp.strip().split("\n")[0].upper() else "MISSING"
        results.append({"id": check["id"], "verdict": verdict, "pass": verdict == "PRESENT"})
        print(f"  {check['id']}: {verdict}")

    present = sum(1 for r in results if r["pass"])
    passed = present >= threshold
    print(f"  checks: {present}/{len(checks)}  threshold={threshold}  → {'PASS' if passed else 'FAIL'}")

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = RESULTS_DIR / scenario_name / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "output.txt").write_text(answer)
    summary = {
        "scenario": scenario_name,
        "timestamp": ts,
        "mode": "baseline",
        "llm_endpoint": endpoint,
        "answer_chars": len(answer),
        "elapsed_s": round(elapsed, 1),
        "checks": results,
        "checks_present": present,
        "checks_total": len(checks),
        "checks_threshold": threshold,
        "pass": passed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  results: {out_dir}")
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Replay H2AI scenarios, capture results for regression analysis.")
    parser.add_argument("tasks", nargs="*", metavar="SCENARIO",
                        help="scenario name(s) (e.g. dsp-onboarding); default: all")
    parser.add_argument("--list", action="store_true", help="list available scenarios and exit")
    parser.add_argument("--baseline", action="store_true",
                        help="direct LLM mode — call LLM without H2AI, score against _expected.checks")
    args = parser.parse_args()

    if args.list:
        for path in sorted(SCENARIOS_DIR.glob("*/task.json")):
            t = json.loads(path.read_text())
            n_checks = len(t.get("_expected", {}).get("checks", []))
            print(f"  {path.parent.name}  checks={n_checks}")
        return

    scenarios = load_scenarios(args.tasks or None)
    overall: dict[str, str] = {}

    if args.baseline:
        for scenario_name, task in scenarios:
            print(f"{'='*60}")
            print(f"BASELINE: {scenario_name}")
            print(f"{'='*60}")
            try:
                result = run_baseline(scenario_name, task)
                overall[scenario_name] = "PASS" if result["pass"] else "FAIL"
            except Exception as e:
                overall[scenario_name] = f"ERROR: {e}"
                print(f"  → ERROR: {e}")
                traceback.print_exc()
            print()
    else:
        for scenario_name, task in scenarios:
            print(f"{'='*60}")
            print(f"SCENARIO: {scenario_name}")
            print(f"{'='*60}")
            proc = None
            try:
                proc = start_server(scenario_name)
                wait_for_health()
                print("  server ready")
                result = run_scenario(scenario_name, task)
                assertions = check_assertions(result, task.get("_expected", {}), task)
                out_dir = save_results(scenario_name, task, result, assertions)
                passed = all(c["pass"] for c in assertions.values())
                failed_checks = [k for k, v in assertions.items() if not v["pass"]]
                overall[scenario_name] = "PASS" if passed else "FAIL"
                if failed_checks:
                    print(f"  → FAIL  failed: {', '.join(failed_checks)}")
                    _analysis = pathlib.Path(__file__).parent / "scripts" / "analyze_failed_run.py"
                    if _analysis.exists():
                        import subprocess as _sub
                        _sub.run([sys.executable, str(_analysis), str(out_dir)], check=False)
                else:
                    print(f"  → PASS")
                print(f"  results: {out_dir}")
            except Exception as e:
                overall[scenario_name] = f"ERROR: {e}"
                print(f"  → ERROR: {e}")
                traceback.print_exc()
            finally:
                if proc:
                    stop_server(proc)
                    print(f"  server stopped")
            print()

    print(f"{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    failed = 0
    for name, verdict in overall.items():
        mark = "✓" if verdict == "PASS" else "✗"
        print(f"  {mark} {name}: {verdict}")
        if verdict != "PASS":
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
