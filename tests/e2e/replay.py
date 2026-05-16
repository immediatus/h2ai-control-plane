#!/usr/bin/env python3
"""
H2AI scenario replay — regression and improvement analysis.

Starts the server with the scenario-specific config, submits the task,
captures the full SSE event stream, checks assertions, saves artifacts.

Usage:
  python3 tests/e2e/replay.py                                    # all scenarios
  python3 tests/e2e/replay.py benchmark                          # single scenario by name
  python3 tests/e2e/replay.py features/01-thinking-loop          # feature isolation scenario
  python3 tests/e2e/replay.py --list                             # list available scenarios
  python3 tests/e2e/replay.py --baseline benchmark               # direct LLM (no H2AI)
  python3 tests/e2e/replay.py --config baseline.toml benchmark   # run with alternate config
  python3 tests/e2e/replay.py --trials 3 benchmark               # run k times, report pass^k
  python3 tests/e2e/replay.py --compare benchmark                # h2ai vs baseline delta table

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

from client import submit_task, stream_events, wait_for_health, DEFAULT_TENANT

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
SCENARIOS_DIR = pathlib.Path(__file__).parent / "scenarios"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"
SERVER_BIN = REPO_ROOT / "target" / "release" / "h2ai-control-plane"
MULTIFAMILY = os.environ.get("H2AI_E2E_MULTIFAMILY", "").strip() == "1"


# ── Scenario loading ──────────────────────────────────────────────────────────

def load_scenarios(names: list[str] | None) -> list[tuple[str, pathlib.Path, dict]]:
    """Return list of (display_name, scenario_dir, task_dict).

    Scans recursively under SCENARIOS_DIR so features/ and benchmark/ subdirs
    are discovered automatically.  The display name is the path relative to
    SCENARIOS_DIR with slashes preserved (e.g. "features/01-thinking-loop").
    Names passed on the CLI can match either the full relative path or the
    leaf directory name.
    """
    paths = sorted(SCENARIOS_DIR.glob("**/task.json"))
    if not paths:
        raise RuntimeError(f"no scenarios under {SCENARIOS_DIR}")
    result = []
    for path in paths:
        scenario_dir = path.parent
        rel = scenario_dir.relative_to(SCENARIOS_DIR)
        display_name = str(rel)
        leaf_name = scenario_dir.name
        if names:
            if display_name not in names and leaf_name not in names:
                continue
        result.append((display_name, scenario_dir, json.loads(path.read_text())))
    if names:
        found_display = {n for n, _, _ in result}
        found_leaf = {d.name for _, d, _ in result}
        missing = [n for n in names if n not in found_display and n not in found_leaf]
        if missing:
            available = [str(p.parent.relative_to(SCENARIOS_DIR)) for p in paths]
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


def start_server(scenario_dir: pathlib.Path, config_file: str = "h2ai.toml") -> subprocess.Popen:
    check_no_server_running()
    if not SERVER_BIN.exists():
        raise RuntimeError(f"binary not found: {SERVER_BIN} — run: cargo build --release")
    config_path = scenario_dir / config_file
    if not config_path.exists():
        raise RuntimeError(f"config not found: {config_path}")
    env = os.environ.copy()
    env["H2AI_CONFIG"] = str(config_path)
    log_path = REPO_ROOT / "tests" / "e2e" / "results" / "_server_logs" / f"{config_path.stem}-{scenario_dir.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [str(SERVER_BIN)],
        env=env,
        stdout=log_fh,
        stderr=log_fh,
    )
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    proc._log_path = log_path  # type: ignore[attr-defined]
    print(f"  server pid={proc.pid}  config={config_path.relative_to(REPO_ROOT)}  log={log_path.relative_to(REPO_ROOT)}")
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    fh = getattr(proc, "_log_fh", None)
    if fh:
        fh.close()


# ── H2AI replay ───────────────────────────────────────────────────────────────

def run_scenario(scenario_name: str, task: dict) -> dict:
    tenant_id = task.get("_tenant_id", DEFAULT_TENANT)
    payload = {k: v for k, v in task.items() if not k.startswith("_")}
    task_id = submit_task(payload, tenant_id=tenant_id)
    print(f"  tenant_id: {tenant_id}  task_id: {task_id}")

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

    hitl_gate_fired = False
    leader_elected = False
    leader_elected_events: list[dict] = []

    for event in stream_events(task_id, tenant_id=tenant_id, timeout_s=timeout_s):
        kind = event.get("kind", "")
        events_raw.append(event)
        suffix = ""

        if kind == "PendingApproval":
            approve_url = f"http://localhost:8080/v1/{tenant_id}/tasks/{task_id}/approve"
            approve_body = json.dumps({
                "approved": True,
                "reviewer_note": "auto-approved by e2e harness",
                "operator_id": "e2e-harness",
            }).encode()
            req = urllib.request.Request(
                approve_url,
                data=approve_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    suffix = f"  → auto-approved (HTTP {resp.status})"
            except Exception as exc:
                suffix = f"  → approval failed: {exc}"
            hitl_gate_fired = True

        elif kind == "VerificationScored":
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

        elif kind == "LeaderElected":
            leader_elected = True
            leader_elected_events.append(event)
            suffix = (
                f"  term={event.get('term')}  leader={str(event.get('leader_explorer_id', ''))[:8]}"
                f"  credibility={event.get('credibility_score', 0.0):.2f}"
            )

        elif kind == "SocraticDiagnosis":
            suffix = f"  term={event.get('term')}  eig_rank={event.get('eig_rank')}"

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
    if leader_elected:
        print(f"  leader_elections={len(leader_elected_events)}", end="")
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
        "hitl_gate_fired": hitl_gate_fired,
        "leader_elected": leader_elected,
        "leader_elected_events": leader_elected_events,
    }


# ── Assertions ────────────────────────────────────────────────────────────────

def check_assertions(result: dict, expected: dict, task_json: dict) -> dict[str, dict]:
    """Evaluate _expected assertions against a run result.

    Design rules:
    - terminal: only asserted when explicitly present in `_expected`; omitting it lets
      feature-isolation scenarios pass on TaskFailed (the feature activated; LLM quality
      is tested separately via checks_pass).
    - Feature activation assertions (thinking_loop_ran, hitl_gate_fired, srani_active):
      always evaluated regardless of which config file was used.  In compare mode this
      creates the h2ai-vs-baseline differentiation: h2ai passes them, baseline fails them.
    - checks_pass: only evaluated when the task actually produced a merged output
      (terminal_kind == "MergeResolved"); TaskFailed means no content to score.
    """
    out: dict[str, dict] = {}

    terminal_kind = result.get("terminal_kind", "")

    # terminal is opt-in — only assert when the scenario explicitly requires it
    if "terminal" in expected:
        exp = expected["terminal"]
        terminal_ok = terminal_kind == exp
        out["terminal"] = {"expected": exp, "actual": terminal_kind, "pass": terminal_ok}

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

    if "hitl_gate_fired" in expected:
        actual = result.get("hitl_gate_fired", False)
        exp = expected["hitl_gate_fired"]
        out["hitl_gate_fired"] = {"expected": exp, "actual": actual, "pass": actual == exp}

    if "leader_election_ran" in expected:
        exp = expected["leader_election_ran"]
        actual = result.get("leader_elected", False)
        out["leader_election_ran"] = {"expected": exp, "actual": actual, "pass": actual == exp}

    if "srani_active" in expected:
        # Tests that SRANI was evaluated on at least one proposal (emitted any event).
        # Use srani_active instead of srani_cfi_check to avoid dependency on fabrication firing.
        exp = expected["srani_active"]
        actual = result.get("srani_events_count", 0) > 0 or result.get("srani_cfi") is not None
        out["srani_active"] = {"expected": exp, "actual": actual, "pass": actual == exp}

    if "srani_cfi_check" in expected:
        # Legacy: checks that at least one CorrelatedFabrication event fired.
        exp = expected["srani_cfi_check"]
        actual = len(result["srani_events"]) > 0
        out["srani_cfi_check"] = {"expected": exp, "actual": actual, "pass": actual == exp}

    if "checks_pass_threshold" in expected and terminal_kind == "MergeResolved":
        # Only assert content quality when there is actually a merged output to score.
        exp = expected["checks_pass_threshold"]
        actual = result.get("checks_present", 0)
        out["checks_pass"] = {"expected": exp, "actual": actual, "pass": actual >= exp}

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
        "hitl_gate_fired": result.get("hitl_gate_fired", False),
        # Epistemic leader signals
        "leader_elected": result.get("leader_elected", False),
        "leader_election_count": len(result.get("leader_elected_events", [])),
        # Content check results
        "checks_results": result.get("checks_results", []),
        "checks_present": result.get("checks_present", 0),
        "checks_total": result.get("checks_total", 0),
        "checks_threshold": result.get("checks_threshold", 0),
        "assertions": assertions,
        "pass": all(c["pass"] for c in assertions.values()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return out_dir


# ── Constraint context loader ─────────────────────────────────────────────────

def _load_constraint_context(scenario_dir: pathlib.Path, task: dict) -> str:
    """Load constraint definitions from the wiki corpus and format them for LLM context.

    Used by --context-augmented to give the bare LLM the same constraint knowledge
    that h2ai injects into explorer context via the constraint wiki.  This isolates
    whether h2ai's improvement comes from constraint knowledge or from orchestration.

    Returns a formatted string, empty string if no constraints found.
    """
    import re

    constraint_ids: list[str] = task.get("constraints", [])
    if not constraint_ids:
        return ""

    # Resolve corpus path from the scenario's h2ai.toml
    toml_path = scenario_dir / "h2ai.toml"
    if not toml_path.exists():
        return ""
    cfg = tomllib.loads(toml_path.read_text())
    wiki_cfg = cfg.get("constraint_wiki", {})
    corpus_path_str = wiki_cfg.get("corpus_path", "")
    if not corpus_path_str:
        return ""

    corpus_dir = REPO_ROOT / corpus_path_str

    blocks: list[str] = []
    for cid in constraint_ids:
        # Match e.g. CONSTRAINT-004-budget-pacing-idempotency.yaml
        matches = sorted(corpus_dir.glob(f"{cid}*.yaml"))
        if not matches:
            continue
        raw = matches[0].read_text()

        # Parse key fields with simple regex (avoids pyyaml dependency)
        def _field(key: str) -> str:
            m = re.search(rf'^{key}:\s*["\']?(.*?)["\']?\s*$', raw, re.MULTILINE)
            return m.group(1).strip() if m else ""

        def _block(key: str) -> str:
            # Find "  key: |" line and collect only the block content lines
            # (lines whose indentation exceeds the key's own indentation level)
            lines = raw.splitlines()
            key_indent = None
            content: list[str] = []
            in_block = False
            for line in lines:
                if not in_block:
                    stripped = line.lstrip()
                    if stripped.startswith(f"{key}:") and "|" in line:
                        key_indent = len(line) - len(stripped)
                        in_block = True
                else:
                    if line.strip() == "":
                        content.append("")
                        continue
                    line_indent = len(line) - len(line.lstrip())
                    if line_indent <= key_indent:
                        break  # back to sibling or parent level — stop
                    content.append(line)
            if not content:
                return ""
            min_indent = min((len(l) - len(l.lstrip()) for l in content if l.strip()), default=0)
            return "\n".join(l[min_indent:] for l in content).strip()

        title = _field("title")
        pass_criteria = _block("pass") or _field("pass")
        hint = _field("remediation_hint")

        block = f"### {cid}: {title}\n**Required:** {pass_criteria}"
        if hint:
            block += f"\n**Guidance:** {hint}"
        blocks.append(block)

    if not blocks:
        return ""

    return (
        "The following constraints MUST ALL be satisfied in your design. "
        "Violating any one of them makes the solution unacceptable:\n\n"
        + "\n\n".join(blocks)
    )


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

    toml_path = SCENARIOS_DIR / scenario_name.split("/")[-1] / "h2ai.toml"
    if not toml_path.exists():
        # Try full relative path
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


def _eval_checks_against_output(output: str, checks: list[dict], scenario_name: str) -> list[dict]:
    """Evaluate content checks against merged h2ai output using the LLM judge.

    Uses the same prompt and PRESENT/MISSING parsing as run_baseline().
    Returns early with an empty list if output is empty.
    """
    if not output:
        return []
    endpoint, model = _llm_endpoint_for_scenario(scenario_name)
    results = []
    for check in checks:
        prompt = (
            f"Does the following ANSWER satisfy the CHECK?\n\n"
            f"CHECK: {check['text']}\n\n"
            f"ANSWER:\n{output[:8000]}\n\n"
            f"Respond with exactly one word on its own line: PRESENT (if the answer clearly satisfies the check) or MISSING (if it does not)."
        )
        resp = _llm_call(endpoint, model, [{"role": "user", "content": prompt}], max_tokens=16)
        # Accept PRESENT anywhere in the full response to handle models that add preamble
        verdict = "PRESENT" if "PRESENT" in resp.upper() else "MISSING"
        results.append({"id": check["id"], "verdict": verdict, "pass": verdict == "PRESENT"})
        print(f"  check {check['id']}: {verdict}")
    return results


def run_baseline(scenario_name: str, task: dict, constraint_context: str = "") -> dict:
    endpoint, model = _llm_endpoint_for_scenario(scenario_name)
    mode_label = "context-augmented" if constraint_context else "bare LLM"
    print(f"  LLM endpoint: {endpoint}  model: {model}  mode: {mode_label}")

    expected = task.get("_expected", {})
    checks = expected.get("checks", [])
    threshold = expected.get("checks_pass_threshold", len(checks))

    system_parts = ["You are a senior distributed systems engineer. Be concrete and precise."]
    if constraint_context:
        system_parts.append(constraint_context)
    system_prompt = "\n\n".join(system_parts)

    user_content = task["description"]
    if task.get("context"):
        user_content += "\n\nContext: " + task["context"]

    print(f"  Generating {mode_label} answer via LLM…")
    if constraint_context:
        print(f"  Constraints injected: {len(task.get('constraints', []))} definitions ({len(constraint_context)} chars)")
    t0 = time.time()
    answer = _llm_call(endpoint, model, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ])
    elapsed = time.time() - t0
    print(f"  Answer: {len(answer)} chars in {elapsed:.0f}s")

    results = _eval_checks_against_output(answer, checks, scenario_name)

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
        "mode": "context-augmented" if constraint_context else "baseline",
        "constraints_injected": task.get("constraints", []) if constraint_context else [],
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

def _run_h2ai_trials(
    scenario_name: str,
    scenario_dir: pathlib.Path,
    task: dict,
    config_file: str,
    trials: int,
) -> dict:
    """Run h2ai for `trials` iterations, return aggregated metrics."""
    results = []
    for trial in range(1, trials + 1):
        if trials > 1:
            print(f"  ── trial {trial}/{trials} ──")
        proc = None
        try:
            proc = start_server(scenario_dir, config_file)
            wait_for_health()
            print("  server ready")
            result = run_scenario(scenario_name, task)
            checks = task.get("_expected", {}).get("checks", [])
            threshold = task.get("_expected", {}).get("checks_pass_threshold", len(checks))
            if checks and result["merged_output"]:
                print(f"  evaluating {len(checks)} content checks against merged output…")
                check_results = _eval_checks_against_output(result["merged_output"], checks, scenario_name)
                present = sum(1 for r in check_results if r["pass"])
                print(f"  checks: {present}/{len(checks)}  threshold={threshold}  → {'PASS' if present >= threshold else 'FAIL'}")
                result["checks_results"] = check_results
                result["checks_present"] = present
                result["checks_total"] = len(checks)
                result["checks_threshold"] = threshold
            else:
                result["checks_results"] = []
                result["checks_present"] = 0
                result["checks_total"] = 0
                result["checks_threshold"] = 0
            assertions = check_assertions(result, task.get("_expected", {}), task)
            out_dir = save_results(scenario_name, task, result, assertions)
            passed = all(c["pass"] for c in assertions.values())
            results.append({"passed": passed, "result": result, "assertions": assertions, "out_dir": str(out_dir)})
            mark = "PASS" if passed else "FAIL"
            failed = [k for k, v in assertions.items() if not v["pass"]]
            print(f"  → {mark}" + (f"  failed: {', '.join(failed)}" if failed else ""))
        except Exception as e:
            results.append({"passed": False, "error": str(e)})
            print(f"  → ERROR: {e}")
            traceback.print_exc()
        finally:
            if proc:
                stop_server(proc)
                print("  server stopped")

    passing = sum(1 for r in results if r["passed"])
    pass_k = passing / trials
    last = next((r for r in reversed(results) if "result" in r), {})
    last_result = last.get("result", {})
    return {
        "trials": trials,
        "passing": passing,
        "pass_k": pass_k,
        "j_eff": last_result.get("j_eff"),
        "avg_verification_score": last_result.get("avg_verification_score", 0.0),
        "valid_proposals": len(last_result.get("verification_scores", [])),
        "thinking_loop_iters": (last_result.get("thinking_loop_event") or {}).get("iterations_run"),
        "hitl_fired": last_result.get("hitl_gate_fired", False),
        "leader_elected": last_result.get("leader_elected", False),
        "leader_election_count": len(last_result.get("leader_elected_events", [])),
        "srani_events": len(last_result.get("srani_events", [])),
        "checks_pass_rate": (
            last_result.get("checks_present", 0) / last_result["checks_total"]
            if last_result.get("checks_total", 0) > 0
            else None
        ),
    }


def _print_delta_table(h2ai_metrics: dict, baseline_metrics: dict) -> None:
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        if isinstance(v, bool):
            return "yes" if v else "no"
        return str(v)

    def delta(a, b):
        if a is None or b is None:
            return "—"
        if isinstance(a, float) and isinstance(b, float):
            d = a - b
            return f"{'+' if d >= 0 else ''}{d:.3f}"
        if isinstance(a, bool) and isinstance(b, bool):
            return "✓" if a and not b else ("✗" if not a and b else "=")
        if isinstance(a, int) and isinstance(b, int):
            d = a - b
            return f"{'+' if d >= 0 else ''}{d}"
        return "—"

    rows = [
        ("pass^k",          h2ai_metrics["pass_k"],                baseline_metrics["pass_k"]),
        ("j_eff",           h2ai_metrics["j_eff"],                 baseline_metrics["j_eff"]),
        ("avg_verif_score", h2ai_metrics["avg_verification_score"],baseline_metrics["avg_verification_score"]),
        ("constraint_pass", h2ai_metrics.get("checks_pass_rate"),  baseline_metrics.get("checks_pass_rate")),
        ("valid_proposals", h2ai_metrics["valid_proposals"],       baseline_metrics["valid_proposals"]),
        ("thinking_iters",  h2ai_metrics["thinking_loop_iters"],   baseline_metrics["thinking_loop_iters"]),
        ("hitl_fired",      h2ai_metrics["hitl_fired"],            baseline_metrics["hitl_fired"]),
        ("leader_elected",  h2ai_metrics.get("leader_elected"),    baseline_metrics.get("leader_elected")),
        ("srani_events",    h2ai_metrics["srani_events"],          baseline_metrics["srani_events"]),
    ]
    col_w = 18
    print(f"\n{'─'*72}")
    print(f"  {'Metric':<{col_w}}  {'baseline':>{col_w}}  {'h2ai':>{col_w}}  {'delta':>{col_w}}")
    print(f"{'─'*72}")
    for name, h2ai_val, base_val in rows:
        print(f"  {name:<{col_w}}  {fmt(base_val):>{col_w}}  {fmt(h2ai_val):>{col_w}}  {delta(h2ai_val, base_val):>{col_w}}")
    print(f"{'─'*72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay H2AI scenarios, capture results for regression analysis.")
    parser.add_argument("tasks", nargs="*", metavar="SCENARIO",
                        help="scenario name(s) (e.g. benchmark, features/01-thinking-loop); default: all")
    parser.add_argument("--list", action="store_true", help="list available scenarios and exit")
    parser.add_argument("--baseline", action="store_true",
                        help="direct LLM mode — call LLM without H2AI, score against _expected.checks")
    parser.add_argument("--context-augmented", action="store_true", dest="context_augmented",
                        help="bare LLM + constraint definitions injected into system prompt; isolates framework value from constraint knowledge")
    parser.add_argument("--config", default="h2ai.toml", metavar="FILE",
                        help="toml config filename to load from scenario dir (default: h2ai.toml)")
    parser.add_argument("--trials", type=int, default=1, metavar="K",
                        help="run each scenario K times and report pass^k (default: 1)")
    parser.add_argument("--compare", action="store_true",
                        help="run h2ai.toml then baseline.toml and print delta table")
    args = parser.parse_args()

    if args.list:
        for path in sorted(SCENARIOS_DIR.glob("**/task.json")):
            t = json.loads(path.read_text())
            rel = path.parent.relative_to(SCENARIOS_DIR)
            n_checks = len(t.get("_expected", {}).get("checks", []))
            has_baseline = (path.parent / "baseline.toml").exists()
            print(f"  {str(rel):<45}  checks={n_checks}  {'[baseline.toml]' if has_baseline else ''}")
        return

    scenarios = load_scenarios(args.tasks or None)
    overall: dict[str, str] = {}

    if args.baseline or args.context_augmented:
        for scenario_name, scenario_dir, task in scenarios:
            mode = "CONTEXT-AUGMENTED" if args.context_augmented else "BASELINE (bare LLM)"
            print(f"{'='*60}")
            print(f"{mode}: {scenario_name}")
            print(f"{'='*60}")
            try:
                constraint_ctx = _load_constraint_context(scenario_dir, task) if args.context_augmented else ""
                result = run_baseline(scenario_name, task, constraint_context=constraint_ctx)
                overall[scenario_name] = "PASS" if result["pass"] else "FAIL"
            except Exception as e:
                overall[scenario_name] = f"ERROR: {e}"
                print(f"  → ERROR: {e}")
                traceback.print_exc()
            print()

    elif args.compare:
        for scenario_name, scenario_dir, task in scenarios:
            print(f"{'='*60}")
            print(f"COMPARE: {scenario_name}")
            print(f"{'='*60}")
            try:
                print(f"\n[h2ai.toml — framework]")
                h2ai_m = _run_h2ai_trials(scenario_name, scenario_dir, task, "h2ai.toml", args.trials)
                print(f"\n[baseline.toml — feature OFF]")
                base_m = _run_h2ai_trials(scenario_name, scenario_dir, task, "baseline.toml", args.trials)
                _print_delta_table(h2ai_m, base_m)
                compare_semantics = task.get("_compare_semantics", "strict")
                if compare_semantics == "no_regression":
                    # Feature adds safety/correctness without measurable output uplift.
                    # Pass when h2ai is at least as good as baseline and h2ai itself passes.
                    verdict = "PASS" if h2ai_m["pass_k"] > 0 and h2ai_m["pass_k"] >= base_m["pass_k"] else "FAIL/WORSE"
                else:
                    verdict = "PASS" if h2ai_m["pass_k"] > base_m["pass_k"] else "SAME/WORSE"
                overall[scenario_name] = verdict
            except Exception as e:
                overall[scenario_name] = f"ERROR: {e}"
                print(f"  → ERROR: {e}")
                traceback.print_exc()
            print()

    else:
        for scenario_name, scenario_dir, task in scenarios:
            print(f"{'='*60}")
            print(f"SCENARIO: {scenario_name}  config={args.config}  trials={args.trials}")
            print(f"{'='*60}")
            try:
                metrics = _run_h2ai_trials(scenario_name, scenario_dir, task, args.config, args.trials)
                if args.trials > 1:
                    print(f"  pass^{args.trials} = {metrics['passing']}/{args.trials} = {metrics['pass_k']:.2f}")
                overall[scenario_name] = "PASS" if metrics["pass_k"] >= 1.0 / args.trials else "FAIL"
            except Exception as e:
                overall[scenario_name] = f"ERROR: {e}"
                print(f"  → ERROR: {e}")
                traceback.print_exc()
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
