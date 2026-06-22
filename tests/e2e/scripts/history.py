#!/usr/bin/env python3
"""
Show run history per scenario with score trend and regression detection.

Usage:
  python3 tests/e2e/scripts/history.py           # last 5 runs per scenario
  python3 tests/e2e/scripts/history.py --n 10    # last 10 runs per scenario
  python3 tests/e2e/scripts/history.py --scenario compliance-lite
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict

RESULTS = pathlib.Path("tests/e2e/results")

# Map old result-directory paths to current canonical scenario names.
CANONICAL = {
    "benchmark/compliance-lite":              "compliance-lite",
    "benchmark/complexity-routing":           "complexity-routing",
    "benchmark/terminal-bft-adversarial":     "terminal-bft-adversarial",
    "benchmark/tau-bench-constraint-drift":   "tau-bench-constraint-drift",
    "benchmark/hle-expert-consensus":         "hle-expert-consensus",
    "innovation-5/i5-tier2-multi-constraint-billing": "multi-constraint-billing",
    # current flat paths
    "compliance-lite":              "compliance-lite",
    "complexity-routing":           "complexity-routing",
    "terminal-bft-adversarial":     "terminal-bft-adversarial",
    "tau-bench-constraint-drift":   "tau-bench-constraint-drift",
    "hle-expert-consensus":         "hle-expert-consensus",
    "multi-constraint-billing":     "multi-constraint-billing",
}

EXPECTED_PASS = {
    "compliance-lite":          True,
    "complexity-routing":       True,
    "terminal-bft-adversarial": True,
    "tau-bench-constraint-drift": True,
    "hle-expert-consensus":     False,   # expected VerificationExhaustion
    "multi-constraint-billing": True,
}

ORDER = [
    "compliance-lite",
    "complexity-routing",
    "terminal-bft-adversarial",
    "tau-bench-constraint-drift",
    "hle-expert-consensus",
    "multi-constraint-billing",
]


def load_runs():
    by_scenario = defaultdict(list)
    for f in sorted(RESULTS.glob("**/summary.json")):
        try:
            s = json.loads(f.read_text())
        except Exception:
            continue
        raw = s.get("scenario", "")
        name = CANONICAL.get(raw)
        if name is None:
            continue   # removed / unknown scenario
        s["_canonical"] = name
        s["_path"] = str(f.parent)
        by_scenario[name].append(s)
    # sort each list by timestamp ascending
    for name in by_scenario:
        by_scenario[name].sort(key=lambda x: x.get("timestamp", ""))
    return by_scenario


def result_char(s):
    terminal = s.get("terminal_kind", "") or ""
    assertions = s.get("assertions", {})
    all_pass = all(v.get("pass", False) for v in assertions.values()) if assertions else None
    if terminal in ("", None):
        return "INTR", "⏸"
    if all_pass is True:
        return "PASS", "✅"
    if all_pass is False:
        if terminal == "TaskFailed":
            return "FAIL", "❌"
        return "FAIL", "❌"
    return "?", "?"


def score_trend(scores):
    """Arrow showing last-3 trend."""
    valid = [s for s in scores if s is not None]
    if len(valid) < 2:
        return ""
    delta = valid[-1] - valid[-2]
    if delta > 0.05:
        return "↑"
    if delta < -0.05:
        return "↓"
    return "→"


def print_scenario(name, runs, n):
    expected = EXPECTED_PASS.get(name, True)
    exp_label = "PASS expected" if expected else "FAIL expected (ceiling)"
    print(f"\n{'─'*90}")
    print(f"  {name}  [{exp_label}]")
    print(f"{'─'*90}")
    print(f"  {'timestamp':<22} {'sha':<8} {'R':<5} {'score':<7} {'j_eff':<6} {'iter':<5} {'cov':<5} {'cause'}")
    print(f"  {'-'*80}")

    recent = runs[-n:]
    scores = [r.get("avg_verification_score") for r in recent]
    trend = score_trend(scores)

    prev_score = None
    for r in recent:
        ts = r.get("timestamp", "?")
        sha = r.get("git_sha", "?")
        terminal = r.get("terminal_kind", "") or ""
        score = r.get("avg_verification_score")
        j_eff = r.get("j_eff")
        iters = r.get("thinking_loop_iterations", "?")
        cov = r.get("thinking_loop_coverage")
        assertions = r.get("assertions", {})
        all_pass = all(v.get("pass", False) for v in assertions.values()) if assertions else None

        result, icon = result_char(r)

        # regression / improvement flag
        flag = ""
        if prev_score is not None and score is not None:
            delta = score - prev_score
            if delta <= -0.1:
                flag = " ⚠ REGRESSION"
            elif delta >= 0.1:
                flag = " ★ IMPROVEMENT"
        prev_score = score if score is not None else prev_score

        score_s = f"{score:.3f}" if isinstance(score, float) else "-"
        j_s = f"{j_eff:.3f}" if isinstance(j_eff, float) else "-"
        cov_s = f"{cov:.2f}" if isinstance(cov, float) else "-"
        iters_s = str(iters) if iters is not None else "-"

        cause = ""
        if terminal == "TaskFailed":
            ev_f = pathlib.Path(r["_path"]) / "events.jsonl"
            if ev_f.exists():
                for line in ev_f.read_text().splitlines():
                    try:
                        e = json.loads(line)
                        if e.get("kind") == "TaskFailed":
                            cause = str(e.get("cause", ""))[:25]
                            break
                    except Exception:
                        pass

        print(f"  {ts:<22} {sha:<8} {icon}{result:<4} {score_s:<7} {j_s:<6} {iters_s:<5} {cov_s:<5} {cause}{flag}")

    if len(runs) > n:
        print(f"  ... ({len(runs) - n} older runs omitted)")

    # summary line
    pass_runs = [r for r in runs if result_char(r)[0] == "PASS"]
    fail_runs = [r for r in runs if result_char(r)[0] == "FAIL"]
    intr_runs = [r for r in runs if result_char(r)[0] == "INTR"]
    best = max((r.get("avg_verification_score", 0) or 0 for r in runs), default=0)
    last_score = recent[-1].get("avg_verification_score") if recent else None
    last_s = f"{last_score:.3f}" if isinstance(last_score, float) else "-"
    print(f"\n  Total runs: {len(runs)}  PASS:{len(pass_runs)}  FAIL:{len(fail_runs)}  INTR:{len(intr_runs)}")
    print(f"  Best score: {best:.3f}   Last score: {last_s}   Trend: {trend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="last N runs to show per scenario")
    ap.add_argument("--scenario", default=None, help="show only this scenario")
    args = ap.parse_args()

    by_scenario = load_runs()

    names = [args.scenario] if args.scenario else ORDER
    for name in names:
        runs = by_scenario.get(name, [])
        if not runs:
            print(f"\n  {name}: no results yet")
            continue
        print_scenario(name, runs, args.n)

    print(f"\n{'═'*90}")


if __name__ == "__main__":
    main()
