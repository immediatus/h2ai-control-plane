#!/usr/bin/env python3
"""
Analyze events.jsonl files from e2e scenario runs.

Usage:
  python3 tests/e2e/analyze_events.py tau-bench-constraint-drift
  python3 tests/e2e/analyze_events.py complexity-routing --run latest
  python3 tests/e2e/analyze_events.py --all
  python3 tests/e2e/analyze_events.py hle-expert-consensus --checks
"""

import argparse
import json
import pathlib
import sys

RESULTS = pathlib.Path("tests/e2e/results")


def _short(s, n=100):
    if not s:
        return ""
    s = str(s).replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s


def analyze_run(run_dir: pathlib.Path, show_checks=False):
    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"

    print(f"\n{'='*70}")
    print(f"RUN: {run_dir}")

    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        terminal = s.get("terminal_kind") or "(none)"
        avg = s.get("avg_verification_score", 0.0)
        passed = s.get("pass", False)
        iters = s.get("thinking_loop_iterations")
        cov = s.get("thinking_loop_coverage")
        print(f"  terminal={terminal}  avg_score={avg:.3f}  pass={passed}")
        print(f"  thinking_iterations={iters}  coverage={cov}")
        agg = s.get("assertions", {})
        for k, v in agg.items():
            status = "✓" if v.get("pass") else "✗"
            print(f"  assertion[{k}]: {status}  expected={v.get('expected')}  actual={v.get('actual')}")

    if not events_path.exists() or events_path.stat().st_size == 0:
        print("  events.jsonl: EMPTY")
        return

    events = []
    for line in events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  [parse error: {e}]")

    print(f"\n  {'KIND':<28} {'SCORE':>6}  {'CSTR/REASON'}")
    print(f"  {'-'*28} {'-'*6}  {'-'*50}")

    for e in events:
        kind = e.get("kind", "?")
        ts = e.get("timestamp", "")[:19].replace("T", " ")

        if kind == "ThinkingLoopCompleted":
            iters = e.get("iterations_run")
            cov = e.get("coverage_score")
            print(f"  {kind:<28}  iters={iters} coverage={cov}  [{ts}]")

        elif kind == "ComplexityProbe":
            c = e.get("complexity")
            rat = _short(e.get("rationale", ""), 80)
            rec = e.get("decompose_recommended")
            lat = e.get("probe_latency_ms", 0) / 1000
            print(f"  {kind:<28}  complexity={c} decompose={rec} ({lat:.1f}s)  [{ts}]")
            if rat:
                print(f"    rationale: {rat}")

        elif kind == "ComplexityCeilingDetected":
            retry = e.get("retry_count")
            sig = e.get("signals_fired")
            print(f"  {kind:<28}  retry={retry} signals={sig}  [{ts}]")

        elif kind == "VerificationScored":
            score = e.get("score", 0.0)
            passed = e.get("passed", False)
            reason = _short(e.get("reason", ""), 60)
            passed_chk = e.get("passed_checks", 0)
            total_chk = e.get("total_checks", 0)
            explorer = (e.get("explorer_id") or "")[:8]
            icon = "✓" if passed else "✗"
            print(f"  {kind:<28} {score:>6.3f}  {icon} explorer={explorer}  checks={passed_chk}/{total_chk}  [{ts}]")
            if reason:
                print(f"    violated: {reason}")
            if show_checks and e.get("per_check_verdicts"):
                for chk in e["per_check_verdicts"]:
                    idx = chk.get("index", "?")
                    ckind = chk.get("kind", "?")
                    txt = _short(chk.get("text", ""), 80)
                    print(f"      check[{idx}]: {ckind}  {txt}")

        elif kind == "TaskFailed":
            cause = e.get("primary_cause", "?")
            top_v = e.get("top_violated_constraints", [])
            violated_str = ", ".join(f"{c}×{n}" for c, n in top_v) if top_v else "(none)"
            print(f"  {kind:<28}         cause={cause}  violated=[{violated_str}]  [{ts}]")

        elif kind == "MergeResolved":
            winner = (e.get("winner_explorer_id") or "")[:8]
            score = e.get("final_score", "?")
            print(f"  {kind:<28}         winner={winner}  final_score={score}  [{ts}]")

        else:
            print(f"  {kind:<28}  [{ts}]")


def list_runs(scenario: str) -> list[pathlib.Path]:
    base = RESULTS / scenario
    if not base.exists():
        return []
    return sorted(base.iterdir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", help="scenario path, e.g. benchmark/tau-bench-constraint-drift")
    ap.add_argument("--run", default="latest", help="'latest', 'all', or timestamp dir name")
    ap.add_argument("--all", action="store_true", help="analyze all scenarios")
    ap.add_argument("--checks", action="store_true", help="show per-check verdicts in VerificationScored events")
    args = ap.parse_args()

    scenarios = []
    if args.all:
        for category in RESULTS.iterdir():
            if category.is_dir() and not category.name.startswith("_"):
                for sc in category.iterdir():
                    if sc.is_dir():
                        scenarios.append(f"{category.name}/{sc.name}")
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        ap.print_help()
        sys.exit(1)

    for scenario in sorted(scenarios):
        runs = list_runs(scenario)
        if not runs:
            print(f"\n[no runs found for {scenario}]")
            continue

        if args.run == "latest":
            analyze_run(runs[-1], show_checks=args.checks)
        elif args.run == "all":
            for r in runs:
                analyze_run(r, show_checks=args.checks)
        else:
            matched = [r for r in runs if r.name == args.run]
            if not matched:
                print(f"\n[no run '{args.run}' in {scenario}]")
            else:
                analyze_run(matched[0], show_checks=args.checks)


if __name__ == "__main__":
    main()
