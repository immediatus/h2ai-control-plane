#!/usr/bin/env python3
"""
baseline_eval.py

Measures per-adapter output accuracy against a reference answer set.
Outputs a baseline_accuracy_proxy value suitable for H2AIConfig.

Usage (dry run — shows proxy table without a live endpoint):
    python3 scripts/baseline_eval.py

Usage (live measurement):
    python3 scripts/baseline_eval.py --endpoint http://localhost:8080 \\
        --eval-file scripts/eval_questions.jsonl

Eval file format (JSONL, one JSON object per line):
    {"question": "What is 2+2?", "correct_answer": "4", "keywords": ["4", "four"]}
"""
import argparse
import json
import sys
from typing import Any, Dict, List


def load_eval_set(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def score_response(response: str, item: Dict[str, Any]) -> float:
    resp_lower = response.lower()
    for kw in item.get("keywords", [item.get("correct_answer", "")]):
        if kw.lower() in resp_lower:
            return 1.0
    return 0.0


def run_dry_run():
    print("=== Baseline Eval — Dry Run (no endpoint) ===\n")
    print("Relationship between CG_mean (calibration output similarity) and p_proxy:\n")
    print(f"  {'CG_mean':>8}  {'p_proxy':>8}  {'rho_proxy':>10}  note")
    print("  " + "-" * 60)
    for cg_int in range(10, 100, 10):
        cg = cg_int / 100.0
        p = 0.5 + cg / 2.0
        rho = 1.0 - cg
        note = "override if measured accuracy differs by >0.05"
        print(f"  {cg:8.2f}  {p:8.3f}  {rho:10.3f}  {note}")
    print("\nTo measure actual accuracy:")
    print("  1. Create scripts/eval_questions.jsonl with question/keyword pairs")
    print("  2. Run: python3 scripts/baseline_eval.py --endpoint http://localhost:8080 \\")
    print("              --eval-file scripts/eval_questions.jsonl")
    print("  3. Add the reported baseline_accuracy_proxy to your H2AI config JSON")
    print("\nExample eval_questions.jsonl entries:")
    for item in [
        {"question": "What is 2+2?", "correct_answer": "4", "keywords": ["4", "four"]},
        {"question": "Capital of France?", "correct_answer": "Paris", "keywords": ["Paris", "paris"]},
    ]:
        print(f"  {json.dumps(item)}")


def run_eval(endpoint: str, eval_file: str):
    try:
        import requests
    except ImportError:
        print("ERROR: requests library required. Run: pip install requests")
        sys.exit(1)

    items = load_eval_set(eval_file)
    if not items:
        print(f"ERROR: No items in {eval_file}")
        sys.exit(1)

    print(f"=== Baseline Accuracy Evaluation ===")
    print(f"Endpoint: {endpoint}  |  Questions: {len(items)}\n")

    scores = []
    for i, item in enumerate(items):
        payload = {"task": item["question"],
                   "system_context": "Answer briefly and directly.",
                   "max_tokens": 64}
        try:
            resp = requests.post(f"{endpoint}/eval_single", json=payload, timeout=30)
            resp.raise_for_status()
            output = resp.json().get("output", "")
        except Exception as e:
            print(f"  [{i+1}/{len(items)}] ERROR: {e}")
            scores.append(0.0)
            continue
        score = score_response(output, item)
        scores.append(score)
        icon = "✓" if score > 0 else "✗"
        print(f"  [{i+1}/{len(items)}] {icon} {item['question'][:60]}")

    accuracy = sum(scores) / len(scores) if scores else 0.0
    print(f"\n=== Result ===")
    print(f"Accuracy: {accuracy:.3f} ({sum(scores):.0f}/{len(scores)} correct)")
    print(f'\nAdd to H2AI config: "baseline_accuracy_proxy": {accuracy:.3f}')
    if accuracy < 0.5:
        print("\nWARNING: accuracy < 0.5 — ensemble will not improve over single agent.")
        print("Condorcet JT requires p > 0.5.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure per-adapter baseline accuracy")
    parser.add_argument("--endpoint", default="", help="H2AI endpoint URL")
    parser.add_argument("--eval-file", default="", help="Path to JSONL eval questions file")
    args = parser.parse_args()

    if not args.endpoint:
        run_dry_run()
    else:
        if not args.eval_file:
            print("ERROR: --eval-file required when --endpoint is given")
            sys.exit(1)
        run_eval(args.endpoint, args.eval_file)
