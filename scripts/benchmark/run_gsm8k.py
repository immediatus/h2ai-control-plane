"""GSM8K evaluation runner.

Usage:
    python -m scripts.benchmark.run_gsm8k [--smoke] [--n-samples 500] [--model gpt-4o-mini]
    python -m scripts.benchmark.run_gsm8k --baselines b0 b1 b2 b3 h2

Outputs JSON results to scripts/benchmark/results/gsm8k_<run_id>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from .baselines.majority_vote import majority_vote
from .baselines.moa import moa
from .baselines.self_moa import self_moa
from .h2ai_client import H2AIClient, TokenUsage

_openai = OpenAI()

RESULTS_DIR = Path(__file__).parent / "results"

GSM8K_SYSTEM = (
    "You are a math tutor. Solve the problem step by step, then state the final "
    "numeric answer on its own line prefixed with '####'."
)


def _extract_gsm8k_answer(text: str) -> str:
    """Extract the final numeric answer from GSM8K-style output."""
    # GSM8K answers follow '####'
    match = re.search(r"####\s*([\d,\-\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last number in the text
    nums = re.findall(r"[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else text.strip()


@dataclass
class TaskResult:
    problem_id: str
    correct_answer: str
    predicted_answer: str
    correct: bool
    usage: dict[str, Any]
    latency_s: float


@dataclass
class RunResult:
    baseline: str
    model: str
    accuracy: float
    mean_cost_usd: float
    total_cost_usd: float
    mean_latency_s: float
    n_tasks: int
    task_results: list[TaskResult] = field(default_factory=list)


def _load_gsm8k(n_samples: int, smoke: bool) -> list[dict]:
    """Load GSM8K problems from HuggingFace datasets."""
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("gsm8k", "main", split="test")
        data = list(ds)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load GSM8K. Install: pip install datasets\n"
            f"Original error: {exc}"
        ) from exc

    if smoke:
        return data[:5]

    # Stratified sample: evenly spaced indices to cover difficulty spread
    step = max(1, len(data) // n_samples)
    return data[::step][:n_samples]


def _single_call(prompt: str, model: str) -> tuple[str, TokenUsage, float]:
    t0 = time.monotonic()
    resp = _openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GSM8K_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    latency = time.monotonic() - t0
    text = resp.choices[0].message.content or ""
    usage = TokenUsage(
        prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        model=model,
    )
    return text, usage, latency


def _run_baseline(
    baseline: str,
    problems: list[dict],
    model: str,
    h2ai_client: H2AIClient | None = None,
) -> RunResult:
    task_results: list[TaskResult] = []

    for item in problems:
        question = item["question"]
        correct_raw = item["answer"]
        correct_answer = _extract_gsm8k_answer(correct_raw)

        t0 = time.monotonic()

        if baseline == "b0":
            raw, usage, latency = _single_call(question, model)
            predicted = _extract_gsm8k_answer(raw)
        elif baseline == "b1":
            res = majority_vote(
                question, model=model, n=6,
                system=GSM8K_SYSTEM,
                extract_fn=_extract_gsm8k_answer,
            )
            predicted = res.answer
            usage = res.usage
            latency = time.monotonic() - t0
        elif baseline == "b2":
            res = moa(question, extract_fn=_extract_gsm8k_answer)
            predicted = res.answer
            usage = res.usage
            latency = time.monotonic() - t0
        elif baseline == "b3":
            res = self_moa(
                question, model=model, n=5,
                system=GSM8K_SYSTEM,
                extract_fn=_extract_gsm8k_answer,
            )
            predicted = res.answer
            usage = res.usage
            latency = time.monotonic() - t0
        elif baseline == "h2":
            if h2ai_client is None:
                raise ValueError("H2AI client required for 'h2' baseline")
            prompt = f"{GSM8K_SYSTEM}\n\n{question}"
            resp = h2ai_client.submit_task(prompt, model=model)
            predicted = _extract_gsm8k_answer(resp.answer)
            usage = resp.usage
            latency = resp.latency_s
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        correct = predicted == correct_answer
        task_results.append(TaskResult(
            problem_id=item.get("id", str(len(task_results))),
            correct_answer=correct_answer,
            predicted_answer=predicted,
            correct=correct,
            usage=asdict(usage),
            latency_s=latency,
        ))

    accuracy = statistics.mean(r.correct for r in task_results)
    costs = [
        TokenUsage(**r.usage).cost_usd() if isinstance(r.usage, dict)
        else r.usage.cost_usd()
        for r in task_results
    ]
    # Recalculate cost from stored dict
    costs_usd = []
    for r in task_results:
        u = r.usage if isinstance(r.usage, TokenUsage) else TokenUsage(**r.usage)
        costs_usd.append(u.cost_usd())

    return RunResult(
        baseline=baseline,
        model=model,
        accuracy=accuracy,
        mean_cost_usd=statistics.mean(costs_usd),
        total_cost_usd=sum(costs_usd),
        mean_latency_s=statistics.mean(r.latency_s for r in task_results),
        n_tasks=len(task_results),
        task_results=task_results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSM8K benchmark")
    parser.add_argument("--smoke", action="store_true", help="5-problem smoke test")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=["b0", "b1", "b2", "b3", "h2"],
        default=["b0", "h2"],
    )
    parser.add_argument("--h2ai-url", default="http://localhost:8080")
    parser.add_argument("--runs", type=int, default=1, help="Repeat runs for σ estimation")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    problems = _load_gsm8k(args.n_samples, args.smoke)
    print(f"Loaded {len(problems)} GSM8K problems")

    h2ai_client: H2AIClient | None = None
    if "h2" in args.baselines:
        h2ai_client = H2AIClient(base_url=args.h2ai_url, default_model=args.model)
        if not h2ai_client.health():
            print(f"WARNING: H2AI runtime not reachable at {args.h2ai_url}")

    all_results: list[dict] = []
    for run_idx in range(args.runs):
        print(f"\n=== Run {run_idx + 1}/{args.runs} ===")
        for baseline in args.baselines:
            print(f"Running baseline '{baseline}'...")
            result = _run_baseline(baseline, problems, args.model, h2ai_client)
            print(
                f"  accuracy={result.accuracy:.3f}  "
                f"cost=${result.total_cost_usd:.4f}  "
                f"latency={result.mean_latency_s:.1f}s/task"
            )
            all_results.append({
                "run": run_idx,
                "baseline": baseline,
                "accuracy": result.accuracy,
                "mean_cost_usd": result.mean_cost_usd,
                "total_cost_usd": result.total_cost_usd,
                "mean_latency_s": result.mean_latency_s,
                "n_tasks": result.n_tasks,
                "task_results": [asdict(r) for r in result.task_results],
            })

    run_id = uuid.uuid4().hex[:8]
    out_path = RESULTS_DIR / f"gsm8k_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump({"benchmark": "gsm8k", "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
