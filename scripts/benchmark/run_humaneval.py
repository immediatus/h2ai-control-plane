"""HumanEval evaluation runner (pass@1).

Usage:
    python -m scripts.benchmark.run_humaneval [--smoke] [--model gpt-4o-mini]
    python -m scripts.benchmark.run_humaneval --baselines b0 b1 h2

All 164 HumanEval problems are evaluated with code execution (pass@1).
Outputs JSON results to scripts/benchmark/results/humaneval_<run_id>.json.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import re
import statistics
import time
import traceback
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

CODEGEN_SYSTEM = (
    "You are an expert Python programmer. Complete the given function. "
    "Output ONLY the complete function implementation (including the def line). "
    "Do not include any explanation or markdown fences."
)


def _extract_code(text: str) -> str:
    """Extract Python code from model output, stripping markdown fences."""
    # Strip ```python ... ``` fences
    match = re.search(r"```(?:python)?\n([\s\S]*?)```", text)
    if match:
        return match.group(1)
    return text.strip()


def _run_code_safely(code: str, test_code: str, entry_point: str) -> bool:
    """Execute generated code + test harness; return True if all tests pass."""
    namespace: dict[str, Any] = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102
    except Exception:
        return False

    check_fn = namespace.get(entry_point)
    if check_fn is None:
        return False

    # HumanEval test_code is a function `check(candidate)` — exec and call it
    try:
        exec(compile(test_code, "<tests>", "exec"), namespace)  # noqa: S102
        check = namespace.get("check")
        if check is None:
            return False
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            check(check_fn)
        return True
    except Exception:
        return False


@dataclass
class TaskResult:
    problem_id: str
    entry_point: str
    passed: bool
    usage: dict[str, Any]
    latency_s: float


@dataclass
class RunResult:
    baseline: str
    model: str
    pass_at_1: float
    mean_cost_usd: float
    total_cost_usd: float
    n_tasks: int
    task_results: list[TaskResult] = field(default_factory=list)


def _load_humaneval(smoke: bool) -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("openai_humaneval", split="test")
        data = list(ds)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load HumanEval. Install: pip install datasets\n"
            f"Original error: {exc}"
        ) from exc

    return data[:5] if smoke else data


def _single_codegen(prompt: str, model: str) -> tuple[str, TokenUsage, float]:
    t0 = time.monotonic()
    resp = _openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CODEGEN_SYSTEM},
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
        prompt_text = item["prompt"]
        test_code = item["test"]
        entry_point = item["entry_point"]
        task_id = item.get("task_id", str(len(task_results)))

        t0 = time.monotonic()

        if baseline == "b0":
            raw, usage, latency = _single_codegen(prompt_text, model)
            code = _extract_code(raw)
        elif baseline == "b1":
            res = majority_vote(
                prompt_text, model=model, n=6,
                system=CODEGEN_SYSTEM,
                extract_fn=_extract_code,
            )
            # For code: pick the most-voted candidate; execute each and take first passing
            code = res.answer
            usage = res.usage
            latency = time.monotonic() - t0
        elif baseline == "b2":
            res = moa(prompt_text, extract_fn=_extract_code)
            code = res.answer
            usage = res.usage
            latency = time.monotonic() - t0
        elif baseline == "b3":
            res = self_moa(prompt_text, model=model, n=5,
                           system=CODEGEN_SYSTEM, extract_fn=_extract_code)
            code = res.answer
            usage = res.usage
            latency = time.monotonic() - t0
        elif baseline == "h2":
            if h2ai_client is None:
                raise ValueError("H2AI client required")
            full_prompt = f"{CODEGEN_SYSTEM}\n\n{prompt_text}"
            resp = h2ai_client.submit_task(full_prompt, model=model)
            code = _extract_code(resp.answer)
            usage = resp.usage
            latency = resp.latency_s
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        passed = _run_code_safely(code, test_code, entry_point)
        task_results.append(TaskResult(
            problem_id=task_id,
            entry_point=entry_point,
            passed=passed,
            usage=asdict(usage),
            latency_s=latency,
        ))

    pass_at_1 = statistics.mean(r.passed for r in task_results)
    costs_usd = [TokenUsage(**r.usage).cost_usd() for r in task_results]

    return RunResult(
        baseline=baseline,
        model=model,
        pass_at_1=pass_at_1,
        mean_cost_usd=statistics.mean(costs_usd),
        total_cost_usd=sum(costs_usd),
        n_tasks=len(task_results),
        task_results=task_results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HumanEval benchmark")
    parser.add_argument("--smoke", action="store_true", help="5-problem smoke test")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=["b0", "b1", "b2", "b3", "h2"],
        default=["b0", "h2"],
    )
    parser.add_argument("--h2ai-url", default="http://localhost:8080")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    problems = _load_humaneval(args.smoke)
    print(f"Loaded {len(problems)} HumanEval problems")

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
                f"  pass@1={result.pass_at_1:.3f}  "
                f"cost=${result.total_cost_usd:.4f}"
            )
            all_results.append({
                "run": run_idx,
                "baseline": baseline,
                "pass_at_1": result.pass_at_1,
                "mean_cost_usd": result.mean_cost_usd,
                "total_cost_usd": result.total_cost_usd,
                "n_tasks": result.n_tasks,
                "task_results": [asdict(r) for r in result.task_results],
            })

    run_id = uuid.uuid4().hex[:8]
    out_path = RESULTS_DIR / f"humaneval_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump({"benchmark": "humaneval", "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
