"""5-problem GSM8K smoke test.

Validates end-to-end pipeline: dataset load → B0 single call → oracle check →
token count → (optional) H2AI call.

Usage:
    python -m scripts.benchmark.smoke_test
    python -m scripts.benchmark.smoke_test --h2ai-url http://localhost:8080
    python -m scripts.benchmark.smoke_test --skip-h2ai

Expected: all 5 problems complete without exception; oracle check returns bool;
          token counts > 0; total cost reported < $0.50.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from openai import OpenAI

from .h2ai_client import H2AIClient, TokenUsage

_openai = OpenAI()

_SYSTEM = (
    "Solve the math problem step by step. "
    "State the final numeric answer on its own line prefixed with '####'."
)


def _extract_answer(text: str) -> str:
    match = re.search(r"####\s*([\d,\-\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    nums = re.findall(r"[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else text.strip()


def _load_5() -> list[dict]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("gsm8k", "main", split="test")
    return list(ds)[:5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h2ai-url", default="http://localhost:8080")
    parser.add_argument("--skip-h2ai", action="store_true")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    print("Loading 5 GSM8K problems...")
    problems = _load_5()
    assert len(problems) == 5, f"Expected 5 problems, got {len(problems)}"

    total_cost = 0.0
    correct_b0 = 0
    print("\n--- B0 (N=1 single call) ---")
    for i, item in enumerate(problems):
        question = item["question"]
        gold = _extract_answer(item["answer"])

        t0 = time.monotonic()
        resp = _openai.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )
        latency = time.monotonic() - t0
        text = resp.choices[0].message.content or ""
        predicted = _extract_answer(text)
        usage = TokenUsage(
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            model=args.model,
        )
        assert usage.prompt_tokens > 0, f"Problem {i}: prompt_tokens == 0"
        assert usage.completion_tokens > 0, f"Problem {i}: completion_tokens == 0"

        correct = predicted == gold
        cost = usage.cost_usd()
        total_cost += cost
        correct_b0 += int(correct)
        marker = "PASS" if correct else "FAIL"
        print(
            f"  [{marker}] Q{i + 1}: gold={gold!r}  predicted={predicted!r}  "
            f"tokens={usage.total_tokens}  cost=${cost:.5f}  {latency:.1f}s"
        )

    print(f"\nB0 accuracy: {correct_b0}/5   total_cost: ${total_cost:.4f}")
    assert total_cost < 0.50, f"Smoke test cost ${total_cost:.4f} exceeded $0.50 budget"

    if not args.skip_h2ai:
        h2 = H2AIClient(base_url=args.h2ai_url, default_model=args.model)
        if not h2.health():
            print(f"\nH2AI runtime not reachable at {args.h2ai_url} — skipping H2 check")
        else:
            print("\n--- H2 (H2AI runtime) ---")
            correct_h2 = 0
            for i, item in enumerate(problems):
                question = item["question"]
                gold = _extract_answer(item["answer"])
                prompt = f"{_SYSTEM}\n\n{question}"
                try:
                    resp = h2.submit_task(prompt, model=args.model)
                    predicted = _extract_answer(resp.answer)
                    correct = predicted == gold
                    correct_h2 += int(correct)
                    marker = "PASS" if correct else "FAIL"
                    print(f"  [{marker}] Q{i + 1}: gold={gold!r}  predicted={predicted!r}")
                except Exception as exc:
                    print(f"  [ERROR] Q{i + 1}: {exc}")
            print(f"H2 accuracy: {correct_h2}/5")

    print("\nSmoke test PASSED — infrastructure validated.")


if __name__ == "__main__":
    main()
