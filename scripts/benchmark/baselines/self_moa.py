"""B3 baseline: Self-MoA — N samples from one model + aggregation call."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field

from openai import OpenAI

from ..h2ai_client import TokenUsage

_client = OpenAI()


@dataclass
class SelfMoAResult:
    answer: str
    proposals: list[str]
    usage: TokenUsage


_AGGREGATOR_SYSTEM = (
    "You are an expert aggregator. Given several candidate answers to the same question, "
    "synthesise the best single answer. Output ONLY the final answer, no explanation."
)


def _sample(prompt: str, model: str, system: str) -> tuple[str, int, int]:
    resp = _client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.8,
    )
    text = resp.choices[0].message.content or ""
    pt = resp.usage.prompt_tokens if resp.usage else 0
    ct = resp.usage.completion_tokens if resp.usage else 0
    return text, pt, ct


def self_moa(
    prompt: str,
    model: str = "gpt-4o-mini",
    n: int = 5,
    system: str = "You are a helpful assistant. Answer concisely.",
    extract_fn=None,
) -> SelfMoAResult:
    """Sample `n` proposals from `model`, then aggregate via a single aggregator call.

    `extract_fn(raw_text) -> str` normalises the aggregator output.
    """
    if extract_fn is None:
        extract_fn = str.strip

    total_prompt = 0
    total_completion = 0
    proposals: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(_sample, prompt, model, system) for _ in range(n)]
        for f in concurrent.futures.as_completed(futures):
            text, pt, ct = f.result()
            proposals.append(text)
            total_prompt += pt
            total_completion += ct

    # Aggregation call
    candidates_block = "\n\n".join(
        f"Candidate {i + 1}:\n{p}" for i, p in enumerate(proposals)
    )
    agg_prompt = (
        f"Question: {prompt}\n\nCandidate answers:\n{candidates_block}\n\n"
        "Synthesise these into one best answer."
    )
    agg_resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _AGGREGATOR_SYSTEM},
            {"role": "user", "content": agg_prompt},
        ],
        temperature=0.0,
    )
    answer = extract_fn(agg_resp.choices[0].message.content or "")
    if agg_resp.usage:
        total_prompt += agg_resp.usage.prompt_tokens
        total_completion += agg_resp.usage.completion_tokens

    return SelfMoAResult(
        answer=answer,
        proposals=proposals,
        usage=TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            model=model,
        ),
    )
