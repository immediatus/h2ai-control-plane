"""B2 baseline: MoA 3-layer (Wang et al., arxiv 2406.04692).

Layer 1: diverse proposer models generate candidate answers in parallel.
Layer 2: (optional) refinement layer — each proposer sees all Layer-1 outputs.
Layer 3: strong aggregator model synthesises the final answer.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field

from openai import OpenAI

from ..h2ai_client import TokenUsage

_PROPOSER_SYSTEM = "You are a helpful expert. Answer the question concisely and accurately."
_AGGREGATOR_SYSTEM = (
    "You are a meticulous expert synthesiser. You will be given several candidate answers "
    "to the same question from different models. Your task is to produce the single best "
    "answer, correcting any errors and preserving the best insights. Output ONLY the final "
    "answer with no explanation."
)


@dataclass
class MoAResult:
    answer: str
    layer1_proposals: list[str]
    usage: TokenUsage


def _propose(prompt: str, model: str, client: OpenAI) -> tuple[str, int, int]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _PROPOSER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    text = resp.choices[0].message.content or ""
    pt = resp.usage.prompt_tokens if resp.usage else 0
    ct = resp.usage.completion_tokens if resp.usage else 0
    return text, pt, ct


def moa(
    prompt: str,
    proposer_models: list[str] | None = None,
    aggregator_model: str | None = None,
    extract_fn=None,
    client: OpenAI | None = None,
) -> MoAResult:
    """Run MoA: parallel proposers → single aggregator.

    `proposer_models` defaults to three copies of the aggregator model when not
    supplied — matching the Self-MoA regime.  Pass explicit model names for
    true diverse-model MoA.
    `extract_fn(raw_text) -> str` normalises the aggregator output.
    `client` defaults to a plain OpenAI() when not supplied.
    """
    if client is None:
        client = OpenAI()
    if aggregator_model is None:
        aggregator_model = "gpt-4o-mini"
    if proposer_models is None:
        proposer_models = [aggregator_model, aggregator_model, aggregator_model]
    if extract_fn is None:
        extract_fn = str.strip

    total_prompt = 0
    total_completion = 0
    proposals: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(proposer_models)) as ex:
        futures = {ex.submit(_propose, prompt, m, client): m for m in proposer_models}
        for f in concurrent.futures.as_completed(futures):
            text, pt, ct = f.result()
            proposals.append(text)
            total_prompt += pt
            total_completion += ct

    candidates_block = "\n\n".join(
        f"Proposer {i + 1} ({model}):\n{p}"
        for i, (p, model) in enumerate(zip(proposals, proposer_models))
    )
    agg_prompt = (
        f"Question: {prompt}\n\nCandidate answers from different models:\n"
        f"{candidates_block}\n\nSynthesize these into one definitive answer."
    )
    agg_resp = client.chat.completions.create(
        model=aggregator_model,
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

    return MoAResult(
        answer=answer,
        layer1_proposals=proposals,
        usage=TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            model=aggregator_model,
        ),
    )
