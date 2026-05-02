"""B1 baseline: N=6 majority vote with a single model."""

from __future__ import annotations

import concurrent.futures
from collections import Counter
from dataclasses import dataclass, field

from openai import OpenAI

from ..h2ai_client import MODEL_COSTS, TokenUsage

_client = OpenAI()


@dataclass
class MajorityVoteResult:
    answer: str
    votes: dict[str, int]
    usage: TokenUsage
    all_answers: list[str] = field(default_factory=list)


def _call_once(prompt: str, model: str, system: str) -> tuple[str, int, int]:
    resp = _client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = resp.choices[0].message.content or ""
    return (
        text,
        resp.usage.prompt_tokens if resp.usage else 0,
        resp.usage.completion_tokens if resp.usage else 0,
    )


def majority_vote(
    prompt: str,
    model: str = "gpt-4o-mini",
    n: int = 6,
    system: str = "You are a helpful assistant. Answer concisely.",
    extract_fn=None,
) -> MajorityVoteResult:
    """Run `n` parallel calls and pick the plurality answer.

    `extract_fn(raw_text) -> str` normalises each raw answer before voting
    (e.g. strip non-numeric characters for GSM8K).  Defaults to str.strip().
    """
    if extract_fn is None:
        extract_fn = str.strip

    total_prompt = 0
    total_completion = 0
    answers: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(_call_once, prompt, model, system) for _ in range(n)]
        for f in concurrent.futures.as_completed(futures):
            text, pt, ct = f.result()
            answers.append(extract_fn(text))
            total_prompt += pt
            total_completion += ct

    votes = Counter(answers)
    winner = votes.most_common(1)[0][0]
    return MajorityVoteResult(
        answer=winner,
        votes=dict(votes),
        usage=TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            model=model,
        ),
        all_answers=answers,
    )
