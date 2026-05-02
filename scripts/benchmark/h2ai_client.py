"""HTTP client for the H2AI runtime with token cost tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests

# Default model cost table (per 1M tokens, USD)
MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "unknown"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self) -> float:
        costs = MODEL_COSTS.get(self.model, {"input": 0.0, "output": 0.0})
        return (
            self.prompt_tokens * costs["input"] / 1_000_000
            + self.completion_tokens * costs["output"] / 1_000_000
        )


@dataclass
class H2AIResponse:
    task_id: str
    answer: str
    usage: TokenUsage
    latency_s: float
    raw: dict[str, Any] = field(default_factory=dict)


class H2AIClient:
    """Thin HTTP wrapper around POST /tasks for benchmarking."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout_s: float = 120.0,
        default_model: str = "gpt-4o-mini",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_s
        self.default_model = default_model
        self._session = requests.Session()

    def submit_task(
        self,
        description: str,
        model: str | None = None,
        n_agents: int | None = None,
        max_turns: int | None = None,
    ) -> H2AIResponse:
        model = model or self.default_model
        payload: dict[str, Any] = {
            "description": description,
            "model": model,
        }
        if n_agents is not None:
            payload["n_agents"] = n_agents
        if max_turns is not None:
            payload["max_turns"] = max_turns

        t0 = time.monotonic()
        resp = self._session.post(
            f"{self.base_url}/tasks",
            json=payload,
            timeout=self.timeout,
        )
        latency = time.monotonic() - t0
        resp.raise_for_status()
        body = resp.json()

        usage = TokenUsage(
            prompt_tokens=body.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=body.get("usage", {}).get("completion_tokens", 0),
            model=model,
        )
        return H2AIResponse(
            task_id=body.get("task_id", ""),
            answer=body.get("answer", body.get("output", "")),
            usage=usage,
            latency_s=latency,
            raw=body,
        )

    def health(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=5.0)
            return r.status_code == 200
        except requests.RequestException:
            return False
