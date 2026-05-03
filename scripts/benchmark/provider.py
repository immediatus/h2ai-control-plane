"""LLM provider configuration for H2AI benchmarks.

Supports OpenAI, Google Gemini (via OpenAI-compatible endpoint), and
local llama.cpp servers (via OpenAI-compatible /v1 endpoint).

Usage:
    from .provider import make_client, default_model, MODEL_COSTS

    client = make_client("gemini")          # reads GEMINI_API_KEY
    client = make_client("openai")          # reads OPENAI_API_KEY
    client = make_client("llamacpp",        # reads LLAMACPP_BASE_URL (default localhost:8000)
                         base_url="http://localhost:8000/v1")

CLI arg to add to every runner:
    parser.add_argument("--provider", default="openai",
                        choices=list(PROVIDERS), help="LLM provider")
    parser.add_argument("--base-url", default=None,
                        help="Override provider base URL (e.g. for llama.cpp)")
"""

from __future__ import annotations

import os

from openai import OpenAI

# Per-provider defaults
PROVIDERS: dict[str, dict] = {
    "openai": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
    },
    "llamacpp": {
        # llama.cpp --server exposes an OpenAI-compatible /v1 endpoint.
        # Set LLAMACPP_BASE_URL or pass --base-url.
        "base_url": os.environ.get("LLAMACPP_BASE_URL", "http://localhost:8000/v1"),
        "api_key_env": None,   # no auth needed for local server
        "default_model": "local",
    },
}

# Model cost table — per 1M tokens, USD (2026-05 pricing)
MODEL_COSTS: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Google Gemini
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    # llama.cpp local (zero API cost — compute cost is out of band)
    "local": {"input": 0.0, "output": 0.0},
}


def make_client(provider: str = "openai", base_url: str | None = None) -> OpenAI:
    """Return a configured OpenAI-compatible client for the given provider.

    Args:
        provider: "openai", "gemini", or "llamacpp".
        base_url: Override the provider's default base URL.  Required when
            pointing at a non-default llama.cpp server address.
    """
    cfg = PROVIDERS[provider]
    resolved_url = base_url or cfg["base_url"]

    key_env = cfg["api_key_env"]
    api_key = os.environ.get(key_env, "none") if key_env else "none"

    kwargs: dict = {"api_key": api_key}
    if resolved_url:
        kwargs["base_url"] = resolved_url
    return OpenAI(**kwargs)


def default_model(provider: str = "openai") -> str:
    return PROVIDERS[provider]["default_model"]


def add_provider_args(parser) -> None:
    """Add --provider, --base-url, and --model args to an argparse parser."""
    parser.add_argument(
        "--provider",
        default="openai",
        choices=list(PROVIDERS),
        help="LLM provider: openai | gemini | llamacpp",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override provider base URL (e.g. http://localhost:8000/v1 for llama.cpp)",
    )
