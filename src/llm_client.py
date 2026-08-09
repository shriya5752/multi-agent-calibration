"""
Thin factory for the LLM client so run_experiment.py doesn't care which
provider you use. Works with any OpenAI-compatible API (Groq, OpenAI,
Together, Fireworks, local vLLM server, etc) — just set the base URL
and API key via environment variables.

Groq is recommended for this project: it has a generous free tier and
is fast enough to run multi-round, multi-agent debates over 30-50
questions in a few minutes.
"""

import os
from openai import OpenAI


def get_client():
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY not set. Copy .env.example to .env and fill in "
            "your API key (e.g. a free Groq key from console.groq.com), "
            "or run with --mock to test the pipeline without an API key."
        )

    return OpenAI(api_key=api_key, base_url=base_url)
