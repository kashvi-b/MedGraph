"""Configuration. Model ids and providers live here, never in node code.

Free tiers churn — models get deprecated and quotas get cut without notice — so
swapping a provider must be an environment change, not a code change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env at import, so every entry point sees it.

    Done here rather than in each script because config is the one module all of
    them import. The project shipped a .env.example and python-dotenv as a
    dependency while nothing actually read the file, so every documented command
    failed with a missing-variable error for anyone following the README.

    Existing environment variables always win, so an explicit export or a CI
    secret is never overridden by a stale local file.
    """
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env, override=False)
        return
    except ImportError:
        pass
    for line in env.read_text().splitlines():                # no hard dependency
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class NodeModel:
    provider: str   # "groq" | "google"
    model: str


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# classify/extract MUST stay on Groq's GPT-OSS models: langchain-groq hard-codes
# the set of models it will send strict=True for, and silently drops strict for
# anything else — turning constrained decoding into prompt-and-pray with no warning.
# Each node has a CHAIN, tried in order. The first entry is the intended model;
# the rest exist because free tiers fail in ways retrying the same model cannot
# fix. A pinned gemini-2.5-flash-lite 404'd mid-evaluation and killed 8 of 20
# cases — Google had retired it for this tier while still returning it from the
# models endpoint, so "listed" and "callable" are not the same thing.
#
# Two defences, because they cover different failures: the Gemini entries use the
# floating "-latest" alias so a version retirement resolves itself, and every
# node falls back across PROVIDERS so an outage or an exhausted daily quota on
# one vendor degrades the run instead of ending it.
#
# The fallbacks are chosen to preserve schema-constrained decoding, not merely to
# return something: Gemini's json_schema mode is genuine constrained decoding,
# which is why it can stand in for Groq on classify and extract.
NODE_CHAINS: dict[str, list[NodeModel]] = {
    "classify": [
        NodeModel("groq",   _env("GROQ_MODEL_CLASSIFY", "openai/gpt-oss-20b")),
        NodeModel("groq",   _env("GROQ_MODEL_CLASSIFY_ALT", "openai/gpt-oss-120b")),
        NodeModel("google", _env("GEMINI_MODEL_FALLBACK", "gemini-flash-lite-latest")),
    ],
    "extract": [
        NodeModel("groq",   _env("GROQ_MODEL_EXTRACT", "openai/gpt-oss-120b")),
        NodeModel("groq",   _env("GROQ_MODEL_EXTRACT_ALT", "openai/gpt-oss-20b")),
        NodeModel("google", _env("GEMINI_MODEL_FALLBACK", "gemini-flash-lite-latest")),
    ],
    "synthesize": [
        NodeModel("google", _env("GEMINI_MODEL_SYNTHESIZE", "gemini-flash-lite-latest")),
        NodeModel("groq",   _env("GROQ_MODEL_SYNTH_ALT", "openai/gpt-oss-120b")),
    ],
    "verify": [
        NodeModel("google", _env("GEMINI_MODEL_VERIFY", "gemini-flash-lite-latest")),
        NodeModel("groq",   _env("GROQ_MODEL_VERIFY_ALT", "openai/gpt-oss-120b")),
    ],
}

# The primary of each chain, kept for callers that only need to name the model.
NODE_MODELS: dict[str, NodeModel] = {k: v[0] for k, v in NODE_CHAINS.items()}

EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(_env("EMBEDDING_DIM", "384"))

# bge-small retrieval is asymmetric: prefix QUERIES only, never passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

RETRIEVAL_K = int(_env("RETRIEVAL_K", "6"))
SIMILARITY_FLOOR = float(_env("SIMILARITY_FLOOR", "0.35"))

# Groq free tier is 30 requests/min per model. 0.4 req/s = 24 RPM leaves headroom.
GROQ_REQUESTS_PER_SECOND = float(_env("GROQ_REQUESTS_PER_SECOND", "0.4"))

DATABASE_URL = os.environ.get("DATABASE_URL", "")
