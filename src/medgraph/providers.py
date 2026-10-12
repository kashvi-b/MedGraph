"""Provider factory and the failover chain.

Both providers are free tiers. The split across them is not arbitrary: Groq's
free tier caps at 200,000 tokens/day per model, and `synthesize` alone is ~3,800
of the ~6,200 tokens in a full pipeline run. Putting the two token-heavy nodes on
Gemini roughly quadruples daily capacity.

Every node has a chain rather than a single model, because free tiers fail in
ways that retrying the same model cannot fix. This is not hypothetical: a pinned
`gemini-2.5-flash-lite` began returning 404 mid-evaluation — retired for this
tier while still listed by the models endpoint — and took out 8 of 20 cases.

What is and is not retried matters. A model that is gone, rate limited, or
unreachable is a *provider* problem, and the next entry in the chain may well
succeed. Output that fails schema validation is a *prompt or schema* problem: the
next provider will almost certainly fail the same way, and quietly trying another
would hide a real defect behind a slower, more expensive failure.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from .config import GROQ_REQUESTS_PER_SECOND, NODE_CHAINS, NodeModel

log = logging.getLogger(__name__)

# Models langchain-groq will actually send strict=True for. It silently sets
# strict=None for anything outside this set, so routing a node to another Groq
# model is a silent downgrade from constrained decoding to best-effort JSON.
_GROQ_STRICT_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}

# Substrings identifying a failure the NEXT provider might not have. Matched on
# the exception class name and message because the provider SDKs share no
# exception hierarchy, and importing both to catch their types would make this
# module depend on packages a given deployment may not have installed.
_RETRYABLE = (
    "notfound", "not_found", "404",
    "ratelimit", "rate_limit", "429", "quota", "resource_exhausted",
    "unavailable", "503", "502", "500", "internal",
    "timeout", "connection", "apiconnection",
    "permissiondenied", "403",
)


def _is_provider_failure(exc: Exception) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    # A schema failure names the schema, not the transport. Never fail over on it.
    if "validation" in blob or "failed to parse" in blob or "json_validate" in blob:
        return False
    return any(token in blob for token in _RETRYABLE)


@lru_cache(maxsize=None)
def _rate_limiter():
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=GROQ_REQUESTS_PER_SECOND,
        check_every_n_seconds=0.1,
        max_bucket_size=4,
    )


@lru_cache(maxsize=None)
def _build(provider: str, model: str) -> Any:
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=0,
                        rate_limiter=_rate_limiter(), max_retries=5)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=5)

    raise ValueError(f"unknown provider {provider!r}")


def chain_for(node: str) -> list[NodeModel]:
    try:
        return NODE_CHAINS[node]
    except KeyError:
        raise ValueError(
            f"unknown node {node!r}; expected one of {sorted(NODE_CHAINS)}"
        ) from None


def get_llm(node: str) -> Any:
    """The primary model for a node. Cached so rate limiting is shared."""
    spec = chain_for(node)[0]
    return _build(spec.provider, spec.model)


def _bind(spec: NodeModel, node: str, schema: type) -> Any:
    """Bind a schema using genuine constrained decoding.

    `method="json_schema"` is NOT LangChain's default — the default is
    `function_calling`, which carries no schema guarantee. Omitting it is a silent
    downgrade, so it is always passed explicitly.
    """
    llm = _build(spec.provider, spec.model)

    if spec.provider == "groq":
        if spec.model not in _GROQ_STRICT_MODELS:
            raise ValueError(
                f"node {node!r} is routed to {spec.model!r}, which langchain-groq will "
                f"not send strict=True for. Constrained decoding would be silently "
                f"dropped. Use one of {sorted(_GROQ_STRICT_MODELS)}."
            )
        return llm.with_structured_output(
            schema, method="json_schema", strict=True, include_raw=True
        )

    return llm.with_structured_output(schema, method="json_schema", include_raw=True)


def structured(node: str, schema: type) -> Any:
    """Bind the node's PRIMARY model. For callers that do not want failover."""
    return _bind(chain_for(node)[0], node, schema)


def invoke(node: str, schema: type, messages: list) -> tuple[Any, "LLMCall"]:
    """Call the node's chain until a provider answers; return output and usage.

    `include_raw=True` is what makes token accounting possible — without it
    `with_structured_output` returns only the parsed object and the usage metadata
    is discarded, leaving no way to enforce a daily free-tier budget.
    """
    from .schemas import LLMCall

    chain = chain_for(node)
    last: Exception | None = None

    for attempt, spec in enumerate(chain):
        try:
            runnable = _bind(spec, node, schema).with_config(
                {"run_name": f"medgraph.{node}"}
            )
            started = time.monotonic()
            result = runnable.invoke(messages)
            elapsed_ms = int((time.monotonic() - started) * 1000)
        except Exception as exc:                                   # noqa: BLE001
            last = exc
            if attempt + 1 < len(chain) and _is_provider_failure(exc):
                nxt = chain[attempt + 1]
                log.warning(
                    "node %s: %s/%s failed (%s); falling back to %s/%s",
                    node, spec.provider, spec.model, type(exc).__name__,
                    nxt.provider, nxt.model,
                )
                continue
            raise

        parsed = result["parsed"] if isinstance(result, dict) else result
        raw = result.get("raw") if isinstance(result, dict) else None

        if isinstance(result, dict) and result.get("parsing_error"):
            raise ValueError(
                f"node {node!r} returned output that failed schema validation: "
                f"{result['parsing_error']}"
            )
        if parsed is None:
            raise ValueError(f"node {node!r} returned no parsed output")

        usage = getattr(raw, "usage_metadata", None) or {}
        call = LLMCall(
            node=node,
            provider=spec.provider,
            model=spec.model,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            latency_ms=elapsed_ms,
            fell_back=attempt > 0,
        )
        return parsed, call

    raise last if last else RuntimeError(f"node {node!r} produced no result")
