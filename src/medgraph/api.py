"""FastAPI surface. One endpoint, typed in and out."""
from __future__ import annotations

import uuid
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .schemas import Answer

app = FastAPI(title="MedGraph", version="0.1.0")


@lru_cache(maxsize=1)
def _graph():
    """Built on first request, not at import, so the module imports without
    a database, an API key, or the ~1GB embedding stack present."""
    from .graph import build_graph

    return build_graph()


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    run_id = str(uuid.uuid4())
    result = _graph().invoke(
        {"query": req.query, "run_id": run_id},
        config={"configurable": {"thread_id": run_id}},
    )
    return result["answer"]
