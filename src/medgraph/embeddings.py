"""Local embeddings — no API key, no quota, no network hop.

Chosen over a hosted embedding API for a reason that is not cost: chunking and
the similarity floor get retuned several times during development, and each pass
re-embeds the whole corpus. That is millions of tokens against a free quota you
cannot see, spent exactly when iteration matters most. Local has no cap.

bge-small-en-v1.5: 384 dimensions, 512-token window. Chosen over the ~2x faster
all-MiniLM-L6-v2 because MiniLM truncates at 256 word pieces *silently* — and
clinical policy chunks run 400-500 tokens, so half of each would vanish with no
error raised, in a system whose whole premise is citation-grounded answers.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from .config import EMBEDDING_MODEL, QUERY_INSTRUCTION


@lru_cache(maxsize=None)
def _model(name: str):
    # Imported lazily: torch + sentence-transformers is ~1GB and is not needed to
    # import the graph, run the schema tests, or start the API in a read-only mode.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class LocalEmbeddings(Embeddings):
    """sentence-transformers behind LangChain's Embeddings interface.

    Written directly against sentence-transformers rather than using
    langchain-huggingface, whose current release requires sentence-transformers
    5.x -> transformers 5.x -> torch >= 2.5, none of which have Intel-Mac wheels.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = _model(self.model_name).encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        out = vectors.tolist()
        # A batched encoder returning one fused vector instead of N is a classic
        # silent failure; retrieval then looks broken for unrelated reasons.
        if len(out) != len(texts):
            raise RuntimeError(
                f"embedding count mismatch: {len(texts)} texts produced {len(out)} vectors"
            )
        return out

    def embed_query(self, text: str) -> list[float]:
        vector = _model(self.model_name).encode(
            [QUERY_INSTRUCTION + text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()
