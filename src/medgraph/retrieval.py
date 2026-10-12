"""pgvector-backed retrieval over the public guideline corpus.

Corpus source (public, no PHI): MedlinePlus health topics.

CMS coverage determinations and FDA drug labels are NOT ingested — see corpus.py.
Coverage questions therefore find nothing and the graph refuses, which the
evaluation set asserts.
"""
from __future__ import annotations

from .config import DATABASE_URL, RETRIEVAL_K, SIMILARITY_FLOOR
from .schemas import Citation


class Retriever:
    """Exact cosine search. No ANN index by design.

    At corpus scale (~5k chunks x 384 dims) an exact scan is single-digit
    milliseconds with 100% recall. An IVFFlat index here would actively hurt:
    pgvector's guidance is lists ~= rows/1000, so `lists = 100` over 5k rows puts
    ~40 rows per list and the default `ivfflat.probes = 1` scans ~1% of the
    corpus — poor recall that presents as "the embeddings are broken".
    Revisit with HNSW past ~50k chunks.
    """

    def __init__(self, dsn: str | None = None, k: int = RETRIEVAL_K) -> None:
        self.dsn = dsn or DATABASE_URL
        self.k = k

    def search(self, embedding: list[float]) -> list[Citation]:
        """Return citations above the similarity floor, best first.

        An empty result is a valid outcome that drives the graph to refuse rather
        than answer from nothing.
        """
        import psycopg
        from pgvector import Vector
        from pgvector.psycopg import register_vector

        sql = """
            SELECT source_id, title, url, chunk,
                   1 - (embedding <=> %(q)s) AS score
            FROM documents
            WHERE 1 - (embedding <=> %(q)s) > %(floor)s
            ORDER BY embedding <=> %(q)s
            LIMIT %(k)s
        """
        if not self.dsn:
            raise RuntimeError("DATABASE_URL is not set")

        with psycopg.connect(self.dsn) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                # Vector(), not a bare list. register_vector only registers
                # dumpers for Vector and numpy.ndarray; a plain list falls through
                # to psycopg's array adapter and is sent as a Postgres array
                # literal, which the vector operators reject.
                cur.execute(sql, {"q": Vector(embedding),
                                  "floor": SIMILARITY_FLOOR, "k": self.k})
                rows = cur.fetchall()

        return [
            Citation(
                source_id=r[0],
                title=r[1],
                url=r[2],
                snippet=r[3][:1200],
                score=max(0.0, min(1.0, float(r[4]))),
            )
            for r in rows
        ]
