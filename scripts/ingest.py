#!/usr/bin/env python
"""Corpus ingestion: fetch -> chunk -> embed -> load.

Run as a standalone script, never inside the API process: the embedding model
holds ~600MB resident and the full corpus pass takes tens of minutes on a CPU.

Progress is checkpointed to disk so an interrupted run resumes instead of
repeating work — on a memory-constrained machine a swap-death partway through a
45-minute embed is a realistic outcome, not a hypothetical one.

    python scripts/ingest.py --dry-run          # fetch + chunk only, no model, no DB
    python scripts/ingest.py --limit 20         # small real run
    python scripts/ingest.py                    # full corpus
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import medgraph.config  # noqa: E402,F401  — loads .env for every entry point
from medgraph.chunking import chunk_document                      # noqa: E402
from medgraph.corpus import DEFAULT_TERMS, SOURCES, Document      # noqa: E402

CHECKPOINT = Path("data/ingest_checkpoint.jsonl")
BATCH = 16


def fetch(source_name: str, terms: list[str], per_term: int) -> list[Document]:
    source = SOURCES[source_name]()
    print(f"fetching from {source_name}: {len(terms)} terms", flush=True)
    docs = source.fetch(terms, per_term=per_term)
    print(f"  {len(docs)} unique documents", flush=True)
    return docs


def token_counter():
    """Return the embedding model's own tokenizer, or None if unavailable.

    Chunking against the real tokenizer rather than a words-per-token rule of
    thumb: the rule of thumb under-counts token-dense clinical text badly enough
    to push chunks past the window, where they are truncated silently.
    """
    try:
        from transformers import AutoTokenizer

        from medgraph.config import EMBEDDING_MODEL
        tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        return lambda text: len(tok.encode(text, add_special_tokens=True))
    except Exception as exc:                                   # noqa: BLE001
        print(f"  warning: tokenizer unavailable ({type(exc).__name__}); "
              f"falling back to the word ceiling", flush=True)
        return None


def to_chunks(docs: list[Document]) -> list[dict]:
    counter = token_counter()
    chunks: list[dict] = []
    for doc in docs:
        for c in chunk_document(doc.source_id, doc.title, doc.text, doc.url,
                                count_tokens=counter):
            chunks.append({
                "source_id": f"{c.source_id}#{c.ordinal}",
                "title": c.title,
                "url": c.url,
                "chunk": c.text,
            })
    return chunks


def load(rows: list[dict], dsn: str, dim: int) -> int:
    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector

    sql = """
        INSERT INTO documents (source_id, title, url, chunk, embedding)
        VALUES (%(source_id)s, %(title)s, %(url)s, %(chunk)s, %(embedding)s)
    """
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for row in rows:
                if len(row["embedding"]) != dim:
                    raise ValueError(
                        f"{row['source_id']}: embedding has {len(row['embedding'])} "
                        f"dims, schema expects {dim}"
                    )
                # See retrieval.py: a bare list is adapted as a Postgres array.
                cur.execute(sql, {**row, "embedding": Vector(row["embedding"])})
        conn.commit()
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="medlineplus", choices=sorted(SOURCES))
    ap.add_argument("--per-term", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="cap chunks (for a smoke run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and chunk only — no embedding model, no database")
    ap.add_argument("--resume", action="store_true", help="reuse the checkpoint file")
    ap.add_argument("--out", default=None, metavar="PATH",
                    help="write embedded chunks to JSONL instead of a database. "
                         "Lets the corpus be embedded on one machine (or a free GPU "
                         "notebook) and loaded elsewhere.")
    ap.add_argument("--load", default=None, metavar="PATH",
                    help="load a JSONL produced by --out into the database")
    args = ap.parse_args()

    if args.load:
        dsn = os.environ.get("DATABASE_URL")
        if not dsn:
            print("DATABASE_URL is not set", file=sys.stderr)
            return 1
        from medgraph.config import EMBEDDING_DIM
        rows = [json.loads(l) for l in Path(args.load).read_text().splitlines() if l.strip()]
        print(f"loaded {load(rows, dsn, EMBEDDING_DIM)} chunks from {args.load}")
        return 0

    docs = fetch(args.source, DEFAULT_TERMS, args.per_term)
    rows = to_chunks(docs)
    if args.limit:
        rows = rows[: args.limit]

    words = sum(len(r["chunk"].split()) for r in rows)
    print(f"chunked into {len(rows)} chunks (~{words:,} words)", flush=True)

    if args.dry_run:
        print("dry run — stopping before embedding")
        for r in rows[:3]:
            print(f"  {r['source_id']:<46} {len(r['chunk'].split()):>4}w  {r['chunk'][:70]}...")
        return 0

    dsn = os.environ.get("DATABASE_URL")
    if not dsn and not args.out:
        print("DATABASE_URL is not set (or pass --out to write vectors to a file)",
              file=sys.stderr)
        return 1

    from medgraph.config import EMBEDDING_DIM
    from medgraph.embeddings import LocalEmbeddings

    done: set[str] = set()
    if args.resume and CHECKPOINT.exists():
        done = {json.loads(line)["source_id"] for line in CHECKPOINT.open()}
        print(f"resuming: {len(done)} chunks already embedded", flush=True)

    pending = [r for r in rows if r["source_id"] not in done]
    embedder = LocalEmbeddings()
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    embedded: list[dict] = []
    with CHECKPOINT.open("a") as ckpt:
        for i in range(0, len(pending), BATCH):
            batch = pending[i:i + BATCH]
            vectors = embedder.embed_documents([b["chunk"] for b in batch])
            for row, vector in zip(batch, vectors):
                record = {**row, "embedding": vector}
                embedded.append(record)
                ckpt.write(json.dumps({"source_id": row["source_id"]}) + "\n")
            ckpt.flush()
            print(f"  embedded {min(i + BATCH, len(pending))}/{len(pending)}",
                  end="\r", flush=True)

    print()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in embedded:
                fh.write(json.dumps(row) + "\n")
        size_mb = out.stat().st_size / 1_048_576
        print(f"wrote {len(embedded)} embedded chunks to {out} ({size_mb:.1f} MB)")
        print(f"load them later with:  python scripts/ingest.py --load {out}")
        return 0

    inserted = load(embedded, dsn, EMBEDDING_DIM)
    print(f"loaded {inserted} chunks into documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
