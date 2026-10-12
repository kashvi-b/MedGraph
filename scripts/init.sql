CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source_id   TEXT NOT NULL,          -- e.g. CMS-NCD-290.1, DAILYMED-xxxx
    title       TEXT NOT NULL,
    url         TEXT,
    chunk       TEXT NOT NULL,
    embedding   vector(384) NOT NULL,   -- bge-small-en-v1.5; must match EMBEDDING_DIM
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- No ANN index by design. At ~5k chunks x 384 dims an exact cosine scan is
-- single-digit milliseconds with 100% recall, while IVFFlat at this size would
-- hurt: pgvector's guidance is lists ~= rows/1000, so lists=100 over 5k rows
-- gives ~40 rows per list and the default ivfflat.probes=1 scans ~1% of the
-- corpus. Add HNSW past ~50k chunks:
--   CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source_id);

CREATE TABLE IF NOT EXISTS runs (
    id            BIGSERIAL PRIMARY KEY,
    run_id        TEXT NOT NULL UNIQUE,   -- "<uuid>:<call index>": one row per LLM call
    query         TEXT NOT NULL,
    nodes_visited TEXT[] NOT NULL DEFAULT '{}',
    model         TEXT NOT NULL,          -- daily free-tier budgets are per MODEL
    refused       BOOLEAN NOT NULL DEFAULT false,
    input_tokens  INT NOT NULL DEFAULT 0,
    output_tokens INT NOT NULL DEFAULT 0,
    cost_usd      NUMERIC(10,6) NOT NULL DEFAULT 0,
    latency_ms    INT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Supports the per-model daily budget query in budget.py.
--
-- Deliberately a plain btree on (model, created_at) rather than an expression
-- index on (created_at::date): casting timestamptz to date depends on the
-- session TimeZone, which makes it STABLE rather than IMMUTABLE, and Postgres
-- rejects non-immutable expressions in an index. The budget query uses a range
-- predicate instead, which this index serves directly.
CREATE INDEX IF NOT EXISTS runs_model_day_idx
    ON runs (model, created_at);
