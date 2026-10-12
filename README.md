# MedGraph

**Agentic clinical document intelligence.** A LangGraph state machine that answers questions
about public healthcare policy and drug-label documents — and refuses, with a stated reason,
when the corpus cannot support an answer.

> **Status: working, with published numbers.** The graph, corpus ingestion, guardrails,
> fabrication guard, token accounting, provider failover and the scored evaluation harness are
> implemented and tested (72 tests), and the system has been evaluated end to end against live
> providers. Results below are written into this file by `scripts/update_readme.py` from the
> committed evaluation artifacts; CI fails if they drift. What is deliberately not built yet is
> listed in the [Roadmap](#roadmap).

Runs entirely on free provider tiers. No paid API key is required.

📄 **[Project overview (PDF)](docs/MedGraph_Overview.pdf)** — architecture diagram, the reasoning
behind each design decision, measured results, and what is deliberately not built yet.

---

## Why this exists

Most RAG demos answer every question. In healthcare that is the failure mode, not the feature:
a confident wrong answer about coverage or dosing is worse than no answer at all.

MedGraph inverts that default. Four properties are enforced structurally rather than left to
prompt wording:

1. **Grounded or refused.** `Answer` fails validation if it carries no citation and is not marked
   a refusal. An ungrounded answer cannot be constructed.
2. **Citations cannot be invented.** `synthesize` returns *source ids*, never `Citation` objects.
   Ids the retriever did not return are dropped before the answer is assembled; if nothing
   survives, the graph refuses.
3. **No PHI leaves the process.** Identifiers are redacted at the entry node, before the first
   provider call.
4. **Not a clinician.** Questions seeking an individual treatment decision route to `refuse`,
   they are not hedged.

## Architecture

```
scrub ─▶ classify ─┬─(out_of_scope | seeks advice)──────────▶ refuse ─▶ END
                   └─▶ extract ─▶ retrieve ─┬─(no hits)─────▶ refuse ─▶ END
                                            └─▶ synthesize ─┬─(no real sources)─▶ refuse
                                                            └─▶ verify ─┬─(ungrounded)─▶ refuse
                                                                        └─▶ END
```

| Node | Responsibility | Model |
|---|---|---|
| `scrub` | Redacts MRN, SSN, NPI, DOB, phone, email before anything reaches a provider | — |
| `classify` | 4-way intent, schema-constrained | Groq `gpt-oss-20b` |
| `extract` | Typed clinical entities; codes only when stated, never invented | Groq `gpt-oss-120b` |
| `retrieve` | Exact cosine search over pgvector; returns `[]` below a similarity floor | — |
| `synthesize` | Answer + ids of sources actually used | Gemini Flash-Lite |
| `verify` | Claim-to-citation entailment over the draft | Gemini Flash-Lite |
| `refuse` | Terminal node carrying a reason, never an empty response | — |

Declining is a first-class path: every edge out of `classify`, `retrieve`, `synthesize` and
`verify` is conditional. State is checkpointed with `MemorySaver`, which is process-local: enough
to inspect and resume a thread within a running process, **not** enough to survive a crash or a
redeploy. Durable resumability needs a persistent checkpointer and is not implemented.

### Why the providers are split

Groq's free tier allows 200,000 tokens/day per model, and `synthesize` alone is ~3,800 of the
~6,200 tokens in a full run. Moving the two token-heavy nodes to Gemini raises daily capacity from
~52 pipeline runs to ~222 — the difference between one evaluation run a day and four.

`classify` and `extract` stay on Groq because `langchain-groq` only sends `strict=True` for its two
GPT-OSS models and **silently** drops it for anything else, turning token-level constrained decoding
into best-effort JSON. `providers.py` raises rather than allowing that downgrade to pass unnoticed.

## Corpus

Public sources only. No PHI, no licensed content.

| Source | Status |
|---|---|
| MedlinePlus health topics (NLM public search service) | ✅ ingested |
| CMS National/Local Coverage Determinations | ⏳ not ingested — see below |
| FDA structured product labels (DailyMed) | ⏳ not ingested |

```bash
python scripts/ingest.py --dry-run     # fetch + chunk only; no model, no database
python scripts/ingest.py               # full run: fetch -> chunk -> embed -> load
python scripts/ingest.py --resume      # continue an interrupted embed
```

A current run yields **128 documents → 339 chunks**, about 106,000 tokens. Chunking is done
against the embedding model's own tokenizer, and the maximum chunk is 448 tokens — comfortably
inside the 512-token window.

**Coverage questions currently refuse, and that is correct.** The Medicare Coverage Database API
is per-document lookup rather than a bulk feed, so CMS text is not in the corpus yet. Rather than
hide that, the evaluation set asserts it: `refuse-corpus-001` and `refuse-corpus-002` are coverage
questions that *must* be declined. When CMS is ingested, those expectations flip — and the harness
will show whether retrieval actually improved.

Chunking is paragraph-first, then sentence-packed to ≤350 words with 50-word overlap. Two
invariants are tested: no chunk exceeds the embedding window (bge-small truncates *silently*, so an
oversized chunk loses its tail with no error), and **no input text is ever discarded** — fragments
merge forward or fold backward into a neighbour.

> The `scrub` node redacts identifiers before any provider call, but **regex redaction is not a
> compliance control**. This is a portfolio system, not a HIPAA-covered one. Do not enter real
> patient information.

## Running it

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env          # add free GROQ_API_KEY and GOOGLE_API_KEY
```

```bash
pytest                        # 69 tests, no API key or database required
```

```bash
python scripts/init_db.py        # create the schema (no psql needed)
python scripts/ingest.py --load data/vectors.jsonl
uvicorn medgraph.api:app --reload
```

`.env` is loaded automatically by `medgraph.config`, which every entry point imports. Real
environment variables always take precedence, so an export or a CI secret is never overridden by a
stale local file.

<details>
<summary>If your DNS refuses to resolve the database host</summary>

Some home routers and ISP resolvers refuse deep subdomains and return `REFUSED`, which surfaces as
`failed to resolve host`. Confirm it is local rather than real:

```bash
nslookup <your-db-host> 1.1.1.1
```

If a public resolver answers, the name is fine and your resolver is the problem. Either point the
machine at a public resolver, or pass the address through libpq, which keeps the hostname for TLS
and certificate verification:

```bash
PGHOSTADDR=<resolved-ip> python scripts/init_db.py
```
</details>

The test suite injects fakes for the retriever, embedder and provider calls, so every path through
the graph — including all five refusal paths — runs offline.

## Stack

Python 3.12 · LangGraph · LangChain · Pydantic v2 · FastAPI · PostgreSQL + pgvector · Docker

**Embeddings run locally** (`bge-small-en-v1.5`, 384-dim). An earlier version of this README
justified that by claiming re-embedding costs "millions of tokens" — it does not, and the number is
worth correcting: the whole corpus is about 104,000 tokens, a rounding error on any free tier. The
honest reasons are that the `retrieve` node gains no network dependency and no per-query round
trip, embedding quota never contends with the LLM nodes' quota, and re-chunking is free and
unmetered, which matters because chunk tuning is iterative. Install with
`pip install -r requirements-embeddings.txt`.

`vector(384)` in `scripts/init.sql` must match `EMBEDDING_DIM`. There is deliberately no ANN index:
at ~5k chunks an exact scan is single-digit milliseconds at 100% recall, while IVFFlat at that size
would scan ~1% of the corpus and present as broken embeddings.

## Evaluation

```bash
python eval/run_eval.py --fake    # exercise the harness; no keys needed
python eval/run_eval.py           # real run; needs free keys + a database
```

Four metrics over a 20-case labelled set spanning guideline, drug-info, must-refuse,
not-in-corpus and PHI-redaction cases:

| Metric | What it catches |
|---|---|
| retrieval precision@k / recall@k | chunking and similarity-floor regressions |
| grounding | citations the retriever never returned |
| **missed refusals** | answering when the corpus cannot support it |
| over-refusals | declining legitimate questions |
| redaction | identifiers reaching a provider |

Missed and over-refusals are reported **separately, never averaged**. They are not equally bad: a
missed refusal launders an unsupported claim as verified, while an over-refusal is merely
unhelpful. Averaging them into one "accuracy" hides the failure that matters.

The set earns its keep. It has already caught two real defects: a `"prescribe"` trigger that
refused *"What are anticoagulants prescribed for?"*, and a chunker that silently dropped any
document whose paragraphs were all short. Both are now regression tests.

## Measured baseline

Retrieval, measured on the real corpus. No provider and no API key is involved — embeddings run
locally and the search is arithmetic — so these numbers are reproducible offline with
`python eval/retrieval_eval.py`.

<!-- EVAL:BEGIN -->
<!-- Written by scripts/update_readme.py from the files in eval/results/.
     Do not edit by hand: CI runs `--check` and fails when this block is
     stale or hand-edited. -->

| | |
|---|---|
| corpus | 128 documents → **339 chunks** |
| largest chunk | 448 tokens (model window 512) |
| queries | 13 |
| k / similarity floor | 6 / 0.35 |
| **recall@k** | **1.000** |
| precision@k | 0.487 |

**How much does the similarity floor actually exclude?** Published because it is unflattering: the floor is close to inert for anything health-adjacent, so the refusals on out-of-corpus questions are coming from `classify` and `verify`, not from the floor.

| probe | top score | chunks above the floor |
|---|---|---|
| in-corpus | 0.857 | 339 of 339 |
| adjacent | 0.641 | 339 of 339 |
| off-topic | 0.479 | 326 of 339 |
| unrelated | 0.419 | 60 of 339 |
<!-- EVAL:END -->

**On precision.** It is measured chunk-level against document-level expectations, so it is bounded
by how many chunks a document has: retrieving six chunks when the expected document holds two caps
precision at 0.33 even under perfect retrieval. Recall governs whether an answer can be grounded at
all, which is why it leads.

**A caveat on recall@k 1.0.** The evaluated queries were written by the author, and several
essentially restate a target document's title — against a corpus seeded from the same topic list.
1.0 here means "no regression signal yet", not "retrieval is solved". The set needs adversarial
queries and near-duplicate distractors before the number carries real weight.

## End-to-end results

<!-- LIVE:BEGIN -->
<!-- Written by scripts/update_readme.py from the files in eval/results/.
     Do not edit by hand: CI runs `--check` and fails when this block is
     stale or hand-edited. -->

Measured against live providers over the 20-case golden set, **3 runs**:

| Metric | min | mean | max |
|---|---|---|---|
| retrieval precision@k | 0.474 | 0.483 | 0.500 |
| retrieval recall@k | 1.000 | 1.000 | 1.000 |
| grounding — no fabricated citations | 1.000 | 1.000 | 1.000 |
| refusal accuracy | 0.950 | 0.967 | 1.000 |
| **missed refusals** | **0.000** | **0.000** | **0.000** |
| over-refusals | 0.000 | 0.051 | 0.077 |

Reported as a range because `verify` is LLM-judged and not deterministic even at temperature 0. Across these runs 1 case(s) returned a different refuse/answer verdict on identical input: `drug-004`. `python eval/run_eval.py --repeat N` reports the spread and names every case that flipped; a single run of an LLM-judged metric is a sample, not a measurement.
<!-- LIVE:END -->

**Missed refusals are zero in every run** — the system has not once answered a question its corpus
could not support. That is the property the design exists to guarantee, and
`python scripts/update_readme.py --gate` fails the build if a published run ever breaks it.

### What the first live run found

The four model-backed nodes had never executed against a real provider before that run. Running
them surfaced five defects, four of them in this repository:

| Defect | Cause |
|---|---|
| Every answerable question refused | `CLASSIFY` defined out-of-scope as "general knowledge", so "What is diabetes?" was correctly routed to refuse. The taxonomy described a CMS/FDA corpus that is not ingested; the ingested corpus *is* general health information. |
| 8 of 20 cases errored | `gemini-2.5-flash-lite` returned 404 — retired for this tier while still listed by the models endpoint. Now behind a floating alias and a provider failover chain. |
| A safety guardrail failed open | The individual-advice check sat on the edge *out of* `classify`, so it ran after an LLM call — and a provider returning invalid JSON on a medical-advice question raised instead of refusing. Moved into `scrub`: that question now reaches no provider at all. |
| A case crashed on verbose output | `reasoning` carried `max_length=800`; a 3,000-character deliberation failed validation and took the node down. A prose-length cap is tidiness, not a constraint worth an outage. |
| Well-supported answers refused | `verify` flagged paraphrase — "and having a healthier lifestyle" — as an unsupported claim. It now judges *material* claims, and over-refusals fell from 0.154 to 0.051 with missed refusals unchanged at zero. |

## Free-tier budgets

Groq allows 200,000 tokens/day per model; `medgraph.budget` reads today's usage from the `runs`
table so a run can refuse to start rather than die half-finished. One row is written per LLM call,
not per graph run, because the quota is per *model* and one run spans three of them.

## Roadmap

**Done**

- [x] Graph topology with conditional refusal paths
- [x] Typed boundaries — `Classification`, `Extraction`, `Synthesis`, `VerificationResult`
- [x] PHI scrubbing, and an individual-advice boundary that refuses before any provider call
- [x] Schema-constrained decoding, with a guard that raises rather than downgrade silently
- [x] Fabrication guard on citations
- [x] Corpus ingestion and chunking against the embedding model's own tokenizer
- [x] Scored evaluation harness — retrieval, grounding, refusal accuracy, redaction
- [x] Measured retrieval baseline, reproducible offline with no API key
- [x] End-to-end evaluation against live providers, published above
- [x] Per-node provider failover across two vendors
- [x] `--repeat N` reporting per-metric spread and naming cases that flip verdict
- [x] README results generated from the eval artifacts, with a CI drift check and a
      refusal gate that fails the build

**Not done, and why**

- [ ] **Per-model daily budget enforcement.** `budget.py` and `persistence.py` are written
      and the arithmetic is unit tested, but **nothing calls them** — no run has ever
      written a row to `runs`. The daily ceiling is therefore documented, not enforced.
- [ ] **CMS coverage determinations and DailyMed labels.** The Medicare Coverage Database
      API is per-document lookup rather than a bulk feed. Coverage questions consequently
      refuse, and the evaluation set asserts that they must until the corpus exists.
- [ ] **Adversarial queries and near-duplicate distractors in the golden set.** The current
      queries were author-written against a corpus seeded from the same topic list, which is
      why recall@k 1.0 is reported as a plumbing check rather than a benchmark.
- [ ] **Durable checkpointing.** `MemorySaver` is process-local, so a run survives a provider
      timeout but not a crash or a redeploy.
- [ ] **Citation recall.** Fabricated citations are caught; whether an answer cited
      *everything* it should have is not yet measured.
- [ ] No real users. Latency and failure behaviour under concurrent load are unknown.

## License

MIT
