### MOSAIC — Multi-Document RAG System

**Stack:** FastAPI 0.115 · Python 3.12 · Next.js 16.1 · React 19 · Tailwind v4 · ChromaDB 0.6 (HNSW cosine) · sentence-transformers · Groq (`llama-3.3-70b-versatile`) · rank-bm25 · Docker · HuggingFace Spaces · Netlify

---

#### Core (always keep — headline bullets)

- Engineered and shipped **Mosaic**, a production-grade multi-document RAG system on FastAPI 0.115 / Python 3.12 and Next.js 16.1 / React 19, letting users upload heterogeneous documents (PDF, DOCX, HTML, Markdown, plaintext, CSV) and receive cited, streamed answers synthesized across the full corpus.
- Architected a four-stage ingestion pipeline (**Parse → Semantic-Chunk → Embed → Persist**) streamed to the UI via Server-Sent Events for live per-step progress.
- Designed and ran a three-way retrieval ablation (vector-only, hybrid, hybrid+multi-query) on a 2,781-chunk / ~390K-character ML-textbook corpus: **achieved 100% `hit@1` and `hit@5`**, and surfaced and fixed a silent retrieval-filter bug in the process.

---

#### AI / RAG / Retrieval flavor

- Implemented **semantic chunking** that encodes each sentence with `all-MiniLM-L6-v2` (384-dim, local CPU) and splits where consecutive cosine similarity drops below 0.65, bounded to 100–800 chars with one-sentence overlap to preserve cross-boundary context.
- Built **hybrid retrieval** fusing ChromaDB HNSW cosine vector search with `rank-bm25` BM25Okapi via **Reciprocal Rank Fusion** at the canonical `k=60` (Cormack et al.), fetching 2× the requested top-k from each retriever before fusion.
- Layered **multi-query expansion** via Groq LLM, generating 2 paraphrases per question and running all 3 retrievals concurrently with `asyncio.gather`, deduping by chunk and keeping the max score per chunk.
- Enforced **dual cosine-distance gates against hallucination**: a 0.85 no-answer threshold that forces an honest refusal when the corpus isn't relevant, and a 0.80 per-chunk filter that drops noisy context before generation.
- Used **local CPU embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) to eliminate per-ingest embedding-API cost entirely.
- Wired **streamed LLM generation** via Groq's `AsyncGroq` client (`llama-3.3-70b-versatile`, temp=0.3, max_tokens=2048) with strict formatting prompts and per-session chat memory (LRU of 100 sessions, last 10 messages injected per call).

---

#### Evaluation / Measurement / Debugging flavor

- Built an **eval harness** reporting `hit@k` and `keyword@k` against an annotated Q&A dataset, with a three-way ablation mode (vector-only vs. hybrid vs. hybrid+multi-query) gated by a runtime feature flag.
- **Measured 100% `hit@1` and `hit@5`** across all three retrieval configurations on a 10-question eval against a 2,781-chunk ML textbook corpus with a distractor document.
- Diagnosed and fixed a silent **retrieval-filter bug** exposed by the ablation: BM25-only chunks were being assigned a placeholder cosine distance of 1.0 and immediately discarded by the 0.80 relevance gate — meaning every pure-lexical BM25 match was thrown away before fusion.
- Refactored the filtering logic in the retrieval layer to **apply gates per-source** (cosine threshold for vector hits, zero-score threshold for BM25 hits) and made the no-answer decision use the OR of both signals, restoring real hybrid behavior without weakening the hallucination guard.
- Reported the multi-query ablation honestly: measured **−10% `keyword@1` precision trade** on this benchmark in exchange for broader recall against underspecified queries — kept as a deliberate precision-vs-recall choice rather than hiding a null result.

---

#### Backend / API Design flavor

- Designed a FastAPI backend with **token-streaming SSE endpoints** (`/query/stream`, `/query/compare/stream`, `/documents/upload`) using `sse-starlette`, delivering both progress events and generation tokens over the same protocol.
- Implemented **per-device data isolation** via an `X-Device-ID` header enforced by a FastAPI dependency, propagated into every vector-DB `where` clause and composed with document-ID and user-supplied metadata filters through a safe `$and` builder.
- Validated user-supplied metadata filters against an **allow-list** (`filename`, `page`, `section`) with type-checked values to prevent filter injection.
- Built a **compare mode** that runs independent retrievals across 2–10 selected documents in parallel and generates a structured cross-document diff.
- Managed **in-memory session state** (LRU of 100 chat sessions, 20-message rolling history per session) with O(1) eviction via `OrderedDict`.
- Modeled all request/response payloads with **Pydantic v2** for boundary-only validation (`max_length=2000`, `ge=1`, `le=20`), keeping internal code trust-based and lean.

---

#### Data Engineering / Pipelines flavor

- Built an **async ingestion pipeline** yielding typed `ProcessingEvent` objects at every stage, giving the frontend granular progress UX without polling.
- Added **SHA-256 content hashing** for duplicate-upload detection across filenames and a 50 MB upload cap enforced at the request boundary.
- Pre-cleaned parsed text with a regex pass to **strip `[Page N]` markers into metadata** instead of chunk bodies, eliminating noise in BM25 tokens and embedding inputs.
- Implemented a **per-device BM25 cache** with dirty-bit invalidation on upload/delete, avoiding full re-tokenization of the corpus on every query.
- Refactored the filtering logic within the **database abstraction layer** to compose device-scope, document-scope, and user metadata filters into safe `$and` clauses without SQL/filter injection risk.

---

#### DevOps / Deployment / Infrastructure flavor

- Containerized the backend with a **CPU-only PyTorch Docker image** pinned to PyTorch's `cpu` wheel index-URL, skipping ~2 GB of unnecessary CUDA libraries from the final image.
- Deployed the **backend to HuggingFace Spaces** (Docker runtime, port 7860) and the **frontend to Netlify**, resolving deployment networking issues to maintain cross-origin security and keep both tiers on free infrastructure.
- Made the backend fully **12-factor-style configurable** via env (`GROQ_API_KEY`, `ALLOWED_ORIGINS`, `CHROMA_DATA_PATH`, `ENVIRONMENT`) with development-friendly defaults.
- Explicitly loaded `.env` from the module directory rather than CWD, eliminating a class of "env not picked up by uvicorn" deployment bugs.

---

#### Testing flavor

- Wrote a **pytest suite of ~1,160 lines across 5 files** covering RRF math properties (idempotence, accumulation, boundary cases), metadata-filter composition, distance-threshold gating, FastAPI TestClient integration, and chunker invariants.
- **Mocked heavy dependencies** (sentence-transformers, ChromaDB, Groq, sse-starlette) at import time via `sys.modules` stubs, letting unit tests run in milliseconds without loading models or hitting disk.
- Used a feature-flag-driven runtime switch to enable **A/B ablation testing** in the live service without code changes between eval runs.

---

#### Frontend flavor

- Built the Next.js 16 / React 19 client with a custom **SSE stream parser** handling multi-line `data:` payloads and CRLF normalization, feeding live tokens into a streaming markdown renderer.
- Designed a **per-browser device-ID system** with `crypto.randomUUID()` + `localStorage`, giving every user a private document library with zero authentication overhead.
- Implemented a **thinking-step UI** that mirrors the backend's SSE progress events into a real-time checklist (Decomposing → Retrieving → Ranking → Synthesizing) rather than a generic spinner.

---

#### LLM / Prompt Engineering flavor

- Tuned per-task Groq parameters: `temp=0.3 / max_tokens=2048` for answer generation, `temp=0.5 / max_tokens=120` for the query-variant step — **two separate calls optimized for different objectives** (faithful synthesis vs. divergent paraphrasing).
- Wrote a **strict formatting system prompt** with a worked example that enforces bold section headings, separated bullet lines, and end-of-response citations — eliminating the single-paragraph failure mode common with instruction-tuned models.
- Added a **dedicated no-answer prompt** that the model is given when retrieval confidence falls below the 0.85 gate, explicitly telling it to refuse rather than fabricate.

---

### How to mix flavors for a target JD

| Job type | Pull from |
|---|---|
| **AI / ML / RAG Engineer** | Core + AI/RAG + Eval + LLM |
| **Backend / Platform Engineer** | Core + Backend API + Data Eng |
| **Full-Stack** | Core + Backend API + Frontend + LLM |
| **DevOps / Infra** | Core + DevOps + Data Eng + Testing |
