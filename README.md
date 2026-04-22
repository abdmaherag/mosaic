# Mosaic — Documents RAG System

Mosaic lets you upload multiple documents and ask questions that are answered by synthesizing information across all of them — with every source cited.

<img src="Mosaic%20GIF.gif" alt="Mosaic demo" width="800"/>

## Features

- **Multi-format ingestion** — PDF, DOCX, HTML, TXT, Markdown.
- **Semantic chunking** — Splits at topic boundaries using sentence-embedding cosine similarity, not arbitrary character counts
- **Hybrid retrieval** — BM25 keyword search + vector similarity merged via Reciprocal Rank Fusion (RRF)
- **Multi-query expansion** — Auto-generates 2 query variants per question for broader recall
- **Per-document filtering** — Query all documents or select a subset; Compare mode diffs 2–10 docs side-by-side
- **Streaming** — Document ingestion (step-by-step progress) and query responses both stream via SSE
- **Conversation memory** — Follow-up questions use per-session chat history (last 10 turns)
- **Honest refusal** — Returns "no relevant documents" when retrieval confidence is below threshold

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI 0.115 · Python 3.12 |
| Frontend | Next.js 16.1 (App Router) · Tailwind CSS v4 |
| Vector DB | ChromaDB 0.6 (local, persistent, HNSW cosine) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` · 384-dim · local CPU |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Keyword search | `rank-bm25` — BM25Okapi with per-device cache |
| Parsing | PyPDF2 · python-docx · BeautifulSoup4 |

## Architecture

```
Upload  →  Parse  →  Semantic chunk (100–800 chars, 1-sentence overlap)  →  Embed (local)  →  ChromaDB

Query   →  Expand to 3 variants (Groq)
        →  Per variant: vector search (ChromaDB) + BM25
        →  Reciprocal Rank Fusion (k=60) across all results
        →  Threshold filter (distance < 0.80) → top-5 chunks
        →  Stream answer via Groq with source citations (SSE)
```

Documents are embedded locally — no API cost at ingestion time. Queries combine BM25 and vector search, run across 3 query phrasings, and fuse results via RRF before the LLM generates a streamed, cited answer.

## Project Structure

```text
mosaic/
├── backend/
│   ├── main.py                 # FastAPI app, CORS, security headers
│   ├── Dockerfile              # CPU-only PyTorch image, port 7860
│   ├── requirements.txt
│   ├── routers/
│   │   ├── documents.py        # Upload (SSE progress), list, delete
│   │   └── query.py            # Stream, compare, vector-only, session clear
│   ├── services/
│   │   ├── parser.py           # Format-aware dispatch (PDF/DOCX/HTML/text/code)
│   │   ├── chunker.py          # Semantic chunking via sentence similarity
│   │   ├── embedder.py         # SentenceTransformer singleton + batch encode
│   │   ├── retriever.py        # Hybrid BM25 + vector, RRF fusion, multi-query
│   │   ├── generator.py        # Groq streaming, query variant generation
│   │   └── pipeline.py         # Orchestrates parse → chunk → embed → store
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
│   ├── db/
│   │   └── chroma_client.py    # ChromaDB CRUD wrapper
│   └── tests/                  # pytest suite (API, chunker, parser, retriever)
├── frontend/
│   ├── app/
│   │   ├── page.tsx            # Main app — state, sidebar, chat panel
│   │   ├── layout.tsx          # Root layout, fonts, metadata
│   │   └── globals.css         # Tailwind theme + custom design tokens
│   ├── lib/
│   │   ├── api.ts              # Typed fetch wrappers + SSE stream parsers
│   │   └── types.ts            # Shared TypeScript interfaces
│   └── components/             # ChatPanel, DocumentList, UploadZone, Citations, Toast
└── eval/
    ├── eval.py                 # Retrieval eval script (hit@k, keyword@k)
    └── eval_dataset.json       # Q&A pairs for evaluation
```

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- A [Groq API key](https://console.groq.com) (free tier available)

### Backend

```bash
cd backend

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

# Create .env
cp ../.env.example .env
# Set GROQ_API_KEY in .env

uvicorn main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

```bash
npm run dev
```

Open `http://localhost:3000`

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Groq API key (generation + query expansion) |
| `ALLOWED_ORIGINS` | No | `http://localhost:3000,http://localhost:3001` | Comma-separated CORS origins |
| `CHROMA_DATA_PATH` | No | `./chroma_data` | ChromaDB persistence directory |
| `ENVIRONMENT` | No | `dev` | Set to `production` to disable Swagger UI |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/documents/` | List all documents for the device |
| `POST` | `/documents/upload` | Upload a file — streams SSE progress events |
| `DELETE` | `/documents/{id}` | Delete document and all its embeddings |
| `POST` | `/query/stream` | Ask a question — streams citations then tokens |
| `POST` | `/query/compare/stream` | Compare a question across 2–10 specific docs |
| `DELETE` | `/query/session/{id}` | Clear conversation history for a session |

All endpoints require the `X-Device-ID` header for per-user data isolation.

## Evaluation

Add Q&A pairs to `eval/eval_dataset.json` after uploading your documents, then:

```bash
cd eval
python eval.py --api http://localhost:8000 --k 5
```

Reports `hit@k` (was the expected document in the top-k citations?) and `keyword@k` (did expected keywords appear in the retrieved chunks?).

## Key Design Decisions

**Hybrid retrieval (BM25 + vector, RRF)** — Vector search captures semantic similarity but misses exact-match keywords. BM25 is the opposite. Rather than tuning score scales between them, Reciprocal Rank Fusion combines the ranked lists directly: a chunk that ranks highly in either method gets a boosted fused score. RRF_K=60 follows the standard from Cormack et al.

**Multi-query expansion** — A single phrasing often misses relevant chunks that surface under a different wording. The query is sent to Groq to generate 2 alternative phrasings; retrieval runs for all 3 in parallel. Results are deduplicated by chunk, keeping the highest score per chunk. Adds one LLM round-trip (~100ms) in exchange for meaningfully broader recall.

**Semantic chunking over fixed-size** — Each sentence is embedded and consecutive similarity is tracked. Splits happen where similarity drops below 0.65, keeping topically coherent text together in one chunk (100–800 char bounds). Each chunk carries one sentence of overlap with the next to preserve cross-boundary context. This improves retrieval precision compared to naive character-based splitting.

**Local embeddings** — `all-MiniLM-L6-v2` runs on CPU, producing 384-dim vectors in ~10ms per chunk. No embedding API means zero per-ingestion cost and no external dependency for the most latency-sensitive pipeline step.

**Per-device isolation** — Every chunk is stored with a `device_id` metadata field derived from the `X-Device-ID` request header. All queries and deletes are scoped to that ID, giving each browser session its own private document library without authentication overhead.

**SSE over WebSockets** — Both use cases (ingestion progress events, token streaming) are server-to-client only. SSE is a better fit: simpler to implement, works through proxies without upgrade headers, and requires no client-side state machine.
