# DocRAG — Multi-Document RAG Pipeline

A full-stack Retrieval-Augmented Generation (RAG) system that lets you upload multiple documents and ask natural language questions answered by synthesizing information across all of them — with source citations and streaming responses.

Built as a portfolio project to demonstrate end-to-end RAG architecture with a production-quality frontend.

## Features

- **Multi-format ingestion** — PDF, DOCX, HTML, TXT, Markdown, CSV, and code files
- **Semantic chunking** — Splits documents at topic boundaries using embedding cosine similarity (not arbitrary character counts)
- **Hybrid retrieval** — BM25 keyword search + vector similarity merged via Reciprocal Rank Fusion for better precision and recall
- **Multi-query** — Auto-generates 2 query variants per question, retrieves for each, deduplicates results for broader coverage
- **Cross-document retrieval** — Queries all documents simultaneously, surfaces top-K chunks
- **Source citations** — Every answer links back to the exact document excerpt it came from
- **Streaming** — Both document ingestion (step-by-step progress) and query responses stream via SSE
- **Conversation memory** — Follow-up questions use session chat history
- **Compare mode** — Select 2+ documents and get a structured per-doc comparison answer
- **Honest refusal** — When no relevant content is found, says so and shows the closest low-confidence chunks

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python 3.12+) |
| Frontend | Next.js 15 (App Router) + Tailwind CSS |
| Vector DB | ChromaDB (local, persistent) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (local, no API) |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Keyword search | BM25 via `rank-bm25` |
| Document parsing | PyPDF2, python-docx, BeautifulSoup4 |

## Architecture

```
Upload → Parse → Semantic Chunk → Embed (local) → Store in ChromaDB

Query  → Generate 2 query variants (Groq)
       → Hybrid retrieve per variant: BM25 + vector similarity (RRF merge)
       → Deduplicate & merge top-5 chunks
       → Stream answer (Groq llama-3.3-70b-versatile) with [Source] citations
```

Documents are embedded locally with sentence-transformers (no API cost). Queries combine BM25 keyword matching with ChromaDB vector similarity via Reciprocal Rank Fusion, run across multiple query variants for better recall. Groq's fast inference generates the answer with streaming.

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
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# Create .env from template
cp ../.env.example .env
# Edit .env and set GROQ_API_KEY

uvicorn main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`

## Environment Variables

Copy `.env.example` to `backend/.env`:

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Groq API key for LLM generation and query expansion |
| `ALLOWED_ORIGINS` | No | `http://localhost:3000` | Comma-separated list of allowed frontend origins |
| `CHROMA_DATA_PATH` | No | `./chroma_data` | Path for ChromaDB persistent storage |

## Deployment

### Backend — Railway

1. Create a new Railway project and connect this repo, with **root directory** set to `backend/`
2. Add a **Volume** and mount it at `/data` (keeps ChromaDB data across deploys)
3. Set environment variables in Railway dashboard:
   - `GROQ_API_KEY` = your key
   - `CHROMA_DATA_PATH` = `/data/chroma`
   - `ALLOWED_ORIGINS` = `https://your-app.vercel.app`
4. Railway picks up `railway.toml` automatically — deploy.

### Frontend — Vercel

1. Import the repo and set **root directory** to `frontend/`
2. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = `https://your-backend.up.railway.app`
3. Deploy.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/documents/` | List all uploaded documents |
| `POST` | `/documents/upload` | Upload a document (SSE progress stream) |
| `DELETE` | `/documents/{id}` | Delete a document and its embeddings |
| `POST` | `/query/stream` | Query with streaming response (SSE) |
| `POST` | `/query/compare/stream` | Compare a question across specific docs (SSE) |
| `DELETE` | `/query/session/{id}` | Clear conversation session |

## Evaluation

Add Q&A pairs to `eval/eval_dataset.json` (after uploading your documents), then:

```bash
cd eval
python eval.py --api http://localhost:8000 --k 5
```

Reports `hit@k` (was the expected document in the top-k citations?) and `keyword@k` (did expected keywords appear in retrieved chunks?).

## Project Structure

```
multi-doc-rag/
  backend/
    main.py                  # FastAPI app, CORS config
    Procfile                 # For Railway/Heroku deployment
    railway.toml             # Railway-specific deployment config
    routers/
      documents.py           # Upload (SSE progress), list, delete
      query.py               # Query stream, compare stream, session clear
    services/
      parser.py              # Format-aware document parsing
      chunker.py             # Semantic chunking via sentence similarity
      embedder.py            # Local sentence-transformers embedding
      retriever.py           # Hybrid retrieval (BM25 + vector, RRF) + multi-query
      generator.py           # Groq streaming generation, query variant expansion
      pipeline.py            # Orchestrates parse → chunk → embed → store
    models/
      schemas.py             # Pydantic models
    db/
      chroma_client.py       # ChromaDB CRUD operations
  frontend/
    app/
      page.tsx               # Main layout, state, responsive sidebar
      layout.tsx             # Root layout + metadata
    components/
      ChatPanel.tsx          # Message stream, compare mode UI
      DocumentList.tsx       # Doc list with compare-mode checkboxes
      UploadZone.tsx         # Drag-and-drop with SSE progress
      SourceCitations.tsx    # Expandable citation cards
    lib/
      api.ts                 # Typed fetch wrappers + SSE parsers
      types.ts               # Shared TypeScript interfaces
  eval/
    eval.py                  # Retrieval evaluation script (hit@k, keyword@k)
    eval_dataset.json        # Q&A pairs for evaluation
```

## Key Design Decisions

**Hybrid retrieval (BM25 + vector)** — Vector search excels at semantic similarity but misses exact keyword matches. BM25 is the opposite. RRF merges both ranked lists without needing to tune score scales — a chunk scoring well in either list gets boosted.

**Multi-query expansion** — A single question phrasing often misses relevant chunks that would surface under a different wording. Generating 2 variants via Groq, retrieving independently, then deduplicating by highest score gives broader coverage with minimal added latency.

**Semantic chunking over fixed-size** — Sentences are embedded and compared pairwise. Splits happen where cosine similarity drops below a threshold (0.4), keeping topically coherent chunks together. This improves retrieval relevance at the cost of slightly more compute during ingestion.

**Local embeddings** — `all-MiniLM-L6-v2` runs entirely on CPU, producing 384-dimensional vectors. No embedding API calls means no per-token cost and no latency on ingestion.

**Groq for LLM** — Fast inference (often 200+ tokens/sec), free API tier, and `llama-3.3-70b-versatile` follows citation instructions reliably.

**SSE over WebSockets** — SSE is unidirectional (server → client) which matches both use cases (ingestion progress, token streaming). Simpler to implement and debug than WebSockets for this pattern.
