import json
import uuid
from collections import OrderedDict

from fastapi import APIRouter, Depends, Header, HTTPException
from sse_starlette.sse import EventSourceResponse

from models.schemas import CompareRequest, QueryRequest, QueryResponse
from services.retriever import deduplicate_sources, retrieve, retrieve_multi_query
from services.generator import generate, generate_compare_stream, generate_stream

router = APIRouter(prefix="/query", tags=["query"])

NO_RELEVANT_MSG = "No relevant documents found in the knowledge base."

# In-memory session store for conversation history (OrderedDict for LRU eviction)
_sessions: OrderedDict[str, list[dict]] = OrderedDict()
MAX_SESSIONS = 100
MAX_HISTORY_MESSAGES = 20


def _require_device_id(x_device_id: str = Header(..., alias="X-Device-ID")) -> str:
    if not x_device_id or not x_device_id.strip():
        raise HTTPException(status_code=400, detail="X-Device-ID header is required")
    return x_device_id.strip()


def _validate_question(question: str) -> None:
    if not question or not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a question before searching the knowledge base.",
        )


@router.post("/", response_model=QueryResponse)
async def query_documents(req: QueryRequest, x_device_id: str = Depends(_require_device_id)):
    _validate_question(req.question)
    session_id = req.session_id or str(uuid.uuid4())
    chat_history = _sessions.get(session_id, [])

    citations, has_relevant = await retrieve_multi_query(
        req.question, n_results=req.n_results, device_id=x_device_id,
        metadata_filter=req.validated_metadata_filter(),
        doc_ids=req.doc_ids,
    )

    if not has_relevant or not citations:
        return QueryResponse(answer=NO_RELEVANT_MSG, citations=[], sources=[], session_id=session_id)

    answer = await generate(req.question, citations, has_relevant, chat_history)

    chat_history.append({"role": "user", "content": req.question})
    chat_history.append({"role": "assistant", "content": answer})
    _sessions[session_id] = chat_history[-MAX_HISTORY_MESSAGES:]
    _sessions.move_to_end(session_id)

    if len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)

    sources = deduplicate_sources(citations)
    return QueryResponse(answer=answer, citations=citations, sources=sources, session_id=session_id)


@router.post("/stream")
async def query_documents_stream(req: QueryRequest, x_device_id: str = Depends(_require_device_id)):
    _validate_question(req.question)
    session_id = req.session_id or str(uuid.uuid4())
    chat_history = _sessions.get(session_id, [])

    citations, has_relevant = await retrieve_multi_query(
        req.question, n_results=req.n_results, device_id=x_device_id,
        metadata_filter=req.validated_metadata_filter(),
        doc_ids=req.doc_ids,
    )

    if not has_relevant or not citations:
        async def no_results_stream():
            yield {
                "event": "citations",
                "data": json.dumps({
                    "citations": [],
                    "sources": [],
                    "has_relevant": False,
                    "session_id": session_id,
                }),
            }
            yield {"event": "token", "data": NO_RELEVANT_MSG}
            yield {"event": "done", "data": json.dumps({"session_id": session_id})}
        return EventSourceResponse(no_results_stream())

    sources = deduplicate_sources(citations)

    async def event_stream():
        yield {
            "event": "citations",
            "data": json.dumps({
                "citations": [c.model_dump() for c in citations],
                "sources": [s.model_dump() for s in sources],
                "has_relevant": has_relevant,
                "session_id": session_id,
            }),
        }

        full_answer = ""
        async for token in generate_stream(req.question, citations, has_relevant, chat_history):
            full_answer += token
            yield {"event": "token", "data": token}

        chat_history.append({"role": "user", "content": req.question})
        chat_history.append({"role": "assistant", "content": full_answer})
        _sessions[session_id] = chat_history
        _sessions.move_to_end(session_id)

        if len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)

        yield {"event": "done", "data": json.dumps({"session_id": session_id})}

    return EventSourceResponse(event_stream())


@router.post("/vector-only", response_model=QueryResponse)
async def query_vector_only(req: QueryRequest, x_device_id: str = Depends(_require_device_id)):
    """Vector-only retrieval (no BM25, no multi-query) for evaluation baseline."""
    _validate_question(req.question)
    session_id = req.session_id or str(uuid.uuid4())

    from services.embedder import embed_query
    from db.chroma_client import query_chunks

    query_embedding = embed_query(req.question)
    results = query_chunks(
        query_embedding, n_results=req.n_results, device_id=x_device_id,
        metadata_filter=req.validated_metadata_filter(),
    )

    if not results["documents"] or not results["documents"][0]:
        return QueryResponse(answer=NO_RELEVANT_MSG, citations=[], sources=[], session_id=session_id)

    from models.schemas import SourceCitation
    citations = []
    best_distance = 1.0
    for doc_text, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        if dist > 0.80:
            continue
        best_distance = min(best_distance, dist)
        citations.append(SourceCitation(
            document=meta["filename"],
            chunk_text=doc_text,
            score=round(1 - dist, 4),
            page=meta.get("page"),
            section=meta.get("section"),
        ))

    has_relevant = best_distance < 0.85
    if not has_relevant or not citations:
        return QueryResponse(answer=NO_RELEVANT_MSG, citations=[], sources=[], session_id=session_id)

    answer = await generate(req.question, citations, has_relevant)
    sources = deduplicate_sources(citations)
    return QueryResponse(answer=answer, citations=citations, sources=sources, session_id=session_id)


@router.post("/compare/stream")
async def compare_documents_stream(req: CompareRequest, x_device_id: str = Depends(_require_device_id)):
    _validate_question(req.question)

    doc_citations: dict[str, list] = {}
    all_citations = []

    for doc_id in req.doc_ids:
        citations, _ = retrieve(req.question, n_results=req.n_results, doc_ids=[doc_id], device_id=x_device_id)
        if citations:
            doc_citations[citations[0].document] = citations
            all_citations.extend(citations)

    sources = deduplicate_sources(all_citations)

    async def event_stream():
        yield {
            "event": "citations",
            "data": json.dumps({
                "citations": [c.model_dump() for c in all_citations],
                "sources": [s.model_dump() for s in sources],
                "has_relevant": len(doc_citations) > 0,
                "session_id": "",
            }),
        }
        async for token in generate_compare_stream(req.question, doc_citations):
            yield {"event": "token", "data": token}
        yield {"event": "done", "data": json.dumps({"session_id": ""})}

    return EventSourceResponse(event_stream())


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
