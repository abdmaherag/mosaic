import asyncio

from rank_bm25 import BM25Okapi

from db.chroma_client import query_chunks, get_all_chunks
from models.schemas import DocumentSource, SourceChunk, SourceCitation
from services.embedder import embed_query

NO_ANSWER_THRESHOLD = 0.85  # ChromaDB cosine distance: lower = more similar
CHUNK_RELEVANCE_THRESHOLD = 0.80  # Per-chunk filter: drop chunks with distance above this

# Reciprocal Rank Fusion constant (standard default from Cormack et al.)
RRF_K = 60

# Per-device BM25 cache — invalidated whenever documents are mutated
_bm25_cache: dict[str, tuple[BM25Okapi, list[dict], dict[str, dict]]] = {}
_bm25_dirty: dict[str, bool] = {}


def invalidate_bm25_cache() -> None:
    _bm25_cache.clear()
    _bm25_dirty.clear()


def _get_bm25(device_id: str = "") -> tuple[BM25Okapi | None, list[dict], dict[str, dict]]:
    if not _bm25_dirty.get(device_id, True) and device_id in _bm25_cache:
        return _bm25_cache[device_id]
    corpus = get_all_chunks(device_id=device_id)
    if not corpus:
        return None, [], {}
    corpus_by_id = {c["id"]: c for c in corpus}
    tokenized = [c["text"].lower().split() for c in corpus]
    index = BM25Okapi(tokenized)
    _bm25_cache[device_id] = (index, corpus, corpus_by_id)
    _bm25_dirty[device_id] = False
    return index, corpus, corpus_by_id


def _reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, float]],
) -> dict[str, float]:
    """Reciprocal Rank Fusion: combine multiple ranked lists using 1/(k + rank).

    Each ranked_list is [(chunk_id, score)] sorted by score descending.
    Higher RRF score = more relevant.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        sorted_list = sorted(ranked, key=lambda x: x[1], reverse=True)
        for rank, (cid, _score) in enumerate(sorted_list, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return fused


def retrieve(
    question: str,
    n_results: int = 5,
    doc_ids: list[str] | None = None,
    device_id: str = "",
    metadata_filter: dict | None = None,
) -> tuple[list[SourceCitation], bool]:
    fetch_n = n_results * 2

    query_embedding = embed_query(question)
    results = query_chunks(
        query_embedding, n_results=fetch_n, doc_ids=doc_ids,
        device_id=device_id, metadata_filter=metadata_filter,
    )

    if not results["documents"] or not results["documents"][0]:
        return [], False

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # Map chunk_id -> data for later lookup
    chunk_map: dict[str, dict] = {}
    vector_scored: list[tuple[str, float]] = []
    for doc_text, meta, dist in zip(docs, metas, dists):
        cid = f"{meta['doc_id']}_chunk_{meta['chunk_index']}"
        chunk_map[cid] = {"text": doc_text, "meta": meta, "distance": dist}
        vector_scored.append((cid, 1 - dist))  # similarity = 1 - distance

    bm25_scored: list[tuple[str, float]] = []
    if not doc_ids:
        bm25_index, bm25_corpus, corpus_by_id = _get_bm25(device_id)
        if bm25_index and bm25_corpus:
            tokenized_q = question.lower().split()
            raw_scores = bm25_index.get_scores(tokenized_q)
            ranked = sorted(
                zip([c["id"] for c in bm25_corpus], raw_scores),
                key=lambda x: x[1],
                reverse=True,
            )[:fetch_n]
            for cid, score in ranked:
                match = corpus_by_id.get(cid)
                if metadata_filter and match:
                    if not all(match["meta"].get(k) == v for k, v in metadata_filter.items()):
                        continue
                bm25_scored.append((cid, score))
                if cid not in chunk_map and match:
                    chunk_map[cid] = {"text": match["text"], "meta": match["meta"], "distance": 1.0}

    # Combine scores via Reciprocal Rank Fusion
    if bm25_scored:
        hybrid_scores = _reciprocal_rank_fusion(vector_scored, bm25_scored)
    else:
        hybrid_scores = _reciprocal_rank_fusion(vector_scored)

    top_ids = sorted(hybrid_scores, key=lambda x: hybrid_scores[x], reverse=True)[:n_results]

    citations: list[SourceCitation] = []
    best_distance = 1.0
    for cid in top_ids:
        data = chunk_map.get(cid)
        if not data:
            continue
        dist = data["distance"]
        if dist > CHUNK_RELEVANCE_THRESHOLD:
            continue
        best_distance = min(best_distance, dist)
        citations.append(SourceCitation(
            document=data["meta"]["filename"],
            chunk_text=data["text"],
            score=round(hybrid_scores.get(cid, 1 - dist), 4),
            page=data["meta"].get("page"),
            section=data["meta"].get("section"),
        ))

    has_relevant = best_distance < NO_ANSWER_THRESHOLD
    return citations, has_relevant


def deduplicate_sources(
    citations: list[SourceCitation],
    max_sources: int = 5,
) -> list[DocumentSource]:
    """Group chunk-level citations into unique document sources with all chunks."""
    doc_map: dict[str, dict] = {}
    for c in citations:
        if c.document not in doc_map:
            doc_map[c.document] = {"pages": set(), "best_score": c.score, "chunks": []}
        entry = doc_map[c.document]
        if c.page is not None:
            entry["pages"].add(c.page)
        if c.score > entry["best_score"]:
            entry["best_score"] = c.score
        entry["chunks"].append(SourceChunk(
            text=c.chunk_text,
            score=round(c.score, 4),
            page=c.page,
            section=c.section,
        ))

    sources = []
    for doc_name, data in doc_map.items():
        chunks_sorted = sorted(data["chunks"], key=lambda ch: ch.score, reverse=True)
        sources.append(DocumentSource(
            document=doc_name,
            pages=sorted(data["pages"]),
            score=round(data["best_score"], 4),
            chunks=chunks_sorted,
        ))

    sources.sort(key=lambda s: s.score, reverse=True)
    return sources[:max_sources]


async def retrieve_multi_query(
    question: str,
    n_results: int = 5,
    device_id: str = "",
    metadata_filter: dict | None = None,
    doc_ids: list[str] | None = None,
) -> tuple[list[SourceCitation], bool]:
    from services.generator import generate_query_variants

    variants = await generate_query_variants(question)
    queries = [question] + variants

    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        *(loop.run_in_executor(None, retrieve, q, n_results, doc_ids, device_id, metadata_filter) for q in queries)
    )

    seen: dict[str, SourceCitation] = {}
    has_relevant = False

    for cits, relevant in results:
        if relevant:
            has_relevant = True
        for c in cits:
            key = f"{c.document}::{c.chunk_text[:80]}"
            if key not in seen or c.score > seen[key].score:
                seen[key] = c

    merged = sorted(seen.values(), key=lambda c: c.score, reverse=True)[:n_results]
    return merged, has_relevant
