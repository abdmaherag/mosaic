import asyncio
import math

from rank_bm25 import BM25Okapi

from db.chroma_client import query_chunks, get_all_chunks, get_embeddings_by_ids
from models.schemas import DocumentSource, SourceChunk, SourceCitation
from services.embedder import embed_query

NO_ANSWER_THRESHOLD = 0.85  # ChromaDB cosine distance: lower = more similar
CHUNK_RELEVANCE_THRESHOLD = 0.80  # Per-chunk filter: drop chunks with distance above this
# Final display floor: only show citations whose true cosine similarity to the
# query is at least this. Prevents weak BM25-only matches from being shown
# with inflated scores. 0.35 is empirically the floor below which chunks are
# rarely actually relevant on this corpus.
DISPLAY_SIMILARITY_FLOOR = 0.35


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

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


def _dedup_near_duplicates(
    ranked_ids: list[str],
    chunk_map: dict[str, dict],
    jaccard_threshold: float = 0.7,
) -> list[str]:
    """Drop near-duplicate chunks from a ranked list, keeping the higher-ranked
    one. Uses token-set Jaccard — cheap, parser-noise-resilient, and good
    enough to catch the common "same page extracted twice" / "overlapping
    chunks share most sentences" failure modes.

    jaccard_threshold=0.7: chunks sharing 70%+ of their tokens are duplicates.
    Lower would over-collapse legitimately related chunks; higher misses the
    parser-induced near-dupes we actually see.
    """
    kept: list[str] = []
    kept_token_sets: list[set[str]] = []
    for cid in ranked_ids:
        text = chunk_map.get(cid, {}).get("text", "")
        # Token set on lowercased words >=4 chars — short tokens (math symbols,
        # stopwords) inflate Jaccard between unrelated chunks.
        tokens = {w for w in text.lower().split() if len(w) >= 4}
        if not tokens:
            kept.append(cid)
            kept_token_sets.append(tokens)
            continue
        is_dupe = False
        for prev in kept_token_sets:
            if not prev:
                continue
            overlap = len(tokens & prev)
            union = len(tokens | prev)
            if union and overlap / union >= jaccard_threshold:
                is_dupe = True
                break
        if not is_dupe:
            kept.append(cid)
            kept_token_sets.append(tokens)
    return kept


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

    # chunk_map tracks text/meta and the signals that pulled the chunk in:
    #   distance  — cosine distance from vector search (None if BM25-only)
    #   bm25      — raw BM25 score (None if vector-only)
    # Filtering happens per-source: a vector chunk is kept if its distance
    # clears CHUNK_RELEVANCE_THRESHOLD; a BM25 chunk is kept if its raw score
    # is positive (at least one query term hit). Both lists then feed RRF.
    chunk_map: dict[str, dict] = {}
    vector_scored: list[tuple[str, float]] = []
    best_vector_distance = 1.0
    for doc_text, meta, dist in zip(docs, metas, dists):
        cid = f"{meta['doc_id']}_chunk_{meta['chunk_index']}"
        chunk_map[cid] = {"text": doc_text, "meta": meta, "distance": dist, "bm25": None}
        best_vector_distance = min(best_vector_distance, dist)
        if dist <= CHUNK_RELEVANCE_THRESHOLD:
            vector_scored.append((cid, 1 - dist))

    bm25_scored: list[tuple[str, float]] = []
    best_bm25_score = 0.0
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
                if score <= 0:
                    continue  # no query term matched — don't pollute the fusion
                match = corpus_by_id.get(cid)
                if metadata_filter and match:
                    if not all(match["meta"].get(k) == v for k, v in metadata_filter.items()):
                        continue
                bm25_scored.append((cid, score))
                best_bm25_score = max(best_bm25_score, score)
                if cid in chunk_map:
                    chunk_map[cid]["bm25"] = score
                elif match:
                    chunk_map[cid] = {
                        "text": match["text"], "meta": match["meta"],
                        "distance": None, "bm25": score,
                    }

    # Nothing survived either filter → no-answer
    if not vector_scored and not bm25_scored:
        return [], False

    if bm25_scored:
        hybrid_scores = _reciprocal_rank_fusion(vector_scored, bm25_scored)
    else:
        hybrid_scores = _reciprocal_rank_fusion(vector_scored)

    ranked_ids = sorted(hybrid_scores, key=lambda x: hybrid_scores[x], reverse=True)
    # Dedup near-duplicates before slicing top-k — otherwise a page extracted
    # twice (or two heavily-overlapped chunks) consume slots that should go
    # to genuinely different evidence.
    deduped = _dedup_near_duplicates(ranked_ids, chunk_map)
    top_ids = deduped[:n_results]

    # Compute REAL cosine similarity for every displayed chunk against the
    # query embedding — fixes the "score=1.00 always" bug caused by
    # max-normalizing BM25 against itself. For vector-search hits we already
    # have a distance; for BM25-only hits we pull the stored embedding from
    # Chroma and compute cosine on the spot.
    bm25_only_ids = [
        cid for cid in top_ids
        if chunk_map.get(cid, {}).get("distance") is None
    ]
    fetched_embeddings = get_embeddings_by_ids(bm25_only_ids) if bm25_only_ids else {}

    citations: list[SourceCitation] = []
    for cid in top_ids:
        data = chunk_map.get(cid)
        if not data:
            continue
        dist = data["distance"]
        if dist is not None:
            similarity = max(0.0, min(1.0, 1 - dist))
        else:
            emb = fetched_embeddings.get(cid)
            similarity = max(0.0, _cosine(query_embedding, emb)) if emb else 0.0

        # Adaptive cutoff: drop weak matches from the visible set even if
        # that means returning fewer than n_results citations. Better to
        # show 1 strong source than pad the list with noise.
        if similarity < DISPLAY_SIMILARITY_FLOOR:
            continue

        citations.append(SourceCitation(
            document=data["meta"]["filename"],
            chunk_text=data["text"],
            score=round(similarity, 4),
            page=data["meta"].get("page"),
            section=data["meta"].get("section"),
        ))

    # has_relevant: either vector search found a semantically close chunk OR
    # BM25 found strong lexical overlap. Either signal is enough to answer.
    has_relevant = (best_vector_distance < NO_ANSWER_THRESHOLD) or (best_bm25_score > 0)
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

    # If query-variant generation fails (rate limit, API error), fall back to
    # the original question alone. Retrieval still works without paraphrases.
    try:
        variants = await generate_query_variants(question)
    except Exception:
        variants = []
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
