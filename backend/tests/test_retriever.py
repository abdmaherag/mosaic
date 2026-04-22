"""
Tests for recent backend changes:
1. _reciprocal_rank_fusion() -- pure function in services/retriever.py
2. query_chunks() metadata_filter passthrough -- db/chroma_client.py
3. retrieve() metadata_filter passthrough -- services/retriever.py
4. /query/vector-only endpoint -- routers/query.py
"""

import sys
import os
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ---------------------------------------------------------------------------
# Helpers -- stub heavy deps so imports don't touch disk/GPU/network
# ---------------------------------------------------------------------------

def _ensure_stubs():
    if "rank_bm25" not in sys.modules:
        rank_bm25 = types.ModuleType("rank_bm25")
        rank_bm25.BM25Okapi = MagicMock()
        sys.modules["rank_bm25"] = rank_bm25

    if "sentence_transformers" not in sys.modules:
        st = types.ModuleType("sentence_transformers")
        st.SentenceTransformer = MagicMock()
        sys.modules["sentence_transformers"] = st

    if "groq" not in sys.modules:
        groq = types.ModuleType("groq")
        groq.AsyncGroq = MagicMock()
        sys.modules["groq"] = groq

    if "sse_starlette" not in sys.modules:
        sse = types.ModuleType("sse_starlette")
        sse_sse = types.ModuleType("sse_starlette.sse")
        sse_sse.EventSourceResponse = MagicMock()
        sys.modules["sse_starlette"] = sse
        sys.modules["sse_starlette.sse"] = sse_sse

    if "chromadb" not in sys.modules:
        chroma = types.ModuleType("chromadb")
        chroma.PersistentClient = MagicMock()
        chroma.ClientAPI = MagicMock()
        chroma.Collection = MagicMock()
        sys.modules["chromadb"] = chroma


_ensure_stubs()

import services.retriever as retriever_module
from services.retriever import _reciprocal_rank_fusion, RRF_K
import db.chroma_client as chroma_client


# ===========================================================================
# 1. Reciprocal Rank Fusion
# ===========================================================================

class TestReciprocalRankFusion:

    def test_single_list_rank1_gets_highest_score(self):
        ranked = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
        result = _reciprocal_rank_fusion(ranked)
        assert result["a"] == pytest.approx(1 / (RRF_K + 1))
        assert result["b"] == pytest.approx(1 / (RRF_K + 2))
        assert result["c"] == pytest.approx(1 / (RRF_K + 3))

    def test_single_list_sorted_by_score_desc(self):
        # Input in ascending order -- must be sorted desc before ranking
        ranked = [("low", 0.1), ("mid", 0.5), ("high", 0.9)]
        result = _reciprocal_rank_fusion(ranked)
        assert result["high"] > result["mid"] > result["low"]

    def test_two_lists_shared_chunk_accumulates(self):
        # A chunk present in both lists accumulates 1/(k+rank) from each
        list1 = [("a", 0.9), ("b", 0.5)]
        list2 = [("b", 0.8), ("a", 0.3)]
        result = _reciprocal_rank_fusion(list1, list2)
        expected_a = 1 / (RRF_K + 1) + 1 / (RRF_K + 2)
        expected_b = 1 / (RRF_K + 2) + 1 / (RRF_K + 1)
        assert result["a"] == pytest.approx(expected_a)
        assert result["b"] == pytest.approx(expected_b)
        assert result["a"] == pytest.approx(result["b"])

    def test_chunk_in_both_lists_beats_chunk_in_one(self):
        list1 = [("shared", 0.9), ("only_vec", 0.5)]
        list2 = [("shared", 0.8), ("only_bm25", 0.3)]
        result = _reciprocal_rank_fusion(list1, list2)
        assert result["shared"] > result["only_vec"]
        assert result["shared"] > result["only_bm25"]

    def test_empty_single_list_returns_empty(self):
        assert _reciprocal_rank_fusion([]) == {}

    def test_no_lists_returns_empty(self):
        assert _reciprocal_rank_fusion() == {}

    def test_single_item_single_list(self):
        result = _reciprocal_rank_fusion([("only", 1.0)])
        assert result == {"only": pytest.approx(1 / (RRF_K + 1))}

    def test_three_lists_triple_contribution(self):
        shared = [("x", 0.9)]
        result = _reciprocal_rank_fusion(shared, shared, shared)
        assert result["x"] == pytest.approx(3 * (1 / (RRF_K + 1)))

    def test_all_scores_positive(self):
        result = _reciprocal_rank_fusion([("a", 0.8), ("b", 0.0)])
        assert all(v > 0 for v in result.values())

    def test_rrf_k_is_60(self):
        assert RRF_K == 60

    def test_large_list_first_beats_last(self):
        n = 100
        ranked = [(f"chunk_{i}", float(n - i)) for i in range(n)]
        result = _reciprocal_rank_fusion(ranked)
        assert result["chunk_0"] > result["chunk_99"]

    def test_returns_dict_type(self):
        result = _reciprocal_rank_fusion([("a", 1.0)])
        assert isinstance(result, dict)


# ===========================================================================
# 2. query_chunks() -- metadata_filter passthrough
# ===========================================================================

class TestQueryChunksMetadataFilter:

    def _mock_collection(self):
        col = MagicMock()
        col.query.return_value = {
            "documents": [["text"]],
            "metadatas": [[{"doc_id": "d1", "filename": "f.pdf", "chunk_index": 0}]],
            "distances": [[0.1]],
        }
        return col

    def test_no_conditions_passes_where_none(self):
        col = self._mock_collection()
        with patch.object(chroma_client, "get_collection", return_value=col):
            chroma_client.query_chunks([0.1, 0.2], n_results=3)
        _, kwargs = col.query.call_args
        assert kwargs.get("where") is None

    def test_metadata_filter_alone_becomes_where(self):
        col = self._mock_collection()
        mf = {"category": "legal"}
        with patch.object(chroma_client, "get_collection", return_value=col):
            chroma_client.query_chunks([0.1], n_results=3, metadata_filter=mf)
        _, kwargs = col.query.call_args
        assert kwargs["where"] == mf

    def test_device_id_and_metadata_filter_uses_and(self):
        col = self._mock_collection()
        mf = {"year": 2024}
        with patch.object(chroma_client, "get_collection", return_value=col):
            chroma_client.query_chunks([0.1], n_results=3, device_id="dev-1", metadata_filter=mf)
        _, kwargs = col.query.call_args
        where = kwargs["where"]
        assert "$and" in where
        assert {"device_id": "dev-1"} in where["$and"]
        assert mf in where["$and"]

    def test_doc_ids_and_metadata_filter_uses_and(self):
        col = self._mock_collection()
        mf = {"source": "internal"}
        with patch.object(chroma_client, "get_collection", return_value=col):
            chroma_client.query_chunks([0.1], n_results=3, doc_ids=["d1"], metadata_filter=mf)
        _, kwargs = col.query.call_args
        where = kwargs["where"]
        assert "$and" in where
        assert {"doc_id": {"$in": ["d1"]}} in where["$and"]
        assert mf in where["$and"]

    def test_all_three_filters_all_in_and(self):
        col = self._mock_collection()
        mf = {"type": "contract"}
        with patch.object(chroma_client, "get_collection", return_value=col):
            chroma_client.query_chunks(
                [0.1], n_results=3,
                device_id="dev-1", doc_ids=["d1", "d2"], metadata_filter=mf
            )
        _, kwargs = col.query.call_args
        assert len(kwargs["where"]["$and"]) == 3

    def test_metadata_filter_none_means_no_extra_condition(self):
        col = self._mock_collection()
        with patch.object(chroma_client, "get_collection", return_value=col):
            chroma_client.query_chunks([0.1], n_results=3, metadata_filter=None)
        _, kwargs = col.query.call_args
        assert kwargs.get("where") is None

    def test_device_id_only_no_and_wrapper(self):
        # Single condition should not be wrapped in $and
        col = self._mock_collection()
        with patch.object(chroma_client, "get_collection", return_value=col):
            chroma_client.query_chunks([0.1], n_results=3, device_id="dev-1")
        _, kwargs = col.query.call_args
        where = kwargs["where"]
        assert where == {"device_id": "dev-1"}


# ===========================================================================
# 3. retrieve() -- metadata_filter forwarded to query_chunks
# ===========================================================================

class TestRetrieveMetadataFilterPassthrough:

    def _chroma_result(self, distance=0.3):
        return {
            "documents": [["relevant text about the topic"]],
            "metadatas": [[{
                "doc_id": "doc1", "filename": "report.pdf",
                "chunk_index": 0, "page": 1, "section": "intro"
            }]],
            "distances": [[distance]],
        }

    def test_metadata_filter_forwarded_to_query_chunks(self):
        mf = {"category": "financial"}
        with (
            patch("services.retriever.embed_query", return_value=[0.1] * 384),
            patch("services.retriever.query_chunks", return_value=self._chroma_result()) as mock_qc,
            patch("services.retriever._get_bm25", return_value=(None, [], {})),
        ):
            retriever_module.retrieve("what is revenue?", metadata_filter=mf)
        _, kwargs = mock_qc.call_args
        assert kwargs.get("metadata_filter") == mf

    def test_no_metadata_filter_passes_none(self):
        with (
            patch("services.retriever.embed_query", return_value=[0.1] * 384),
            patch("services.retriever.query_chunks", return_value=self._chroma_result()) as mock_qc,
            patch("services.retriever._get_bm25", return_value=(None, [], {})),
        ):
            retriever_module.retrieve("what is revenue?")
        _, kwargs = mock_qc.call_args
        assert kwargs.get("metadata_filter") is None

    def test_empty_chroma_result_returns_empty(self):
        empty = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        with (
            patch("services.retriever.embed_query", return_value=[0.1] * 384),
            patch("services.retriever.query_chunks", return_value=empty),
            patch("services.retriever._get_bm25", return_value=(None, [], {})),
        ):
            citations, has_relevant = retriever_module.retrieve("anything?")
        assert citations == []
        assert has_relevant is False

    def test_missing_documents_key_returns_empty(self):
        empty = {"documents": [], "metadatas": [], "distances": []}
        with (
            patch("services.retriever.embed_query", return_value=[0.1] * 384),
            patch("services.retriever.query_chunks", return_value=empty),
            patch("services.retriever._get_bm25", return_value=(None, [], {})),
        ):
            citations, has_relevant = retriever_module.retrieve("anything?")
        assert citations == []
        assert has_relevant is False

    def test_high_distance_chunk_excluded(self):
        # distance > CHUNK_RELEVANCE_THRESHOLD (0.80) -> dropped from results
        with (
            patch("services.retriever.embed_query", return_value=[0.1] * 384),
            patch("services.retriever.query_chunks", return_value=self._chroma_result(distance=0.95)),
            patch("services.retriever._get_bm25", return_value=(None, [], {})),
        ):
            citations, has_relevant = retriever_module.retrieve("anything?")
        assert citations == []
        assert has_relevant is False

    def test_low_distance_chunk_included_and_relevant(self):
        # distance=0.3 -> included, has_relevant=True (below NO_ANSWER_THRESHOLD 0.85)
        with (
            patch("services.retriever.embed_query", return_value=[0.1] * 384),
            patch("services.retriever.query_chunks", return_value=self._chroma_result(distance=0.3)),
            patch("services.retriever._get_bm25", return_value=(None, [], {})),
        ):
            citations, has_relevant = retriever_module.retrieve("anything?")
        assert len(citations) == 1
        assert has_relevant is True
        assert citations[0].document == "report.pdf"


# ===========================================================================
# 4. /query/vector-only endpoint
# ===========================================================================

# Load the FastAPI app once at module level so patch.object targets the live
# module reference. Dropping and re-importing inside each test would unbind
# any patches applied before the re-import.

for _mod in ["main", "routers.query", "services.generator"]:
    sys.modules.pop(_mod, None)

from fastapi.testclient import TestClient
import main as _app_module
import routers.query as _query_router

_test_client = TestClient(_app_module.app)


class TestVectorOnlyEndpoint:

    def _chroma_result(self, distance=0.3):
        return {
            "documents": [["This is relevant text about the topic."]],
            "metadatas": [[{
                "doc_id": "doc1", "filename": "report.pdf",
                "chunk_index": 0, "page": 2, "section": None
            }]],
            "distances": [[distance]],
        }

    def test_happy_path_returns_answer_and_citations(self):
        with (
            patch("services.embedder.embed_query", return_value=[0.1] * 384),
            patch("db.chroma_client.query_chunks", return_value=self._chroma_result(0.3)),
            patch.object(_query_router, "generate", new_callable=AsyncMock, return_value="Here is the answer."),
        ):
            response = _test_client.post(
                "/query/vector-only",
                json={"question": "What is revenue?", "n_results": 3}
            )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Here is the answer."
        assert len(body["citations"]) == 1
        assert body["citations"][0]["document"] == "report.pdf"
        assert body["citations"][0]["score"] == pytest.approx(round(1 - 0.3, 4))

    def test_no_chroma_results_returns_no_relevant_msg(self):
        empty = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        with (
            patch("services.embedder.embed_query", return_value=[0.1] * 384),
            patch("db.chroma_client.query_chunks", return_value=empty),
        ):
            response = _test_client.post(
                "/query/vector-only",
                json={"question": "Irrelevant question?", "n_results": 3}
            )
        assert response.status_code == 200
        assert "No relevant" in response.json()["answer"]
        assert response.json()["citations"] == []

    def test_empty_question_returns_400(self):
        with (
            patch("services.embedder.embed_query", return_value=[0.1] * 384),
            patch("db.chroma_client.query_chunks", return_value=self._chroma_result()),
        ):
            response = _test_client.post(
                "/query/vector-only",
                json={"question": "   ", "n_results": 3}
            )
        assert response.status_code == 400

    def test_whitespace_only_question_returns_400(self):
        with (
            patch("services.embedder.embed_query", return_value=[0.1] * 384),
            patch("db.chroma_client.query_chunks", return_value=self._chroma_result()),
        ):
            response = _test_client.post(
                "/query/vector-only",
                json={"question": "\t\n", "n_results": 3}
            )
        assert response.status_code == 400

    def test_high_distance_chunks_filtered_returns_no_relevant(self):
        # Chunks with distance > 0.80 dropped; best_distance stays 1.0 -> no-answer
        with (
            patch("services.embedder.embed_query", return_value=[0.1] * 384),
            patch("db.chroma_client.query_chunks", return_value=self._chroma_result(distance=0.95)),
        ):
            response = _test_client.post(
                "/query/vector-only",
                json={"question": "What is revenue?", "n_results": 3}
            )
        assert response.status_code == 200
        assert "No relevant" in response.json()["answer"]

    def test_session_id_returned_in_response(self):
        with (
            patch("services.embedder.embed_query", return_value=[0.1] * 384),
            patch("db.chroma_client.query_chunks", return_value=self._chroma_result(0.3)),
            patch.object(_query_router, "generate", new_callable=AsyncMock, return_value="Answer here."),
        ):
            response = _test_client.post(
                "/query/vector-only",
                json={"question": "Tell me something?", "n_results": 3}
            )
        assert response.status_code == 200
        assert "session_id" in response.json()
        assert response.json()["session_id"] != ""

    def test_sources_deduplicated_in_response(self):
        with (
            patch("services.embedder.embed_query", return_value=[0.1] * 384),
            patch("db.chroma_client.query_chunks", return_value=self._chroma_result(0.3)),
            patch.object(_query_router, "generate", new_callable=AsyncMock, return_value="Answer."),
        ):
            response = _test_client.post(
                "/query/vector-only",
                json={"question": "What is revenue?", "n_results": 3}
            )
        body = response.json()
        assert "sources" in body
        assert len(body["sources"]) == 1
        assert body["sources"][0]["document"] == "report.pdf"
