"""Tests for FastAPI endpoints using TestClient.

All external services (ChromaDB, embedder, generator, pipeline) are mocked
so tests run without any real infrastructure.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_lifespan_reset():
    """Prevent reset_collection() from running on app startup during tests."""
    with patch("db.chroma_client.reset_collection", return_value=None):
        yield


@pytest.fixture
def client():
    """Return a TestClient with all heavy dependencies mocked."""
    with patch("db.chroma_client.get_client"), \
         patch("db.chroma_client.get_collection"):
        from main import app
        return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /documents/
# ---------------------------------------------------------------------------

class TestDocumentsList:
    def test_list_documents_empty(self, client):
        with patch("routers.documents.get_all_chunks", return_value=[]), \
             patch("routers.documents._documents", {}), \
             patch("routers.documents._registry_loaded", True):
            resp = client.get("/documents/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_documents_returns_registered_docs(self, client):
        import routers.documents as docs_router
        original = dict(docs_router._documents)
        original_loaded = docs_router._registry_loaded

        docs_router._documents.clear()
        docs_router._documents["doc1"] = {
            "id": "doc1",
            "filename": "test.txt",
            "content_type": "text/plain",
            "chunk_count": 3,
            "status": "ready",
        }
        docs_router._registry_loaded = True

        try:
            resp = client.get("/documents/")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["filename"] == "test.txt"
            assert data[0]["chunk_count"] == 3
        finally:
            docs_router._documents.clear()
            docs_router._documents.update(original)
            docs_router._registry_loaded = original_loaded


class TestDocumentsDelete:
    def test_delete_nonexistent_returns_404(self, client):
        import routers.documents as docs_router
        original = dict(docs_router._documents)
        original_loaded = docs_router._registry_loaded

        docs_router._documents.clear()
        docs_router._registry_loaded = True

        try:
            resp = client.delete("/documents/nonexistent-id")
            assert resp.status_code == 404
        finally:
            docs_router._documents.clear()
            docs_router._documents.update(original)
            docs_router._registry_loaded = original_loaded

    def test_delete_existing_returns_deleted(self, client):
        import routers.documents as docs_router
        original = dict(docs_router._documents)
        original_loaded = docs_router._registry_loaded

        docs_router._documents.clear()
        docs_router._documents["doc-to-delete"] = {
            "id": "doc-to-delete",
            "filename": "old.txt",
            "content_type": "text/plain",
            "chunk_count": 1,
            "status": "ready",
        }
        docs_router._registry_loaded = True

        try:
            with patch("routers.documents.delete_document") as mock_del, \
                 patch("routers.documents.invalidate_bm25_cache"):
                resp = client.delete("/documents/doc-to-delete")
                mock_del.assert_called_once_with("doc-to-delete")

            assert resp.status_code == 200
            assert resp.json()["status"] == "deleted"
            assert "doc-to-delete" not in docs_router._documents
        finally:
            docs_router._documents.clear()
            docs_router._documents.update(original)
            docs_router._registry_loaded = original_loaded


# ---------------------------------------------------------------------------
# /query/ — session management
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def _mock_retrieve_and_generate(self):
        from models.schemas import SourceCitation
        citation = SourceCitation(document="doc.txt", chunk_text="relevant text", score=0.9)
        return [citation], True

    def test_query_returns_valid_response(self, client):
        from models.schemas import SourceCitation
        citation = SourceCitation(document="doc.txt", chunk_text="chunk", score=0.85)

        with patch("routers.query.retrieve_multi_query", new=AsyncMock(return_value=([citation], True))), \
             patch("routers.query.generate", new=AsyncMock(return_value="The answer is 42.")):
            resp = client.post("/query/", json={"question": "What is 42?"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "The answer is 42."
        assert len(data["citations"]) == 1
        assert "session_id" in data

    def test_query_creates_session_id_when_not_provided(self, client):
        with patch("routers.query.retrieve_multi_query", new=AsyncMock(return_value=([], False))), \
             patch("routers.query.generate", new=AsyncMock(return_value="No results.")):
            resp = client.post("/query/", json={"question": "Q?"})

        data = resp.json()
        assert data["session_id"]  # not empty

    def test_query_uses_provided_session_id(self, client):
        with patch("routers.query.retrieve_multi_query", new=AsyncMock(return_value=([], False))), \
             patch("routers.query.generate", new=AsyncMock(return_value="Answer.")):
            resp = client.post("/query/", json={"question": "Q?", "session_id": "my-session"})

        assert resp.json()["session_id"] == "my-session"

    def test_query_missing_question_returns_422(self, client):
        resp = client.post("/query/", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /query/session/{session_id} — clear session
# ---------------------------------------------------------------------------

class TestClearSession:
    def test_clear_existing_session(self, client):
        import routers.query as query_router
        query_router._sessions["test-session"] = [{"role": "user", "content": "hello"}]

        resp = client.delete("/query/session/test-session")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
        assert "test-session" not in query_router._sessions

    def test_clear_nonexistent_session_ok(self, client):
        # Should not raise — just a no-op
        resp = client.delete("/query/session/does-not-exist")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"


# ---------------------------------------------------------------------------
# /query/compare/stream — validation
# ---------------------------------------------------------------------------

class TestCompareEndpoint:
    def test_compare_requires_at_least_two_doc_ids(self, client):
        resp = client.post("/query/compare/stream", json={"question": "Q?", "doc_ids": ["only-one"]})
        assert resp.status_code == 400

    def test_compare_empty_doc_ids_returns_400(self, client):
        resp = client.post("/query/compare/stream", json={"question": "Q?", "doc_ids": []})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /documents/upload — size limit
# ---------------------------------------------------------------------------

class TestDocumentUpload:
    def test_upload_too_large_returns_413(self, client):
        # 51 MB > MAX_UPLOAD_SIZE (50 MB)
        large_content = b"x" * (51 * 1024 * 1024)
        resp = client.post(
            "/documents/upload",
            files={"file": ("big.txt", large_content, "text/plain")},
        )
        assert resp.status_code == 413
