"""Tests for models/schemas.py — Pydantic validation."""
import pytest
from pydantic import ValidationError

from models.schemas import (
    DocumentInfo,
    QueryRequest,
    SourceCitation,
    QueryResponse,
    CompareRequest,
    ProcessingEvent,
)


# ---------------------------------------------------------------------------
# DocumentInfo
# ---------------------------------------------------------------------------

class TestDocumentInfo:
    def test_valid_construction(self):
        doc = DocumentInfo(id="abc", filename="file.txt", content_type="text/plain", chunk_count=5, status="ready")
        assert doc.id == "abc"
        assert doc.chunk_count == 5

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            DocumentInfo(filename="file.txt", content_type="text/plain", chunk_count=5, status="ready")

    def test_all_fields_present(self):
        doc = DocumentInfo(id="x", filename="a.pdf", content_type="application/pdf", chunk_count=0, status="processing")
        assert doc.status == "processing"


# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------

class TestQueryRequest:
    def test_minimal_request(self):
        req = QueryRequest(question="What is RAG?")
        assert req.question == "What is RAG?"
        assert req.session_id is None
        assert req.n_results == 5

    def test_explicit_session_id(self):
        req = QueryRequest(question="Q?", session_id="session-123")
        assert req.session_id == "session-123"

    def test_custom_n_results(self):
        req = QueryRequest(question="Q?", n_results=10)
        assert req.n_results == 10

    def test_missing_question_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest()

    def test_n_results_default_is_5(self):
        req = QueryRequest(question="Q?")
        assert req.n_results == 5


# ---------------------------------------------------------------------------
# SourceCitation
# ---------------------------------------------------------------------------

class TestSourceCitation:
    def test_required_fields(self):
        c = SourceCitation(document="file.txt", chunk_text="Some text.", score=0.85)
        assert c.document == "file.txt"
        assert c.score == 0.85
        assert c.page is None
        assert c.section is None

    def test_optional_page_and_section(self):
        c = SourceCitation(document="f.pdf", chunk_text="text", score=0.7, page=3, section="Introduction")
        assert c.page == 3
        assert c.section == "Introduction"

    def test_missing_required_raises(self):
        with pytest.raises(ValidationError):
            SourceCitation(document="f.txt", score=0.5)  # chunk_text missing

    def test_score_is_float(self):
        c = SourceCitation(document="f.txt", chunk_text="t", score=1)
        assert isinstance(c.score, float)


# ---------------------------------------------------------------------------
# QueryResponse
# ---------------------------------------------------------------------------

class TestQueryResponse:
    def _citation(self):
        return SourceCitation(document="doc.txt", chunk_text="chunk", score=0.9)

    def test_valid_response(self):
        resp = QueryResponse(answer="42", citations=[self._citation()], sources=[], session_id="sid-1")
        assert resp.answer == "42"
        assert len(resp.citations) == 1

    def test_empty_citations_allowed(self):
        resp = QueryResponse(answer="No results.", citations=[], sources=[], session_id="sid-2")
        assert resp.citations == []

    def test_missing_session_id_raises(self):
        with pytest.raises(ValidationError):
            QueryResponse(answer="A", citations=[], sources=[])


# ---------------------------------------------------------------------------
# CompareRequest
# ---------------------------------------------------------------------------

class TestCompareRequest:
    def test_defaults(self):
        req = CompareRequest(question="Compare?", doc_ids=["a", "b"])
        assert req.n_results == 3

    def test_custom_n_results(self):
        req = CompareRequest(question="Q?", doc_ids=["x", "y"], n_results=7)
        assert req.n_results == 7

    def test_missing_doc_ids_raises(self):
        with pytest.raises(ValidationError):
            CompareRequest(question="Q?")

    def test_empty_doc_ids_allowed_by_schema(self):
        # Schema does not enforce minimum length (that's router logic)
        req = CompareRequest(question="Q?", doc_ids=[])
        assert req.doc_ids == []


# ---------------------------------------------------------------------------
# ProcessingEvent
# ---------------------------------------------------------------------------

class TestProcessingEvent:
    def test_required_fields(self):
        e = ProcessingEvent(step="parse", status="running")
        assert e.step == "parse"
        assert e.status == "running"
        assert e.detail is None
        assert e.progress is None
        assert e.chunk_count is None
        assert e.doc_id is None

    def test_all_optional_fields(self):
        e = ProcessingEvent(
            step="complete",
            status="done",
            detail="All good",
            progress=1.0,
            chunk_count=12,
            doc_id="doc-abc",
        )
        assert e.chunk_count == 12
        assert e.doc_id == "doc-abc"

    def test_model_dump_includes_all_fields(self):
        e = ProcessingEvent(step="embed", status="running", progress=0.5)
        d = e.model_dump()
        assert "step" in d
        assert "status" in d
        assert "progress" in d
        assert d["progress"] == 0.5
