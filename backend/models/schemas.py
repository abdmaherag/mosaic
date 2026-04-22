from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    id: str
    filename: str
    content_type: str
    chunk_count: int
    status: str


ALLOWED_FILTER_KEYS = {"filename", "page", "section"}


class QueryRequest(BaseModel):
    question: str = Field(max_length=2000)
    session_id: str | None = None
    n_results: int = Field(default=5, ge=1, le=20)
    metadata_filter: dict | None = None
    doc_ids: list[str] | None = None

    def validated_metadata_filter(self) -> dict | None:
        """Return metadata_filter with only allowed keys and safe value types."""
        if not self.metadata_filter:
            return None
        safe = {}
        for key, val in self.metadata_filter.items():
            if key not in ALLOWED_FILTER_KEYS:
                continue
            if not isinstance(val, (str, int, float)):
                continue
            safe[key] = val
        return safe or None


class SourceCitation(BaseModel):
    document: str
    chunk_text: str
    score: float
    page: int | None = None
    section: str | None = None


class SourceChunk(BaseModel):
    text: str
    score: float
    page: int | None = None
    section: str | None = None


class DocumentSource(BaseModel):
    document: str
    pages: list[int]
    score: float
    chunks: list[SourceChunk]


class QueryResponse(BaseModel):
    answer: str
    citations: list[SourceCitation]
    sources: list[DocumentSource]
    session_id: str


class CompareRequest(BaseModel):
    question: str = Field(max_length=2000)
    doc_ids: list[str] = Field(min_length=2, max_length=10)
    n_results: int = Field(default=3, ge=1, le=20)


class ProcessingEvent(BaseModel):
    step: str
    status: str
    detail: str | None = None
    progress: float | None = None
    chunk_count: int | None = None
    doc_id: str | None = None
    file_hash: str | None = None
