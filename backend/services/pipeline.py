import hashlib
import uuid
from collections.abc import AsyncGenerator

from models.schemas import ProcessingEvent
from services.parser import parse_document
from services.chunker import semantic_chunk
from services.embedder import embed_texts
from db.chroma_client import add_chunks


async def ingest_document(
    content: bytes,
    filename: str,
    content_type: str | None = None,
    device_id: str = "",
) -> AsyncGenerator[ProcessingEvent, None]:
    """Ingest a document through the full pipeline, yielding progress events."""
    doc_id = str(uuid.uuid4())
    file_hash = hashlib.sha256(content).hexdigest()

    # Step 1: Parse
    yield ProcessingEvent(step="parsing", status="started", detail=f"Parsing {filename}")
    try:
        text = parse_document(content, filename, content_type)
    except ValueError:
        yield ProcessingEvent(step="parsing", status="error", detail="Failed to parse document. Check the file format.")
        return
    yield ProcessingEvent(step="parsing", status="done", detail=f"Extracted {len(text)} characters")

    # Step 2: Chunk
    yield ProcessingEvent(step="chunking", status="started", detail="Splitting into semantic chunks")
    chunks = semantic_chunk(text)
    if not chunks:
        yield ProcessingEvent(step="chunking", status="error", detail="No chunks produced")
        return
    yield ProcessingEvent(step="chunking", status="done", detail=f"Created {len(chunks)} chunks")

    # Step 3: Embed
    yield ProcessingEvent(step="embedding", status="started", detail=f"Embedding {len(chunks)} chunks")
    try:
        embeddings = embed_texts(chunks)
    except Exception:
        yield ProcessingEvent(step="embedding", status="error", detail="Embedding failed. Please try again.")
        return
    yield ProcessingEvent(step="embedding", status="done", detail="Embeddings computed")

    # Step 4: Store
    yield ProcessingEvent(step="storing", status="started", detail="Saving to ChromaDB")
    try:
        count = add_chunks(doc_id, filename, chunks, embeddings, file_hash, device_id)
    except Exception:
        yield ProcessingEvent(step="storing", status="error", detail="Failed to store document. Please try again.")
        return
    yield ProcessingEvent(step="storing", status="done", detail=f"Stored {count} chunks")

    yield ProcessingEvent(
        step="complete",
        status="done",
        detail=f"Document '{filename}' ingested: {count} chunks",
        progress=1.0,
        chunk_count=count,
        doc_id=doc_id,
        file_hash=file_hash,
    )
