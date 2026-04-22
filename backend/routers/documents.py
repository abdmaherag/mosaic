import hashlib
import json
import asyncio
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, UploadFile, File, Header, HTTPException
from sse_starlette.sse import EventSourceResponse

from db.chroma_client import delete_document, get_all_chunks
from models.schemas import DocumentInfo
from services.pipeline import ingest_document
from services.retriever import invalidate_bm25_cache

router = APIRouter(prefix="/documents", tags=["documents"])


def _require_device_id(x_device_id: str = Header("")) -> str:
    if not x_device_id or not x_device_id.strip():
        raise HTTPException(status_code=400, detail="X-Device-ID header is required")
    return x_device_id.strip()

# Per-device document registry: device_id -> {doc_id -> metadata}
_devices: dict[str, dict[str, dict]] = {}
_loaded_devices: set[str] = set()
_registry_lock = asyncio.Lock()


def _rebuild_registry(device_id: str) -> dict[str, dict]:
    """Rebuild the document registry for a device from ChromaDB."""
    if device_id in _loaded_devices:
        return _devices.get(device_id, {})
    docs: dict[str, dict] = {}
    chunks = get_all_chunks(device_id=device_id)
    for chunk in chunks:
        doc_id = chunk["meta"]["doc_id"]
        if doc_id not in docs:
            docs[doc_id] = {
                "id": doc_id,
                "filename": chunk["meta"]["filename"],
                "content_type": chunk["meta"].get("content_type", "unknown"),
                "chunk_count": 0,
                "status": "ready",
                "file_hash": chunk["meta"].get("file_hash", ""),
            }
        docs[doc_id]["chunk_count"] += 1
    _devices[device_id] = docs
    _loaded_devices.add(device_id)
    return docs


MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), device_id: str = Depends(_require_device_id)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB)")

    safe_filename = PurePosixPath(file.filename).name if file.filename else "unnamed"
    file_hash = hashlib.sha256(content).hexdigest()

    async with _registry_lock:
        docs = _rebuild_registry(device_id)
    for doc in docs.values():
        if doc["filename"] == safe_filename or doc.get("file_hash") == file_hash:
            raise HTTPException(status_code=409, detail="File already uploaded.")

    async def event_stream():
        async for event in ingest_document(content, safe_filename, file.content_type, device_id):
            if event.step == "complete" and event.status == "done":
                dev_docs = _devices.setdefault(device_id, {})
                dev_docs[event.doc_id] = {
                    "id": event.doc_id,
                    "filename": safe_filename,
                    "content_type": file.content_type or "unknown",
                    "chunk_count": event.chunk_count or 0,
                    "status": "ready",
                    "file_hash": file_hash,
                }
                invalidate_bm25_cache()

            yield {
                "event": event.step,
                "data": json.dumps(event.model_dump()),
            }

    return EventSourceResponse(event_stream())


@router.get("/", response_model=list[DocumentInfo])
async def list_documents(device_id: str = Depends(_require_device_id)):
    async with _registry_lock:
        docs = _rebuild_registry(device_id)
    return [DocumentInfo(**doc) for doc in docs.values()]


@router.delete("/{doc_id}")
async def delete_doc(doc_id: str, device_id: str = Depends(_require_device_id)):
    async with _registry_lock:
        docs = _rebuild_registry(device_id)
    if doc_id not in docs:
        raise HTTPException(status_code=404, detail="Document not found")

    delete_document(doc_id, device_id=device_id)
    del docs[doc_id]
    invalidate_bm25_cache()
    return {"status": "deleted", "doc_id": doc_id}
