import os

import chromadb

_client: chromadb.ClientAPI | None = None
COLLECTION_NAME = "documents"
_DATA_PATH = os.getenv("CHROMA_DATA_PATH", "./chroma_data")


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=_DATA_PATH)
    return _client


def get_collection() -> chromadb.Collection:
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    doc_id: str,
    filename: str,
    chunks: list[str],
    embeddings: list[list[float]],
    file_hash: str = "",
    device_id: str = "",
) -> int:
    collection = get_collection()
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "filename": filename, "chunk_index": i, "file_hash": file_hash, "device_id": device_id}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(ids)


def query_chunks(
    query_embedding: list[float],
    n_results: int = 5,
    doc_ids: list[str] | None = None,
    device_id: str = "",
    metadata_filter: dict | None = None,
) -> dict:
    collection = get_collection()
    conditions = []
    if device_id:
        conditions.append({"device_id": device_id})
    if doc_ids:
        conditions.append({"doc_id": {"$in": doc_ids}})
    if metadata_filter:
        conditions.append(metadata_filter)
    if len(conditions) > 1:
        where = {"$and": conditions}
    elif conditions:
        where = conditions[0]
    else:
        where = None
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def delete_document(doc_id: str, device_id: str = "") -> None:
    collection = get_collection()
    conditions = [{"doc_id": doc_id}]
    if device_id:
        conditions.append({"device_id": device_id})
    where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
    collection.delete(where=where)


def get_document_count(doc_id: str) -> int:
    collection = get_collection()
    result = collection.get(where={"doc_id": doc_id}, include=[])
    return len(result["ids"])


def reset_collection() -> None:
    """Delete and recreate the collection, wiping all stored documents."""
    client = get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    get_collection()


def get_all_chunks(device_id: str = "") -> list[dict]:
    """Return chunks for BM25 index construction, filtered by device."""
    collection = get_collection()
    where = {"device_id": device_id} if device_id else None
    result = collection.get(where=where, include=["documents", "metadatas"])
    if not result["ids"]:
        return []
    return [
        {"id": cid, "text": doc, "meta": meta}
        for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]
