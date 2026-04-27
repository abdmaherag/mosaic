import threading

from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model with a lock so concurrent retrieval
    threads (e.g. multi-query fan-out via run_in_executor) don't race on
    initialization — torch's meta-tensor handling fails under that race."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
