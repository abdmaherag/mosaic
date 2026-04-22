import re

import numpy as np

from services.embedder import _get_model


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(text: str, similarity_threshold: float = 0.4, min_chunk_size: int = 100, max_chunk_size: int = 2000) -> list[str]:
    """Split text into chunks based on semantic similarity between consecutive sentence groups."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    if len(sentences) <= 3:
        return [text.strip()] if text.strip() else []

    model = _get_model()
    embeddings = model.encode(sentences, show_progress_bar=False)

    # Compare consecutive sentences using cosine similarity
    similarities = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        similarities.append(float(cos_sim))

    # Find split points where similarity drops below threshold
    chunks = []
    current_chunk: list[str] = [sentences[0]]

    for i, sim in enumerate(similarities):
        current_text = " ".join(current_chunk)

        if sim < similarity_threshold and len(current_text) >= min_chunk_size:
            chunks.append(current_text)
            current_chunk = [sentences[i + 1]]
        elif len(current_text) >= max_chunk_size:
            chunks.append(current_text)
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        remaining = " ".join(current_chunk)
        if chunks and len(remaining) < min_chunk_size:
            chunks[-1] = chunks[-1] + " " + remaining
        else:
            chunks.append(remaining)

    return chunks
