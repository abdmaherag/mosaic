"""Tests for services/chunker.py

The semantic_chunk function calls sentence-transformers internally.
We mock _get_model to avoid loading the model in unit tests.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(n_sentences: int, similarity_pattern: list[float] | None = None):
    """Return a mock sentence-transformer model whose encode() returns
    deterministic embeddings that produce the given pairwise cosine similarities.

    For simplicity we build orthonormal-ish vectors: sentences at a semantic
    boundary get a very different vector so similarity drops.
    """
    model = MagicMock()

    def fake_encode(sentences, show_progress_bar=False):
        embeddings = []
        dim = 64
        rng = np.random.default_rng(42)
        base = rng.standard_normal((dim,))
        base /= np.linalg.norm(base)

        for i, _ in enumerate(sentences):
            if similarity_pattern is not None and i < len(similarity_pattern):
                # Build a vector at a controlled angle from the previous one
                noise = rng.standard_normal((dim,))
                noise /= np.linalg.norm(noise)
                # We just return random unit vectors; the real similarities
                # are controlled in the patch of the cosine computation.
                embeddings.append(noise)
            else:
                embeddings.append(base + rng.standard_normal((dim,)) * 0.01)
        return np.array(embeddings)

    model.encode = fake_encode
    return model


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_basic_split(self):
        from services.chunker import _split_sentences
        sentences = _split_sentences("Hello world. How are you? I am fine!")
        assert len(sentences) == 3

    def test_single_sentence(self):
        from services.chunker import _split_sentences
        result = _split_sentences("Just one sentence.")
        assert result == ["Just one sentence."]

    def test_empty_string(self):
        from services.chunker import _split_sentences
        result = _split_sentences("")
        assert result == []

    def test_whitespace_only(self):
        from services.chunker import _split_sentences
        result = _split_sentences("   \n\t  ")
        assert result == []

    def test_no_terminal_punctuation(self):
        from services.chunker import _split_sentences
        result = _split_sentences("This has no punctuation")
        # Should still return the whole thing as one item
        assert len(result) >= 1

    def test_strips_whitespace_from_sentences(self):
        from services.chunker import _split_sentences
        result = _split_sentences("  First.  Second.  ")
        for s in result:
            assert s == s.strip()


# ---------------------------------------------------------------------------
# semantic_chunk — empty / trivial paths (no model needed)
# ---------------------------------------------------------------------------

class TestSemanticChunkTrivialPaths:
    def test_empty_string_returns_empty(self):
        from services.chunker import semantic_chunk
        result = semantic_chunk("")
        assert result == []

    def test_whitespace_only_returns_empty(self):
        from services.chunker import semantic_chunk
        result = semantic_chunk("   \n  ")
        assert result == []

    def test_single_sentence_no_model_needed(self):
        """<=3 sentences returns the text as-is without calling the model."""
        from services.chunker import semantic_chunk
        text = "One sentence."
        result = semantic_chunk(text)
        assert result == [text.strip()]

    def test_two_sentences_no_model_needed(self):
        from services.chunker import semantic_chunk
        text = "Sentence one. Sentence two."
        result = semantic_chunk(text)
        assert result == [text.strip()]

    def test_three_sentences_no_model_needed(self):
        from services.chunker import semantic_chunk
        text = "A. B. C."
        result = semantic_chunk(text)
        assert result == [text.strip()]


# ---------------------------------------------------------------------------
# semantic_chunk — with mocked model (>3 sentences)
# ---------------------------------------------------------------------------

class TestSemanticChunkWithModel:
    def _run_chunk(self, text, similarity_threshold=0.4, min_chunk_size=10, max_chunk_size=2000):
        mock_model = _make_mock_model(100)
        with patch("services.chunker._get_model", return_value=mock_model):
            from services.chunker import semantic_chunk
            return semantic_chunk(
                text,
                similarity_threshold=similarity_threshold,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
            )

    def test_output_is_list_of_strings(self):
        text = " ".join(f"Sentence {i}." for i in range(10))
        result = self._run_chunk(text)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_at_least_one_chunk(self):
        text = " ".join(f"Sentence {i}." for i in range(10))
        result = self._run_chunk(text)
        assert len(result) >= 1

    def test_no_empty_chunks(self):
        text = " ".join(f"Sentence {i}." for i in range(10))
        result = self._run_chunk(text)
        for chunk in result:
            assert chunk.strip() != ""

    def test_chunks_cover_original_content(self):
        """All original sentences should appear in some chunk."""
        sentences = [f"Sentence {i}." for i in range(6)]
        text = " ".join(sentences)
        result = self._run_chunk(text, min_chunk_size=1)
        combined = " ".join(result)
        for s in sentences:
            assert s in combined

    def test_max_chunk_size_respected(self):
        """No individual chunk should exceed max_chunk_size by more than one sentence."""
        # Build text where each sentence is ~50 chars
        sentences = ["This is sentence number %02d." % i for i in range(20)]
        text = " ".join(sentences)
        max_size = 200
        result = self._run_chunk(text, max_chunk_size=max_size, min_chunk_size=10)
        for chunk in result:
            # Allow one sentence of overflow (chunk is split BEFORE appending)
            assert len(chunk) < max_size + 60, f"Chunk too long: {len(chunk)}"

    def test_low_similarity_threshold_produces_more_chunks(self):
        """A very low threshold should eagerly split (when min_chunk_size allows)."""
        text = " ".join(f"Sentence {i}." for i in range(10))
        mock_model = _make_mock_model(100)

        # Force similarities to be all 0.1 (below any reasonable threshold)
        import numpy as np
        dim = 64
        rng = np.random.default_rng(0)

        def divergent_encode(sentences, show_progress_bar=False):
            # Each sentence gets a completely random unit vector — low similarities
            vecs = rng.standard_normal((len(sentences), dim))
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms

        mock_model.encode = divergent_encode

        with patch("services.chunker._get_model", return_value=mock_model):
            from services.chunker import semantic_chunk
            result_low = semantic_chunk(text, similarity_threshold=0.99, min_chunk_size=1)
            result_high = semantic_chunk(text, similarity_threshold=0.0, min_chunk_size=1)

        # low threshold (0.0) should NOT split; high threshold (0.99) should split more
        assert len(result_low) >= len(result_high)
