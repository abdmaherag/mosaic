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
        """<=3 sentences returns the text without calling the model.
        Use prose-shaped input so the noise filter doesn't drop it."""
        from services.chunker import semantic_chunk
        text = "Machine learning is a useful technique for data analysis."
        result = semantic_chunk(text)
        assert result == [text.strip()]

    def test_two_sentences_no_model_needed(self):
        from services.chunker import semantic_chunk
        text = "Machine learning is a useful technique. We can apply it to many problems."
        result = semantic_chunk(text)
        assert result == [text.strip()]

    def test_three_sentences_no_model_needed(self):
        from services.chunker import semantic_chunk
        text = (
            "Machine learning is a useful technique. "
            "We can apply it to many problems. "
            "It generalizes from training data to new examples."
        )
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

    # The chunker's noise filter drops short header-like fragments such as
    # "Sentence 0." The tests below use prose-shaped inputs that survive
    # the filter while still exercising the chunking logic.
    @staticmethod
    def _make_prose(n: int) -> list[str]:
        """Build n prose-shaped sentences that pass the noise filter
        (each has a verb and >= 4 words)."""
        return [
            f"the model is described by parameter number {i} in the example."
            for i in range(n)
        ]

    def test_output_is_list_of_strings(self):
        text = " ".join(self._make_prose(10))
        result = self._run_chunk(text)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_at_least_one_chunk(self):
        text = " ".join(self._make_prose(10))
        result = self._run_chunk(text)
        assert len(result) >= 1

    def test_no_empty_chunks(self):
        text = " ".join(self._make_prose(10))
        result = self._run_chunk(text)
        for chunk in result:
            assert chunk.strip() != ""

    def test_chunks_cover_original_content(self):
        """All original sentences should appear in some chunk."""
        sentences = self._make_prose(6)
        text = " ".join(sentences)
        result = self._run_chunk(text, min_chunk_size=1)
        combined = " ".join(result)
        for s in sentences:
            assert s in combined

    def test_max_chunk_size_respected(self):
        """No individual chunk should exceed max_chunk_size by more than one sentence."""
        # Each sentence ~50 chars and noise-filter-safe.
        sentences = ["this is sentence number %02d in the example list." % i for i in range(20)]
        text = " ".join(sentences)
        max_size = 200
        result = self._run_chunk(text, max_chunk_size=max_size, min_chunk_size=10)
        for chunk in result:
            # Allow one sentence of overflow (chunk is split BEFORE appending)
            assert len(chunk) < max_size + 60, f"Chunk too long: {len(chunk)}"

    def test_low_similarity_threshold_produces_more_chunks(self):  # noqa: E501
        pass  # placeholder to keep diff small — original test below

    def test_noise_filter_drops_serp_chunks(self):
        """Sentence-level filter should drop SERP/Amazon screenshot text."""
        from services.chunker import _is_noise_sentence
        serp = "ai.stanford.edu/~nilsson/mlbook.html - 15k - Cached - Similar pages"
        amazon = "List Price:$87.47 Average Customer Review (30 customer reviews) 5 star: (23)"
        prose = "Machine learning can appear in many guises and we now formalize the problems."
        assert _is_noise_sentence(serp) is True
        assert _is_noise_sentence(amazon) is True
        assert _is_noise_sentence(prose) is False

    def test_noise_filter_keeps_prose_with_one_url(self):
        """A single citation URL inside real prose must NOT be flagged."""
        from services.chunker import _is_noise_chunk
        ok = (
            "In their seminal paper available at https://arxiv.org/abs/1909.12345 "
            "the authors demonstrate that gradient descent converges under mild "
            "assumptions, which extends earlier work in this area."
        )
        assert _is_noise_chunk(ok) is False

    def test_noise_filter_drops_link_dense_chunk(self):
        """A chunk packed with URLs + Cached markers should be dropped."""
        from services.chunker import _is_noise_chunk
        bad = (
            "www.springer.com/computer/artificial/journal/10994 - 39k - Cached - "
            "Similar pageshunch.net/ - 94k - Cached - Similar pages"
            "www.amazon.com/Machine-Learning - 210k - Cached"
        )
        assert _is_noise_chunk(bad) is True

    def test_filter_drops_course_syllabus_text(self):
        """Course-page noise (CS229, Autumn YYYY, Announcements) should drop."""
        from services.chunker import _is_noise_sentence
        assert _is_noise_sentence("CS229 Machine Learning Autumn 2007.") is True
        assert _is_noise_sentence("Announcements.") is True
        assert _is_noise_sentence(
            "Final reports from this year's class projects have been posted here."
        ) is True

    def test_filter_drops_title_case_headers(self):
        """Title-case fragments without verbs are dropped (TOCs, cover pages)."""
        from services.chunker import _is_noise_sentence
        assert _is_noise_sentence("INTRODUCTION TO MACHINE LEARNING") is True
        assert _is_noise_sentence("1.1 A Taste of Machine Learning") is True
        assert _is_noise_sentence(
            "Introduction Density Estimation Graphical Models Kernels Optimization"
        ) is True

    def test_filter_keeps_short_math_sentences(self):
        """Short math/notation sentences without verbs must NOT be dropped."""
        from services.chunker import _is_noise_sentence
        assert _is_noise_sentence("f(x) = x^2 + 1.") is False
        assert _is_noise_sentence("Let X be a random variable.") is False

    def test_filter_drops_amazon_book_description(self):
        """Amazon product-page section markers (Book Info, Editorial Reviews)."""
        from services.chunker import _is_noise_sentence
        assert _is_noise_sentence(
            "Book InfoPresents the key algorithms and theory of machine learning."
        ) is True
        assert _is_noise_sentence(
            "Editorial Reviews Book Description This exciting addition to McGraw-Hill."
        ) is True


class TestSemanticChunkLowSimilarityFlag:
    """Original parametric test for similarity-threshold splitting."""
    def test_low_similarity_threshold_produces_more_chunks(self):
        """A very low threshold should eagerly split (when min_chunk_size allows)."""
        text = " ".join(
            f"the model is described by parameter number {i} in the example."
            for i in range(10)
        )
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
