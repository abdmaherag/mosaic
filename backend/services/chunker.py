import re

import numpy as np

from services.embedder import _get_model

# Strip [Page N] markers injected by the PDF parser before chunking.
# These belong in metadata, not in chunk text — they pollute BM25 tokens
# and add noise to embeddings.
_PAGE_MARKER = re.compile(r'\[Page \d+\]\s*')

# SERP / e-commerce screenshot boilerplate that PDF parsers occasionally
# extract from figures (a Google search result captured as an illustrative
# example, an Amazon product page, etc.). When ingested as prose these chunks
# pollute retrieval — querying the corpus surfaces the screenshot text as if
# it were the author's definition. We do TWO passes:
#   1. Sentence-level: drop sentences with strong noise signals before they
#      ever reach the embedder/chunker.
#   2. Chunk-level: catch any residual noisy chunks the sentence pass missed
#      (e.g. when noise sentences glued together via run-on punctuation).
# We deliberately do NOT strip noise markers inline — keeping them visible
# is what lets the scorer recognize the surrounding text as a screenshot.
_URL_RE = re.compile(
    r'(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|edu|gov|au|uk|de|jp)\b/?\S*',
    re.IGNORECASE,
)
_PRICE_RE = re.compile(r'\$\d{1,4}(?:[.,]\d{2})?')
_SIZE_CACHED_RE = re.compile(r'\b\d{1,4}\s*k\b\s*-\s*Cached', re.IGNORECASE)
_STAR_RE = re.compile(r'\b\d+\s+star\s*[:!]', re.IGNORECASE)
# Phrases that almost-always indicate UI/SERP/e-commerce/syllabus text rather than prose.
_NOISE_PHRASES = re.compile(
    r'\b('
    # SERP
    r'Cached|Similar\s+pages?|Sponsored\s+Links?|Sponsored|'
    r'Advanced\s+Search\s+Preferences|Search\s+within\s+results|'
    r'Results\s+\d+\s*-\s*\d+\s+of|Search\s+Tips|Try\s+Google|'
    # E-commerce / Amazon product page
    r'Customer\s+Reviews?|customer\s+reviews?|Was\s+this\s+review\s+helpful|'
    r'Add\s+to\s+Cart|Buy\s+Together|Buy\s+this\s+book|1-Click|Wish\s+Lists?|'
    r'Gift\s+Cards?|List\s+Price|Sales\s+Rank|Average\s+Customer\s+Review|'
    r'Sign\s+in\s+to|Recently\s+Viewed\s+Products|This\s+text\s+refers\s+to|'
    r'Hardcover\s+edition|Paperback\s+edition|ISBN-1[03]|Product\s+Dimensions|'
    r'Shipping\s+Weight|Listmania|Look\s+for\s+Similar\s+Items|All\s+Editions|'
    r'helpful\s+to\s+you\?|Permalink|Report\s+this|Editorial\s*Reviews?|'
    r'Book\s*Description|Book\s*Info(?:Presents)?|Inside\s+This\s+Book|'
    r'Kindle\s+Books?|Better\s+Together|Buy\s+now\s+with\s+1-Click|'
    r'See\s+all\s+\d+\s+(?:customer\s+)?reviews?|'
    r'Customers\s+Who\s+Bought|See\s+all\s+my\s+reviews|'
    r'reference\s+tool\s+for\s+software\s+developers|'
    # Course syllabus / academic-page boilerplate
    r'(?:Autumn|Fall|Spring|Winter|Summer)\s+(?:19|20)\d{2}|'
    r'CS\s*\d{2,4}\b|EE\s*\d{2,4}\b|MATH\s*\d{2,4}\b|'
    r'Final\s+reports?|Lecture\s+notes?|Class\s+projects?|Course\s+syllabus|'
    r'Office\s+hours?|Problem\s+sets?|Homework\s+\d|TA\s+sessions?|'
    r'pointers?\s+to\s+(?:my\s+)?(?:draft\s+)?book|individual\s*chapters?|'
    r'Adobe\s+Acrobat\s+format|Announcements?(?:\.|$)'
    r')\b',
    re.IGNORECASE,
)

# Common English verbs/copulas — a sentence with at least one is structurally
# likely to be prose rather than a header/title/fragment list. Cheap heuristic
# but works against course-page noise like "CS229 Machine Learning Autumn 2007.
# Announcements. Final reports from this year's class projects have been..."
_VERB_RE = re.compile(
    r'\b(?:is|are|was|were|be|been|being|am|'
    r'has|have|had|do|does|did|done|'
    r'can|could|should|would|will|may|might|must|shall|'
    r'use|uses|used|using|make|makes|made|making|'
    r'show|shows|shown|showed|showing|find|finds|found|finding|'
    r'see|sees|saw|seen|give|gives|given|gave|giving|'
    r'take|takes|took|taken|taking|need|needs|needed|needing|'
    r'allow|allows|allowed|consider|considers|considered|'
    r'represent|represents|represented|represents|denote|denotes|denoted|'
    r'define|defines|defined|defining|describe|describes|described|'
    r'become|becomes|became|becoming|appear|appears|appeared|appearing|'
    r'concern|concerns|concerned|involve|involves|involved|'
    r'reduce|reduces|reduced|reducing|provide|provides|provided|providing|'
    r'solve|solves|solved|solving'
    # Note: "learn/learning" intentionally excluded — overwhelmingly used as
    # nouns ("machine learning", "deep learning") in this corpus.
    r')\b',
    re.IGNORECASE,
)


def _signal_count(text: str) -> tuple[int, int, int, int, int]:
    """Return (urls, phrases, prices, stars, size_marks) for a text segment."""
    return (
        len(_URL_RE.findall(text)),
        len(_NOISE_PHRASES.findall(text)),
        len(_PRICE_RE.findall(text)),
        1 if _STAR_RE.search(text) else 0,
        1 if _SIZE_CACHED_RE.search(text) else 0,
    )


def _is_header_fragment(s: str) -> bool:
    """Detect header/title fragments dressed up as sentences:
        - "CS229 Machine Learning Autumn 2007."
        - "Announcements."
        - "Introduction Density Estimation Graphical Models Kernels Optimization"
        - "INTRODUCTION TO MACHINE LEARNING"
    Conservative: only kill obvious title-case fragments. Math-only short
    sentences (lowercase, no verb) are kept — they're rare and not worth the
    false-positive risk against real prose.
    """
    words = s.split()
    n = len(words)
    if n < 4:
        # 1-3 word "sentences" are almost always headers, captions or list items.
        return True
    has_verb = bool(_VERB_RE.search(s))
    if has_verb:
        return False
    # No verb. Distinguish title-case-headers from non-header fragments by
    # capitalized-word ratio. Real English prose averages ~1 capped word per
    # sentence (just the first word), so cap_ratio in prose is roughly 1/n.
    caps = sum(1 for w in words if w[:1].isupper())
    cap_ratio = caps / n
    return cap_ratio > 0.5


def _is_noise_sentence(s: str) -> bool:
    """Sentence-level kill: pre-chunk filter to drop SERP/UI/header fragments
    before they hit the embedder or get glued onto adjacent prose chunks."""
    urls, phrases, prices, stars, size_marks = _signal_count(s)
    score = 0
    if urls >= 1: score += 1
    if phrases >= 1: score += 2
    if prices >= 1: score += 1
    score += stars * 2
    score += size_marks * 2
    # Very short sentences with any noise marker are almost always UI fragments.
    if score >= 1 and len(s.split()) < 8:
        score += 1
    if score >= 2:
        return True
    # Structural fragment kill — catches headers/syllabus lines that have no
    # SERP markers but aren't prose either.
    return _is_header_fragment(s)


def _noise_score(chunk: str) -> int:
    """Chunk-level score: higher = more screenshot/UI-noise, less prose.

        +3  3+ URLs in the chunk  (SERP / link-list style)
        +1  a single URL  (one citation in real prose is fine)
        +3  3+ noise phrases  (Cached / Customer Reviews / List Price / ...)
        +1  1-2 noise phrases
        +2  2+ price patterns
        +1  1 price pattern
        +2  any star-rating token
        +1  any "39k - Cached"-style size annotation
    """
    urls, phrases, prices, stars, size_marks = _signal_count(chunk)
    score = 0
    if urls >= 3: score += 3
    elif urls >= 1: score += 1
    if phrases >= 3: score += 3
    elif phrases >= 1: score += 1
    if prices >= 2: score += 2
    elif prices >= 1: score += 1
    score += stars * 2
    score += size_marks * 1
    return score


def _is_noise_chunk(chunk: str, threshold: int = 3) -> bool:
    """Drop chunks whose noise signal dominates. Threshold of 3 means a chunk
    needs at least two independent noise indicators (e.g. 3+ URLs alone, or
    1 URL + 1 noise phrase + 1 price) to be discarded."""
    return _noise_score(chunk) >= threshold


def _clean_text(text: str) -> str:
    return _PAGE_MARKER.sub('', text)


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.65,
    min_chunk_size: int = 100,
    max_chunk_size: int = 800,
    overlap_sentences: int = 1,
) -> list[str]:
    """Split text into semantically coherent chunks.

    Splits when cosine similarity between consecutive sentences drops below
    similarity_threshold (higher = more splits = smaller, more focused chunks).
    overlap_sentences carries the tail of each chunk into the next to preserve
    cross-boundary context.
    """
    text = _clean_text(text)
    sentences = _split_sentences(text)
    # Sentence-level noise kill — drop SERP/UI fragments before they reach
    # the embedder or get glued onto adjacent prose at chunk boundaries.
    sentences = [s for s in sentences if not _is_noise_sentence(s)]
    if not sentences:
        return []

    if len(sentences) <= 3:
        joined = " ".join(sentences).strip()
        return [joined] if joined else []

    model = _get_model()
    embeddings = model.encode(sentences, show_progress_bar=False)

    similarities = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        similarities.append(float(cos_sim))

    chunks = []
    current_chunk: list[str] = [sentences[0]]

    for i, sim in enumerate(similarities):
        current_text = " ".join(current_chunk)

        if sim < similarity_threshold and len(current_text) >= min_chunk_size:
            chunks.append(current_text)
            overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk = overlap + [sentences[i + 1]]
        elif len(current_text) >= max_chunk_size:
            chunks.append(current_text)
            overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk = overlap + [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        remaining = " ".join(current_chunk)
        if chunks and len(remaining) < min_chunk_size:
            chunks[-1] = chunks[-1] + " " + remaining
        else:
            chunks.append(remaining)

    # Hard cap fallback: a single "sentence" can run hundreds of chars when
    # the upstream parser failed to detect periods (math-heavy PDFs are the
    # common case). The semantic loop only checks length BEFORE appending,
    # so it can't split such mega-sentences. Force a character-window split
    # at word boundaries so no chunk grossly exceeds max_chunk_size.
    hard_cap = int(max_chunk_size * 1.5)
    capped: list[str] = []
    for c in chunks:
        if len(c) <= hard_cap:
            capped.append(c)
            continue
        words = c.split()
        cur, cur_len = [], 0
        for w in words:
            if cur_len + len(w) + 1 > max_chunk_size and cur:
                capped.append(" ".join(cur))
                cur, cur_len = [], 0
            cur.append(w)
            cur_len += len(w) + 1
        if cur:
            capped.append(" ".join(cur))

    # Drop chunks dominated by SERP / e-commerce screenshot text (Cached,
    # customer reviews, prices, dense URL lists). The semantic chunker has
    # already grouped them away from real prose, so dropping is safe.
    return [c for c in capped if not _is_noise_chunk(c)]
