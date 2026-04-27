import os
from collections.abc import AsyncGenerator

from groq import AsyncGroq

from models.schemas import SourceCitation

_client: AsyncGroq | None = None

COMPARE_SYSTEM_PROMPT = """You are an analytical assistant that compares information across multiple documents.

Formatting (STRICT — follow exactly):
1. Start with a brief synthesis paragraph
2. For each document, use a **Bold Heading** (NOT ## markdown headings)
   - Use bullet points (* ) for facts, one per line
   - End each section with source on its own line: Source: filename.pdf
3. End with a **Key Differences** section using bullet points

NEVER put source citations inline within sentences. Each bullet point must be on its own line."""

SYSTEM_PROMPT = """[System Role]
You are a helpful and expert research assistant. Your task is to provide accurate, concise answers based *only* on the provided context.

[Context]
<context>
{retrieved_documents}
</context>

[User Query]
{user_query}

[Instructions & Constraints]
1. Read the <context> carefully.
2. If the answer is not in the context, state that you do not know. Do not make up information.
3. Cite the source of information using [1] or [2] based on the context provided.
4. Keep the answer professional and concise.

[Final Output]"""

NO_ANSWER_PROMPT = """The retrieved document chunks don't appear to be relevant to the user's question.
Tell the user honestly that you couldn't find relevant information in their documents.
Suggest they try rephrasing their question or uploading more relevant documents."""


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        _client = AsyncGroq(api_key=api_key)
    return _client


def _build_context(citations: list[SourceCitation]) -> str:
    parts = []
    for i, c in enumerate(citations, 1):
        parts.append(
            f"Source: {c.document}\n"
            f"Relevance: {c.score:.2f}\n\n"
            f"{c.chunk_text}"
        )
    return "\n\n---\n\n".join(parts)


def _build_messages(
    question: str,
    citations: list[SourceCitation],
    has_relevant: bool,
    chat_history: list[dict] | None = None,
) -> list[dict]:
    if has_relevant:
        context = _build_context(citations)
    else:
        context = _build_context(citations)
        question = f"{NO_ANSWER_PROMPT}\n\nUser's question: {question}"

    system_content = SYSTEM_PROMPT.format(
        retrieved_documents=context,
        user_query=question,
    )
    messages = [{"role": "system", "content": system_content}]

    if chat_history:
        messages.extend(chat_history[-10:])

    messages.append({"role": "user", "content": question})
    return messages


async def generate_query_variants(question: str, n: int = 2) -> list[str]:
    """Generate n alternative phrasings of a question for multi-query retrieval."""
    client = _get_client()
    prompt = (
        f"Generate {n} alternative search queries for the following question. "
        f"Return only the queries, one per line, no numbering or extra text.\n\n"
        f"Question: {question}"
    )
    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=120,
    )
    raw = response.choices[0].message.content.strip()
    variants = [line.strip() for line in raw.split("\n") if line.strip()]
    return variants[:n]


async def generate(
    question: str,
    citations: list[SourceCitation],
    has_relevant: bool,
    chat_history: list[dict] | None = None,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    client = _get_client()
    messages = _build_messages(question, citations, has_relevant, chat_history)

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content


def _build_compare_context(doc_citations: dict[str, list[SourceCitation]]) -> str:
    parts = []
    for filename, citations in doc_citations.items():
        excerpts = "\n\n".join(
            f"- {c.chunk_text}" for c in citations
        )
        parts.append(f"Source: {filename}\n\n{excerpts}")
    return "\n\n---\n\n".join(parts)


async def generate_compare_stream(
    question: str,
    doc_citations: dict[str, list[SourceCitation]],
    model: str = "llama-3.3-70b-versatile",
) -> AsyncGenerator[str, None]:
    client = _get_client()
    context = _build_compare_context(doc_citations)
    messages = [
        {"role": "system", "content": COMPARE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\n\nDocument excerpts:\n\n{context}",
        },
    ]
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def generate_stream(
    question: str,
    citations: list[SourceCitation],
    has_relevant: bool,
    chat_history: list[dict] | None = None,
    model: str = "llama-3.3-70b-versatile",
) -> AsyncGenerator[str, None]:
    client = _get_client()
    messages = _build_messages(question, citations, has_relevant, chat_history)

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
