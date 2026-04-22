import io
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from PyPDF2 import PdfReader


def parse_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(pages)


def parse_docx(content: bytes) -> str:
    doc = DocxDocument(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def parse_html(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def parse_text(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


PARSERS = {
    "application/pdf": parse_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": parse_docx,
    "text/html": parse_html,
    "text/plain": parse_text,
    "text/markdown": parse_text,
    "text/csv": parse_text,
}

EXTENSION_MAP = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".py": "text/plain",
    ".js": "text/plain",
    ".ts": "text/plain",
    ".json": "text/plain",
}


def parse_document(content: bytes, filename: str, content_type: str | None = None) -> str:
    if not content_type or content_type == "application/octet-stream":
        ext = Path(filename).suffix.lower()
        content_type = EXTENSION_MAP.get(ext)

    parser = PARSERS.get(content_type)
    if not parser:
        raise ValueError(f"Unsupported file type: {content_type} ({filename})")

    return parser(content)
