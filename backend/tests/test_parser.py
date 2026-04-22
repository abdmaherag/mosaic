"""Tests for services/parser.py"""
import io
import struct
import pytest

from services.parser import (
    parse_text,
    parse_html,
    parse_document,
    PARSERS,
    EXTENSION_MAP,
)


# ---------------------------------------------------------------------------
# parse_text
# ---------------------------------------------------------------------------

class TestParseText:
    def test_basic_utf8(self):
        result = parse_text(b"Hello, world!")
        assert result == "Hello, world!"

    def test_empty_bytes(self):
        result = parse_text(b"")
        assert result == ""

    def test_multiline(self):
        content = b"line one\nline two\nline three"
        result = parse_text(content)
        assert result == "line one\nline two\nline three"

    def test_unicode_text(self):
        text = "Caf\u00e9 au lait \u00e9t\u00e9"
        result = parse_text(text.encode("utf-8"))
        assert result == text

    def test_invalid_utf8_replaced(self):
        # bytes that are not valid UTF-8 should be replaced, not crash
        result = parse_text(b"valid \xff\xfe invalid")
        assert "valid" in result  # partial content survives

    def test_large_content(self):
        content = ("word " * 10_000).encode("utf-8")
        result = parse_text(content)
        assert result.count("word") == 10_000


# ---------------------------------------------------------------------------
# parse_html
# ---------------------------------------------------------------------------

class TestParseHtml:
    def test_basic_paragraph(self):
        html = b"<html><body><p>Hello world</p></body></html>"
        result = parse_html(html)
        assert "Hello world" in result

    def test_strips_script_tags(self):
        html = b"<html><body><p>Content</p><script>alert('xss')</script></body></html>"
        result = parse_html(html)
        assert "alert" not in result
        assert "Content" in result

    def test_strips_style_tags(self):
        html = b"<html><head><style>body{color:red}</style></head><body><p>Text</p></body></html>"
        result = parse_html(html)
        assert "color" not in result
        assert "Text" in result

    def test_strips_nav_footer_header(self):
        html = (
            b"<html><body>"
            b"<nav>Navigation</nav>"
            b"<header>Header</header>"
            b"<main><p>Main content</p></main>"
            b"<footer>Footer</footer>"
            b"</body></html>"
        )
        result = parse_html(html)
        assert "Navigation" not in result
        assert "Header" not in result
        assert "Footer" not in result
        assert "Main content" in result

    def test_empty_html(self):
        result = parse_html(b"<html><body></body></html>")
        assert isinstance(result, str)

    def test_plain_text_html(self):
        result = parse_html(b"just text no tags")
        assert "just text no tags" in result

    def test_nested_tags(self):
        html = b"<div><p><strong>Bold</strong> and <em>italic</em></p></div>"
        result = parse_html(html)
        assert "Bold" in result
        assert "italic" in result


# ---------------------------------------------------------------------------
# parse_document — dispatch logic
# ---------------------------------------------------------------------------

class TestParseDocument:
    def test_dispatch_by_content_type_text(self):
        result = parse_document(b"plain text", "file.txt", "text/plain")
        assert result == "plain text"

    def test_dispatch_by_content_type_html(self):
        html = b"<p>Hello</p>"
        result = parse_document(html, "page.html", "text/html")
        assert "Hello" in result

    def test_dispatch_by_extension_when_no_content_type(self):
        result = parse_document(b"markdown content", "README.md", None)
        assert result == "markdown content"

    def test_dispatch_by_extension_with_octet_stream(self):
        result = parse_document(b"csv data", "data.csv", "application/octet-stream")
        assert result == "csv data"

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document(b"data", "file.xyz", "application/xyz")

    def test_unsupported_extension_no_content_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document(b"data", "file.xyz", None)

    def test_markdown_extension_maps_to_text_parser(self):
        result = parse_document(b"# heading\ncontent", "doc.md", None)
        assert "# heading" in result

    def test_py_extension_maps_to_text_parser(self):
        result = parse_document(b"def foo(): pass", "script.py", None)
        assert "def foo" in result

    def test_js_extension_maps_to_text_parser(self):
        result = parse_document(b"const x = 1;", "app.js", None)
        assert "const x" in result

    def test_json_extension_maps_to_text_parser(self):
        result = parse_document(b'{"key": "value"}', "data.json", None)
        assert '"key"' in result


# ---------------------------------------------------------------------------
# EXTENSION_MAP / PARSERS completeness
# ---------------------------------------------------------------------------

class TestMappings:
    def test_all_extension_map_values_have_parsers(self):
        for ext, mime in EXTENSION_MAP.items():
            assert mime in PARSERS, f"Extension {ext} maps to {mime} which has no parser"

    def test_extension_map_has_expected_keys(self):
        expected = {".pdf", ".docx", ".html", ".htm", ".txt", ".md", ".csv", ".py", ".js", ".ts", ".json"}
        assert expected.issubset(set(EXTENSION_MAP.keys()))
