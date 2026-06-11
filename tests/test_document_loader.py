"""
Unit tests for document_loader module.
Run with: pytest tests/test_document_loader.py -v
"""

import pytest
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from src.document_loader import load_document


def _write_temp(suffix: str, content: bytes | str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    if isinstance(content, str):
        tmp.write(content.encode("utf-8"))
    else:
        tmp.write(content)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


class TestTxtLoader:
    def test_loads_plain_text(self):
        path = _write_temp(".txt", "Hello world.\n\nSecond paragraph here.")
        docs = load_document(path)
        assert len(docs) >= 1
        full_text = " ".join(d.page_content for d in docs)
        assert "Hello world" in full_text

    def test_metadata_fields(self):
        path = _write_temp(".txt", "Some content.\n\nAnother block.")
        docs = load_document(path)
        for doc in docs:
            assert "source" in doc.metadata
            assert "doc_type" in doc.metadata
            assert doc.metadata["doc_type"] == "txt"
            assert "upload_date" in doc.metadata
            assert "file_hash" in doc.metadata

    def test_empty_file(self):
        path = _write_temp(".txt", "")
        docs = load_document(path)
        assert docs == []

    def test_large_text(self):
        large = ("word " * 1000 + "\n\n") * 20
        path = _write_temp(".txt", large)
        docs = load_document(path)
        assert len(docs) >= 1
        total_chars = sum(len(d.page_content) for d in docs)
        assert total_chars > 1000


class TestMarkdownLoader:
    def test_loads_markdown(self):
        md = "# Title\n\nSome **bold** text.\n\n## Section\n\nMore content here."
        path = _write_temp(".md", md)
        docs = load_document(path)
        assert len(docs) >= 1
        full = " ".join(d.page_content for d in docs)
        assert "Title" in full

    def test_doc_type_markdown(self):
        path = _write_temp(".md", "# Test\nContent")
        docs = load_document(path)
        if docs:
            assert docs[0].metadata["doc_type"] == "markdown"


class TestHtmlLoader:
    def test_strips_html_tags(self):
        html = "<html><body><h1>Title</h1><p>Hello world</p></body></html>"
        path = _write_temp(".html", html)
        docs = load_document(path)
        full = " ".join(d.page_content for d in docs)
        assert "<html>" not in full
        assert "Hello world" in full

    def test_strips_scripts(self):
        html = "<html><body><script>alert('x')</script><p>Clean text</p></body></html>"
        path = _write_temp(".html", html)
        docs = load_document(path)
        full = " ".join(d.page_content for d in docs)
        assert "alert" not in full
        assert "Clean text" in full


class TestPdfLoader:
    def test_unsupported_extension_raises(self):
        path = _write_temp(".xyz", b"binary data")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(path)

    def test_file_hash_consistent(self):
        content = b"consistent content for hashing test"
        path1 = _write_temp(".txt", content)
        path2 = _write_temp(".txt", content)
        docs1 = load_document(path1)
        docs2 = load_document(path2)
        if docs1 and docs2:
            assert docs1[0].metadata["file_hash"] == docs2[0].metadata["file_hash"]

    def test_different_content_different_hash(self):
        path1 = _write_temp(".txt", "Content A unique 12345")
        path2 = _write_temp(".txt", "Content B unique 67890")
        docs1 = load_document(path1)
        docs2 = load_document(path2)
        if docs1 and docs2:
            assert docs1[0].metadata["file_hash"] != docs2[0].metadata["file_hash"]
