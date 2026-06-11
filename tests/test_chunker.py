"""
Unit tests for the chunker module.
Run with: pytest tests/test_chunker.py -v
"""

import pytest
from langchain_core.documents import Document
from src.chunker import ChunkConfig, chunk_documents


def _make_doc(text: str, page: int = 1, source: str = "test.pdf") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "page": page, "doc_type": "pdf", "upload_date": "2024-01-01"},
    )


class TestRecursiveChunking:
    def test_basic_split(self):
        doc = _make_doc("word " * 400)  # ~2000 chars
        cfg = ChunkConfig(strategy="recursive", chunk_size=500, chunk_overlap=50)
        chunks = chunk_documents([doc], cfg)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= cfg.chunk_size + 50  # small tolerance

    def test_metadata_preserved(self):
        doc = _make_doc("Hello world. " * 100, page=3, source="sample.pdf")
        cfg = ChunkConfig(strategy="recursive", chunk_size=300, chunk_overlap=30)
        chunks = chunk_documents([doc], cfg)
        for chunk in chunks:
            assert chunk.metadata["source"] == "sample.pdf"
            assert chunk.metadata["page"] == 3

    def test_chunk_index_set(self):
        doc = _make_doc("sentence. " * 200)
        cfg = ChunkConfig(strategy="recursive", chunk_size=200, chunk_overlap=20)
        chunks = chunk_documents([doc], cfg)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices[0] == 0
        assert all(isinstance(i, int) for i in indices)

    def test_empty_document(self):
        doc = _make_doc("")
        cfg = ChunkConfig(strategy="recursive")
        chunks = chunk_documents([doc], cfg)
        # Empty doc may produce 0 or 1 chunk — both acceptable
        assert isinstance(chunks, list)

    def test_single_short_doc(self):
        doc = _make_doc("Short text.")
        cfg = ChunkConfig(strategy="recursive", chunk_size=1000)
        chunks = chunk_documents([doc], cfg)
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."

    def test_multiple_documents(self):
        docs = [_make_doc("text " * 200, page=i) for i in range(1, 6)]
        cfg = ChunkConfig(strategy="recursive", chunk_size=300, chunk_overlap=30)
        chunks = chunk_documents(docs, cfg)
        sources = {c.metadata["page"] for c in chunks}
        assert sources == {1, 2, 3, 4, 5}

    def test_overlap_creates_shared_content(self):
        long_text = "The quick brown fox jumps over the lazy dog. " * 100
        doc = _make_doc(long_text)
        cfg = ChunkConfig(strategy="recursive", chunk_size=200, chunk_overlap=100)
        chunks = chunk_documents([doc], cfg)
        if len(chunks) >= 2:
            # Last chars of chunk[0] should appear somewhere in chunk[1]
            end_of_first = chunks[0].page_content[-50:].strip()
            assert end_of_first in chunks[1].page_content or len(end_of_first) == 0


class TestChunkConfig:
    def test_default_config(self):
        cfg = ChunkConfig()
        assert cfg.strategy == "recursive"
        assert cfg.chunk_size > 0
        assert cfg.chunk_overlap >= 0
        assert cfg.chunk_overlap < cfg.chunk_size

    def test_custom_config(self):
        cfg = ChunkConfig(strategy="semantic", chunk_size=800, chunk_overlap=100)
        assert cfg.strategy == "semantic"
        assert cfg.chunk_size == 800


class TestChunkMetadata:
    def test_strategy_label(self):
        doc = _make_doc("Testing metadata. " * 50)
        cfg = ChunkConfig(strategy="recursive", chunk_size=200)
        chunks = chunk_documents([doc], cfg)
        for c in chunks:
            assert c.metadata.get("chunk_strategy") == "recursive"

    def test_chunk_size_recorded(self):
        doc = _make_doc("A " * 500)
        cfg = ChunkConfig(strategy="recursive", chunk_size=200, chunk_overlap=20)
        chunks = chunk_documents([doc], cfg)
        for c in chunks:
            assert "chunk_size" in c.metadata
            assert c.metadata["chunk_size"] > 0
