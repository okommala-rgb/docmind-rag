"""
Integration tests for VectorStore.
Uses a temporary ChromaDB directory to avoid polluting production data.
Run with: pytest tests/test_vector_store.py -v
"""

import pytest
import tempfile
import os
from pathlib import Path
from langchain_core.documents import Document


# Patch config before importing VectorStore
@pytest.fixture(autouse=True)
def temp_chroma_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma_test"))
    # Also patch the imported config object
    from src import utils
    utils.config.CHROMA_PERSIST_DIR = str(tmp_path / "chroma_test")
    (tmp_path / "chroma_test").mkdir(parents=True, exist_ok=True)
    yield tmp_path


def _make_docs(n: int = 5) -> list[Document]:
    return [
        Document(
            page_content=f"This is document number {i}. It contains unique content about topic {i}.",
            metadata={
                "source": f"doc_{i}.txt",
                "file_hash": f"hash_{i:04d}",
                "doc_type": "txt",
                "page": i,
                "total_pages": n,
                "upload_date": "2024-01-01T00:00:00",
                "char_count": 60,
            },
        )
        for i in range(1, n + 1)
    ]


class TestVectorStoreBasics:
    def test_add_and_count(self, temp_chroma_dir):
        from src.vector_store import VectorStore
        vs = VectorStore(collection_name="test_add")
        docs = _make_docs(3)
        added = vs.add_documents(docs)
        assert added == 3
        assert vs.count() == 3

    def test_similarity_search_returns_results(self, temp_chroma_dir):
        from src.vector_store import VectorStore
        vs = VectorStore(collection_name="test_search")
        docs = _make_docs(5)
        vs.add_documents(docs)
        results = vs.similarity_search("document content topic", k=3, score_threshold=0.0)
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc, _ in results)
        assert all(isinstance(score, float) for _, score in results)

    def test_search_returns_k_or_fewer(self, temp_chroma_dir):
        from src.vector_store import VectorStore
        vs = VectorStore(collection_name="test_topk")
        vs.add_documents(_make_docs(10))
        results = vs.similarity_search("unique content", k=3, score_threshold=0.0)
        assert len(results) <= 3

    def test_delete_document(self, temp_chroma_dir):
        from src.vector_store import VectorStore
        vs = VectorStore(collection_name="test_delete")
        docs = _make_docs(3)
        vs.add_documents(docs)
        count_before = vs.count()
        deleted = vs.delete_document("doc_1.txt")
        assert deleted >= 1
        assert vs.count() < count_before

    def test_get_indexed_hashes(self, temp_chroma_dir):
        from src.vector_store import VectorStore
        vs = VectorStore(collection_name="test_hashes")
        vs.add_documents(_make_docs(3))
        hashes = vs.get_indexed_hashes()
        assert "hash_0001" in hashes
        assert "hash_0002" in hashes
        assert "hash_0003" in hashes

    def test_get_indexed_sources(self, temp_chroma_dir):
        from src.vector_store import VectorStore
        vs = VectorStore(collection_name="test_sources")
        vs.add_documents(_make_docs(3))
        sources = vs.get_indexed_sources()
        assert "doc_1.txt" in sources

    def test_collection_stats(self, temp_chroma_dir):
        from src.vector_store import VectorStore
        vs = VectorStore(collection_name="test_stats")
        vs.add_documents(_make_docs(4))
        stats = vs.get_collection_stats()
        assert stats["total_chunks"] == 4
        assert stats["total_documents"] == 4
        assert "storage_size" in stats
        assert isinstance(stats["documents"], list)

    def test_rebuild_collection(self, temp_chroma_dir):
        from src.vector_store import VectorStore
        vs = VectorStore(collection_name="test_rebuild")
        vs.add_documents(_make_docs(5))
        assert vs.count() == 5
        vs.rebuild_collection()
        assert vs.count() == 0

    def test_empty_store_stats(self, temp_chroma_dir):
        from src.vector_store import VectorStore
        vs = VectorStore(collection_name="test_empty_stats")
        stats = vs.get_collection_stats()
        assert stats["total_chunks"] == 0
        assert stats["total_documents"] == 0
