"""
ChromaDB vector store management.
Handles persistence, incremental indexing, duplicate detection,
metadata filtering, collection statistics, and document deletion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings import get_embedder
from src.utils import config, logger, now_iso, human_size


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def _sanitise_metadata(meta: dict) -> dict:
    """ChromaDB only accepts str / int / float / bool metadata values."""
    clean: dict = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean


# ── VectorStore class ─────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin wrapper around LangChain Chroma that adds:
      - duplicate detection via file_hash
      - incremental indexing
      - per-document and per-collection management
    """

    def __init__(
        self,
        collection_name: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self._store: Optional[Chroma] = None
        self._client: Optional[chromadb.PersistentClient] = None

    # ── Initialise / connect ──────────────────────────────────────────────────

    def _get_store(self) -> Chroma:
        if self._store is None:
            embedder = get_embedder(self.embedding_model)
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embedder,
                persist_directory=config.CHROMA_PERSIST_DIR,
                collection_metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
        return self._store

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = _get_chroma_client()
        return self._client

    # ── Indexing ──────────────────────────────────────────────────────────────

    def get_indexed_hashes(self) -> set[str]:
        """Return set of file_hash values already stored in the collection."""
        try:
            store = self._get_store()
            result = store.get(include=["metadatas"])
            hashes = {m.get("file_hash", "") for m in result["metadatas"] if m}
            return hashes
        except Exception:
            return set()

    def get_indexed_sources(self) -> set[str]:
        """Return set of source filenames already indexed."""
        try:
            store = self._get_store()
            result = store.get(include=["metadatas"])
            return {m.get("source", "") for m in result["metadatas"] if m}
        except Exception:
            return set()

    def add_documents(
        self,
        chunks: list[Document],
        progress_callback=None,
        batch_size: int = 100,
    ) -> int:
        """
        Add chunks to the vector store in batches.
        Returns the number of chunks successfully added.
        """
        if not chunks:
            return 0

        store = self._get_store()
        total = len(chunks)
        added = 0

        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            # Sanitise metadata for ChromaDB
            for doc in batch:
                doc.metadata = _sanitise_metadata(doc.metadata)
            try:
                store.add_documents(batch)
                added += len(batch)
            except Exception as exc:
                logger.error(f"Batch add failed at offset {start}: {exc}")

            if progress_callback:
                progress_callback(min(start + batch_size, total), total)

        logger.info(f"Added {added}/{total} chunks to '{self.collection_name}'")
        return added

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = config.DEFAULT_TOP_K,
        filter_dict: dict | None = None,
        score_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[tuple[Document, float]]:
        """
        Return (document, score) pairs sorted by relevance.
        Score is cosine similarity in [0, 1].
        """
        store = self._get_store()
        try:
            results = store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter=filter_dict,
            )
            # Filter by threshold
            filtered = [(doc, score) for doc, score in results if score >= score_threshold]
            return filtered
        except Exception as exc:
            logger.error(f"Similarity search failed: {exc}")
            return []

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = config.DEFAULT_TOP_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_dict: dict | None = None,
    ) -> list[Document]:
        """MMR search for diversity-aware retrieval."""
        store = self._get_store()
        try:
            return store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter_dict,
            )
        except Exception as exc:
            logger.error(f"MMR search failed: {exc}")
            return []

    # ── Document management ───────────────────────────────────────────────────

    def delete_document(self, source_filename: str) -> int:
        """Delete all chunks belonging to a given source filename."""
        store = self._get_store()
        try:
            result = store.get(
                where={"source": source_filename},
                include=["metadatas"],
            )
            ids_to_delete = result.get("ids", [])
            if ids_to_delete:
                store.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for '{source_filename}'")
            return len(ids_to_delete)
        except Exception as exc:
            logger.error(f"Delete failed for '{source_filename}': {exc}")
            return 0

    def rebuild_collection(self) -> None:
        """Drop and recreate the collection (full re-index)."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' dropped for rebuild")
        except Exception:
            pass
        self._store = None  # force reconnect

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_collection_stats(self) -> dict[str, Any]:
        """Return counts, per-document breakdown, and storage info."""
        try:
            store = self._get_store()
            result = store.get(include=["metadatas"])
            metadatas = result.get("metadatas") or []
            total_chunks = len(metadatas)

            per_doc: dict[str, dict] = {}
            for meta in metadatas:
                src = meta.get("source", "unknown")
                if src not in per_doc:
                    per_doc[src] = {
                        "source": src,
                        "doc_type": meta.get("doc_type", "unknown"),
                        "chunk_count": 0,
                        "total_pages": meta.get("total_pages", "?"),
                        "upload_date": meta.get("upload_date", ""),
                        "char_count": 0,
                    }
                per_doc[src]["chunk_count"] += 1
                per_doc[src]["char_count"] += int(meta.get("char_count", 0))

            # Storage size estimate
            persist_path = Path(config.CHROMA_PERSIST_DIR)
            total_bytes = sum(
                f.stat().st_size for f in persist_path.rglob("*") if f.is_file()
            )

            return {
                "total_chunks": total_chunks,
                "total_documents": len(per_doc),
                "documents": list(per_doc.values()),
                "storage_size": human_size(total_bytes),
                "storage_bytes": total_bytes,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
            }
        except Exception as exc:
            logger.error(f"Stats collection failed: {exc}")
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "documents": [],
                "storage_size": "0 B",
                "storage_bytes": 0,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
            }

    def get_document_list(self) -> list[dict]:
        """Return a list of dicts describing each indexed document."""
        stats = self.get_collection_stats()
        return stats.get("documents", [])

    def count(self) -> int:
        """Total number of chunks in the store."""
        try:
            store = self._get_store()
            return store._collection.count()
        except Exception:
            return 0
