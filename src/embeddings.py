"""
Embeddings: wraps sentence-transformers via LangChain interface.
Provides batch embedding with progress reporting and a simple in-memory cache.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings

from src.utils import config, logger

# ── Module-level singleton ────────────────────────────────────────────────────
_embedder_lock = threading.Lock()
_current_model_name: str = ""
_embedder_instance: Optional[HuggingFaceEmbeddings] = None


def get_embedder(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """
    Return (and cache) a HuggingFaceEmbeddings instance.
    Re-initialises if a different model_name is requested.
    """
    global _current_model_name, _embedder_instance

    model = model_name or config.EMBEDDING_MODEL

    with _embedder_lock:
        if _embedder_instance is None or _current_model_name != model:
            logger.info(f"Loading embedding model: {model}")
            _embedder_instance = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 64,
                    "show_progress_bar": False,
                },
            )
            _current_model_name = model
            logger.info(f"Embedding model ready: {model}")

    return _embedder_instance


def embed_texts_batch(
    texts: list[str],
    model_name: str | None = None,
    progress_callback=None,
    batch_size: int = 64,
) -> list[list[float]]:
    """
    Embed texts in batches, optionally reporting progress.
    progress_callback(done, total) is called after each batch.
    """
    embedder = get_embedder(model_name)
    all_embeddings: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
        if progress_callback:
            progress_callback(min(start + batch_size, total), total)

    logger.info(f"Embedded {total} texts with model '{_current_model_name}'")
    return all_embeddings


def embed_query(text: str, model_name: str | None = None) -> list[float]:
    """Embed a single query string."""
    return get_embedder(model_name).embed_query(text)
