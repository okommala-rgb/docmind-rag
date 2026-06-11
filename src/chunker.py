"""
Chunking strategies:
  1. Recursive character splitting (default, very fast)
  2. Semantic chunking (groups sentences by embedding similarity)
Both preserve page references and section hierarchy from metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import config, logger


@dataclass
class ChunkConfig:
    strategy: Literal["recursive", "semantic"] = "recursive"
    chunk_size: int = config.DEFAULT_CHUNK_SIZE
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    semantic_threshold: float = 0.85  # cosine similarity threshold for semantic merging


# ── Recursive chunker ─────────────────────────────────────────────────────────

def _recursive_chunk(docs: list[Document], cfg: ChunkConfig) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
        length_function=len,
        add_start_index=True,
    )
    chunks: list[Document] = []
    for doc in docs:
        sub_docs = splitter.split_documents([doc])
        for chunk_idx, chunk in enumerate(sub_docs):
            chunk.metadata["chunk_index"] = chunk_idx
            chunk.metadata["chunk_total"] = len(sub_docs)
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            chunks.append(chunk)
    return chunks


# ── Semantic chunker ──────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _semantic_chunk(docs: list[Document], cfg: ChunkConfig, embedder) -> list[Document]:
    """
    Split each document into sentences, embed them, then greedily merge
    adjacent sentences whose embedding similarity exceeds the threshold.
    Falls back to recursive chunking if embedder is unavailable.
    """
    if embedder is None:
        logger.warning("Semantic chunking requested but no embedder provided; falling back to recursive.")
        return _recursive_chunk(docs, cfg)

    import re

    sentence_splitter = re.compile(r"(?<=[.!?])\s+")
    chunks: list[Document] = []

    for doc in docs:
        sentences = sentence_splitter.split(doc.page_content.strip())
        if len(sentences) <= 1:
            chunks.append(doc)
            continue

        try:
            embeddings = embedder.embed_documents(sentences)
        except Exception as exc:
            logger.warning(f"Embedding failed during semantic chunking: {exc}")
            chunks.extend(_recursive_chunk([doc], cfg))
            continue

        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = _cosine_similarity(
                np.array(embeddings[i - 1]),
                np.array(embeddings[i]),
            )
            if sim >= cfg.semantic_threshold:
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])

        for chunk_idx, group in enumerate(groups):
            text = " ".join(group)
            # Respect max chunk size even in semantic mode
            if len(text) > cfg.chunk_size * 1.5:
                sub_doc = Document(page_content=text, metadata=doc.metadata.copy())
                sub_chunks = _recursive_chunk([sub_doc], cfg)
                for sc in sub_chunks:
                    sc.metadata["chunk_index"] = chunk_idx
                chunks.extend(sub_chunks)
            else:
                new_meta = doc.metadata.copy()
                new_meta["chunk_index"] = chunk_idx
                new_meta["chunk_total"] = len(groups)
                new_meta["chunk_size"] = len(text)
                new_meta["chunk_strategy"] = "semantic"
                chunks.append(Document(page_content=text, metadata=new_meta))

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_documents(
    docs: list[Document],
    cfg: ChunkConfig | None = None,
    embedder=None,
) -> list[Document]:
    """
    Split a list of Documents into chunks according to ChunkConfig.
    Returns a new list of Document chunks with augmented metadata.
    """
    if cfg is None:
        cfg = ChunkConfig()

    logger.info(
        f"Chunking {len(docs)} documents | strategy={cfg.strategy} "
        f"size={cfg.chunk_size} overlap={cfg.chunk_overlap}"
    )

    if cfg.strategy == "semantic":
        result = _semantic_chunk(docs, cfg, embedder)
    else:
        result = _recursive_chunk(docs, cfg)

    # Add consistent chunk_strategy label
    for chunk in result:
        chunk.metadata.setdefault("chunk_strategy", cfg.strategy)
        chunk.metadata.setdefault("chunk_index", 0)
        chunk.metadata.setdefault("chunk_total", 1)

    logger.info(f"Chunking complete: {len(docs)} docs → {len(result)} chunks")
    return result
