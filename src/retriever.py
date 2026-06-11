"""
Retrieval pipeline:
  - Dense (vector similarity) retrieval
  - Sparse (BM25) retrieval
  - Hybrid fusion (Reciprocal Rank Fusion)
  - Cross-encoder re-ranking
  - Metadata filtering
  - Confidence scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.vector_store import VectorStore
from src.utils import config, logger, truncate


@dataclass
class RetrievalResult:
    document: Document
    score: float
    rank: int
    retrieval_method: str
    source: str = ""
    page: int | str = ""
    chunk_index: int = 0

    def __post_init__(self):
        self.source = self.document.metadata.get("source", "unknown")
        self.page = self.document.metadata.get("page", "")
        self.chunk_index = self.document.metadata.get("chunk_index", 0)

    def to_dict(self) -> dict:
        return {
            "content": self.document.page_content,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "rank": self.rank,
            "method": self.retrieval_method,
            "doc_type": self.document.metadata.get("doc_type", ""),
            "section": self.document.metadata.get("section_heading", ""),
        }


@dataclass
class RetrieverConfig:
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = config.DEFAULT_TOP_K
    similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD
    rerank: bool = True
    mmr_diversity: bool = False
    mmr_lambda: float = 0.5
    metadata_filter: dict = field(default_factory=dict)
    bm25_weight: float = 0.3       # weight for BM25 in hybrid fusion
    vector_weight: float = 0.7     # weight for vector in hybrid fusion


# ── BM25 index (built on demand from current store contents) ──────────────────

class BM25Index:
    """Lightweight in-memory BM25 index over the current vector store corpus."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        if not self.bm25 or not self.documents:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        # Normalise to [0, 1]
        max_score = scores[top_indices[0]] if len(top_indices) > 0 else 1.0
        if max_score == 0:
            max_score = 1.0
        return [
            (self.documents[i], float(scores[i]) / max_score)
            for i in top_indices
            if scores[i] > 0
        ]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    weights: list[float],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """
    Fuse multiple ranked lists using weighted RRF.
    Returns deduplicated list sorted by fused score descending.
    """
    from collections import defaultdict

    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked, weight in zip(ranked_lists, weights):
        for rank, (doc, _score) in enumerate(ranked):
            doc_id = doc.page_content[:200]  # use content prefix as dedup key
            scores[doc_id] += weight * (1.0 / (k + rank + 1))
            doc_map[doc_id] = doc

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(doc_map[doc_id], scores[doc_id]) for doc_id in sorted_ids]


# ── Simple cross-encoder re-ranker (scikit-learn cosine as proxy) ─────────────

def _rerank(
    query: str,
    results: list[tuple[Document, float]],
    top_k: int,
) -> list[tuple[Document, float]]:
    """
    Re-rank using query-document cosine similarity on TF-IDF vectors.
    This is a fast proxy for a cross-encoder, avoiding extra model loading.
    """
    if len(results) <= 1:
        return results[:top_k]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = [query] + [doc.page_content for doc, _ in results]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vectorizer.fit_transform(texts)
        sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        # Blend original score (70%) with TF-IDF similarity (30%)
        reranked = [
            (doc, 0.7 * orig_score + 0.3 * float(sims[i]))
            for i, (doc, orig_score) in enumerate(results)
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
    except Exception as exc:
        logger.warning(f"Re-ranking failed: {exc}")
        return results[:top_k]


# ── Main Retriever ────────────────────────────────────────────────────────────

class Retriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vs = vector_store
        self._bm25_index: Optional[BM25Index] = None

    def _get_bm25_index(self) -> BM25Index:
        """Lazily build BM25 index from the full corpus."""
        if self._bm25_index is None:
            logger.info("Building BM25 index from corpus…")
            try:
                raw = self.vs._get_store().get(include=["documents", "metadatas"])
                docs = []
                for content, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
                    docs.append(Document(page_content=content, metadata=meta or {}))
                self._bm25_index = BM25Index(docs)
                logger.info(f"BM25 index built: {len(docs)} documents")
            except Exception as exc:
                logger.error(f"BM25 index build failed: {exc}")
                self._bm25_index = BM25Index([])
        return self._bm25_index

    def invalidate_cache(self) -> None:
        """Call after new documents are added."""
        self._bm25_index = None

    def retrieve(
        self,
        query: str,
        cfg: RetrieverConfig | None = None,
    ) -> list[RetrievalResult]:
        if cfg is None:
            cfg = RetrieverConfig()

        filter_dict = cfg.metadata_filter if cfg.metadata_filter else None
        fetch_k = max(cfg.top_k * 3, 20)

        # ── Dense retrieval ────────────────────────────────────────────────
        if cfg.mmr_diversity:
            dense_docs = self.vs.max_marginal_relevance_search(
                query=query,
                k=fetch_k,
                fetch_k=fetch_k * 2,
                lambda_mult=cfg.mmr_lambda,
                filter_dict=filter_dict,
            )
            dense_results: list[tuple[Document, float]] = [(d, 0.9) for d in dense_docs]
        else:
            dense_results = self.vs.similarity_search(
                query=query,
                k=fetch_k,
                filter_dict=filter_dict,
                score_threshold=0.0,  # apply threshold after fusion
            )

        if cfg.mode == "dense":
            combined = dense_results
            method = "dense"

        elif cfg.mode == "sparse":
            sparse_results = self._get_bm25_index().search(query, k=fetch_k)
            combined = sparse_results
            method = "sparse"

        else:  # hybrid
            sparse_results = self._get_bm25_index().search(query, k=fetch_k)
            combined = _reciprocal_rank_fusion(
                [dense_results, sparse_results],
                weights=[cfg.vector_weight, cfg.bm25_weight],
            )
            method = "hybrid"

        # ── Re-ranking ─────────────────────────────────────────────────────
        if cfg.rerank and len(combined) > 1:
            combined = _rerank(query, combined, top_k=cfg.top_k * 2)

        # ── Threshold + top-k ──────────────────────────────────────────────
        filtered = [
            (doc, score)
            for doc, score in combined
            if score >= cfg.similarity_threshold
        ]
        top = filtered[: cfg.top_k]

        results = [
            RetrievalResult(
                document=doc,
                score=score,
                rank=rank + 1,
                retrieval_method=method,
            )
            for rank, (doc, score) in enumerate(top)
        ]

        logger.info(
            f"Retrieved {len(results)} chunks for query '{truncate(query, 60)}' "
            f"(mode={cfg.mode}, rerank={cfg.rerank})"
        )
        return results
