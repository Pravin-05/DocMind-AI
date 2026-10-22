"""Hybrid vector + keyword retrieval with optional reranking."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from langchain_core.documents import Document

from config import (
    HYBRID_KEYWORD_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
)


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def keyword_overlap_score(query: str, document_text: str) -> float:
    """Simple BM25-style keyword overlap in [0, 1]."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    d_tokens = _tokenize(document_text)
    if not d_tokens:
        return 0.0
    overlap = len(q_tokens & d_tokens)
    return overlap / len(q_tokens)


@dataclass
class ScoredDocument:
    document: Document
    vector_score: float
    keyword_score: float
    fused_score: float


class HybridRetriever:
    """Combine Chroma vector search with keyword overlap fusion."""

    def __init__(
        self,
        vector_weight: float = HYBRID_VECTOR_WEIGHT,
        keyword_weight: float = HYBRID_KEYWORD_WEIGHT,
    ):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def retrieve(
        self,
        collection,
        query_embedding: list[float],
        question: str,
        top_k: int,
    ) -> list[ScoredDocument]:
        total = collection.count()
        if total == 0:
            return []

        n_candidates = min(total, max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k))
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )

        docs_raw = results["documents"][0]
        metas_raw = results["metadatas"][0]
        dists_raw = results["distances"][0]

        scored: list[ScoredDocument] = []
        for text, meta, dist in zip(docs_raw, metas_raw, dists_raw):
            # Cosine distance in Chroma: lower is better → convert to similarity
            vector_sim = 1.0 / (1.0 + float(dist))
            kw = keyword_overlap_score(question, text)
            fused = self.vector_weight * vector_sim + self.keyword_weight * kw
            scored.append(
                ScoredDocument(
                    document=Document(page_content=text, metadata=meta or {}),
                    vector_score=vector_sim,
                    keyword_score=kw,
                    fused_score=fused,
                )
            )

        scored.sort(key=lambda s: s.fused_score, reverse=True)
        return scored[:top_k]


class Reranker:
    """Modular reranker — default uses fused hybrid scores with length penalty."""

    def rerank(
        self,
        question: str,
        candidates: list[ScoredDocument],
        top_k: int,
    ) -> list[ScoredDocument]:
        if not candidates:
            return []

        reranked: list[tuple[float, ScoredDocument]] = []
        q_len = max(len(question.split()), 1)
        for item in candidates:
            text = item.document.page_content
            # Mild preference for chunks that are informative but not huge
            length_factor = 1.0 - min(0.15, len(text) / 8000)
            relevance_boost = 1.0 + 0.1 * math.log1p(item.keyword_score * q_len)
            score = item.fused_score * length_factor * relevance_boost
            reranked.append((score, item))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in reranked[:top_k]]
