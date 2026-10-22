"""Lightweight deterministic query routing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryRoute(str, Enum):
    DOCUMENT_QA = "document_qa"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    GENERAL = "general"


@dataclass(frozen=True)
class RouteDecision:
    route: QueryRoute
    reason: str


_SUMMARY_PATTERNS = (
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\btl;dr\b",
    r"\boverview\b",
    r"\bkey points\b",
    r"\bmain findings\b",
)

_COMPARE_PATTERNS = (
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bdifference(s)? between\b",
    r"\bvs\.?\b",
    r"\bversus\b",
)


def route_query(question: str) -> RouteDecision:
    """Classify a user question into a retrieval/generation strategy."""
    q = (question or "").strip().lower()
    if not q:
        return RouteDecision(QueryRoute.GENERAL, "empty query")

    for pat in _COMPARE_PATTERNS:
        if re.search(pat, q):
            return RouteDecision(QueryRoute.COMPARISON, f"matched comparison pattern: {pat}")

    for pat in _SUMMARY_PATTERNS:
        if re.search(pat, q):
            return RouteDecision(QueryRoute.SUMMARIZATION, f"matched summary pattern: {pat}")

    if q.endswith("?"):
        return RouteDecision(QueryRoute.DOCUMENT_QA, "question form")

    return RouteDecision(QueryRoute.GENERAL, "default document question")


def top_k_for_route(route: QueryRoute, default: int = 5) -> int:
    if route == QueryRoute.SUMMARIZATION:
        return max(default, 8)
    if route == QueryRoute.COMPARISON:
        return max(default, 10)
    return default
