"""
Lightweight RAG evaluation framework (no fabricated scores).

Run:
    uv run python evaluation/evaluate_rag.py

Requires GOOGLE_API_KEY and optional indexed content via the app first,
or uses synthetic in-memory indexing for the example dataset below.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

from dotenv import load_dotenv

from rag import MultimodalRAG

load_dotenv()


@dataclass
class EvalExample:
    question: str
    expected_source_substrings: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class EvalResult:
    question: str
    latency_ms: float
    retrieval_hit: bool
    citation_hit: bool
    answer_length: int
    route: str
    notes: str = ""


EXAMPLE_DATASET: list[EvalExample] = [
    EvalExample(
        question="What are the main topics in the sample document?",
        expected_source_substrings=["Sample DocMind"],
        notes="Add sample text via evaluate_rag bootstrap.",
    ),
]


def _bootstrap_rag(rag: MultimodalRAG) -> None:
    if rag.chunk_count() > 0:
        return
    rag.add_text(
        "Sample DocMind document: DocMind AI supports PDF, URLs, images, audio, and video. "
        "It uses hybrid retrieval and source citations.",
        label="Sample DocMind Notes",
    )


def evaluate(rag: MultimodalRAG, examples: list[EvalExample]) -> list[EvalResult]:
    results: list[EvalResult] = []
    for ex in examples:
        start = time.perf_counter()
        out = rag.query(ex.question)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_labels = " ".join(
            d.metadata.get("source_label", "") for d in out.retrieved_docs
        ).lower()
        retrieval_hit = all(
            s.lower() in retrieved_labels for s in ex.expected_source_substrings
        ) if ex.expected_source_substrings else bool(out.retrieved_docs)

        answer_lower = out.answer.lower()
        citation_hit = any(
            s.lower() in answer_lower or s.lower() in retrieved_labels
            for s in ex.expected_source_substrings
        ) if ex.expected_source_substrings else True

        results.append(
            EvalResult(
                question=ex.question,
                latency_ms=round(latency_ms, 2),
                retrieval_hit=retrieval_hit,
                citation_hit=citation_hit,
                answer_length=len(out.answer),
                route=out.route,
                notes=ex.notes,
            )
        )
    return results


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY to run evaluation.")

    rag = MultimodalRAG(api_key=api_key, persist=False)
    _bootstrap_rag(rag)
    report = evaluate(rag, EXAMPLE_DATASET)
    print(json.dumps([asdict(r) for r in report], indent=2))


if __name__ == "__main__":
    main()
