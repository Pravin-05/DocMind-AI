"""Document summarization via sampled retrieval context."""

from __future__ import annotations

from google import genai
from google.genai import types

from config import GEN_MODEL
from services.gemini import GeminiServiceError, generate_text


def summarize_text(client: genai.Client, label: str, source_type: str, content: str) -> str:
    """Generate a structured summary from representative document text."""
    if not content or not content.strip():
        raise ValueError("Cannot summarize empty document content.")
    excerpt = content[:12000]
    if len(content) > 12000:
        excerpt += "\n\n[... content truncated for summarization ...]"

    prompt = (
        f"Summarize the following {source_type} document titled '{label}'.\n"
        "Include:\n"
        "1. Overview\n"
        "2. Key points (bullets)\n"
        "3. Important entities/topics\n"
        "4. Conclusion\n"
        "Use only the text below.\n\n"
        f"{excerpt}"
    )
    try:
        summary = generate_text(
            client,
            GEN_MODEL,
            [types.Part(text=prompt)],
            operation="document summarization",
        )
    except GeminiServiceError:
        raise
    if not summary:
        return "Unable to generate a summary for this document."
    return summary
