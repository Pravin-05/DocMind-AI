"""Shared Google Gemini client creation and response normalization."""

from __future__ import annotations

import json
import os
from typing import Any

from google import genai


class GeminiServiceError(RuntimeError):
    """A configuration or provider error raised by the Gemini adapter layer."""


def create_gemini_client(api_key: str | None = None) -> genai.Client:
    """Create a client from an explicit runtime value or GOOGLE_API_KEY."""
    configured_key = (api_key or os.getenv("GOOGLE_API_KEY", "")).strip()
    if not configured_key:
        raise GeminiServiceError("Google API key is required. Set GOOGLE_API_KEY.")
    try:
        return genai.Client(api_key=configured_key)
    except Exception as exc:
        raise GeminiServiceError(f"Could not initialize the Gemini client: {exc}") from exc


def response_to_text(response: Any) -> str:
    """Extract text from standard or structured Gemini response shapes."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, str):
            return parsed.strip()
        if hasattr(parsed, "model_dump"):
            parsed = parsed.model_dump()
        try:
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(parsed)

    candidate_text: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                candidate_text.append(part_text.strip())
    return "\n".join(candidate_text)


def generate_text(
    client: genai.Client,
    model: str,
    contents: Any,
    *,
    operation: str,
    config: Any = None,
) -> str:
    """Run one Gemini generation request and return normalized response text."""
    try:
        kwargs: dict[str, Any] = {"model": model, "contents": contents}
        if config is not None:
            kwargs["config"] = config
        response = client.models.generate_content(**kwargs)
    except Exception as exc:
        raise GeminiServiceError(f"Gemini {operation} failed: {exc}") from exc
    return response_to_text(response)