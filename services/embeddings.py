"""Gemini embedding wrapper for LangChain-compatible usage."""

from __future__ import annotations

import math

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

from config import EMBED_MODEL
from services.gemini import GeminiServiceError


class EmbeddingServiceError(GeminiServiceError):
    """Raised when Gemini cannot return a valid embedding vector."""


class GeminiEmbeddings(Embeddings):
    """Adapt Gemini's embedding endpoint to LangChain's document/query contract."""

    def __init__(self, client: genai.Client):
        self.client = client

    def _embed(self, text: str, task_type: str) -> list[float]:
        if not text or not text.strip():
            raise EmbeddingServiceError("Cannot embed empty text.")
        try:
            response = self.client.models.embed_content(
                model=EMBED_MODEL,
                contents=text.strip(),
                config=types.EmbedContentConfig(task_type=task_type),
            )
            if not response.embeddings or not response.embeddings[0].values:
                raise EmbeddingServiceError("Gemini returned an empty embedding vector.")
            vector = [float(value) for value in response.embeddings[0].values]
        except EmbeddingServiceError:
            raise
        except Exception as exc:
            raise EmbeddingServiceError(f"Gemini embedding request failed: {exc}") from exc
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingServiceError("Gemini returned a non-finite embedding value.")
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = [self._embed(text, "RETRIEVAL_DOCUMENT") for text in texts]
        if vectors and any(len(vector) != len(vectors[0]) for vector in vectors):
            raise EmbeddingServiceError("Gemini returned inconsistent embedding dimensions.")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_QUERY")
