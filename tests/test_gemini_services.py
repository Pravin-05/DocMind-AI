"""Offline contract tests for the Gemini and embedding adapters."""

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from services.embeddings import EmbeddingServiceError, GeminiEmbeddings
from services.gemini import (
    GeminiServiceError,
    create_gemini_client,
    response_to_text,
)


class TestGeminiClientFactory(unittest.TestCase):
    def test_uses_google_api_key_environment_variable(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-test-key"}), patch(
            "services.gemini.genai.Client"
        ) as client_factory:
            client = create_gemini_client()

        self.assertIs(client, client_factory.return_value)
        client_factory.assert_called_once_with(api_key="env-test-key")

    def test_explicit_runtime_key_is_not_saved(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("services.gemini.genai.Client") as client_factory:
            create_gemini_client("transient-key")

        client_factory.assert_called_once_with(api_key="transient-key")
        self.assertNotIn("GOOGLE_API_KEY", os.environ)

    def test_missing_key_has_actionable_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            GeminiServiceError, "GOOGLE_API_KEY"
        ):
            create_gemini_client()


class TestEmbeddingAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Mock()
        self.adapter = GeminiEmbeddings(self.client)

    def test_document_embedding_uses_document_task_type(self) -> None:
        self.client.models.embed_content.return_value.embeddings = [
            Mock(values=[0.25, -0.5, 1.0])
        ]

        vectors = self.adapter.embed_documents(["document text"])

        self.assertEqual(vectors, [[0.25, -0.5, 1.0]])
        kwargs = self.client.models.embed_content.call_args.kwargs
        self.assertEqual(kwargs["contents"], "document text")
        self.assertEqual(kwargs["config"].task_type, "RETRIEVAL_DOCUMENT")

    def test_query_embedding_uses_query_task_type(self) -> None:
        self.client.models.embed_content.return_value.embeddings = [Mock(values=[1.0, 0.0])]

        vector = self.adapter.embed_query("question")

        self.assertEqual(vector, [1.0, 0.0])
        self.assertEqual(
            self.client.models.embed_content.call_args.kwargs["config"].task_type,
            "RETRIEVAL_QUERY",
        )

    def test_rejects_empty_and_invalid_embedding_results(self) -> None:
        with self.assertRaisesRegex(EmbeddingServiceError, "empty text"):
            self.adapter.embed_query("  ")

        self.client.models.embed_content.return_value.embeddings = [Mock(values=[float("nan")])]
        with self.assertRaisesRegex(EmbeddingServiceError, "non-finite"):
            self.adapter.embed_query("valid input")

    def test_wraps_provider_errors(self) -> None:
        self.client.models.embed_content.side_effect = RuntimeError("provider unavailable")

        with self.assertRaisesRegex(EmbeddingServiceError, "embedding request failed"):
            self.adapter.embed_query("question")


class TestGeminiResponseNormalization(unittest.TestCase):
    def test_prefers_text_and_normalizes_whitespace(self) -> None:
        self.assertEqual(response_to_text(Mock(text="  hello\nworld  ")), "hello\nworld")

    def test_serializes_structured_parsed_response(self) -> None:
        response = Mock(text=None, parsed={"answer": "grounded", "sources": ["report.pdf"]})

        self.assertEqual(
            response_to_text(response),
            '{"answer": "grounded", "sources": ["report.pdf"]}',
        )

    def test_falls_back_to_candidate_text(self) -> None:
        part_a = Mock(text="first")
        part_b = Mock(text="second")
        response = Mock(
            text=None,
            parsed=None,
            candidates=[Mock(content=Mock(parts=[part_a, part_b]))],
        )

        self.assertEqual(response_to_text(response), "first\nsecond")


class TestGenerationAndSummaryAdapters(unittest.TestCase):
    def test_structured_answer_sets_json_response_configuration(self) -> None:
        from services.generation import generate_answer

        client = Mock()
        schema = {"type": "OBJECT", "properties": {"answer": {"type": "STRING"}}}
        with patch("services.generation.generate_text", return_value='{"answer":"ok"}') as generate:
            answer = generate_answer(client, [], response_schema=schema)

        self.assertEqual(answer, '{"answer":"ok"}')
        config = generate.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_schema, schema)

    def test_summary_uses_shared_generation_adapter(self) -> None:
        from services.summarization import summarize_text

        client = Mock()
        with patch("services.summarization.generate_text", return_value="Summary text") as generate:
            result = summarize_text(client, "Notes", "text", "Document body")

        self.assertEqual(result, "Summary text")
        self.assertEqual(generate.call_args.args[0], client)
        self.assertEqual(generate.call_args.kwargs["operation"], "document summarization")
        self.assertIn("Document body", generate.call_args.args[2][0].text)

    def test_summary_rejects_empty_content(self) -> None:
        from services.summarization import summarize_text

        with self.assertRaisesRegex(ValueError, "empty document content"):
            summarize_text(Mock(), "Notes", "text", "  ")


if __name__ == "__main__":
    unittest.main()