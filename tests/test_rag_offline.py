"""Offline unit tests (no API key required)."""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from services.retrieval import keyword_overlap_score
from services.routing import QueryRoute, route_query


class TestQueryRouting(unittest.TestCase):
    def test_summarization_route(self) -> None:
        decision = route_query("Please summarize the document")
        self.assertEqual(decision.route, QueryRoute.SUMMARIZATION)

    def test_comparison_route(self) -> None:
        decision = route_query("Compare section A vs section B")
        self.assertEqual(decision.route, QueryRoute.COMPARISON)


class TestKeywordOverlap(unittest.TestCase):
    def test_overlap_score(self) -> None:
        score = keyword_overlap_score("machine learning models", "machine learning is useful")
        self.assertGreater(score, 0.0)

    def test_empty_query(self) -> None:
        self.assertEqual(keyword_overlap_score("", "text"), 0.0)


class TestRAGErrorHandling(unittest.TestCase):
    def test_missing_api_key(self) -> None:
        from rag import MultimodalRAG, RAGError

        with self.assertRaises(RAGError):
            MultimodalRAG(api_key="", persist=False)

    def test_empty_text_ingest(self) -> None:
        from rag import MultimodalRAG, RAGError

        rag = MultimodalRAG(api_key="test-key-not-used", persist=False)
        with self.assertRaises(RAGError):
            rag.add_text("   ")


class TestRAGOfflinePipeline(unittest.TestCase):
    def test_gemini_generation_wrapper_uses_configured_model(self) -> None:
        from config import GEN_MODEL
        from rag import generate_answer

        client = Mock()
        client.models.generate_content.return_value.text = "  Grounded answer.  "

        answer = generate_answer(client, [])

        self.assertEqual(answer, "Grounded answer.")
        client.models.generate_content.assert_called_once()
        self.assertEqual(client.models.generate_content.call_args.kwargs["model"], GEN_MODEL)

    def test_text_ingestion_retrieval_and_answer_wiring(self) -> None:
        from rag import MultimodalRAG
        from services.embeddings import GeminiEmbeddings

        with (
            patch.object(GeminiEmbeddings, "embed_documents", return_value=[[1.0, 0.0, 0.0]]),
            patch.object(GeminiEmbeddings, "embed_query", return_value=[1.0, 0.0, 0.0]),
            patch("rag.generate_answer", return_value="The notes discuss project milestones."),
        ):
            rag = MultimodalRAG(api_key="test-key-not-used", persist=False)
            chunks = rag.add_text(
                "Project milestones include planning, implementation, and review.",
                label="Project notes",
            )
            result = rag.query("What are the project milestones?")

        self.assertEqual(chunks, 1)
        self.assertEqual(result.answer, "The notes discuss project milestones.")
        self.assertEqual(result.citations[0].source_label, "Project notes")
        self.assertEqual(result.retrieved_docs[0].metadata["source_type"], "text")

    def test_index_commit_keeps_source_catalog_consistent_on_embedding_failure(self) -> None:
        from rag import MultimodalRAG, RAGError
        from services.embeddings import EmbeddingServiceError

        rag = MultimodalRAG(api_key="test-key-not-used", persist=False)
        with (
            patch.object(rag.embeddings, "embed_documents", side_effect=EmbeddingServiceError("offline")),
            self.assertRaisesRegex(RAGError, "Embedding failed"),
        ):
            rag.add_text("This source must not enter the index.", label="Failed source")

        self.assertEqual(rag.chunk_count(), 0)
        self.assertEqual(rag.source_count(), 0)

    def test_index_commit_does_not_register_source_after_vector_write_failure(self) -> None:
        from rag import MultimodalRAG, RAGError
        from services.embeddings import GeminiEmbeddings

        with patch.object(GeminiEmbeddings, "embed_documents", return_value=[[0.5, 0.5]]):
            rag = MultimodalRAG(api_key="test-key-not-used", persist=False)
            with (
                patch.object(rag._collection, "add", side_effect=RuntimeError("disk unavailable")),
                self.assertRaisesRegex(RAGError, "Could not save source"),
            ):
                rag.add_text("A vector write should fail.", label="Unstored source")

        self.assertEqual(rag.chunk_count(), 0)
        self.assertEqual(rag.source_count(), 0)

    def test_nonpersistent_clients_use_distinct_temporary_chroma_stores(self) -> None:
        import chromadb
        from rag import MultimodalRAG

        first = MultimodalRAG(api_key="test-key-not-used", persist=False)
        second = MultimodalRAG(api_key="test-key-not-used", persist=False)
        first._collection.add(
            ids=["isolated-record"],
            embeddings=[[0.1, 0.2]],
            documents=["only in the first client"],
            metadatas=[{"source_label": "isolated"}],
        )
        first_database = first._ephemeral_database_name
        second_database = second._ephemeral_database_name

        self.assertNotEqual(first_database, second_database)
        self.assertNotEqual(first._collection_name, second._collection_name)
        self.assertEqual(first.chunk_count(), 1)
        self.assertEqual(second.chunk_count(), 0)
        admin = first._ephemeral_admin

        first.close()
        second.close()

        with self.assertRaises(chromadb.errors.NotFoundError):
            admin.get_database(name=first_database)
        with self.assertRaises(chromadb.errors.NotFoundError):
            admin.get_database(name=second_database)

    def test_persistent_client_keeps_configured_storage_path(self) -> None:
        from config import CHROMA_PERSIST_DIR
        from rag import MultimodalRAG

        with patch("rag.chromadb.PersistentClient") as persistent_client:
            persistent_client.return_value.get_or_create_collection.return_value.count.return_value = 0
            rag = MultimodalRAG(api_key="test-key-not-used", persist=True)

        persistent_client.assert_called_once_with(path=str(CHROMA_PERSIST_DIR))
        rag.close()

    def test_chunker_preserves_text_and_source_metadata(self) -> None:
        from rag import MultimodalRAG

        with patch("rag.CHUNK_SIZE", 40), patch("rag.CHUNK_OVERLAP", 8):
            chunks = MultimodalRAG._split_text(
                "First sentence carries context. Second sentence adds details. Third sentence closes.",
                {"source_id": "source-1", "source_type": "text"},
            )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.page_content) <= 40 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["source_id"] == "source-1" for chunk in chunks))
        self.assertIn("Third sentence closes.", " ".join(chunk.page_content for chunk in chunks))

    def test_html_extractor_omits_non_content_regions(self) -> None:
        from rag import _PageTextParser

        parser = _PageTextParser()
        parser.feed(
            "<header>Site title</header><main><h1>Report</h1><p>Useful evidence.</p>"
            "<script>secret text</script><nav>Menu item</nav></main>"
        )

        extracted = parser.get_text()
        self.assertIn("Report", extracted)
        self.assertIn("Useful evidence.", extracted)
        self.assertNotIn("secret text", extracted)
        self.assertNotIn("Menu item", extracted)
        self.assertNotIn("Site title", extracted)

    def test_url_ingestion_extracts_and_indexes_visible_text(self) -> None:
        from rag import MultimodalRAG
        from services.embeddings import GeminiEmbeddings

        response = type(
            "PageResponse",
            (),
            {
                "text": "<nav>Menu</nav><main><h1>Research</h1><p>Evidence content.</p></main>",
                "raise_for_status": Mock(),
            },
        )()
        with (
            patch("rag.httpx.get", return_value=response),
            patch.object(GeminiEmbeddings, "embed_documents", return_value=[[1.0, 0.0, 0.0]]),
        ):
            rag = MultimodalRAG(api_key="test-key-not-used", persist=False)
            count = rag.add_url("https://example.test/research")
            stored = rag._collection.get(include=["documents", "metadatas"])

        self.assertEqual(count, 1)
        self.assertIn("Research", stored["documents"][0])
        self.assertIn("Evidence content.", stored["documents"][0])
        self.assertNotIn("Menu", stored["documents"][0])
        self.assertEqual(stored["metadatas"][0]["source_type"], "url")

    def test_pdf_ingestion_keeps_page_and_source_metadata(self) -> None:
        from rag import MultimodalRAG

        fake_page = Document(page_content="A test page has extracted content.", metadata={"page": 0})
        with TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "sample.pdf"
            pdf_path.touch()
            with (
                patch("rag.PyPDFLoader") as loader,
                patch.object(MultimodalRAG, "_ingest", return_value=1) as ingest,
            ):
                loader.return_value.load.return_value = [fake_page]
                rag = MultimodalRAG(api_key="test-key-not-used", persist=False)
                count = rag.add_pdf(str(pdf_path))

        self.assertEqual(count, 1)
        prepared_source = ingest.call_args.args[0]
        self.assertEqual(prepared_source.source_type, "pdf")
        self.assertEqual(prepared_source.segments[0].page, 0)
        self.assertTrue(prepared_source.source_id)
        self.assertEqual(prepared_source.pages, 1)

    def test_citations_deduplicate_same_source_page(self) -> None:
        from rag import MultimodalRAG

        docs = [
            Document(page_content="chunk one", metadata={"source_label": "report.pdf", "source_type": "pdf", "page": 0}),
            Document(page_content="chunk two", metadata={"source_label": "report.pdf", "source_type": "pdf", "page": 0}),
            Document(page_content="another page", metadata={"source_label": "report.pdf", "source_type": "pdf", "page": 1}),
        ]

        citations = MultimodalRAG._build_citations(docs)

        self.assertEqual([citation.page for citation in citations], [1, 2])
        self.assertEqual(citations[0].excerpt, "chunk one")

    def test_generation_context_attaches_media_once_and_cites_pages(self) -> None:
        from rag import MultimodalRAG

        docs = [
            Document(
                page_content="first excerpt",
                metadata={
                    "source_label": "diagram.png",
                    "source_type": "image",
                    "file_uri": "gs://media/diagram",
                    "mime_type": "image/png",
                },
            ),
            Document(
                page_content="second excerpt",
                metadata={
                    "source_label": "diagram.png",
                    "source_type": "image",
                    "file_uri": "gs://media/diagram",
                    "mime_type": "image/png",
                },
            ),
            Document(page_content="page excerpt", metadata={"source_label": "notes.pdf", "source_type": "pdf", "page": 2}),
        ]
        rag = MultimodalRAG(api_key="test-key-not-used", persist=False)

        parts = rag._generation_parts("What is shown?", docs, QueryRoute.DOCUMENT_QA)

        media_parts = [part for part in parts if part.file_data]
        prompt_text = "\n".join(part.text or "" for part in parts)
        self.assertEqual(len(media_parts), 1)
        self.assertIn("notes.pdf (Page 3)", prompt_text)
        self.assertIn("first excerpt", prompt_text)
        self.assertIn("page excerpt", prompt_text)

    def test_media_ingestion_records_uri_and_type(self) -> None:
        from rag import MultimodalRAG
        from services.embeddings import GeminiEmbeddings

        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "diagram.png"
            image_path.write_bytes(b"not parsed because media analysis is mocked")
            with (
                patch.object(MultimodalRAG, "_upload_media", return_value="gs://media/diagram"),
                patch.object(MultimodalRAG, "_describe_media", return_value="A process diagram."),
                patch.object(GeminiEmbeddings, "embed_documents", return_value=[[1.0, 0.0, 0.0]]),
            ):
                rag = MultimodalRAG(api_key="test-key-not-used", persist=False)
                count = rag.add_image(str(image_path))

        stored = rag._collection.get(include=["metadatas"])["metadatas"][0]
        self.assertEqual(count, 1)
        self.assertEqual(stored["file_uri"], "gs://media/diagram")
        self.assertEqual(stored["mime_type"], "image/png")
        self.assertEqual(rag.sources[0].source_type, "image")

    def test_reset_clears_index_and_rebuilds_catalog(self) -> None:
        from rag import MultimodalRAG
        from services.embeddings import GeminiEmbeddings

        with patch.object(GeminiEmbeddings, "embed_documents", return_value=[[1.0, 0.0, 0.0]]):
            rag = MultimodalRAG(api_key="test-key-not-used", persist=False)
            rag.add_text("The catalog tracks this source.", label="Catalog notes")
            rag.sources.clear()
            rag._restore_catalog()
            self.assertEqual(rag.source_count(), 1)
            self.assertEqual(rag.sources[0].label, "Catalog notes")
            rag.reset_knowledge_base()

        self.assertEqual(rag.chunk_count(), 0)
        self.assertEqual(rag.source_count(), 0)
        self.assertEqual(rag.insights()["queries"], 0)


if __name__ == "__main__":
    unittest.main()
