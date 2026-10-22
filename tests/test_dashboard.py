"""Offline dashboard workflow checks using a fake RAG service boundary."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from rag import RAGResult, SourceCitation


class FakeRAG:
    def __init__(self, api_key: str | None = None):
        self.sources: list[SimpleNamespace] = []
        self.query_count = 0

    def insights(self) -> dict[str, int]:
        return {
            "documents": len(self.sources),
            "pages": sum(source.pages or 0 for source in self.sources),
            "chunks": sum(source.chunks for source in self.sources),
            "queries": self.query_count,
        }

    def add_text(self, text: str, label: str = "Pasted text") -> int:
        self.sources.append(
            SimpleNamespace(
                label=label,
                source_id=f"fake-{len(self.sources) + 1}",
                source_type="text",
                chunks=1,
                pages=None,
                status="indexed",
                file_uri="",
                mime_type="",
            )
        )
        return 1

    def summarize_source(self, source_index: int) -> str:
        return f"Summary for {self.sources[source_index].label}"

    def query(self, question: str) -> RAGResult:
        self.query_count += 1
        document = Document(
            page_content="Recorded evidence passage.",
            metadata={"source_label": "Test Notes", "source_type": "text"},
        )
        return RAGResult(
            answer="Answer grounded in Test Notes.",
            question=question,
            retrieved_docs=[document],
            route="document_qa",
            citations=[
                SourceCitation(
                    source_label="Test Notes",
                    source_type="text",
                    excerpt="Recorded evidence passage.",
                )
            ],
            latency_ms=12.0,
        )

    def reset_knowledge_base(self) -> None:
        self.sources.clear()


class TestDashboardWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patcher = patch.dict(os.environ, {"GOOGLE_API_KEY": "offline-test-key"})
        self.env_patcher.start()
        self.rag_patcher = patch("services.dashboard.MultimodalRAG", FakeRAG)
        self.rag_patcher.start()

        self.app = AppTest.from_file("app.py").run(timeout=40)
        self.app.session_state["auth_user"] = {
            "id": 1,
            "full_name": "Dashboard Tester",
            "email": "dashboard@example.test",
        }
        self.app.run(timeout=40)

    def tearDown(self) -> None:
        self.rag_patcher.stop()
        self.env_patcher.stop()

    def test_upload_chat_sources_feedback_summary_and_session_controls(self) -> None:
        self.app.selectbox(key="upload_kind").select("Text").run(timeout=40)
        self.app.text_area(key="upload_text").set_value("Project details for verification.").run(timeout=40)
        self.app.text_input(key="upload_text_label").set_value("Test Notes").run(timeout=40)
        self.app.button(key="dashboard_index_text").click().run(timeout=40)

        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state.rag.sources[0].label, "Test Notes")

        self.app.chat_input[0].set_value("What do the notes say?").run(timeout=40)
        self.assertFalse(self.app.exception)
        rendered = "\n".join(element.value for element in self.app.markdown)
        self.assertIn("Answer grounded in Test Notes.", rendered)
        self.assertIn("Test Notes", rendered)
        self.assertTrue(any(button.key and button.key.startswith("feedback_up_") for button in self.app.button))
        self.assertEqual(self.app.session_state.rag.query_count, 1)
        self.app.button(key="feedback_up_1").click().run(timeout=40)
        self.assertEqual(self.app.session_state.feedback[1], "up")

        self.app.radio(key="dashboard_view").set_value("Documents").run(timeout=40)
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Generate summary" in button.label for button in self.app.button))
        self.app.button(key="dashboard_summary_0").click().run(timeout=40)
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Summary for Test Notes" in element.value for element in self.app.markdown))

        self.app.button(key="dashboard_new_conversation").click().run(timeout=40)
        self.assertFalse(self.app.session_state.messages)
        self.assertFalse(self.app.session_state.feedback)

        self.app.button(key="dashboard_logout").click().run(timeout=40)
        self.assertFalse(self.app.exception)
        self.assertIsNone(self.app.session_state["auth_user"])
        self.assertTrue(any("Understand Your Documents with AI" in element.value for element in self.app.markdown))


if __name__ == "__main__":
    unittest.main()