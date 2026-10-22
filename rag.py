"""DocMind AI's document-to-evidence pipeline and public RAG interface."""

from __future__ import annotations

import mimetypes
import time
import uuid
import weakref
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import chromadb
import httpx
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    GEN_MODEL,
    MEDIA_ANALYSIS_INSTRUCTIONS,
    MAX_PDF_PAGES,
    MIME_MAP,
    SOURCE_ICONS,
    VECTOR_TOP_K,
)
from services.embeddings import GeminiEmbeddings
from services.gemini import GeminiServiceError, create_gemini_client, generate_text
from services.generation import build_multimodal_parts, generate_answer
from services.retrieval import HybridRetriever, Reranker
from services.routing import QueryRoute, route_query, top_k_for_route
from services.summarization import summarize_text

__all__ = [
    "SOURCE_ICONS",
    "MultimodalRAG",
    "RAGResult",
    "IndexedSource",
    "SourceCitation",
]


class RAGError(Exception):
    """A pipeline failure that can be reported safely in the DocMind UI."""


@dataclass
class IndexedSource:
    label: str
    source_type: str
    chunks: int
    file_uri: str = ""
    mime_type: str = ""
    pages: int | None = None
    status: str = "indexed"
    source_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class SourceCitation:
    source_label: str
    source_type: str
    page: int | None = None
    excerpt: str = ""


@dataclass
class RAGResult:
    answer: str
    question: str
    retrieved_docs: list[Document] = field(default_factory=list)
    route: str = QueryRoute.GENERAL.value
    citations: list[SourceCitation] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass(frozen=True)
class _TextSegment:
    """Extracted text plus optional source-local page location."""

    text: str
    page: int | None = None


@dataclass(frozen=True)
class _PreparedSource:
    """Format-neutral extraction output passed to the indexing pipeline."""

    label: str
    source_type: str
    segments: tuple[_TextSegment, ...]
    source_id: str
    pages: int | None = None
    file_uri: str = ""
    mime_type: str = ""


class _PageTextParser(HTMLParser):
    """Collect readable page text while excluding non-content regions."""

    _HIDDEN_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript"}
    _BOUNDARY_TAGS = {"article", "br", "div", "h1", "h2", "h3", "li", "p", "section"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_level = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        if tag in self._HIDDEN_TAGS:
            self._hidden_level += 1
        elif not self._hidden_level and tag in self._BOUNDARY_TAGS:
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._HIDDEN_TAGS and self._hidden_level:
            self._hidden_level -= 1
        elif not self._hidden_level and tag in self._BOUNDARY_TAGS:
            self._parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._hidden_level:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(" ".join(self._parts).split())


def _close_ephemeral_chroma(database: Any, admin: Any, database_name: str) -> None:
    """Close an isolated in-memory client and remove its temporary database."""
    database.close()
    admin.delete_database(name=database_name)


class MultimodalRAG:
    """Manage extraction, indexing, evidence selection, and grounded answers."""

    def __init__(self, api_key: str | None = None, persist: bool = True):
        try:
            self.client = create_gemini_client(api_key)
        except GeminiServiceError as exc:
            raise RAGError(str(exc)) from exc

        self.embeddings = GeminiEmbeddings(self.client)
        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self._collection_name = COLLECTION_NAME if persist else f"{COLLECTION_NAME}_{uuid.uuid4().hex}"
        self._ephemeral_database_name: str | None = None
        self._ephemeral_admin: Any = None
        self._ephemeral_cleanup: weakref.finalize | None = None
        try:
            if persist:
                CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
                self._database = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
            else:
                self._ephemeral_admin = chromadb.AdminClient()
                self._ephemeral_database_name = f"docmind_test_{uuid.uuid4().hex}"
                self._ephemeral_admin.create_database(name=self._ephemeral_database_name)
                self._database = chromadb.EphemeralClient(database=self._ephemeral_database_name)
            self._collection = self._database.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            if self._ephemeral_database_name is not None:
                self._ephemeral_cleanup = weakref.finalize(
                    self,
                    _close_ephemeral_chroma,
                    self._database,
                    self._ephemeral_admin,
                    self._ephemeral_database_name,
                )
        except Exception as exc:
            database = getattr(self, "_database", None)
            if database is not None:
                database.close()
            if self._ephemeral_database_name is not None and self._ephemeral_admin is not None:
                self._ephemeral_admin.delete_database(name=self._ephemeral_database_name)
            self._ephemeral_database_name = None
            raise RAGError(f"Could not initialize the document index: {exc}") from exc

        self.sources: list[IndexedSource] = []
        self.query_count = 0
        if persist and self._collection.count():
            self._restore_catalog()

    def close(self) -> None:
        """Release Chroma resources and remove this instance's temporary namespace."""
        cleanup = self._ephemeral_cleanup
        if cleanup is not None and cleanup.alive:
            cleanup()
            self._ephemeral_cleanup = None
            self._ephemeral_database_name = None
            return
        if self._ephemeral_database_name is None:
            self._database.close()

    def __enter__(self) -> MultimodalRAG:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # Public ingestion interface ---------------------------------------------

    def add_text(self, text: str, label: str = "Pasted text") -> int:
        body = self._require_text(text, "Text content is empty.")
        return self._ingest(
            _PreparedSource(
                label=(label or "").strip() or "Pasted text",
                source_type="text",
                segments=(_TextSegment(body),),
                source_id=str(uuid.uuid4()),
            )
        )

    def add_url(self, url: str) -> int:
        target = (url or "").strip()
        if not target:
            raise RAGError("URL is empty.")
        page = self._download_page(target)
        extracted = self._extract_page_text(page)
        if not extracted:
            raise RAGError("No readable text found at this URL.")
        return self._ingest(
            _PreparedSource(
                label=target,
                source_type="url",
                segments=(_TextSegment(extracted),),
                source_id=str(uuid.uuid4()),
            )
        )

    def add_pdf(self, file_path: str) -> int:
        path = Path(file_path)
        if not path.is_file():
            raise RAGError("PDF file not found.")
        pages = self._extract_pdf_pages(path)
        segments = tuple(
            _TextSegment(text=page_text, page=page_index)
            for page_index, page_text in enumerate(pages)
            if page_text.strip()
        )
        if not segments:
            raise RAGError("PDF contains no extractable text.")
        return self._ingest(
            _PreparedSource(
                label=path.name,
                source_type="pdf",
                segments=segments,
                source_id=str(uuid.uuid4()),
                pages=len(pages),
            )
        )

    def add_image(self, file_path: str) -> int:
        return self._add_media_file(file_path, "image")

    def add_audio(self, file_path: str) -> int:
        return self._add_media_file(file_path, "audio")

    def add_video(self, file_path: str) -> int:
        return self._add_media_file(file_path, "video")

    def summarize_source(self, source_index: int) -> str:
        if not 0 <= source_index < len(self.sources):
            raise RAGError("Invalid document selection.")
        source = self.sources[source_index]
        text = self._fetch_source_text(source)
        if not text:
            raise RAGError("No indexed content found for this document.")
        try:
            return summarize_text(self.client, source.label, source.source_type, text)
        except Exception as exc:
            raise RAGError(f"Could not summarize document: {exc}") from exc

    # Public query/index interface -------------------------------------------

    def query(self, question: str, top_k: int = VECTOR_TOP_K) -> RAGResult:
        began = time.perf_counter()
        self.query_count += 1
        prompt = (question or "").strip()
        if not prompt:
            return RAGResult("Please enter a question about your documents.", question or "")
        if self.chunk_count() == 0:
            return RAGResult(
                "No documents are indexed yet. Upload or add sources in the sidebar, then ask your question.",
                prompt,
            )

        decision = route_query(prompt)
        desired = top_k_for_route(decision.route, top_k)
        try:
            query_vector = self.embeddings.embed_query(prompt)
            pool = self.retriever.retrieve(
                self._collection,
                query_vector,
                prompt,
                max(desired, 1) * 2,
            )
            ranked = self.reranker.rerank(prompt, pool, max(desired, 1))
            evidence = [entry.document for entry in ranked]
        except Exception as exc:
            raise RAGError(f"Evidence retrieval failed: {exc}") from exc

        elapsed = (time.perf_counter() - began) * 1000
        if not evidence:
            return RAGResult(
                answer="No relevant passages were found. Try rephrasing or adding more documents.",
                question=prompt,
                route=decision.route.value,
                latency_ms=elapsed,
            )

        try:
            request_parts = self._generation_parts(prompt, evidence, decision.route)
            answer_text = generate_answer(self.client, request_parts)
        except Exception as exc:
            raise RAGError(f"Answer generation failed: {exc}") from exc

        return RAGResult(
            answer=answer_text,
            question=prompt,
            retrieved_docs=evidence,
            route=decision.route.value,
            citations=self._build_citations(evidence),
            latency_ms=(time.perf_counter() - began) * 1000,
        )

    def reset_knowledge_base(self) -> None:
        try:
            self._database.delete_collection(self._collection_name)
            self._collection = self._database.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise RAGError(f"Could not reset the document index: {exc}") from exc
        self.sources.clear()
        self.query_count = 0

    def chunk_count(self) -> int:
        return self._collection.count()

    def source_count(self) -> int:
        return len(self.sources)

    def total_pages(self) -> int:
        return sum(source.pages or 0 for source in self.sources if source.source_type == "pdf")

    def insights(self) -> dict[str, int]:
        return {
            "documents": self.source_count(),
            "pages": self.total_pages(),
            "chunks": self.chunk_count(),
            "queries": self.query_count,
        }

    # Extraction and indexing stages -----------------------------------------

    @staticmethod
    def _require_text(value: str, message: str) -> str:
        normalized = " ".join((value or "").split())
        if not normalized:
            raise RAGError(message)
        return normalized

    @staticmethod
    def _download_page(url: str) -> str:
        try:
            response = httpx.get(
                url,
                timeout=20,
                follow_redirects=True,
                headers={"User-Agent": "DocMindAI/1.0"},
            )
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as exc:
            raise RAGError(f"Could not fetch URL: {exc}") from exc

    @staticmethod
    def _extract_page_text(markup: str) -> str:
        parser = _PageTextParser()
        try:
            parser.feed(markup)
            parser.close()
        except Exception as exc:
            raise RAGError(f"Could not extract page text: {exc}") from exc
        return " ".join(parser.get_text().split())

    @staticmethod
    def _extract_pdf_pages(path: Path) -> list[str]:
        try:
            loaded_pages = PyPDFLoader(str(path)).load()
        except Exception as exc:
            raise RAGError(f"Failed to read PDF: {exc}") from exc
        if not loaded_pages:
            raise RAGError("PDF contains no extractable pages.")
        return [page.page_content or "" for page in loaded_pages[:MAX_PDF_PAGES]]

    def _add_media_file(self, file_path: str, media_kind: str) -> int:
        path = Path(file_path)
        if not path.is_file():
            raise RAGError(f"{media_kind.title()} file not found.")
        media_type = self._media_type_for_path(path)
        remote_uri = self._upload_media(path, media_type)
        extracted_text = self._describe_media(remote_uri, media_type, media_kind)
        return self._ingest(
            _PreparedSource(
                label=path.name,
                source_type=media_kind,
                segments=(_TextSegment(extracted_text),),
                source_id=str(uuid.uuid4()),
                file_uri=remote_uri,
                mime_type=media_type,
            )
        )

    @staticmethod
    def _media_type_for_path(path: Path) -> str:
        extension = path.suffix.lower()
        return MIME_MAP.get(extension, mimetypes.guess_type(path.name)[0] or "application/octet-stream")

    def _upload_media(self, path: Path, mime_type: str) -> str:
        try:
            item = self.client.files.upload(
                file=str(path),
                config=types.UploadFileConfig(mime_type=mime_type),
            )
            deadline = time.monotonic() + 120
            while item.state.name == "PROCESSING":
                if time.monotonic() >= deadline:
                    raise RAGError("File processing timed out on Gemini File API.")
                time.sleep(1)
                item = self.client.files.get(name=item.name)
            if item.state.name != "ACTIVE" or not item.uri:
                raise RAGError(f"File processing failed with state: {item.state.name}")
            return item.uri
        except RAGError:
            raise
        except Exception as exc:
            raise RAGError(f"Could not upload media to Gemini: {exc}") from exc

    def _describe_media(self, uri: str, mime_type: str, media_kind: str) -> str:
        instruction = MEDIA_ANALYSIS_INSTRUCTIONS.get(media_kind)
        if instruction is None:
            raise RAGError(f"Unsupported media type: {media_kind}")
        try:
            text = generate_text(
                self.client,
                GEN_MODEL,
                [
                    types.Part(file_data=types.FileData(file_uri=uri, mime_type=mime_type)),
                    types.Part(text=instruction),
                ],
                operation=f"{media_kind} analysis",
            )
        except GeminiServiceError as exc:
            raise RAGError(str(exc)) from exc
        if not text:
            raise RAGError(f"Gemini returned no {media_kind} description.")
        return text

    def _ingest(self, source: _PreparedSource) -> int:
        documents = self._documents_for_source(source)
        if not documents:
            raise RAGError("No usable text was extracted from this source.")

        try:
            vectors = self.embeddings.embed_documents([document.page_content for document in documents])
        except Exception as exc:
            raise RAGError(f"Embedding failed: {exc}") from exc
        if len(vectors) != len(documents):
            raise RAGError("Embedding service returned a mismatched vector count.")

        try:
            self._collection.add(
                ids=[str(uuid.uuid4()) for _ in documents],
                embeddings=vectors,
                documents=[document.page_content for document in documents],
                metadatas=[document.metadata for document in documents],
            )
        except Exception as exc:
            raise RAGError(f"Could not save source to the vector index: {exc}") from exc

        self.sources.append(
            IndexedSource(
                label=source.label,
                source_type=source.source_type,
                chunks=len(documents),
                file_uri=source.file_uri,
                mime_type=source.mime_type,
                pages=source.pages,
                source_id=source.source_id,
            )
        )
        return len(documents)

    def _documents_for_source(self, source: _PreparedSource) -> list[Document]:
        output: list[Document] = []
        for segment in source.segments:
            normalized = " ".join(segment.text.split())
            if not normalized:
                continue
            details: dict[str, Any] = {
                "source_id": source.source_id,
                "source_type": source.source_type,
                "source_label": source.label,
            }
            if segment.page is not None:
                details["page"] = segment.page
            if source.file_uri:
                details["file_uri"] = source.file_uri
                details["mime_type"] = source.mime_type
            output.extend(self._split_text(normalized, details))
        return output

    @staticmethod
    def _split_text(text: str, metadata: dict[str, Any]) -> list[Document]:
        """Normalize text into bounded passages with a small continuity overlap."""
        normalized = " ".join((text or "").split())
        if not normalized:
            return []

        width = max(1, CHUNK_SIZE)
        carry = min(max(0, CHUNK_OVERLAP), width - 1)
        results: list[Document] = []
        offset = 0
        while offset < len(normalized):
            stop = min(offset + width, len(normalized))
            if stop < len(normalized):
                floor = offset + width // 2
                candidates = [normalized.rfind(separator, floor, stop) for separator in (". ", "? ", "! ", "; ", " ")]
                boundary = max(candidates)
                if boundary > offset:
                    stop = boundary + (2 if normalized[boundary:boundary + 2] in {". ", "? ", "! ", "; "} else 1)

            passage = normalized[offset:stop].strip()
            if passage:
                results.append(
                    Document(
                        page_content=passage,
                        metadata={**metadata, "chunk_index": len(results)},
                    )
                )
            if stop == len(normalized):
                break
            offset = max(offset + 1, stop - carry)
        return results

    # Evidence, response, and source-catalog stages ---------------------------

    def _generation_parts(
        self,
        question: str,
        documents: list[Document],
        route: QueryRoute,
    ) -> list[types.Part]:
        return build_multimodal_parts(question, documents, route)

    @staticmethod
    def _build_citations(documents: list[Document]) -> list[SourceCitation]:
        citations: list[SourceCitation] = []
        seen: set[tuple[str, str, int | None]] = set()
        for document in documents:
            metadata = document.metadata or {}
            label = str(metadata.get("source_label", "unknown"))
            source_type = str(metadata.get("source_type", "unknown"))
            page_number = None
            if source_type == "pdf" and metadata.get("page") is not None:
                try:
                    page_number = int(metadata["page"]) + 1
                except (TypeError, ValueError):
                    page_number = None
            key = (label, source_type, page_number)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                SourceCitation(
                    source_label=label,
                    source_type=source_type,
                    page=page_number,
                    excerpt=document.page_content[:280],
                )
            )
        return citations

    def _restore_catalog(self) -> None:
        try:
            metadata_rows = self._collection.get(include=["metadatas"]).get("metadatas") or []
        except Exception as exc:
            raise RAGError(f"Could not rebuild the source catalog: {exc}") from exc

        catalog: dict[tuple[str, str, str], dict[str, Any]] = {}
        for metadata in metadata_rows:
            if not metadata:
                continue
            label = str(metadata.get("source_label", "unknown"))
            source_type = str(metadata.get("source_type", "unknown"))
            source_id = str(metadata.get("source_id") or f"legacy:{source_type}:{label}")
            entry = catalog.setdefault(
                (source_id, source_type, label),
                {"chunks": 0, "pages": set(), "uri": "", "mime": ""},
            )
            entry["chunks"] += 1
            if source_type == "pdf" and metadata.get("page") is not None:
                try:
                    entry["pages"].add(int(metadata["page"]))
                except (TypeError, ValueError):
                    pass
            entry["uri"] = entry["uri"] or str(metadata.get("file_uri") or "")
            entry["mime"] = entry["mime"] or str(metadata.get("mime_type") or "")

        self.sources = [
            IndexedSource(
                label=label,
                source_type=source_type,
                chunks=entry["chunks"],
                pages=len(entry["pages"]) or None,
                file_uri=entry["uri"],
                mime_type=entry["mime"],
                source_id=source_id,
            )
            for (source_id, source_type, label), entry in catalog.items()
        ]

    def _fetch_source_text(self, source: IndexedSource, limit: int = 15) -> str:
        if self.chunk_count() == 0:
            return ""
        selector = (
            {"source_id": source.source_id}
            if not source.source_id.startswith("legacy:")
            else {"source_label": source.label}
        )
        try:
            stored = self._collection.get(where=selector, include=["documents"], limit=limit)
        except Exception as exc:
            raise RAGError(f"Could not load indexed source content: {exc}") from exc
        return "\n\n".join(stored.get("documents") or [])