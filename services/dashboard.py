"""Authenticated DocMind workspace rendered with Streamlit primitives."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from config import APP_NAME, APP_TAGLINE, SUPPORTED_UPLOAD_EXTENSIONS
from rag import RAGError, SOURCE_ICONS, MultimodalRAG, SourceCitation

UPLOAD_OPTIONS = {
    "Text": None,
    "URL": None,
    "PDF": "pdf",
    "Image": "image",
    "Audio": "audio",
    "Video": "video",
}
UPLOAD_METHODS = {
    "Text": "add_text",
    "URL": "add_url",
    "PDF": "add_pdf",
    "Image": "add_image",
    "Audio": "add_audio",
    "Video": "add_video",
}


def _initialize_state() -> None:
    defaults: dict[str, Any] = {
        "rag": None,
        "messages": [],
        "pending_q": None,
        "feedback": {},
        "last_metrics": {},
        "conversation_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _api_key() -> str:
    return (st.session_state.get("api_key_input") or os.getenv("GOOGLE_API_KEY", "")).strip()


def _get_rag(api_key: str) -> MultimodalRAG | None:
    if not api_key:
        return None
    if st.session_state.rag is None:
        try:
            st.session_state.rag = MultimodalRAG(api_key=api_key)
        except RAGError as exc:
            st.sidebar.error(str(exc))
            return None
    return st.session_state.rag


def _reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.feedback = {}
    st.session_state.pending_q = None
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d-%H%M%S")


def _logout() -> None:
    for key in (
        "auth_user",
        "rag",
        "messages",
        "pending_q",
        "feedback",
        "last_metrics",
        "api_key_input",
    ):
        st.session_state.pop(key, None)
    st.session_state.auth_page = "landing"
    st.rerun()


def _persist_upload(uploaded_file: Any) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(uploaded_file.getvalue())
        return temporary.name


def _ingest_from_sidebar(rag: MultimodalRAG | None, kind: str) -> None:
    if kind == "Text":
        content = st.sidebar.text_area("Text content", height=120, key="upload_text")
        label = st.sidebar.text_input("Source label", placeholder="Meeting notes", key="upload_text_label")
        if st.sidebar.button(
            "Index text", type="primary", use_container_width=True,
            disabled=rag is None, key="dashboard_index_text",
        ):
            if not content.strip():
                st.sidebar.error("Enter text before indexing.")
                return
            _run_ingestion(lambda: rag.add_text(content, label=label or "Pasted text"), "Text indexed")
        return

    if kind == "URL":
        url = st.sidebar.text_input("Page URL", placeholder="https://example.com/article", key="upload_url")
        if st.sidebar.button(
            "Index URL", type="primary", use_container_width=True,
            disabled=rag is None, key="dashboard_index_url",
        ):
            if not url.strip():
                st.sidebar.error("Enter a URL before indexing.")
                return
            _run_ingestion(lambda: rag.add_url(url), "URL indexed")
        return

    category = UPLOAD_OPTIONS[kind]
    extensions = SUPPORTED_UPLOAD_EXTENSIONS[category]
    accepted = [extension.lstrip(".") for extension in extensions]
    uploaded_file = st.sidebar.file_uploader(
        f"Choose {kind.lower()} file",
        type=accepted,
        key=f"upload_{category}",
    )
    if uploaded_file is not None:
        st.sidebar.caption(f"{uploaded_file.name} · {uploaded_file.size / 1024:.0f} KB")
    if st.sidebar.button(
        f"Process {kind.lower()}",
        type="primary",
        use_container_width=True,
        disabled=rag is None or uploaded_file is None,
        key=f"dashboard_process_{category}",
    ):
        if uploaded_file is None:
            st.sidebar.error(f"Choose a {kind.lower()} file first.")
            return
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in extensions:
            st.sidebar.error("This file type is not supported.")
            return
        temporary_path = _persist_upload(uploaded_file)
        try:
            method = getattr(rag, UPLOAD_METHODS[kind])
            _run_ingestion(lambda: method(temporary_path), f"{kind} processed")
        finally:
            Path(temporary_path).unlink(missing_ok=True)


def _run_ingestion(operation, success_label: str) -> None:
    try:
        with st.spinner("Extracting, embedding, and indexing source..."):
            chunk_count = operation()
        st.sidebar.success(f"{success_label}: {chunk_count} chunks")
        st.rerun()
    except RAGError as exc:
        st.sidebar.error(str(exc))
    except Exception as exc:
        st.sidebar.error(f"Processing failed: {exc}")


def _render_sidebar(user: dict[str, Any], rag: MultimodalRAG | None) -> None:
    with st.sidebar:
        st.markdown(f"### {APP_NAME}")
        st.caption(APP_TAGLINE)
        st.caption(f"Signed in as {user['full_name']}")
        if st.button("Log out", use_container_width=True, key="dashboard_logout"):
            _logout()
        st.divider()
        if st.button(
            "New conversation", icon=":material/add_comment:",
            use_container_width=True, key="dashboard_new_conversation",
        ):
            _reset_conversation()
            st.rerun()

        st.subheader("Connection")
        st.text_input(
            "Session key override (optional)",
            type="password",
            value="",
            placeholder="Environment key configured" if os.getenv("GOOGLE_API_KEY") else "Enter a key for this session",
            help="GOOGLE_API_KEY is used server-side. An override stays in this browser session only.",
            key="api_key_input",
        )
        key = _api_key()
        if key and rag is not None:
            st.success("Gemini connected", icon=":material/check_circle:")
        else:
            st.warning("Set GOOGLE_API_KEY to process sources and ask questions.", icon=":material/key:")

        st.divider()
        st.subheader("Add a source")
        kind = st.selectbox("Source type", list(UPLOAD_OPTIONS), key="upload_kind")
        _ingest_from_sidebar(rag, kind)


def _format_citation(citation: SourceCitation) -> str:
    icon = SOURCE_ICONS.get(citation.source_type, "📁")
    page = f" · page {citation.page}" if citation.page else ""
    return f"{icon} **{citation.source_label}**{page} · {citation.source_type}"


def _render_message(message_index: int, message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        citations = message.get("citations") or []
        if citations:
            st.markdown("**Sources**")
            for citation in citations:
                st.markdown(_format_citation(citation))
                if citation.excerpt:
                    st.caption(citation.excerpt)
        chunks = message.get("chunks") or []
        if chunks:
            with st.expander(f"Retrieved context · {len(chunks)} chunks"):
                for chunk in chunks:
                    icon = SOURCE_ICONS.get(chunk.get("source_type", "unknown"), "📁")
                    page = f" · page {chunk['page']}" if chunk.get("page") else ""
                    st.markdown(f"**{icon} {chunk.get('source_label', 'unknown')}{page}**")
                    st.text(chunk.get("content", ""))
        if message.get("latency_ms") is not None and message.get("route"):
            st.caption(f"{message['latency_ms']:.0f} ms · {message['route'].replace('_', ' ')}")
        if message["role"] == "assistant" and message.get("show_feedback"):
            _render_feedback(message_index)


def _render_feedback(message_index: int) -> None:
    existing = st.session_state.feedback.get(message_index)
    helpful, not_helpful, _ = st.columns([1, 1, 5])
    if helpful.button("Helpful", icon=":material/thumb_up:", key=f"feedback_up_{message_index}"):
        st.session_state.feedback[message_index] = "up"
        st.rerun()
    if not_helpful.button("Not helpful", icon=":material/thumb_down:", key=f"feedback_down_{message_index}"):
        st.session_state.feedback[message_index] = "down"
        st.rerun()
    if existing:
        st.caption("Feedback saved for this session.")


def _render_metrics(rag: MultimodalRAG | None) -> None:
    if rag is None:
        st.info("Connect Gemini to access the document index and live workspace metrics.")
        return
    metrics = rag.insights()
    columns = st.columns(4)
    for column, label, value in zip(
        columns,
        ("Documents", "PDF pages", "Indexed chunks", "Questions"),
        (metrics["documents"], metrics["pages"], metrics["chunks"], metrics["queries"]),
    ):
        column.metric(label, value)


def _render_document_library(rag: MultimodalRAG | None) -> None:
    st.subheader("Document library")
    if rag is None:
        st.info("Connect Gemini to load the local document catalog.")
        return
    if not rag.sources:
        st.info("No documents indexed yet. Add a source from the sidebar.")
        return

    source_options = list(range(len(rag.sources)))
    selected_index = st.selectbox(
        "Select a document",
        source_options,
        format_func=lambda index: f"{SOURCE_ICONS.get(rag.sources[index].source_type, '📁')} {rag.sources[index].label}",
        key="selected_source_index",
    )
    selected = rag.sources[selected_index]
    st.markdown(f"### {selected.label}")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Type", selected.source_type.upper())
    meta_cols[1].metric("Chunks", selected.chunks)
    meta_cols[2].metric("Pages", selected.pages if selected.pages is not None else "—")
    meta_cols[3].metric("Status", selected.status)
    if selected.file_uri:
        st.caption(f"Gemini media reference available · {selected.mime_type or 'media'}")
    if st.button(
        "Generate summary", icon=":material/summarize:",
        disabled=rag is None, key=f"dashboard_summary_{selected_index}",
    ):
        try:
            with st.spinner("Summarizing indexed source..."):
                summary = rag.summarize_source(selected_index)
            st.session_state.last_summary = {
                "source_id": selected.source_id,
                "content": summary,
            }
            st.session_state.messages.append(
                {"role": "assistant", "content": f"**Summary · {selected.label}**\n\n{summary}"}
            )
            st.rerun()
        except RAGError as exc:
            st.error(str(exc))

    last_summary = st.session_state.get("last_summary")
    if last_summary and last_summary.get("source_id") == selected.source_id:
        st.markdown("#### Latest summary")
        st.markdown(last_summary["content"])

    st.divider()
    st.warning("Reset removes all sources from this local knowledge base.")
    confirm_reset = st.checkbox("Confirm knowledge-base reset", key="confirm_kb_reset")
    if st.button(
        "Reset knowledge base",
        type="secondary",
        disabled=not confirm_reset,
        key="dashboard_reset_knowledge",
    ):
        try:
            rag.reset_knowledge_base()
            _reset_conversation()
            st.session_state.pop("last_summary", None)
            st.success("Knowledge base reset.")
            st.rerun()
        except RAGError as exc:
            st.error(str(exc))


def _render_conversation(rag: MultimodalRAG | None) -> None:
    if not st.session_state.messages:
        st.markdown("### Ask your indexed documents")
        st.write("Answers use retrieved source material. Citations and retrieved passages appear with each response.")
        examples = [
            "What are the main topics in my documents?",
            "Summarize the indexed content",
            "Compare key points across sources",
        ]
        example_columns = st.columns(len(examples))
        for index, (column, question) in enumerate(zip(example_columns, examples)):
            if column.button(question, key=f"question_example_{index}", use_container_width=True):
                st.session_state.pending_q = question

    for index, message in enumerate(st.session_state.messages):
        _render_message(index, message)

    pending = st.session_state.pop("pending_q", None)
    prompt = pending or st.chat_input("Ask DocMind AI about your indexed sources...")
    if not prompt:
        return
    if rag is None:
        st.error("Connect Gemini with GOOGLE_API_KEY before asking a question.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        with st.spinner("Retrieving evidence and generating an answer..."):
            result = rag.query(prompt)
        chunks = []
        for document in result.retrieved_docs:
            metadata = document.metadata or {}
            page_number = metadata.get("page")
            if page_number is not None and metadata.get("source_type") == "pdf":
                try:
                    page_number = int(page_number) + 1
                except (TypeError, ValueError):
                    page_number = None
            chunks.append(
                {
                    "content": document.page_content,
                    "source_type": metadata.get("source_type", "unknown"),
                    "source_label": metadata.get("source_label", "unknown"),
                    "page": page_number,
                }
            )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "citations": result.citations,
                "chunks": chunks,
                "latency_ms": result.latency_ms,
                "route": result.route,
                "show_feedback": True,
            }
        )
    except RAGError as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"**Could not complete request:** {exc}"}
        )
    except Exception:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "**Something went wrong.** Check the API key, network connection, and document index.",
            }
        )
    st.rerun()


def render_dashboard() -> None:
    """Render the authenticated DocMind document workspace."""
    _initialize_state()
    user = st.session_state.auth_user
    rag = _get_rag(_api_key())
    _render_sidebar(user, rag)

    st.markdown(f"<div class='landing-kicker'>{APP_NAME} WORKSPACE</div>", unsafe_allow_html=True)
    st.markdown(f"<h1 class='hero-title'>Welcome, {user['full_name']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='hero-sub'>{APP_TAGLINE}</p>", unsafe_allow_html=True)
    _render_metrics(rag)

    active_view = st.radio(
        "Workspace view",
        ("Conversation", "Documents"),
        horizontal=True,
        label_visibility="collapsed",
        key="dashboard_view",
    )
    if active_view == "Conversation":
        _render_conversation(rag)
    else:
        _render_document_library(rag)
