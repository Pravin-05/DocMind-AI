"""
DocMind AI — Streamlit application.

Portfolio customization built on the reference Multimodal RAG project
(see README — Credits & Attribution).
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from html import escape

import streamlit as st
from dotenv import load_dotenv

from config import APP_NAME, APP_TAGLINE
from rag import RAGError, SOURCE_ICONS, MultimodalRAG, SourceCitation
from services.auth import AuthError, authenticate_user, register_user

load_dotenv()

# ── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
:root, .stApp {
    color-scheme: dark;
    --text-color: #f1f5f9;
    --secondary-text-color: #cbd5e1;
    --heading-color: #f8fafc;
    color: #f1f5f9 !important;
}
.stApp { background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%); }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] blockquote,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] strong {
    color: #f1f5f9;
}
[data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #cbd5e1 !important;
}
a { color: #a3e635 !important; }

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] * { color: #e5e7eb !important; }

[data-testid="stChatMessage"] {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
    color: #f1f5f9 !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #1e293b !important;
    border: 1px solid #475569 !important;
    color: #f8fafc !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #cbd5e1 !important; opacity: 1; }
[data-testid="stChatInput"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.25) !important;
}

.stButton > button {
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
    background: #1e293b !important;
    color: #e2e8f0 !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: #6366f1 !important;
    background: #312e81 !important;
}

.brand-title { font-size: 22px; font-weight: 700; color: #f8fafc; margin: 0; }
.brand-tag { font-size: 12px; color: #cbd5e1; line-height: 1.4; margin-top: 4px; }
.hero-title { font-size: 32px; font-weight: 700; color: #f8fafc; margin-bottom: 6px; }
.hero-sub { color: #cbd5e1; font-size: 15px; margin-bottom: 18px; max-width: 820px; }
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.metric-value { font-size: 22px; font-weight: 700; color: #c7d2fe; }
.metric-label { font-size: 12px; color: #cbd5e1; margin-top: 4px; }
.doc-card {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-size: 13px;
}
.doc-meta { color: #cbd5e1; font-size: 11px; margin-top: 4px; }
.status-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
}
.status-ok { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid #10b981; }
.status-warn { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid #f59e0b; }
.citation-card {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
    font-size: 12px;
    color: #c7d2fe;
}
.chunk-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #cbd5e1;
}
.chunk-meta { color: #cbd5e1; font-size: 11px; margin-top: 6px; }
.empty-state { text-align: center; padding: 40px 16px; color: #cbd5e1; }
.empty-state h3 { color: #f1f5f9; }
.landing-header { padding: 10px 0 18px; border-bottom: 1px solid #334155; margin-bottom: 36px; }
.landing-brand { font-size: 23px; font-weight: 750; color: #f8fafc; }
.landing-hero { padding: 28px 0 56px; }
.landing-hero h1 { max-width: 680px; font-size: 48px; line-height: 1.08; color: #f8fafc; }
.landing-hero p { max-width: 620px; font-size: 17px; color: #cbd5e1; line-height: 1.65; }
.landing-kicker { color: #a3e635; text-transform: uppercase; font-size: 12px; font-weight: 700; }
.landing-section { padding: 30px 0; border-top: 1px solid #334155; }
.landing-section h2 { color: #f8fafc; font-size: 27px; }
.feature-tile { min-height: 118px; padding: 18px; background: #172234; border: 1px solid #334155; border-radius: 8px; margin-bottom: 12px; }
.feature-tile strong { color: #f8fafc; font-size: 16px; }
.feature-tile p { margin: 8px 0 0; color: #aab7c8; font-size: 13px; }
.step-number { color: #a3e635; font-size: 13px; font-weight: 700; }
.tech-pill { display: inline-block; border: 1px solid #526174; color: #e2e8f0; border-radius: 999px; padding: 7px 13px; margin: 4px; font-size: 13px; }
.landing-footer { border-top: 1px solid #334155; padding: 20px 0 8px; color: #94a3b8; }
@media (max-width: 700px) {
    .landing-hero h1 { font-size: 36px; }
    .landing-hero { padding: 10px 0 32px; }
}
</style>
""",
    unsafe_allow_html=True,
)


def _set_auth_page(page: str) -> None:
    st.session_state.auth_page = page
    st.rerun()


def _render_landing_page() -> None:
    brand, nav, login_col, register_col = st.columns([1.6, 2.4, 0.7, 0.9])
    with brand:
        st.markdown(f"<div class='landing-brand'>{APP_NAME}</div>", unsafe_allow_html=True)
    with nav:
        st.markdown(
            "<div style='padding-top:8px'>"
            "<a href='#features'>Features</a> &nbsp; · &nbsp; "
            "<a href='#how-it-works'>How It Works</a> &nbsp; · &nbsp; "
            "<a href='#about'>About</a></div>",
            unsafe_allow_html=True,
        )
    if login_col.button("Login", use_container_width=True, key="landing_login"):
        _set_auth_page("login")
    if register_col.button("Register", type="primary", use_container_width=True, key="landing_register"):
        _set_auth_page("register")

    st.markdown("<div class='landing-hero'>", unsafe_allow_html=True)
    hero_copy, hero_visual = st.columns([1.15, 0.85], gap="large")
    with hero_copy:
        st.markdown("<div class='landing-kicker'>Document intelligence, grounded</div>", unsafe_allow_html=True)
        st.markdown("<h1>Understand Your Documents with AI</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p>DocMind AI is an intelligent multimodal document assistant powered by "
            "Retrieval-Augmented Generation.</p>",
            unsafe_allow_html=True,
        )
        start_col, login_action = st.columns([1.2, 0.8])
        if start_col.button("Get Started", type="primary", use_container_width=True, key="hero_start"):
            _set_auth_page("register")
        if login_action.button("Login", use_container_width=True, key="hero_login"):
            _set_auth_page("login")
    with hero_visual:
        st.image("assets/docmind-workflow.svg", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div id='features' class='landing-section'><h2>Features</h2></div>", unsafe_allow_html=True)
    feature_items = [
        ("Multimodal Document Analysis", "Bring text, URLs, PDFs, images, audio, and video into one searchable workspace."),
        ("RAG-powered Question Answering", "Retrieve relevant passages before generating document-grounded answers."),
        ("Semantic Search", "Find related passages using Gemini embeddings and hybrid retrieval."),
        ("AI Document Chat", "Ask follow-up questions in a conversational workspace."),
        ("Source Citations", "Review source labels and available PDF page references alongside answers."),
        ("Smart Summaries", "Generate a structured summary from an indexed document."),
    ]
    feature_cols = st.columns(3)
    for index, (title, description) in enumerate(feature_items):
        with feature_cols[index % 3]:
            st.markdown(
                f"<div class='feature-tile'><strong>{title}</strong><p>{description}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div id='how-it-works' class='landing-section'><h2>How It Works</h2></div>", unsafe_allow_html=True)
    step_cols = st.columns(4)
    for col, number, title, description in zip(
        step_cols,
        ("01", "02", "03", "04"),
        ("Upload", "Process", "Retrieve", "Ask"),
        (
            "Add documents or media sources.",
            "Extract, chunk, and index their content.",
            "Search for relevant context with hybrid retrieval.",
            "Get a grounded answer with available citations.",
        ),
    ):
        with col:
            st.markdown(
                f"<div class='feature-tile'><div class='step-number'>{number}</div>"
                f"<strong>{title}</strong><p>{description}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='landing-section'><h2>Technology</h2></div>", unsafe_allow_html=True)
    st.markdown(
        "<span class='tech-pill'>Python</span><span class='tech-pill'>Streamlit</span>"
        "<span class='tech-pill'>Google Gemini</span><span class='tech-pill'>LangChain</span>"
        "<span class='tech-pill'>ChromaDB</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<div id='about' class='landing-section'><h2>About DocMind AI</h2></div>", unsafe_allow_html=True)
    st.write(
        "DocMind AI is a portfolio project built on a multimodal RAG reference architecture. "
        "Its local account system is intended for a single-instance demonstration and is not production-grade authentication."
    )
    st.markdown("<div class='landing-section'><h2>Ready to chat with your documents?</h2></div>", unsafe_allow_html=True)
    if st.button("Start Using DocMind AI", type="primary", key="landing_cta"):
        _set_auth_page("register")

    st.markdown(
        "<div class='landing-footer'>DocMind AI &nbsp; · &nbsp; "
        "<a href='https://github.com/vishakha-codes/multimodal_rag'>GitHub</a> &nbsp; · &nbsp; "
        "<a href='https://github.com/vishakha-codes/multimodal_rag#readme'>Documentation</a> &nbsp; · &nbsp; "
        "<a href='https://github.com/vishakha-codes/multimodal_rag'>License</a></div>",
        unsafe_allow_html=True,
    )


def _render_login_page() -> None:
    if st.button("← DocMind AI", key="login_home"):
        _set_auth_page("landing")
    st.title("Login")
    with st.form("login_form"):
        email = st.text_input("Email", autocomplete="email")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
    if submitted:
        if not email.strip() or not password:
            st.error("Enter your email and password.")
        else:
            user = authenticate_user(email, password)
            if user is None:
                st.error("Email or password is incorrect.")
            else:
                st.session_state.auth_user = user
                st.session_state.auth_page = "dashboard"
                st.rerun()
    if st.button("Register", key="login_register"):
        _set_auth_page("register")


def _render_register_page() -> None:
    if st.button("← DocMind AI", key="register_home"):
        _set_auth_page("landing")
    st.title("Create your account")
    st.caption("Passwords must be at least 10 characters.")
    with st.form("register_form"):
        full_name = st.text_input("Full Name", autocomplete="name")
        email = st.text_input("Email", autocomplete="email")
        password = st.text_input("Password", type="password", autocomplete="new-password")
        confirm_password = st.text_input("Confirm Password", type="password", autocomplete="new-password")
        submitted = st.form_submit_button("Register", type="primary", use_container_width=True)
    if submitted:
        try:
            st.session_state.auth_user = register_user(
                full_name, email, password, confirm_password
            )
            st.session_state.auth_page = "dashboard"
            st.rerun()
        except AuthError as exc:
            st.error(str(exc))
    if st.button("Login", key="register_login"):
        _set_auth_page("login")


def _require_authentication() -> bool:
    st.session_state.setdefault("auth_user", None)
    st.session_state.setdefault("auth_page", "landing")
    if st.session_state.auth_user:
        return True
    if st.session_state.auth_page == "login":
        _render_login_page()
    elif st.session_state.auth_page == "register":
        _render_register_page()
    else:
        _render_landing_page()
    return False


if not _require_authentication():
    st.stop()

from services.dashboard import render_dashboard

render_dashboard()
st.stop()

# ── Session state ────────────────────────────────────────────────────────────

DEFAULTS = {
    "rag": None,
    "messages": [],
    "pending_q": None,
    "feedback": {},
    "last_metrics": {},
    "conversation_id": datetime.utcnow().strftime("%Y%m%d-%H%M%S"),
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

EXAMPLE_QUESTIONS = [
    "What are the main topics in my documents?",
    "Summarize the indexed content",
    "Compare key points across sources",
    "What does the PDF say on the first pages?",
]


def get_api_key() -> str:
    sidebar_key = st.session_state.get("api_key_input", "")
    env_key = os.getenv("GOOGLE_API_KEY", "")
    return (sidebar_key or env_key).strip()


def ensure_rag(api_key: str) -> MultimodalRAG | None:
    if not api_key:
        return None
    if st.session_state.rag is None:
        try:
            st.session_state.rag = MultimodalRAG(api_key=api_key)
        except RAGError as exc:
            st.sidebar.error(str(exc))
            return None
    return st.session_state.rag


def format_citation_html(c: SourceCitation) -> str:
    icon = SOURCE_ICONS.get(c.source_type, "📁")
    page_line = f"<br><span style='color:#94a3b8;'>Page {c.page}</span>" if c.page else ""
    return (
        f"<div class='citation-card'>{icon} <strong>{c.source_label}</strong>"
        f"{page_line}</div>"
    )


def message_chunk_payload(doc) -> dict:
    meta = doc.metadata or {}
    page = meta.get("page")
    page_display = None
    if page is not None and meta.get("source_type") == "pdf":
        try:
            page_display = int(page) + 1
        except (TypeError, ValueError):
            page_display = None
    return {
        "content": doc.page_content,
        "source_type": meta.get("source_type", "unknown"),
        "source_label": meta.get("source_label", "unknown"),
        "page": page_display,
    }


def render_retrieval(chunks: list[dict], citations: list[SourceCitation] | None = None) -> None:
    if citations:
        st.markdown("**Sources**")
        for cit in citations:
            st.markdown(format_citation_html(cit), unsafe_allow_html=True)

    if chunks:
        with st.expander(f"Retrieved context ({len(chunks)} chunks)"):
            for chunk in chunks:
                icon = SOURCE_ICONS.get(chunk["source_type"], "📁")
                page_txt = f" · Page {chunk['page']}" if chunk.get("page") else ""
                st.markdown(
                    f"<div class='chunk-card'>{chunk['content']}"
                    f"<div class='chunk-meta'>{icon} {chunk['source_label']}{page_txt}</div></div>",
                    unsafe_allow_html=True,
                )


def render_feedback(message_index: int) -> None:
    fb = st.session_state.feedback.get(message_index)
    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("👍 Helpful", key=f"up_{message_index}"):
        st.session_state.feedback[message_index] = "up"
        st.rerun()
    if c2.button("👎 Not helpful", key=f"down_{message_index}"):
        st.session_state.feedback[message_index] = "down"
        st.rerun()
    if fb:
        st.caption(
            "Thanks — feedback saved for this session only."
            if fb == "up"
            else "Noted — we'll use this session feedback to improve prompts later."
        )


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"<p class='brand-title'>{APP_NAME}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='brand-tag'>{APP_TAGLINE}</p>", unsafe_allow_html=True)
    auth_user = st.session_state.auth_user
    st.caption(f"Signed in as {auth_user['full_name']}")
    if st.button("Log out", use_container_width=True, key="logout"):
        for key in (
            "auth_user", "rag", "messages", "pending_q", "feedback",
            "last_metrics", "api_key_input",
        ):
            st.session_state.pop(key, None)
        st.session_state.auth_page = "landing"
        st.rerun()
    st.divider()

    if st.button("✨ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.session_state.conversation_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        st.rerun()

    st.markdown("**Settings**")
    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        placeholder="AIza...",
        help="Set GOOGLE_API_KEY in .env or enter here (session only).",
        key="api_key_input",
    )
    effective_key = get_api_key()
    rag = ensure_rag(effective_key)

    if effective_key and rag:
        st.markdown("<span class='status-pill status-ok'>● API ready</span>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<span class='status-pill status-warn'>○ API key required</span>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Add documents**")

    source_type = st.selectbox(
        "Source type",
        options=["Text", "URL", "PDF", "Image", "Audio", "Video"],
        label_visibility="collapsed",
    )
    stype = source_type.lower()
    ingested = False

    try:
        if stype == "text":
            text_input = st.text_area("Paste text", height=100, label_visibility="collapsed")
            text_label = st.text_input("Label (optional)", placeholder="Meeting notes")
            if st.button("Add text", use_container_width=True, disabled=not rag):
                n = rag.add_text(text_input, label=text_label or "Pasted text")
                st.success(f"Indexed {n} chunks")
                ingested = True

        elif stype == "url":
            url_input = st.text_input("URL", placeholder="https://...")
            if st.button("Add URL", use_container_width=True, disabled=not rag):
                n = rag.add_url(url_input)
                st.success(f"Indexed {n} chunks")
                ingested = True

        elif stype == "pdf":
            pdf_file = st.file_uploader("PDF", type=["pdf"])
            if pdf_file and pdf_file.size / (1024 * 1024) > 10:
                st.warning("Large PDF — only the first 100 pages are indexed.")
            if st.button("Add PDF", use_container_width=True, disabled=not (rag and pdf_file)):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_file.read())
                    tmp_path = tmp.name
                n = rag.add_pdf(tmp_path)
                st.success(f"Indexed {pdf_file.name} ({n} chunks)")
                ingested = True

        elif stype == "image":
            img_file = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp", "gif"])
            if st.button("Add image", use_container_width=True, disabled=not (rag and img_file)):
                suffix = "." + img_file.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(img_file.read())
                    tmp_path = tmp.name
                n = rag.add_image(tmp_path)
                st.success(f"Indexed {img_file.name} ({n} chunks)")
                ingested = True

        elif stype == "audio":
            audio_file = st.file_uploader("Audio", type=["mp3", "wav", "ogg", "m4a", "flac"])
            if st.button("Add audio", use_container_width=True, disabled=not (rag and audio_file)):
                suffix = "." + audio_file.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                n = rag.add_audio(tmp_path)
                st.success(f"Indexed {audio_file.name} ({n} chunks)")
                ingested = True

        elif stype == "video":
            video_file = st.file_uploader("Video", type=["mp4", "mov", "webm", "avi"])
            if st.button("Add video", use_container_width=True, disabled=not (rag and video_file)):
                suffix = "." + video_file.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name
                n = rag.add_video(tmp_path)
                st.success(f"Indexed {video_file.name} ({n} chunks)")
                ingested = True

    except RAGError as exc:
        st.error(str(exc))
    except Exception:
        st.error("Document processing failed. Check the file format and try again.")

    if ingested:
        st.rerun()

    st.divider()
    st.markdown("**Documents**")

    if rag and rag.sources:
        for idx, src in enumerate(rag.sources):
            icon = SOURCE_ICONS.get(src.source_type, "📁")
            pages = f" · {src.pages} pages" if src.pages else ""
            st.markdown(
                f"<div class='doc-card'>{icon} <strong>{src.label}</strong>"
                f"<div class='doc-meta'>{src.source_type.upper()} · {src.chunks} chunks"
                f"{pages} · {src.status}</div></div>",
                unsafe_allow_html=True,
            )
            if st.button(f"Generate summary", key=f"sum_{idx}", use_container_width=True):
                with st.spinner("Generating summary..."):
                    try:
                        summary = rag.summarize_source(idx)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"**Summary — {src.label}**\n\n{summary}"}
                        )
                        st.rerun()
                    except RAGError as exc:
                        st.error(str(exc))
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    if rag:
        ins = rag.insights()
        st.markdown("**Session insights**")
        st.caption(
            f"Documents: {ins['documents']} · Pages: {ins['pages']} · "
            f"Chunks: {ins['chunks']} · Queries: {ins['queries']}"
        )

    if st.button("Reset knowledge base", use_container_width=True, disabled=not rag):
        if rag:
            rag.reset_knowledge_base()
        st.session_state.messages = []
        st.session_state.feedback = {}
        if effective_key:
            st.session_state.rag = MultimodalRAG(api_key=effective_key)
        st.rerun()

    st.divider()
    st.caption("Conversations are kept for this browser session only.")
    st.caption(f"Session ID: {st.session_state.conversation_id}")

# ── Main ─────────────────────────────────────────────────────────────────────

st.markdown(
    f"<div class='hero-title'>Welcome, {escape(auth_user['full_name'])}</div>",
    unsafe_allow_html=True,
)
st.markdown(f"<div class='hero-sub'>{APP_TAGLINE}</div>", unsafe_allow_html=True)

rag_main: MultimodalRAG | None = st.session_state.rag
if rag_main:
    ins = rag_main.insights()
    m1, m2, m3, m4 = st.columns(4)
    for col, label, val in zip(
        [m1, m2, m3, m4],
        ["Documents", "Pages (PDF)", "Chunks", "Queries"],
        [ins["documents"], ins["pages"], ins["chunks"], ins["queries"]],
    ):
        col.markdown(
            f"<div class='metric-card'><div class='metric-value'>{val}</div>"
            f"<div class='metric-label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    if rag_main.sources:
        st.markdown("**Indexed documents**")
        cols = st.columns(min(3, len(rag_main.sources)))
        for i, src in enumerate(rag_main.sources[:6]):
            with cols[i % len(cols)]:
                icon = SOURCE_ICONS.get(src.source_type, "📁")
                st.markdown(f"{icon} **{src.label}**")
                st.caption(f"{src.source_type} · {src.chunks} chunks · {src.status}")

st.divider()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            render_retrieval(msg["chunks"], msg.get("citations"))
        if msg.get("latency_ms"):
            st.caption(
                f"Response latency: {msg['latency_ms']:.0f} ms · Route: {msg.get('route', 'n/a')}"
            )
        if msg["role"] == "assistant" and msg.get("show_feedback"):
            render_feedback(i)

if not st.session_state.messages:
    st.markdown(
        "<div class='empty-state'><h3>Ask questions about your documents</h3>"
        "<p>Upload sources in the sidebar, then chat here. Answers include citations "
        "and retrieved context when available.</p></div>",
        unsafe_allow_html=True,
    )
    ex_cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if ex_cols[i % 2].button(q, use_container_width=True, key=f"ex_{i}"):
            st.session_state.pending_q = q

if st.session_state.pending_q:
    prompt = st.session_state.pending_q
    st.session_state.pending_q = None
else:
    prompt = st.chat_input("Ask DocMind AI about your documents...")

if prompt:
    effective_key = get_api_key()
    if not effective_key:
        st.error("Add your Google API key in the sidebar or .env file.")
        st.stop()

    rag = ensure_rag(effective_key)
    if not rag:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Retrieving context and generating answer..."):
        try:
            result = rag.query(prompt)
            chunk_data = [message_chunk_payload(d) for d in result.retrieved_docs]
            st.session_state.last_metrics = {
                "latency_ms": result.latency_ms,
                "route": result.route,
                "chunks": len(chunk_data),
            }
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "chunks": chunk_data,
                    "citations": result.citations,
                    "show_feedback": True,
                    "latency_ms": result.latency_ms,
                    "route": result.route,
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
                    "content": "**Something went wrong.** Check your API key, network, and document index.",
                }
            )
    st.rerun()
