"""DocMind AI — shared configuration."""

from pathlib import Path

APP_NAME = "DocMind AI"
APP_TAGLINE = "AI-Powered Multimodal Document Intelligence & RAG Assistant"

EMBED_MODEL = "gemini-embedding-2"
GEN_MODEL = "gemini-3-flash-preview"
COLLECTION_NAME = "docmind_ai"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
MAX_PDF_PAGES = 100

VECTOR_TOP_K = 5
RETRIEVAL_CANDIDATE_MULTIPLIER = 4
HYBRID_VECTOR_WEIGHT = 0.65
HYBRID_KEYWORD_WEIGHT = 0.35

CHROMA_PERSIST_DIR = Path(__file__).resolve().parent / "data" / "chroma"

MEDIA_ANALYSIS_INSTRUCTIONS = {
    "image": (
        "Inspect the image as document evidence. Report readable text, visible objects, "
        "layout, and notable relationships; distinguish direct observations from uncertainty."
    ),
    "audio": (
        "Transcribe intelligible speech and summarize salient non-speech sounds. "
        "Do not infer speaker identities; mark unclear words as uncertain."
    ),
    "video": (
        "Summarize the clip in time order. Include scene changes, legible on-screen text, "
        "intelligible speech, and relevant audio, using approximate timestamps when useful."
    ),
}

MIME_MAP: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}

SOURCE_ICONS = {
    "text": "✏️",
    "url": "🔗",
    "pdf": "📚",
    "image": "🌄",
    "audio": "🎙️",
    "video": "📽️",
}

SUPPORTED_UPLOAD_EXTENSIONS = {
    "pdf": [".pdf"],
    "image": [".jpg", ".jpeg", ".png", ".webp", ".gif"],
    "audio": [".mp3", ".wav", ".ogg", ".m4a", ".flac"],
    "video": [".mp4", ".mov", ".webm", ".avi"],
}
