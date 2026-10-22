"""Contract checks for configuration consumed by DocMind services and UI."""

from __future__ import annotations

import unittest

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    GEN_MODEL,
    MEDIA_ANALYSIS_INSTRUCTIONS,
    MIME_MAP,
    SOURCE_ICONS,
    SUPPORTED_UPLOAD_EXTENSIONS,
)


class TestDocMindConfiguration(unittest.TestCase):
    def test_chunk_profile_has_valid_overlap(self) -> None:
        self.assertGreater(CHUNK_SIZE, 0)
        self.assertGreaterEqual(CHUNK_OVERLAP, 0)
        self.assertLess(CHUNK_OVERLAP, CHUNK_SIZE)

    def test_model_ids_are_provider_identifiers(self) -> None:
        self.assertEqual(EMBED_MODEL, "gemini-embedding-2")
        self.assertEqual(GEN_MODEL, "gemini-3-flash-preview")

    def test_all_accepted_media_extensions_have_mime_types(self) -> None:
        media_extensions = {
            extension
            for category, extensions in SUPPORTED_UPLOAD_EXTENSIONS.items()
            if category != "pdf"
            for extension in extensions
        }
        self.assertTrue(media_extensions.issubset(MIME_MAP))

    def test_source_marks_cover_supported_document_types(self) -> None:
        source_types = {"text", "url", "pdf", "image", "audio", "video"}
        self.assertEqual(set(SOURCE_ICONS), source_types)
        self.assertEqual(len(set(SOURCE_ICONS.values())), len(source_types))

    def test_media_instructions_cover_supported_gemini_media(self) -> None:
        self.assertEqual(set(MEDIA_ANALYSIS_INSTRUCTIONS), {"image", "audio", "video"})
        self.assertTrue(all(text.strip() for text in MEDIA_ANALYSIS_INSTRUCTIONS.values()))


if __name__ == "__main__":
    unittest.main()