"""Offline tests for local account registration and login."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from services.auth import AuthError, authenticate_user, register_user


class TestLocalAuthentication(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "users.sqlite3"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_register_and_login(self) -> None:
        user = register_user(
            "Avery Example", "AVERY@example.com", "long-password-1", "long-password-1", self.db_path
        )

        self.assertEqual(user["email"], "avery@example.com")
        self.assertEqual(
            authenticate_user("avery@example.com", "long-password-1", self.db_path),
            user,
        )

    def test_duplicate_email_is_rejected_case_insensitively(self) -> None:
        register_user("Avery", "avery@example.com", "long-password-1", "long-password-1", self.db_path)

        with self.assertRaisesRegex(AuthError, "already exists"):
            register_user("Another Avery", "AVERY@EXAMPLE.COM", "long-password-2", "long-password-2", self.db_path)

    def test_registration_validates_fields(self) -> None:
        invalid_cases = [
            ("", "avery@example.com", "long-password-1", "long-password-1", "full name"),
            ("Avery", "not-an-email", "long-password-1", "long-password-1", "email"),
            ("Avery", "avery@example.com", "short", "short", "at least"),
            ("Avery", "avery@example.com", "long-password-1", "different-password", "do not match"),
        ]
        for name, email, password, confirmation, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(AuthError, message):
                register_user(name, email, password, confirmation, self.db_path)

    def test_invalid_login_does_not_authenticate(self) -> None:
        register_user("Avery", "avery@example.com", "long-password-1", "long-password-1", self.db_path)

        self.assertIsNone(authenticate_user("avery@example.com", "wrong-password", self.db_path))
        self.assertIsNone(authenticate_user("missing@example.com", "long-password-1", self.db_path))

    def test_password_is_not_stored_as_plaintext(self) -> None:
        register_user("Avery", "avery@example.com", "long-password-1", "long-password-1", self.db_path)

        import sqlite3

        with closing(sqlite3.connect(self.db_path)) as connection:
            stored_hash = connection.execute("SELECT password_hash FROM users").fetchone()[0]
        self.assertNotEqual(stored_hash, "long-password-1")
        self.assertTrue(stored_hash.startswith("pbkdf2_sha256$"))


if __name__ == "__main__":
    unittest.main()