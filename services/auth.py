"""Local SQLite authentication for the DocMind AI portfolio application."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

AUTH_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "docmind_users.sqlite3"
PASSWORD_ITERATIONS = 600_000
PASSWORD_MIN_LENGTH = 10
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(ValueError):
    """An input or account error suitable for display to the user."""


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else AUTH_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    return connection


@contextmanager
def _database(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    connection = _connect(db_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _normalize_email(email: str) -> str:
    normalized = (email or "").strip().casefold()
    if not _EMAIL_PATTERN.fullmatch(normalized):
        raise AuthError("Enter a valid email address.")
    return normalized


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_raw, salt_hex, digest_hex = encoded.split("$", 3)
        iterations = int(iterations_raw)
        if algorithm != "pbkdf2_sha256" or not 100_000 <= iterations <= 2_000_000:
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, TypeError):
        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(actual, expected)


def register_user(
    full_name: str,
    email: str,
    password: str,
    confirm_password: str,
    db_path: str | Path | None = None,
) -> dict[str, str | int]:
    name = (full_name or "").strip()
    if not name:
        raise AuthError("Enter your full name.")
    normalized_email = _normalize_email(email)
    if not password:
        raise AuthError("Enter a password.")
    if len(password) < PASSWORD_MIN_LENGTH:
        raise AuthError(f"Password must be at least {PASSWORD_MIN_LENGTH} characters.")
    if len(password) > 1024:
        raise AuthError("Password must be 1024 characters or fewer.")
    if password != confirm_password:
        raise AuthError("Passwords do not match.")

    try:
        with _database(db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
                (name, normalized_email, _hash_password(password)),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise AuthError("An account with this email already exists.") from exc

    return {"id": int(user_id), "full_name": name, "email": normalized_email}


def authenticate_user(
    email: str, password: str, db_path: str | Path | None = None
) -> dict[str, str | int] | None:
    if not email or not password:
        return None
    try:
        normalized_email = _normalize_email(email)
    except AuthError:
        return None

    with _database(db_path) as connection:
        row = connection.execute(
            "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
            (normalized_email,),
        ).fetchone()
    if row is None or not _verify_password(password, row["password_hash"]):
        return None
    return {"id": int(row["id"]), "full_name": row["full_name"], "email": row["email"]}