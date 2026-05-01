"""SQLite persistence layer. All writes go through here.

One file: state/conduit.db
Users keyed by phone_e164. Calls keyed by call_id.
JSON files in state/ remain ground truth for shot detail.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import auth

DB_PATH = Path("state") / "conduit.db"
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    Path("state").mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _lock:
        conn = _conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                phone_e164   TEXT PRIMARY KEY,
                token        TEXT UNIQUE NOT NULL,
                created_at   REAL NOT NULL,
                display_name TEXT
            );
            CREATE TABLE IF NOT EXISTS calls (
                call_id      TEXT PRIMARY KEY,
                phone_e164   TEXT REFERENCES users(phone_e164),
                title        TEXT,
                brief        TEXT,
                created_at   REAL NOT NULL,
                finalized_at REAL,
                state_path   TEXT,
                video_path   TEXT,
                shot_count   INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_calls_phone   ON calls(phone_e164);
            CREATE INDEX IF NOT EXISTS idx_calls_created ON calls(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_users_token   ON users(token);
        """)
        conn.commit()
        conn.close()


def upsert_user(phone_e164: str) -> dict[str, Any]:
    token = auth.phone_to_token(phone_e164)
    normalized = auth.normalize_phone(phone_e164)
    with _lock:
        conn = _conn()
        conn.execute(
            "INSERT OR IGNORE INTO users (phone_e164, token, created_at) VALUES (?,?,?)",
            (normalized, token, time.time()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE phone_e164=?", (normalized,)).fetchone()
        conn.close()
    return dict(row)


def get_user_by_token(token: str) -> dict[str, Any] | None:
    with _lock:
        conn = _conn()
        row = conn.execute("SELECT * FROM users WHERE token=?", (token,)).fetchone()
        conn.close()
    return dict(row) if row else None


def upsert_call(call_id: str, phone_e164: str | None = None,
                title: str | None = None, brief: str | None = None,
                state_path: str | None = None, video_path: str | None = None,
                shot_count: int | None = None, finalized: bool = False) -> None:
    normalized = auth.normalize_phone(phone_e164) if phone_e164 else None
    with _lock:
        conn = _conn()
        existing = conn.execute("SELECT * FROM calls WHERE call_id=?", (call_id,)).fetchone()
        if existing:
            updates: list[str] = []
            params: list[Any] = []
            if title is not None:
                updates.append("title=?"); params.append(title)
            if brief is not None:
                updates.append("brief=?"); params.append(brief)
            if state_path is not None:
                updates.append("state_path=?"); params.append(state_path)
            if video_path is not None:
                updates.append("video_path=?"); params.append(video_path)
            if shot_count is not None:
                updates.append("shot_count=?"); params.append(shot_count)
            if finalized:
                updates.append("finalized_at=?"); params.append(time.time())
            if normalized and not existing["phone_e164"]:
                updates.append("phone_e164=?"); params.append(normalized)
            if updates:
                params.append(call_id)
                conn.execute(f"UPDATE calls SET {','.join(updates)} WHERE call_id=?", params)
        else:
            conn.execute(
                """INSERT INTO calls
                   (call_id, phone_e164, title, brief, created_at, state_path, video_path, shot_count)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (call_id, normalized, title, brief, time.time(),
                 state_path, video_path, shot_count or 0),
            )
        conn.commit()
        conn.close()


def get_calls_for_user(phone_e164: str) -> list[dict[str, Any]]:
    normalized = auth.normalize_phone(phone_e164)
    with _lock:
        conn = _conn()
        rows = conn.execute(
            "SELECT * FROM calls WHERE phone_e164=? ORDER BY created_at DESC",
            (normalized,),
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def get_active_call(phone_e164: str) -> dict[str, Any] | None:
    """Most recent call without a finalized_at for this user."""
    normalized = auth.normalize_phone(phone_e164)
    with _lock:
        conn = _conn()
        row = conn.execute(
            "SELECT * FROM calls WHERE phone_e164=? AND finalized_at IS NULL ORDER BY created_at DESC LIMIT 1",
            (normalized,),
        ).fetchone()
        conn.close()
    return dict(row) if row else None


init()
