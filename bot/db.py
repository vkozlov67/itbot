from __future__ import annotations

import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chats (
    chat_id    INTEGER PRIMARY KEY,
    title      TEXT    NOT NULL DEFAULT '',
    type       TEXT    NOT NULL DEFAULT '',
    username   TEXT,
    status     TEXT    NOT NULL DEFAULT 'member',
    enabled    INTEGER NOT NULL DEFAULT 0,
    transcribe INTEGER NOT NULL DEFAULT 1,
    answer     INTEGER NOT NULL DEFAULT 1,
    download   INTEGER NOT NULL DEFAULT 1,
    added_by   INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    chat_id    INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    user_id    INTEGER,
    author     TEXT    NOT NULL,
    kind       TEXT    NOT NULL,
    text       TEXT    NOT NULL,
    date       INTEGER NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, message_id);
CREATE TABLE IF NOT EXISTS usage (
    day           TEXT    NOT NULL,
    chat_id       INTEGER NOT NULL,
    stt_count     INTEGER NOT NULL DEFAULT 0,
    stt_seconds   REAL    NOT NULL DEFAULT 0,
    llm_requests  INTEGER NOT NULL DEFAULT 0,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, chat_id)
);
"""

ACTIVE_STATUSES = ("creator", "administrator", "member", "restricted")
CHAT_FLAGS = ("enabled", "transcribe", "answer", "download")
HISTORY_KEEP = 300


@dataclass(frozen=True)
class ChatRow:
    chat_id: int
    title: str
    type: str
    username: str | None
    status: str
    enabled: bool
    transcribe: bool
    answer: bool
    download: bool
    added_by: int | None

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def is_admin(self) -> bool:
        return self.status in ("administrator", "creator")


@dataclass(frozen=True)
class StoredMessage:
    chat_id: int
    message_id: int
    user_id: int | None
    author: str
    kind: str
    text: str
    date: int


@dataclass(frozen=True)
class UsageTotals:
    stt_count: int
    stt_seconds: float
    llm_requests: int
    input_tokens: int
    output_tokens: int


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(chats)")}
        if "download" not in cols:
            self._conn.execute("ALTER TABLE chats ADD COLUMN download INTEGER NOT NULL DEFAULT 1")

    def close(self) -> None:
        self._conn.close()

    # ---------- settings ----------
    def get_setting(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def delete_setting(self, key: str) -> None:
        self._conn.execute("DELETE FROM settings WHERE key=?", (key,))

    # ---------- chats ----------
    @staticmethod
    def _chat(row: sqlite3.Row | None) -> ChatRow | None:
        if row is None:
            return None
        return ChatRow(
            chat_id=row["chat_id"],
            title=row["title"],
            type=row["type"],
            username=row["username"],
            status=row["status"],
            enabled=bool(row["enabled"]),
            transcribe=bool(row["transcribe"]),
            answer=bool(row["answer"]),
            download=bool(row["download"]),
            added_by=row["added_by"],
        )

    def get_chat(self, chat_id: int) -> ChatRow | None:
        return self._chat(self._conn.execute("SELECT * FROM chats WHERE chat_id=?", (chat_id,)).fetchone())

    def list_chats(self, active_only: bool = True) -> list[ChatRow]:
        sql = "SELECT * FROM chats"
        params: tuple = ()
        if active_only:
            sql += f" WHERE status IN ({','.join('?' * len(ACTIVE_STATUSES))})"
            params = ACTIVE_STATUSES
        sql += " ORDER BY enabled DESC, title COLLATE NOCASE"
        return [self._chat(r) for r in self._conn.execute(sql, params).fetchall()]  # type: ignore[misc]

    def upsert_chat(
        self,
        chat_id: int,
        title: str,
        type_: str,
        username: str | None,
        status: str | None = None,
        added_by: int | None = None,
    ) -> ChatRow:
        now = int(time.time())
        self._conn.execute(
            """
            INSERT INTO chats(chat_id, title, type, username, status, added_by, created_at, updated_at)
            VALUES(?, ?, ?, ?, COALESCE(?, 'member'), ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                title=excluded.title,
                type=excluded.type,
                username=excluded.username,
                status=COALESCE(?, chats.status),
                added_by=COALESCE(?, chats.added_by),
                updated_at=excluded.updated_at
            """,
            (chat_id, title, type_, username, status, added_by, now, now, status, added_by),
        )
        row = self.get_chat(chat_id)
        assert row is not None
        return row

    def set_status(self, chat_id: int, status: str) -> None:
        self._conn.execute(
            "UPDATE chats SET status=?, updated_at=? WHERE chat_id=?", (status, int(time.time()), chat_id)
        )

    def set_flag(self, chat_id: int, flag: str, value: bool) -> None:
        if flag not in CHAT_FLAGS:
            raise ValueError(flag)
        self._conn.execute(
            f"UPDATE chats SET {flag}=?, updated_at=? WHERE chat_id=?", (int(value), int(time.time()), chat_id)
        )

    def migrate_chat(self, old_id: int, new_id: int) -> None:
        """Group -> supergroup upgrade changes the chat id; carry settings and history over."""
        if old_id == new_id:
            return
        old = self.get_chat(old_id)
        if old is None:
            return
        with self._conn:
            self._conn.execute("BEGIN")
            if self.get_chat(new_id) is None:
                self._conn.execute("UPDATE chats SET chat_id=? WHERE chat_id=?", (new_id, old_id))
            else:
                self._conn.execute(
                    "UPDATE chats SET enabled=?, transcribe=?, answer=?, download=? WHERE chat_id=?",
                    (int(old.enabled), int(old.transcribe), int(old.answer), int(old.download), new_id),
                )
                self._conn.execute("DELETE FROM chats WHERE chat_id=?", (old_id,))
            self._conn.execute("UPDATE OR IGNORE messages SET chat_id=? WHERE chat_id=?", (new_id, old_id))
            self._conn.execute("DELETE FROM messages WHERE chat_id=?", (old_id,))

    # ---------- message history ----------
    def add_message(
        self,
        chat_id: int,
        message_id: int,
        user_id: int | None,
        author: str,
        kind: str,
        text: str,
        date: int | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO messages(chat_id, message_id, user_id, author, kind, text, date) "
            "VALUES(?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(chat_id, message_id) DO UPDATE SET text=excluded.text, kind=excluded.kind",
            (chat_id, message_id, user_id, author, kind, text, date or int(time.time())),
        )
        if random.random() < 0.05:
            self._prune(chat_id)

    def _prune(self, chat_id: int) -> None:
        self._conn.execute(
            "DELETE FROM messages WHERE chat_id=? AND message_id < ("
            "  SELECT COALESCE(MIN(message_id), 0) FROM ("
            "    SELECT message_id FROM messages WHERE chat_id=? ORDER BY message_id DESC LIMIT ?"
            "  )"
            ")",
            (chat_id, chat_id, HISTORY_KEEP),
        )

    def get_message(self, chat_id: int, message_id: int) -> StoredMessage | None:
        row = self._conn.execute(
            "SELECT * FROM messages WHERE chat_id=? AND message_id=?", (chat_id, message_id)
        ).fetchone()
        return StoredMessage(**dict(row)) if row else None

    def recent_messages(self, chat_id: int, limit: int, before_message_id: int | None = None) -> list[StoredMessage]:
        if before_message_id is None:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE chat_id=? ORDER BY message_id DESC LIMIT ?", (chat_id, limit)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE chat_id=? AND message_id<? ORDER BY message_id DESC LIMIT ?",
                (chat_id, before_message_id, limit),
            ).fetchall()
        return [StoredMessage(**dict(r)) for r in reversed(rows)]

    def count_messages(self, chat_id: int) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM messages WHERE chat_id=?", (chat_id,)).fetchone()[0]

    def clear_history(self, chat_id: int) -> None:
        self._conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))

    # ---------- usage ----------
    def add_usage(
        self,
        chat_id: int,
        *,
        stt_count: int = 0,
        stt_seconds: float = 0.0,
        llm_requests: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO usage(day, chat_id, stt_count, stt_seconds, llm_requests, input_tokens, output_tokens)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(day, chat_id) DO UPDATE SET
                stt_count=stt_count+excluded.stt_count,
                stt_seconds=stt_seconds+excluded.stt_seconds,
                llm_requests=llm_requests+excluded.llm_requests,
                input_tokens=input_tokens+excluded.input_tokens,
                output_tokens=output_tokens+excluded.output_tokens
            """,
            (_today(), chat_id, stt_count, stt_seconds, llm_requests, input_tokens, output_tokens),
        )

    def usage_totals(self, day: str | None = None) -> UsageTotals:
        sql = (
            "SELECT COALESCE(SUM(stt_count),0), COALESCE(SUM(stt_seconds),0), COALESCE(SUM(llm_requests),0), "
            "COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) FROM usage"
        )
        params: tuple = ()
        if day == "today":
            sql += " WHERE day=?"
            params = (_today(),)
        row = self._conn.execute(sql, params).fetchone()
        return UsageTotals(int(row[0]), float(row[1]), int(row[2]), int(row[3]), int(row[4]))
