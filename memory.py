"""Conversation memory that stays cheap.

Each session keeps:
  - the last few messages word-for-word (so follow-ups work), and
  - one short rolling summary of everything older (instead of resending the whole chat).
Stored in SQLite, so a conversation survives a restart if you reuse the same session id.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from prompts import SUMMARY_SYSTEM_PROMPT, build_summary_request


@dataclass
class History:
    summary: str
    recent: list[dict]  # [{"role": "user"|"assistant", "content": str}]


class ConversationStore:
    def __init__(self, db_path: Path, history_turns: int = 4, summary_trigger: int = 4):
        db_path = Path(db_path)
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.keep = history_turns * 2          # messages kept verbatim (a turn = question + answer)
        self.trigger = summary_trigger
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);
            CREATE TABLE IF NOT EXISTS summaries (
                session_id TEXT PRIMARY KEY, summary TEXT NOT NULL, upto_id INTEGER NOT NULL);
            """
        )

    def _summary_row(self, session_id: str):
        return self.db.execute("SELECT summary, upto_id FROM summaries WHERE session_id = ?",
                               (session_id,)).fetchone()

    def get(self, session_id: str) -> History:
        row = self._summary_row(session_id)
        upto = row["upto_id"] if row else 0
        msgs = self.db.execute(
            "SELECT role, content FROM messages WHERE session_id = ? AND id > ? ORDER BY id",
            (session_id, upto)).fetchall()
        return History(row["summary"] if row else "", [dict(m) for m in msgs])

    def add(self, session_id: str, role: str, content: str) -> None:
        with self.db:
            self.db.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                            (session_id, role, content))

    def compact(self, session_id: str, provider, max_tokens: int = 200) -> tuple[int, int]:
        """Fold messages older than the verbatim window into the summary (one small model call).
        Returns (input_tokens, output_tokens) used, (0, 0) if nothing was needed."""
        row = self._summary_row(session_id)
        upto = row["upto_id"] if row else 0
        pending = self.db.execute(
            "SELECT id, role, content FROM messages WHERE session_id = ? AND id > ? ORDER BY id",
            (session_id, upto)).fetchall()
        older = pending[:-self.keep] if self.keep else pending
        if len(older) < self.trigger:
            return 0, 0
        request = build_summary_request(row["summary"] if row else "", [dict(m) for m in older])
        result = provider.chat(SUMMARY_SYSTEM_PROMPT, [{"role": "user", "content": request}], max_tokens)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO summaries VALUES (?, ?, ?)",
                            (session_id, result.text.strip(), older[-1]["id"]))
        return result.input_tokens, result.output_tokens

    def clear(self, session_id: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self.db.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))