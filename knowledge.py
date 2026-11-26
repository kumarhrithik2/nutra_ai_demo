"""Persistent company knowledge base.

Documents are read ONCE (ingest), cut into small chunks and stored in a SQLite database file
with a full-text index (FTS5, BM25 ranking). At question time only the few chunks that match
the question are fetched and sent to the model, so the full documents never travel to the API
again and the index survives restarts.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

STOPWORDS = set(
    "a an the and or but if of to in on at for with by from as is are was were be been being it its "
    "this that these those i me my we our you your do does did can could should would will may might "
    "have has had not no so than then there what which who whom how when where why about into over "
    "also just any some more most very".split()
)
CHUNK_CHARS = 800
SUPPORTED = {".md", ".txt", ".pdf", ".docx"}


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to ingest PDF files:  pip install pypdf") from exc
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        try:
            import docx
        except ImportError as exc:
            raise RuntimeError("Install python-docx to ingest Word files:  pip install python-docx") from exc
        return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    raise RuntimeError(f"Unsupported file type: {path.name}")


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[tuple[str, str]]:
    """Split into (heading, chunk) pairs. Markdown headings define sections; long sections are
    cut at paragraph boundaries so each chunk stays about one topic and under max_chars."""
    chunks: list[tuple[str, str]] = []
    heading = "General"
    buffer: list[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            chunks.append((heading, body))
        buffer.clear()

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        first, _, rest = block.partition("\n")
        if first.startswith("#"):
            flush()
            heading = first.lstrip("#").strip() or heading
            block = rest.strip()
            if not block:
                continue
        if sum(len(b) for b in buffer) + len(block) > max_chars:
            flush()
        buffer.append(block)
    flush()
    return chunks


def build_fts_query(question: str) -> str:
    terms = [t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) > 1 and t not in STOPWORDS]
    # OR-join so any matching word can retrieve a chunk; BM25 ranks chunks matching more words higher
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


class KnowledgeBase:
    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (source TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                text, heading, source UNINDEXED, tokenize = 'porter unicode61');
            """
        )

    def ingest_text(self, source: str, text: str) -> int:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = self.db.execute("SELECT sha256 FROM documents WHERE source = ?", (source,)).fetchone()
        if row and row["sha256"] == digest:
            return -1  # unchanged: skip
        with self.db:
            self.db.execute("DELETE FROM chunks WHERE source = ?", (source,))
            pieces = chunk_text(text)
            self.db.executemany(
                "INSERT INTO chunks (text, heading, source) VALUES (?, ?, ?)",
                [(body, heading, source) for heading, body in pieces],
            )
            self.db.execute("INSERT OR REPLACE INTO documents VALUES (?, ?)", (source, digest))
        return len(pieces)

    def ingest_folder(self, folder: Path) -> dict:
        report = {"added": {}, "unchanged": [], "removed": [], "errors": {}}
        seen = set()
        for path in sorted(Path(folder).rglob("*")):
            if path.suffix.lower() not in SUPPORTED or not path.is_file():
                continue
            source = path.relative_to(folder).as_posix()
            seen.add(source)
            try:
                count = self.ingest_text(source, read_file(path))
            except Exception as exc:  # keep going; report the problem file
                report["errors"][source] = str(exc)
                continue
            if count < 0:
                report["unchanged"].append(source)
            else:
                report["added"][source] = count
        for (source,) in self.db.execute("SELECT source FROM documents").fetchall():
            if source not in seen:  # file deleted from the folder -> forget it
                with self.db:
                    self.db.execute("DELETE FROM chunks WHERE source = ?", (source,))
                    self.db.execute("DELETE FROM documents WHERE source = ?", (source,))
                report["removed"].append(source)
        return report

    def search(self, question: str, k: int = 4, max_chars: int = 3500) -> list[dict]:
        query = build_fts_query(question)
        if not query:
            return []
        rows = self.db.execute(
            # heading matches count 3x more than body matches
            "SELECT text, heading, source FROM chunks WHERE chunks MATCH ? "
            "ORDER BY bm25(chunks, 1.0, 3.0) LIMIT ?",
            (query, k),
        ).fetchall()
        results, used = [], 0
        for row in rows:
            if used + len(row["text"]) > max_chars and results:
                break  # stay inside the context budget
            results.append({"text": row["text"], "heading": row["heading"], "source": row["source"]})
            used += len(row["text"])
        return results

    def stats(self) -> dict:
        docs = self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": docs, "chunks": chunks}