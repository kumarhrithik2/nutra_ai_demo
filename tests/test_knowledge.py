from knowledge import KnowledgeBase, chunk_text
from config import get_settings


def make_kb():
    kb = KnowledgeBase(":memory:")
    kb.ingest_folder(get_settings().knowledge_dir)
    return kb


def test_chunking_splits_by_heading():
    chunks = chunk_text("# T\n\n## A\nalpha text\n\n## B\nbeta text")
    assert [h for h, _ in chunks] == ["A", "B"]


def test_search_finds_relevant_chunk_and_ignores_unrelated():
    kb = make_kb()
    hits = kb.search("what is the price of vitamin D3?")
    assert hits and hits[0]["heading"].startswith("Vitamin D3")
    assert kb.search("bitcoin mining rig") == []


def test_ingest_is_idempotent_and_detects_changes():
    kb = KnowledgeBase(":memory:")
    assert kb.ingest_text("a.md", "## X\nhello world") > 0
    assert kb.ingest_text("a.md", "## X\nhello world") == -1
    assert kb.ingest_text("a.md", "## X\nhello changed") > 0
    assert kb.stats()["documents"] == 1


def test_context_budget_is_respected():
    kb = make_kb()
    hits = kb.search("vitamin", k=10, max_chars=300)
    assert sum(len(h["text"]) for h in hits) <= 300 or len(hits) == 1