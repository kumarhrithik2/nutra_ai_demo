from assistant import Assistant, retrieval_query
from config import get_settings
from guardrails import Guard
from knowledge import KnowledgeBase
from llm import FakeProvider
from memory import ConversationStore


def make_bot(history_turns=4, summary_trigger=4):
    s = get_settings()
    kb = KnowledgeBase(":memory:")
    kb.ingest_folder(s.knowledge_dir)
    return Assistant(s, FakeProvider(), kb, ConversationStore(":memory:", history_turns, summary_trigger),
                     Guard.from_file(s.guard_rules_path))


def test_grounded_answer_with_sources():
    a = make_bot().ask("s1", "How much is Vitamin D3?")
    assert a.status == "ok" and "249" in a.text
    assert a.sources and a.sources[0]["source"] == "products.md"


def test_unknown_topic_declines():
    a = make_bot().ask("s1", "Tell me about the bitcoin mining rig hardware")
    assert "don't have verified information" in a.text


def test_emergency_skips_the_model_entirely():
    a = make_bot().ask("s1", "I have chest pain, which supplement should I take?")
    assert a.status == "emergency" and a.llm_calls == 0 and a.input_tokens == 0


def test_bad_model_answer_is_retried_then_replaced_by_fallback():
    bot = make_bot()
    a = bot.ask("s1", "Tell me about vitamin D3 [[bad]]")
    assert a.status == "fallback" and a.llm_calls == 2
    assert "banned_claims" in a.flags
    assert bot.store.get("s1").recent == []   # blocked answers are never remembered


def test_follow_up_borrows_previous_topic():
    recent = [{"role": "user", "content": "Tell me about Magnesium Glycinate"},
              {"role": "assistant", "content": "..."}]
    assert "Magnesium" in retrieval_query("what about the price?", recent)
    assert retrieval_query("Is Omega-3 suitable for vegetarians in general?", []) .startswith("Is Omega-3")
    bot = make_bot()
    bot.ask("s2", "Tell me about Magnesium Glycinate")
    follow = bot.ask("s2", "and its price?")
    assert "349" in follow.text


def test_old_turns_are_folded_into_a_summary():
    bot = make_bot(history_turns=1, summary_trigger=2)
    for q in ["Tell me about Vitamin B12", "And Vitamin D3?", "What about Zinc?"]:
        bot.ask("s3", q)
    hist = bot.store.get("s3")
    assert hist.summary.startswith("Summary:")
    assert len(hist.recent) <= 2 + 2   # window of one turn (+ pending)