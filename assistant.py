"""The pipeline that ties everything together. Call Assistant.ask(session_id, text)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import Settings, get_settings
from guardrails import Guard
from knowledge import KnowledgeBase
from llm import LLMError, make_provider
from memory import ConversationStore
from prompts import SYSTEM_PROMPT, build_user_turn

FOLLOW_UP_WORDS = re.compile(r"\b(it|that|this|they|them|those|these|one|ones|also|too|else|instead)\b", re.I)


@dataclass
class Answer:
    text: str
    status: str                      # ok | fixed | retried | fallback | refused | emergency | error
    sources: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)       # rule ids that fired
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0


def retrieval_query(question: str, recent: list[dict]) -> str:
    """Follow-ups like 'what about its side effects?' have no topic words, so borrow the previous
    customer question. A plain heuristic: no extra model call, no extra cost."""
    words = question.split()
    if recent and (len(words) <= 6 or FOLLOW_UP_WORDS.search(question)):
        previous = [m["content"] for m in recent if m["role"] == "user"]
        if previous:
            return f"{previous[-1]} {question}"
    return question


class Assistant:
    def __init__(self, settings: Settings | None = None, provider=None, kb: KnowledgeBase | None = None,
                 store: ConversationStore | None = None, guard: Guard | None = None):
        self.s = settings or get_settings()
        self.provider = provider or make_provider(self.s)
        self.kb = kb or KnowledgeBase(self.s.db_path)
        self.store = store or ConversationStore(self.s.db_path, self.s.history_turns, self.s.summary_trigger)
        self.guard = guard or Guard.from_file(self.s.guard_rules_path)

    def ask(self, session_id: str, user_message: str) -> Answer:
        question = " ".join(user_message.split())[: self.s.max_input_chars]
        if not question:
            return Answer("Please type a question.", "error")

        # 1) Input checks: emergencies, injection, off-topic. No model call, no cost.
        verdict_in = self.guard.check_input(question)
        if verdict_in.action != "allow":
            status = "emergency" if verdict_in.action == "emergency" else "refused"
            return Answer(verdict_in.reply, status, flags=[verdict_in.rule_id])

        # 2) Memory: summary of old turns + the last few messages word-for-word.
        history = self.store.get(session_id)

        # 3) Retrieval: only the few matching knowledge chunks, never the whole library.
        chunks = self.kb.search(retrieval_query(question, history.recent), self.s.top_k,
                                self.s.max_context_chars)
        context_text = "\n".join(c["text"] for c in chunks)

        answer = Answer("", "ok", sources=[{"source": c["source"], "heading": c["heading"]} for c in chunks])

        def call(correction: str = "") -> str:
            user_turn = build_user_turn(question, chunks, history.summary, correction)
            messages = [*history.recent, {"role": "user", "content": user_turn}]
            result = self.provider.chat(SYSTEM_PROMPT, messages, self.s.max_output_tokens)
            answer.input_tokens += result.input_tokens
            answer.output_tokens += result.output_tokens
            answer.llm_calls += 1
            return result.text

        def judge(text: str):
            return self.guard.check_output(text, user_message=question, context_text=context_text,
                                           has_context=bool(chunks))

        try:
            text = call()
            verdict = judge(text)
            # 4) Wrong-response detection: one retry with the rule messages as feedback.
            if verdict.blocking:
                answer.flags += [v.rule_id for v in verdict.blocking]
                feedback = "Your previous draft broke these rules. Rewrite it so it complies:\n" + "\n".join(
                    f"- {v.message}" for v in verdict.blocking)
                text = call(feedback)
                verdict = judge(text)
                answer.status = "retried"
            if verdict.blocking:
                answer.flags += [v.rule_id for v in verdict.blocking]
                answer.text, answer.status = self.guard.messages["fallback"], "fallback"
            else:
                answer.text = text
                if verdict.appended:
                    answer.text += "\n\n" + " ".join(dict.fromkeys(verdict.appended))
                    answer.status = "fixed" if answer.status == "ok" else answer.status
                answer.flags += [v.rule_id for v in verdict.violations if v.severity in ("fix", "warn")]
        except LLMError as exc:
            return Answer(f"The assistant is temporarily unavailable ({exc}).", "error")

        # 5) Save only what the customer actually saw, then compact old turns into the summary.
        if answer.status != "fallback":
            self.store.add(session_id, "user", question)
            self.store.add(session_id, "assistant", answer.text)
            try:
                used_in, used_out = self.store.compact(session_id, self.provider)
                answer.input_tokens += used_in
                answer.output_tokens += used_out
                answer.llm_calls += 1 if used_in else 0
            except LLMError:
                pass  # summary is an optimisation; try again next turn
        return answer

    def cost_usd(self, answer: Answer) -> float:
        return (answer.input_tokens * self.s.price_in_per_m + answer.output_tokens * self.s.price_out_per_m) / 1e6