"""Wrong-response detection, driven entirely by guard_rules.json (no code edits to add rules).

Two checkpoints:
  1. check_input()  runs BEFORE the model is called (emergencies, prompt injection, off-topic).
     A match returns a ready-made reply and the model is never called (saves money, safer).
  2. check_output() runs on the model's answer. Each rule has a severity:
        block -> the answer is rejected; the model gets ONE retry with the rule messages as
                 feedback; if it still fails, the safe fallback message is shown.
        fix   -> the answer is kept and a fixed sentence is appended (e.g. doctor reminder).
        warn  -> the answer is kept; the problem is only logged/reported.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

NEGATORS = re.compile(r"\b(not|no|never|cannot|can't|cant|don't|dont|doesn't|doesnt|won't|without|nor)\b|n't",
                      re.I)


@dataclass
class Violation:
    rule_id: str
    severity: str
    message: str


@dataclass
class OutputVerdict:
    violations: list[Violation] = field(default_factory=list)
    appended: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "block"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warn"]


@dataclass
class InputVerdict:
    action: str          # "allow" | "emergency" | "refuse_out_of_scope" | "refuse_injection"
    rule_id: str = ""
    reply: str = ""


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _word_re(phrase: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.I)


def _extract_numbers(text: str, units: list[str]) -> list[str]:
    unit_alt = "|".join(re.escape(u) for u in sorted(units, key=len, reverse=True))
    found = [m.group(1) for m in re.finditer(rf"(\d[\d,]*(?:\.\d+)?)\s*(?:{unit_alt})(?!\w)", text, re.I)]
    found += [m.group(1) for m in re.finditer(r"₹\s*(\d[\d,]*(?:\.\d+)?)", text)]
    return [n.replace(",", "") for n in found]


class Guard:
    def __init__(self, config: dict):
        self.config = config
        self.messages = config.get("messages", {})

    @classmethod
    def from_file(cls, path: Path) -> "Guard":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ input
    def check_input(self, text: str) -> InputVerdict:
        for rule in self.config.get("input_rules", []):
            hit = False
            if rule["type"] == "contains_any":
                hit = any(_word_re(p).search(text) for p in rule["patterns"])
            elif rule["type"] == "regex":
                hit = any(re.search(p, text, re.I) for p in rule["patterns"])
            if hit:
                action = rule["action"]  # the action name is also the key in "messages"
                reply = self.messages.get(action, self.messages.get("fallback", ""))
                return InputVerdict(action, rule["id"], reply)
        return InputVerdict("allow")

    # ----------------------------------------------------------------- output
    def check_output(self, answer: str, *, user_message: str, context_text: str,
                     has_context: bool) -> OutputVerdict:
        verdict = OutputVerdict()
        for rule in self.config.get("output_rules", []):
            rid, sev, kind = rule["id"], rule.get("severity", "block"), rule["type"]
            message = rule.get("message", rid)

            def add(extra: str = ""):
                verdict.violations.append(Violation(rid, sev, f"{message} {extra}".strip()))

            if kind == "contains_any":
                for sentence in _sentences(answer):
                    hits = [p for p in rule["patterns"] if _word_re(p).search(sentence)]
                    if hits and not (rule.get("ignore_negated") and NEGATORS.search(sentence)):
                        add(f'(found: "{hits[0]}")')
                        break
            elif kind == "regex":
                for pattern in rule["patterns"]:
                    m = re.search(pattern, answer, re.I)
                    if m:
                        add(f'(found: "{m.group(0)[:40]}")')
                        break
            elif kind == "numbers_in_context":
                context_norm = context_text.replace(",", "")
                for number in _extract_numbers(answer, rule["units"]):
                    if not re.search(rf"(?<![\d.]){re.escape(number)}(?!\d)", context_norm):
                        add(f'(number "{number}" is not in the context)')
                        break
            elif kind == "unlisted_product":
                ctx = context_text.lower()
                for name in rule["known_products"]:
                    if _word_re(name).search(answer) and name.lower() not in ctx:
                        add(f'("{name}" is not in the context)')
                        break
            elif kind == "no_context_must_decline":
                if not has_context and not any(p.lower() in answer.lower() for p in rule["decline_phrases"]):
                    add()
            elif kind == "requires_when":
                # triggered by the customer's words, not by our own answer text
                triggered = any(t in user_message.lower() for t in rule["when_any"])
                satisfied = any(p in answer.lower() for p in rule["must_contain_any"])
                if triggered and not satisfied:
                    verdict.violations.append(Violation(rid, sev, message))
                    if sev == "fix" and rule.get("append"):
                        verdict.appended.append(rule["append"])
            elif kind == "min_chars":
                if len(answer.strip()) < rule["value"]:
                    add()
            elif kind == "max_chars":
                if len(answer) > rule["value"]:
                    add()
        return verdict