"""Talks to the model service over plain HTTPS so you can see exactly what is sent and received.

Three interchangeable providers share one method:   chat(system, messages, max_tokens) -> LLMResult
  - OpenAIProvider     POST {base_url}/chat/completions
  - AnthropicProvider  POST https://api.anthropic.com/v1/messages
  - FakeProvider       offline stand-in so you can run everything without an API key
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import httpx

from config import Settings
from prompts import NO_CONTEXT_MARKER


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


def _post_with_retry(url: str, headers: dict, payload: dict, timeout: float, name: str) -> dict:
    """POST JSON; retry twice on rate limits (429) and server errors (5xx)."""
    last = ""
    for attempt in range(3):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        except httpx.HTTPError as exc:
            last = f"network error: {exc}"
        else:
            if resp.status_code < 400:
                return resp.json()
            last = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code not in (429, 500, 502, 503, 504):
                break
        time.sleep(1.5 * (attempt + 1))
    raise LLMError(f"{name} request failed ({last})")


class OpenAIProvider:
    def __init__(self, s: Settings):
        if not s.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        self.s = s

    def chat(self, system: str, messages: list[dict], max_tokens: int) -> LLMResult:
        payload = {
            "model": self.s.openai_model,
            # the system prompt is just the first message with role "system"
            "messages": [{"role": "system", "content": system}, *messages],
            "max_completion_tokens": max_tokens,
        }
        if self.s.temperature is not None:
            payload["temperature"] = self.s.temperature
        data = _post_with_retry(
            f"{self.s.openai_base_url}/chat/completions",
            {"Authorization": f"Bearer {self.s.openai_api_key}"},
            payload, self.s.timeout_seconds, "OpenAI",
        )
        text = (data["choices"][0]["message"].get("content") or "").strip()
        if not text:
            raise LLMError("Empty reply. If the model uses hidden reasoning, raise MAX_OUTPUT_TOKENS.")
        usage = data.get("usage", {})
        return LLMResult(text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


class AnthropicProvider:
    def __init__(self, s: Settings):
        if not s.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.s = s

    def chat(self, system: str, messages: list[dict], max_tokens: int) -> LLMResult:
        payload = {
            "model": self.s.anthropic_model,
            "max_tokens": max_tokens,
            # cache_control marks the (unchanging) system prompt as cacheable
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
        }
        if self.s.temperature is not None:
            payload["temperature"] = self.s.temperature
        data = _post_with_retry(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.s.anthropic_api_key, "anthropic-version": "2023-06-01"},
            payload, self.s.timeout_seconds, "Anthropic",
        )
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        if not text:
            raise LLMError("Empty reply from Anthropic")
        usage = data.get("usage", {})
        return LLMResult(text, usage.get("input_tokens", 0), usage.get("output_tokens", 0))


class FakeProvider:
    """Deterministic, offline. It reads the <context> it is given and quotes it back.
    Type the marker [[bad]] in a question to make it produce a rule-breaking answer
    (lets you watch the wrong-response checks fire)."""

    def chat(self, system: str, messages: list[dict], max_tokens: int) -> LLMResult:
        last = messages[-1]["content"]
        if system.startswith("You compress chat history"):
            body = last.split("New messages to fold in:")[-1].strip().replace("\n", " ")
            text = "Summary: " + body[:300]
        elif "[[bad]]" in last:
            text = "This supplement will cure your diabetes. Guaranteed results, no side effects!"
        else:
            match = re.search(r"<context>\n(.*?)\n</context>", last, re.S)
            context = match.group(1) if match else NO_CONTEXT_MARKER
            if context.strip() == NO_CONTEXT_MARKER:
                text = ("I don't have verified information on that. Please contact NutraAI "
                        "support or ask a doctor or pharmacist.")
            else:
                first = re.sub(r"^\[\d+\] \([^)]*\) ", "", context.split("\n")[0])
                text = f"Here is what our information says: {first[:350]}"
        in_tokens = (len(system) + sum(len(m["content"]) for m in messages)) // 4
        return LLMResult(text, in_tokens, len(text) // 4)


def make_provider(s: Settings):
    if s.provider == "openai":
        return OpenAIProvider(s)
    if s.provider == "anthropic":
        return AnthropicProvider(s)
    if s.provider == "fake":
        return FakeProvider()
    raise LLMError(f"Unknown LLM_PROVIDER '{s.provider}' (use fake, openai or anthropic)")