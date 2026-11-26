"""All settings in one place. Values come from environment variables or a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

BASE_DIR = Path(__file__).resolve().parent


def _float_or_none(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


@dataclass(frozen=True)
class Settings:
    provider: str
    gemini_api_key: str
    gemini_model: str
    openai_api_key: str
    openai_model: str
    openai_base_url: str
    anthropic_api_key: str
    anthropic_model: str
    temperature: float | None
    max_output_tokens: int
    timeout_seconds: float
    db_path: Path
    knowledge_dir: Path
    guard_rules_path: Path
    top_k: int                 # how many knowledge chunks go into each prompt
    max_context_chars: int     # hard cap on the knowledge text sent per question
    history_turns: int         # recent question/answer pairs kept word-for-word
    summary_trigger: int       # fold older messages into the summary once this many pile up
    max_input_chars: int       # longest customer message accepted
    price_in_per_m: float
    price_out_per_m: float


def get_settings() -> Settings:
    env = os.getenv
    return Settings(
        provider=env("LLM_PROVIDER", "fake").lower(),
        gemini_api_key=env("GEMINI_API_KEY", ""),
        gemini_model=env("GEMINI_MODEL", "gemini-flash-latest"),
        # openai_api_key=env("OPENAI_API_KEY", ""),
        # openai_model=env("OPENAI_MODEL", "gpt-6-luna"),
        # openai_base_url=env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        # anthropic_api_key=env("ANTHROPIC_API_KEY", ""),
        # anthropic_model=env("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        temperature=_float_or_none(env("LLM_TEMPERATURE")),
        max_output_tokens=int(env("MAX_OUTPUT_TOKENS", "600")),
        timeout_seconds=float(env("LLM_TIMEOUT_SECONDS", "30")),
        db_path=Path(env("DB_PATH", str(BASE_DIR / "data" / "nutra.db"))),
        knowledge_dir=Path(env("KNOWLEDGE_DIR", str(BASE_DIR / "data" / "knowledge"))),
        guard_rules_path=Path(env("GUARD_RULES_PATH", str(BASE_DIR / "guard_rules.json"))),
        top_k=int(env("TOP_K", "4")),
        max_context_chars=int(env("MAX_CONTEXT_CHARS", "3500")),
        history_turns=int(env("HISTORY_TURNS", "4")),
        summary_trigger=int(env("SUMMARY_TRIGGER", "4")),
        max_input_chars=int(env("MAX_INPUT_CHARS", "1000")),
        price_in_per_m=float(env("PRICE_INPUT_PER_M", "0.10")),
        price_out_per_m=float(env("PRICE_OUTPUT_PER_M", "0.50")),
    )