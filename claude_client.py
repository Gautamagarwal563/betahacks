"""Claude (Anthropic) wrapper.

Used as the orchestrator LLM. Hackathon rules permit third-party LLMs for orchestration;
all video output still runs through BytePlus Seed models.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

import anthropic

_KEY = os.getenv("ANTHROPIC_API_KEY", "")
_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")


class NoKeyError(RuntimeError):
    pass


def _client() -> anthropic.Anthropic:
    if not _KEY:
        raise NoKeyError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=_KEY)


def chat(system: str, user: str, max_tokens: int = 4096, model: str = None) -> str:
    """Single-turn chat. System + user prompt in, text out."""
    r = _client().messages.create(
        model=model or _MODEL,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in r.content if b.type == "text")


def chat_json(system: str, user: str, max_tokens: int = 4096) -> dict:
    """Chat and parse JSON reply. Strips fences if present."""
    import json
    raw = chat(system, user, max_tokens=max_tokens)
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0]
    return json.loads(s.strip())
