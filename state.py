"""Persistent channel state. Just JSON files in ./state/.

Schema (state/channel.json):
{
  "niche": "...",
  "runs": [
    {
      "run_id": "...",
      "ts": 1777034346,
      "topic": "...",
      "title": "...",
      "video_path": "...",
      "youtube_id": "..." | null,
      "status": "rendered" | "uploaded" | "failed",
      "stats": {"views": 0, "likes": 0, "comments": 0}
    }
  ],
  "last_comment_check": 1777034346
}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

STATE_DIR = Path("state")
STATE_FILE = STATE_DIR / "channel.json"


def load() -> dict[str, Any]:
    STATE_DIR.mkdir(exist_ok=True)
    if not STATE_FILE.exists():
        return {"niche": None, "runs": [], "last_comment_check": 0}
    return json.loads(STATE_FILE.read_text())


def save(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def record_run(run_id: str, topic: str, title: str, video_path: str,
               status: str = "rendered", youtube_id: str | None = None) -> None:
    state = load()
    state["runs"].append({
        "run_id": run_id,
        "ts": int(time.time()),
        "topic": topic,
        "title": title,
        "video_path": video_path,
        "youtube_id": youtube_id,
        "status": status,
        "stats": {"views": 0, "likes": 0, "comments": 0},
    })
    save(state)


def update_run(run_id: str, **fields) -> None:
    state = load()
    for r in state["runs"]:
        if r["run_id"] == run_id:
            r.update(fields)
            break
    save(state)


def recent_topics(limit: int = 10) -> list[str]:
    state = load()
    return [r["topic"] for r in state["runs"][-limit:]]
