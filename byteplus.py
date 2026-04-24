"""BytePlus Seed client — Seedance 2.0 video, Seedream 5.0 image, Seed 2.0 chat.

Handles:
- async video task creation + polling
- key rotation on AccountOverdueError
- clear failure surface (raises OverdueError, callers decide whether to retry later)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE = os.getenv("BYTEPLUS_BASE_URL", "https://ark.ap-southeast.bytepluses.com/api/v3")
KEYS = [k.strip() for k in os.getenv("BYTEPLUS_API_KEYS", "").split(",") if k.strip()]
SEEDANCE = os.getenv("SEEDANCE_MODEL", "dreamina-seedance-2-0-fast-260128")
SEEDREAM = os.getenv("SEEDREAM_MODEL", "seedream-5-0-260128")
SEED2 = os.getenv("SEED2_MODEL", "seed-2-0-pro-260328")


class OverdueError(Exception):
    """All shared keys are AccountOverdue. Caller decides whether to wait or fallback."""


class ByteplusError(Exception):
    pass


def _post(path: str, payload: dict, timeout: float = 30.0) -> dict:
    """POST with key rotation on AccountOverdueError."""
    if not KEYS:
        raise ByteplusError("BYTEPLUS_API_KEYS not set")
    last_err = None
    for key in KEYS:
        r = httpx.post(
            f"{BASE}{path}",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json()
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        code = (body.get("error") or {}).get("code")
        if code == "AccountOverdueError":
            last_err = OverdueError(body)
            continue  # try next key
        raise ByteplusError(f"HTTP {r.status_code}: {body}")
    raise last_err or ByteplusError("all keys failed")


def _get(path: str, timeout: float = 30.0) -> dict:
    for key in KEYS:
        r = httpx.get(
            f"{BASE}{path}",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json()
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        if (body.get("error") or {}).get("code") == "AccountOverdueError":
            continue
        raise ByteplusError(f"HTTP {r.status_code}: {body}")
    raise OverdueError("all keys overdue")


# ---------- Seed 2.0 chat (used by orchestrator) ----------

def chat(system: str, user: str, model: str = SEED2, max_tokens: int = 2048) -> str:
    body = _post("/chat/completions", {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
    })
    return body["choices"][0]["message"]["content"]


# ---------- Seedream 5.0 image ----------

@dataclass
class ImageResult:
    url: str

def generate_image(prompt: str, size: str = "2560x1440", model: str = SEEDREAM) -> ImageResult:
    body = _post("/images/generations", {
        "model": model,
        "prompt": prompt,
        "size": size,
    })
    data = body.get("data") or []
    if not data:
        raise ByteplusError(f"no image data: {body}")
    return ImageResult(url=data[0]["url"])


# ---------- Seedance 2.0 video ----------

@dataclass
class VideoResult:
    url: str
    task_id: str

def _submit_video_task(prompt: str, model: str, image_url: Optional[str] = None) -> str:
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    body = _post("/contents/generations/tasks", {
        "model": model,
        "content": content,
    })
    task_id = body.get("id") or body.get("task_id")
    if not task_id:
        raise ByteplusError(f"no task id: {body}")
    return task_id

def _poll_video_task(task_id: str, timeout: float = 600, interval: float = 5) -> VideoResult:
    start = time.time()
    while time.time() - start < timeout:
        body = _get(f"/contents/generations/tasks/{task_id}")
        status = body.get("status")
        if status == "succeeded":
            url = (body.get("content") or {}).get("video_url") or body.get("video_url")
            if not url:
                raise ByteplusError(f"succeeded but no url: {body}")
            return VideoResult(url=url, task_id=task_id)
        if status == "failed":
            raise ByteplusError(f"task failed: {body}")
        time.sleep(interval)
    raise ByteplusError(f"timed out after {timeout}s (task {task_id})")

def generate_video(prompt: str, image_url: Optional[str] = None,
                   model: str = SEEDANCE, timeout: float = 600) -> VideoResult:
    """Submit Seedance task + poll until done. Raises OverdueError if balance empty."""
    task_id = _submit_video_task(prompt, model=model, image_url=image_url)
    return _poll_video_task(task_id, timeout=timeout)


# ---------- Download helper ----------

def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return dest


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if cmd == "probe":
        try:
            out = chat("You are terse.", "Say hi in 3 words.")
            print("chat OK:", out)
        except OverdueError:
            print("chat: ALL KEYS OVERDUE (expected until Daniel reloads)")
        except Exception as e:
            print("chat ERR:", e)
        try:
            img = generate_image("a cat in a garden, photoreal")
            print("image OK:", img.url)
        except OverdueError:
            print("image: ALL KEYS OVERDUE")
        except Exception as e:
            print("image ERR:", e)
