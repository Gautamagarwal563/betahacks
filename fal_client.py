"""fal.ai Seedance 2.0 gateway — same interface as byteplus.generate_video.

Used when BytePlus shared account is overdue and you want real Seedance motion.
Paid: roughly $0.05 per 5-second 720p clip.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv("FAL_API_KEY", "")
BASE = "https://queue.fal.run"

# fal model paths
T2V = "fal-ai/bytedance/seedance/v1/lite/text-to-video"
I2V = "fal-ai/bytedance/seedance/v1/lite/image-to-video"
PRO_T2V = "fal-ai/bytedance/seedance/v1/pro/text-to-video"
PRO_I2V = "fal-ai/bytedance/seedance/v1/pro/image-to-video"


class FalError(Exception):
    pass


@dataclass
class VideoResult:
    url: str
    request_id: str
    seed: Optional[int] = None


def _headers() -> dict:
    if not KEY:
        raise FalError("FAL_API_KEY not set")
    return {"Authorization": f"Key {KEY}", "Content-Type": "application/json"}


def _submit(path: str, payload: dict) -> tuple[str, str]:
    r = httpx.post(f"{BASE}/{path}", headers=_headers(), json=payload, timeout=30)
    if r.status_code >= 400:
        raise FalError(f"fal submit HTTP {r.status_code}: {r.text[:500]}")
    body = r.json()
    rid = body.get("request_id")
    status_url = body.get("status_url")
    if not rid or not status_url:
        raise FalError(f"fal submit missing request_id/status_url: {body}")
    return rid, status_url


def _poll(status_url: str, response_url: str, timeout: float = 300) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        r = httpx.get(status_url, headers=_headers(), timeout=30)
        if r.status_code >= 400:
            raise FalError(f"fal status HTTP {r.status_code}: {r.text[:500]}")
        s = r.json().get("status")
        if s == "COMPLETED":
            res = httpx.get(response_url, headers=_headers(), timeout=30)
            res.raise_for_status()
            return res.json()
        if s in ("ERROR", "CANCELLED"):
            raise FalError(f"fal job {s}: {r.json()}")
        time.sleep(3)
    raise FalError(f"fal job timeout after {timeout}s")


def generate_video(
    prompt: str,
    image_url: Optional[str] = None,
    duration: int = 5,
    resolution: str = "720p",
    pro: bool = False,
) -> VideoResult:
    """Matches byteplus.generate_video signature."""
    if image_url:
        path = PRO_I2V if pro else I2V
        payload = {
            "prompt": prompt,
            "image_url": image_url,
            "duration": str(duration),
            "resolution": resolution,
        }
    else:
        path = PRO_T2V if pro else T2V
        payload = {
            "prompt": prompt,
            "duration": str(duration),
            "resolution": resolution,
        }
    rid, status_url = _submit(path, payload)
    # response_url is the base request URL (no /status suffix)
    response_url = status_url.rsplit("/status", 1)[0]
    body = _poll(status_url, response_url)
    video = body.get("video") or {}
    url = video.get("url")
    if not url:
        raise FalError(f"fal completed but no video url: {body}")
    return VideoResult(url=url, request_id=rid, seed=body.get("seed"))


def download(url: str, dest) -> None:
    from pathlib import Path
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)


if __name__ == "__main__":
    # smoke test — same Tesla prompt
    print("submitting…")
    res = generate_video(
        prompt="a red tesla roadster driving fast through the desert at golden hour, cinematic",
        duration=5,
        resolution="720p",
    )
    print(f"done: {res.url}")
    download(res.url, "videos/fal_smoke.mp4")
    print("saved to videos/fal_smoke.mp4")
