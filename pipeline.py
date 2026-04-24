"""Parallel render pipeline — takes a Session, renders shots concurrently.

Emits events to a callback (bus.emit) so the dashboard can stream progress.

Modes:
- CLIP_MODE=kenburns — ffmpeg pan/zoom on Seedream stills (works today)
- CLIP_MODE=seedance — real Seedance 2.0 i2v (when account credit returns)
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
load_dotenv()

import byteplus
import fal_client
import director as _director
from director import Session, Shot, ShotStatus

CLIP_MODE = os.getenv("CLIP_MODE", "kenburns")
VIDEOS_DIR = Path("videos")

# Event callback wiring — dashboard / voice subscribe to this
_subscribers: list[Callable[[str, dict], None]] = []


def subscribe(fn: Callable[[str, dict], None]) -> None:
    _subscribers.append(fn)


def emit(event: str, data: dict) -> None:
    for fn in _subscribers:
        try:
            fn(event, data)
        except Exception:
            pass


# ---------- single-shot render ----------

def render_keyframe(session: Session, shot: Shot) -> None:
    shot.status = ShotStatus.KEYFRAME
    _director.dump_session(session)
    emit("shot.status", {"call_id": session.call_id, "shot_id": shot.id, "status": shot.status.value})
    img = byteplus.generate_image(shot.prompt)
    run_dir = VIDEOS_DIR / session.call_id / "keyframes"
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / f"kf_{shot.index:02d}.jpeg"
    byteplus.download(img.url, dest)
    shot.keyframe_url = img.url
    shot.keyframe_path = str(dest)
    _director.dump_session(session)
    emit("shot.keyframe", {
        "call_id": session.call_id, "shot_id": shot.id,
        "keyframe_url": img.url, "keyframe_path": str(dest),
    })


def _kenburns_clip(shot: Shot, out_dir: Path) -> Path:
    out = out_dir / f"clip_{shot.index:02d}.mp4"
    src = Path(shot.keyframe_path)
    dur = max(3.0, float(shot.duration))
    fps = 30
    n = int(dur * fps)
    mode = shot.index % 4
    if mode == 0:   # zoom-in
        zp = f"zoompan=z='min(zoom+0.0008,1.25)':d={n}:s=1920x1080:fps={fps}"
    elif mode == 1: # zoom-out
        zp = f"zoompan=z='if(lte(zoom,1.0),1.25,max(1.001,zoom-0.0008))':d={n}:s=1920x1080:fps={fps}"
    elif mode == 2: # pan left
        zp = f"zoompan=z=1.15:x='if(lte(on,1),0,x-1)':y=0:d={n}:s=1920x1080:fps={fps}"
    else:           # pan right
        zp = f"zoompan=z=1.15:x='if(lte(on,1),iw-iw/zoom,x+1)':y=0:d={n}:s=1920x1080:fps={fps}"
    subprocess.run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(src),
        "-vf", f"scale=3840:2160,{zp}",
        "-t", f"{dur}", "-r", f"{fps}", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        str(out),
    ], check=True, capture_output=True)
    return out


def _seedance_clip(shot: Shot, out_dir: Path) -> Path:
    dur = max(5, int(round(shot.duration)))
    prompt = f"{shot.prompt} --duration {dur} --resolution 720p"
    res = byteplus.generate_video(prompt=prompt, image_url=shot.keyframe_url)
    out = out_dir / f"clip_{shot.index:02d}.mp4"
    byteplus.download(res.url, out)
    return out


def _fal_clip(shot: Shot, out_dir: Path) -> Path:
    """Real Seedance 2.0 via fal.ai gateway (paid, US-accessible)."""
    dur = max(5, int(round(shot.duration)))
    res = fal_client.generate_video(
        prompt=shot.prompt,
        image_url=shot.keyframe_url,
        duration=dur,
        resolution="720p",
    )
    out = out_dir / f"clip_{shot.index:02d}.mp4"
    fal_client.download(res.url, out)
    return out


def render_clip(session: Session, shot: Shot) -> None:
    if not shot.keyframe_path:
        render_keyframe(session, shot)
    shot.status = ShotStatus.RENDERING
    _director.dump_session(session)
    emit("shot.status", {"call_id": session.call_id, "shot_id": shot.id, "status": shot.status.value})
    out_dir = VIDEOS_DIR / session.call_id / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = {
        "seedance": _seedance_clip,
        "fal": _fal_clip,
        "kenburns": _kenburns_clip,
    }.get(CLIP_MODE, _kenburns_clip)
    try:
        path = renderer(shot, out_dir)
        shot.clip_path = str(path)
        shot.status = ShotStatus.DONE
    except Exception as e:
        shot.error = str(e)
        shot.status = ShotStatus.FAILED
    _director.dump_session(session)
    emit("shot.clip", {
        "call_id": session.call_id, "shot_id": shot.id,
        "status": shot.status.value,
        "clip_path": shot.clip_path, "error": shot.error,
    })


# ---------- parallel orchestration ----------

_executor = ThreadPoolExecutor(max_workers=8)
_inflight: dict[str, Future] = {}   # shot_id -> Future


def render_shot_async(session: Session, shot: Shot) -> Future:
    fut = _executor.submit(render_clip, session, shot)
    _inflight[shot.id] = fut
    return fut


def render_shots(session: Session, shots: list[Shot]) -> list[Future]:
    return [render_shot_async(session, s) for s in shots]


def wait_all(session: Session, timeout: float = 300) -> None:
    """Block until every in-flight shot finishes (or timeout)."""
    start = time.time()
    while time.time() - start < timeout:
        pending = [f for f in _inflight.values() if not f.done()]
        if not pending:
            return
        time.sleep(0.5)


# ---------- narration + stitch ----------

def render_narration(session: Session, out_dir: Path) -> Optional[Path]:
    lines = [s.narration.strip() for s in session.shots if s.narration.strip()]
    if not lines:
        return None
    text = "  ".join(lines)
    aiff = out_dir / "narration.aiff"
    mp3 = out_dir / "narration.mp3"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["say", "-v", "Daniel", "-r", "170", "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(aiff), "-b:a", "192k", str(mp3)],
                   check=True, capture_output=True)
    aiff.unlink()
    return mp3


def finalize(session: Session) -> Path:
    """Wait for all shots, stitch, produce final.mp4."""
    wait_all(session)
    run_dir = VIDEOS_DIR / session.call_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # write concat list in shot-index order, skipping failed
    ordered = sorted(
        [s for s in session.shots if s.status == ShotStatus.DONE and s.clip_path],
        key=lambda s: s.index,
    )
    if not ordered:
        raise RuntimeError("no shots rendered — cannot finalize")

    clips_txt = run_dir / "clips.txt"
    clips_txt.write_text("\n".join(f"file '{Path(s.clip_path).resolve()}'" for s in ordered))

    silent = run_dir / "silent.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(clips_txt),
        "-c", "copy", str(silent),
    ], check=True, capture_output=True)

    narration = render_narration(session, run_dir)
    final = run_dir / "final.mp4"
    if narration:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(silent), "-i", str(narration),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest",
            str(final),
        ], check=True, capture_output=True)
    else:
        final = silent.rename(final)

    session.final_video_path = str(final)
    emit("session.finalized", {"call_id": session.call_id, "path": str(final)})
    return final


if __name__ == "__main__":
    # smoke test — load the director test session and render it
    import director, sys
    call_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not call_id:
        # pick most recent
        state_files = sorted(Path("state").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not state_files:
            print("no sessions found — run `python director.py` first")
            sys.exit(1)
        call_id = state_files[0].stem
    sess = director.load_session(call_id)
    print(f"rendering session {call_id}: {sess.title} ({len(sess.shots)} shots)")

    def log(e, d): print(f"  event {e}: {d.get('shot_id', d.get('path', ''))} {d.get('status', '')}")
    subscribe(log)

    render_shots(sess, sess.shots)
    final = finalize(sess)
    print(f"\nfinal: {final}")
    director.dump_session(sess)
