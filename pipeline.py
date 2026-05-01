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

import httpx
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

def render_keyframe(session: Session, shot: Shot) -> bool:
    """Returns True if keyframe succeeded, False if Seedream is unavailable.
    Caller should fall back to text-to-video when this returns False."""
    shot.status = ShotStatus.KEYFRAME
    _director.dump_session(session)
    emit("shot.status", {"call_id": session.call_id, "shot_id": shot.id, "status": shot.status.value})
    try:
        img = byteplus.generate_image(shot.prompt)
    except byteplus.OverdueError:
        print(f"  [keyframe {shot.index}] Seedream overdue — will use T2V fallback")
        return False
    except Exception as e:
        print(f"  [keyframe {shot.index}] failed: {e} — will use T2V fallback")
        return False
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
    return True


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
    dur = max(5, min(10, int(round(shot.duration))))   # fal seedance lite: 5-10s
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
    try:
        if not shot.keyframe_path:
            render_keyframe(session, shot)  # may return False; that's fine
        shot.status = ShotStatus.RENDERING
        _director.dump_session(session)
        emit("shot.status", {"call_id": session.call_id, "shot_id": shot.id, "status": shot.status.value})
        out_dir = VIDEOS_DIR / session.call_id / "clips"
        out_dir.mkdir(parents=True, exist_ok=True)
        # pick renderer; if we have no keyframe and mode is kenburns (needs a still), force fal T2V
        effective_mode = CLIP_MODE
        if CLIP_MODE == "kenburns" and not shot.keyframe_url:
            print(f"  [clip {shot.index}] no keyframe — kenburns impossible, forcing fal T2V")
            effective_mode = "fal"
        renderer = {
            "seedance": _seedance_clip,
            "fal": _fal_clip,
            "kenburns": _kenburns_clip,
        }.get(effective_mode, _fal_clip)
        path = renderer(shot, out_dir)
        path = _bake_shot_audio(shot, path)  # Aura VO + silent fallback
        shot.clip_path = str(path)
        shot.status = ShotStatus.DONE
    except Exception as e:
        shot.error = f"{type(e).__name__}: {e}"
        shot.status = ShotStatus.FAILED
        print(f"  [clip {shot.index}] FAILED: {shot.error}")
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


# ---------- title + outro cards (Pillow-based, works on minimal ffmpeg) ----------

def _find_serif_font() -> str:
    candidates = [
        "/System/Library/Fonts/Supplemental/Baskerville.ttc",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Times.ttc",
        "/Library/Fonts/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return ""


def _find_mono_font() -> str:
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Courier New.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return ""


def _render_card_png(title: str, subtitle: str, out: Path,
                     size: tuple = (1920, 1080)) -> Path:
    """Render a cinematic title/outro card to PNG via Pillow."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = size
    img = Image.new("RGB", (W, H), (5, 5, 5))
    draw = ImageDraw.Draw(img)

    serif_path = _find_serif_font()
    mono_path = _find_mono_font()
    try:
        title_font = ImageFont.truetype(serif_path, 110) if serif_path else ImageFont.load_default()
    except Exception:
        title_font = ImageFont.load_default()
    try:
        sub_font = ImageFont.truetype(mono_path, 26) if mono_path else ImageFont.load_default()
    except Exception:
        sub_font = ImageFont.load_default()

    # subtle radial warmth via gradient: fake with concentric rectangles
    for i in range(60):
        alpha = int(8 * (1 - i / 60))
        if alpha <= 0: break
        draw.rectangle([i, i, W - i, H - i], outline=(20, 20, 25, alpha))

    # title (wrap at ~24 chars per line)
    words = title.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        bbox = draw.textbbox((0, 0), test, font=title_font)
        if bbox[2] > W * 0.8 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)
    text = "\n".join(lines[:3])
    # compute bounding box using textbbox for multiline
    try:
        bbox = draw.multiline_textbbox((0, 0), text, font=title_font, spacing=10, align="center")
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = W // 2, 100
    draw.multiline_text(((W - tw) // 2, (H - th) // 2 - 40), text,
                        font=title_font, fill=(244, 244, 245), spacing=10, align="center")

    # subtitle (cyan mono, below)
    if subtitle:
        bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sw = bbox[2] - bbox[0]
        draw.text(((W - sw) // 2, (H + th) // 2 + 24), subtitle,
                  font=sub_font, fill=(103, 232, 249))

    # thin accent line above subtitle
    line_y = (H + th) // 2 + 12
    draw.line([(W // 2 - 60, line_y), (W // 2 + 60, line_y)], fill=(103, 232, 249), width=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG")
    return out


def _make_card(text: str, subtitle: str, out: Path, dur: float = 2.0,
               variant: str = "intro") -> Path:
    """PNG card → mp4 via ffmpeg -loop (no drawtext needed)."""
    png = out.with_suffix(".png")
    _render_card_png(text, subtitle, png)
    subprocess.run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(png),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", f"{dur}", "-r", "30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        "-vf", "scale=1920:1080",
        str(out),
    ], check=True, capture_output=True)
    return out


def _render_caption_png(text: str, out: Path,
                        size: tuple = (1920, 160)) -> Path:
    """Render a lower-third caption to PNG with alpha."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = size
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # dark translucent bar
    draw.rectangle([0, 0, W, H], fill=(0, 0, 0, 140))
    serif = _find_serif_font()
    try:
        font = ImageFont.truetype(serif, 38) if serif else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((W - tw) // 2, (H - th) // 2 - 4), text, font=font, fill=(255, 255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG")
    return out


def _make_caption_clip(src: Path, caption: str, out: Path) -> Path:
    """Overlay a Pillow-rendered caption PNG and apply the cinematic grade.
    Preserves the per-shot audio track baked in earlier."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if not caption.strip():
        subprocess.run([
            "ffmpeg", "-y", "-i", str(src),
            "-map", "0:v:0", "-map", "0:a?",
            "-vf", GRADE_VF,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p", "-r", "30",
            str(out),
        ], check=True, capture_output=True)
        return out
    png = out.with_suffix(".png")
    _render_caption_png(caption, png)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(src), "-i", str(png),
        "-filter_complex",
            f"[0:v]{GRADE_VF}[v];[v][1:v]overlay=0:main_h-160:format=auto",
        "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p", "-r", "30",
        str(out),
    ], check=True, capture_output=True)
    return out


# ---------- narration + stitch ----------

def _ensure_silent_audio(clip_path: Path) -> Path:
    """If the clip has no audio track, add a silent AAC one. Concat needs uniform streams."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(clip_path)],
        capture_output=True, text=True, check=False,
    )
    if probe.stdout.strip():
        return clip_path
    out = clip_path.with_name(clip_path.stem + "_va.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(clip_path),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
        str(out),
    ], check=True, capture_output=True)
    return out


def _bake_shot_audio(shot: Shot, clip_path: Path) -> Path:
    """Generate Aura TTS for shot.narration and mux onto the clip, padded to clip length.
    No narration → just guarantee a silent audio track so concat works."""
    if not shot.narration.strip():
        return _ensure_silent_audio(clip_path)
    dg_key = os.getenv("DEEPGRAM_API_KEY", "")
    if not dg_key:
        return _ensure_silent_audio(clip_path)
    voice = os.getenv("DEEPGRAM_TTS_VOICE", "aura-asteria-en")
    out_dir = clip_path.parent
    mp3 = out_dir / f"narration_{shot.index:02d}.mp3"
    try:
        r = httpx.post(
            f"https://api.deepgram.com/v1/speak?model={voice}",
            headers={"Authorization": f"Token {dg_key}",
                     "Content-Type": "application/json"},
            json={"text": shot.narration.strip()},
            timeout=30,
        )
        r.raise_for_status()
        mp3.write_bytes(r.content)
    except Exception as e:
        print(f"  [shot {shot.index}] aura tts failed: {e!r} — silent fallback")
        return _ensure_silent_audio(clip_path)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(clip_path)],
        capture_output=True, text=True, check=True,
    )
    try:
        clip_dur = float(probe.stdout.strip())
    except ValueError:
        clip_dur = float(shot.duration or 5)

    out = out_dir / f"clip_{shot.index:02d}_va.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(clip_path), "-i", str(mp3),
        "-filter_complex",
            f"[1:a]apad,atrim=0:{clip_dur:.3f},asetpts=N/SR/TB[a]",
        "-map", "0:v:0", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        str(out),
    ], check=True, capture_output=True)
    return out


# Cinematic teal-orange grade applied per-shot at the caption-clip stage.
GRADE_VF = (
    "scale=1920:1080,"
    "curves=r='0/0 0.5/0.55 1/1':b='0/0 0.5/0.45 1/1',"
    "eq=saturation=1.10:contrast=1.06,"
    "unsharp=lx=3:ly=3:la=0.25"
)


def render_narration(session: Session, out_dir: Path) -> Optional[Path]:
    lines = [s.narration.strip() for s in session.shots if s.narration.strip()]
    if not lines:
        return None
    text = "  ".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp3 = out_dir / "narration.mp3"

    dg_key = os.getenv("DEEPGRAM_API_KEY", "")
    voice = os.getenv("DEEPGRAM_TTS_VOICE", "aura-asteria-en")
    if dg_key:
        try:
            r = httpx.post(
                f"https://api.deepgram.com/v1/speak?model={voice}",
                headers={"Authorization": f"Token {dg_key}",
                         "Content-Type": "application/json"},
                json={"text": text},
                timeout=60,
            )
            r.raise_for_status()
            mp3.write_bytes(r.content)
            return mp3
        except Exception as e:
            print(f"[narration] deepgram tts failed: {e!r} — falling back to macOS say")

    aiff = out_dir / "narration.aiff"
    subprocess.run(["say", "-v", "Daniel", "-r", "170", "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(aiff), "-b:a", "192k", str(mp3)],
                   check=True, capture_output=True)
    aiff.unlink()
    return mp3


ENABLE_CARDS = os.getenv("ENABLE_CARDS", "true").lower() == "true"
ENABLE_CAPTIONS = os.getenv("ENABLE_CAPTIONS", "true").lower() == "true"


def finalize(session: Session) -> Path:
    """Wait for all shots, stitch, produce final.mp4.
    Adds intro card, burned captions, and outro card for a polished ad feel.
    """
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

    emit("stitch.start", {"call_id": session.call_id, "total": len(ordered)})
    # Re-encode each clip with caption burned in (if captions enabled)
    processed_clips: list[Path] = []
    processed_dir = run_dir / "_processed"
    processed_dir.mkdir(exist_ok=True)
    for s in ordered:
        emit("stitch.progress", {"call_id": session.call_id, "shot_index": s.index, "total": len(ordered)})
        src = Path(s.clip_path)
        dst = processed_dir / src.name
        if ENABLE_CAPTIONS and s.narration.strip():
            _make_caption_clip(src, s.narration.strip(), dst)
        else:
            # re-encode with consistent params so concat works cleanly
            subprocess.run([
                "ffmpeg", "-y", "-i", str(src),
                "-map", "0:v:0",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-r", "30",
                str(dst),
            ], check=True, capture_output=True)
        processed_clips.append(dst)

    # Intro + outro cards
    concat_items: list[Path] = []
    if ENABLE_CARDS:
        intro = run_dir / "_intro.mp4"
        outro = run_dir / "_outro.mp4"
        title = (session.title or "Untitled").upper()
        _make_card(title, "DIRECTED BY CONDUIT", intro, dur=1.8, variant="intro")
        _make_card("CONDUIT", "+1 (443) 464-8118", outro, dur=2.0, variant="outro")
        concat_items.append(intro)
    concat_items.extend(processed_clips)
    if ENABLE_CARDS:
        concat_items.append(outro)

    # Re-encode all to same codec params (concat demuxer requires identical streams;
    # caption re-encode already normalized the clips, so we do full re-encode here to be safe)
    clips_txt = run_dir / "clips.txt"
    clips_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in concat_items))

    # Each input already carries audio (per-shot Aura VO baked in during render,
    # silent track on cards/un-narrated shots). Concat preserves audio cleanly.
    final = run_dir / "final.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(clips_txt),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "192k",
        str(final),
    ], check=True, capture_output=True)

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
