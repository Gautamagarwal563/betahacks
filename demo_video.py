"""Build the hero demo video for Conduit — the one that goes on the website / submission.

Scenes (in order):
  1. iPhone calling UI — dialing +1 (443) 464-8118
  2. Call connects — "Conduit" answers with greeting
  3. User types / speaks the prompt (airplane / Eiffel / kid with ice cream)
  4. Director plans — shot list forms on screen
  5. Agent orchestration — seven agent nodes light up in sequence
  6. Generated video plays (the actual Conduit-rendered output)
  7. Outro — brand + phone CTA

All scenes are Pillow-rendered PNG sequences stitched with ffmpeg.
Requires the `demo_airplane` Conduit session to be finalized first.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1920, 1080
FPS = 30

OUT_DIR = Path("videos/_demo")
SCENES_DIR = OUT_DIR / "scenes"
FRAMES_DIR = OUT_DIR / "frames"

# --- Fonts ---
def _font(path_candidates: list[str], size: int):
    for p in path_candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


SERIF = [
    "/System/Library/Fonts/Supplemental/Baskerville.ttc",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Times.ttc",
]
SANS = [
    "/System/Library/Fonts/Supplemental/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
MONO = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

# --- Colors ---
BG = (5, 5, 6)
SURFACE = (14, 14, 16)
TEXT = (244, 244, 245)
MUTED = (161, 161, 170)
DIM = (113, 113, 122)
ACCENT = (103, 232, 249)        # cyan
ACCENT_DIM = (52, 142, 158)
SUCCESS = (74, 222, 128)
AMBER = (250, 204, 21)


def _blur_glow(img: Image.Image, center: tuple, radius: int, color: tuple):
    """Paint a soft glow on img at center (x, y)."""
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
               fill=(*color, 80))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=30))
    img.alpha_composite(glow)


def _new_frame() -> Image.Image:
    img = Image.new("RGBA", (W, H), (*BG, 255))
    # subtle dual-radial ambient
    _blur_glow(img, (0, 0), 800, ACCENT)
    _blur_glow(img, (W, H * 0.2), 600, (168, 85, 247))
    return img


def _save_frame(img: Image.Image, scene: str, idx: int):
    d = FRAMES_DIR / scene
    d.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(d / f"{idx:05d}.jpg", "JPEG", quality=92)


def _scene_to_mp4(scene: str, fps: int = FPS) -> Path:
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    src = FRAMES_DIR / scene / "%05d.jpg"
    out = SCENES_DIR / f"{scene}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-crf", "18", "-r", str(fps),
        str(out),
    ], check=True, capture_output=True)
    return out


# ============================================================
# SCENE 1: phone calling UI
# ============================================================
def scene_phone_call(duration: float = 5.0):
    """iMessage/Phone-style UI dialing the Conduit number, then connected."""
    total = int(duration * FPS)
    title_font = _font(SANS, 44)
    big_font = _font(SANS, 90)
    small_font = _font(SANS, 26)
    mono_font = _font(MONO, 22)

    phone_w, phone_h = 460, 940
    px, py = (W - phone_w) // 2, (H - phone_h) // 2

    for i in range(total):
        img = _new_frame()
        draw = ImageDraw.Draw(img)

        # phone body
        phone = Image.new("RGBA", (phone_w, phone_h), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(phone)
        pdraw.rounded_rectangle([0, 0, phone_w, phone_h], radius=54,
                                 fill=(18, 18, 22, 255), outline=(70, 70, 80, 255), width=3)
        # screen
        pdraw.rounded_rectangle([10, 40, phone_w - 10, phone_h - 40], radius=40,
                                 fill=(8, 8, 10, 255))

        # status bar icons (stylized)
        pdraw.text((30, 55), "9:41", font=_font(SANS, 18), fill=TEXT)

        # call label
        is_connected = i > total * 0.5
        label_y = 140
        if is_connected:
            pdraw.text((phone_w // 2, label_y), "connected",
                       font=_font(SANS, 18), fill=SUCCESS, anchor="mt")
            duration_seconds = int((i - total * 0.5) / FPS)
            pdraw.text((phone_w // 2, label_y + 30), f"00:{duration_seconds:02d}",
                       font=mono_font, fill=MUTED, anchor="mt")
        else:
            # pulsing "calling..."
            fade = 0.5 + 0.5 * ((i // 6) % 2)
            c = tuple(int(x * fade) for x in MUTED) + (255,)
            pdraw.text((phone_w // 2, label_y), "calling…",
                       font=_font(SANS, 18), fill=MUTED, anchor="mt")

        # caller name + number
        pdraw.text((phone_w // 2, 220), "Conduit",
                   font=big_font, fill=TEXT, anchor="mt")
        pdraw.text((phone_w // 2, 335), "+1 (443) 464-8118",
                   font=_font(SANS, 22), fill=DIM, anchor="mt")

        # avatar circle
        av_cx, av_cy, av_r = phone_w // 2, 490, 80
        if not is_connected:
            # pulsing rings while calling
            t = (i % 24) / 24.0
            for k in [1.0, 0.6, 0.3]:
                r = int(av_r + (t * 60 + k * 20))
                alpha = int(100 * (1 - t) * k)
                if alpha > 0:
                    rcol = (*ACCENT, alpha)
                    pdraw.ellipse([av_cx - r, av_cy - r, av_cx + r, av_cy + r],
                                   outline=rcol, width=2)
        pdraw.ellipse([av_cx - av_r, av_cy - av_r, av_cx + av_r, av_cy + av_r],
                       fill=SURFACE + (255,), outline=(*ACCENT, 200), width=3)
        pdraw.text((av_cx, av_cy), "C", font=_font(SERIF, 60),
                   fill=ACCENT, anchor="mm")

        # hang up button
        btn_cy = phone_h - 180
        pdraw.ellipse([av_cx - 45, btn_cy - 45, av_cx + 45, btn_cy + 45],
                       fill=(239, 68, 68, 255))
        pdraw.text((av_cx, btn_cy), "✕", font=_font(SANS, 44),
                   fill=(255, 255, 255), anchor="mm")

        img.alpha_composite(phone, (px, py))

        # caption under phone
        if is_connected:
            cap = "On the line with the Director."
        else:
            cap = "Dial. Describe. Done."
        draw.text((W // 2, H - 80), cap, font=_font(SERIF, 32, ),
                   fill=MUTED, anchor="mm")

        _save_frame(img, "01_phone", i)


# ============================================================
# SCENE 2: Prompt typing
# ============================================================
def scene_prompt_typing(prompt: str, duration: float = 8.0):
    """Typed animation of the user's prompt appearing on screen."""
    total = int(duration * FPS)
    serif_big = _font(SERIF, 64)
    mono_small = _font(MONO, 18)
    sans_label = _font(SANS, 22)

    # Reveal characters over ~70% of duration, hold at end
    reveal_frames = int(total * 0.72)
    for i in range(total):
        img = _new_frame()
        draw = ImageDraw.Draw(img)

        # label
        draw.text((W // 2, 170), "YOU · ON THE CALL",
                   font=_font(MONO, 14), fill=ACCENT, anchor="mm")
        # small cyan underline
        draw.line([(W // 2 - 100, 200), (W // 2 + 100, 200)], fill=(*ACCENT, 120), width=1)

        # typed prompt
        chars_to_show = int((min(i, reveal_frames) / reveal_frames) * len(prompt))
        shown = prompt[:chars_to_show]

        # word-wrap
        words = shown.split()
        lines = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            bbox = draw.textbbox((0, 0), test, font=serif_big)
            if bbox[2] > W - 400 and cur:
                lines.append(cur)
                cur = w
            else:
                cur = test
        if cur:
            lines.append(cur)
        text = "\n".join(lines)

        # multiline bounding box for centering
        try:
            bbox = draw.multiline_textbbox((0, 0), text, font=serif_big, spacing=14, align="center")
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw = th = 0
        draw.multiline_text(((W - tw) // 2, (H - th) // 2), text,
                             font=serif_big, fill=TEXT, spacing=14, align="center")

        # blinking cursor at end (during reveal)
        if i < reveal_frames:
            if (i // 10) % 2 == 0:
                # find end position of last line
                last_line = lines[-1] if lines else ""
                last_bbox = draw.textbbox((0, 0), last_line, font=serif_big)
                last_w = last_bbox[2]
                cursor_x = (W - last_bbox[2]) // 2 + last_w + 8  # approx
                # draw a thin cursor at estimated end
                cy = (H - th) // 2 + (len(lines) - 1) * 78 + 10
                draw.rectangle([cursor_x, cy, cursor_x + 3, cy + 64], fill=ACCENT)

        # bottom footer
        draw.text((W // 2, H - 90), "→ sending to Director",
                   font=mono_small, fill=DIM, anchor="mm")

        _save_frame(img, "02_prompt", i)


# ============================================================
# SCENE 3: Director planning
# ============================================================
def scene_director_planning(title: str, shots: list[dict], duration: float = 6.0):
    """Animated shot list forming as director plans."""
    total = int(duration * FPS)
    serif = _font(SERIF, 54)
    mono = _font(MONO, 16)
    sans = _font(SANS, 22)
    sans_sm = _font(SANS, 18)

    # Each shot appears every ~0.8s
    reveal_per_shot_frames = int(total / max(1, len(shots) + 1))

    for i in range(total):
        img = _new_frame()
        draw = ImageDraw.Draw(img)

        # label
        draw.text((W // 2, 110), "DIRECTOR · PLANNING",
                   font=_font(MONO, 13), fill=ACCENT, anchor="mm")
        draw.line([(W // 2 - 100, 130), (W // 2 + 100, 130)], fill=(*ACCENT, 120), width=1)

        # title
        draw.text((W // 2, 200), title, font=serif, fill=TEXT, anchor="mm")

        # shot list — slide in from bottom, one at a time
        y_start = 330
        row_h = 90
        shots_visible = min(len(shots), i // reveal_per_shot_frames)

        for idx, shot in enumerate(shots[:shots_visible]):
            y = y_start + idx * row_h
            # opacity ramp for newest shot
            if idx == shots_visible - 1:
                sub_i = i - idx * reveal_per_shot_frames
                fade = min(1.0, sub_i / (reveal_per_shot_frames * 0.6))
                offset_y = int((1 - fade) * 20)
                y = y + offset_y
            else:
                fade = 1.0

            # row container
            row_bg = (*SURFACE, int(180 * fade))
            draw.rounded_rectangle([W // 2 - 540, y - 30, W // 2 + 540, y + 35],
                                     radius=12, fill=row_bg, outline=(*ACCENT, int(60 * fade)), width=1)
            # index
            draw.text((W // 2 - 500, y + 2),
                       f"SHOT {str(idx + 1).zfill(2)}",
                       font=mono, fill=tuple(int(c * fade) for c in ACCENT),
                       anchor="lm")
            # intent
            intent = shot.get("intent", "")[:80]
            draw.text((W // 2 - 380, y + 2), intent,
                       font=sans, fill=tuple(int(c * fade) for c in TEXT), anchor="lm")

        # counter at bottom
        draw.text((W // 2, H - 90),
                   f"{shots_visible} / {len(shots)} shots · rendering in parallel",
                   font=_font(MONO, 16), fill=DIM, anchor="mm")

        _save_frame(img, "03_director", i)


# ============================================================
# SCENE 4: Agent orchestration
# ============================================================
AGENTS = [
    {"name": "Vapi + Deepgram", "role": "voice I/O",     "model": "nova-3 · aura"},
    {"name": "Director",         "role": "planner",        "model": "claude-sonnet-4-6"},
    {"name": "Storyboard",       "role": "keyframes",     "model": "seedream-5-0"},
    {"name": "Cinematographer", "role": "motion",         "model": "seedance-2-0"},
    {"name": "Voice",            "role": "narration",      "model": "seed-speech"},
    {"name": "Stitcher",         "role": "final assembly", "model": "ffmpeg · remotion"},
]

def scene_orchestration(duration: float = 8.0):
    """Agent nodes light up sequentially, connecting lines trace between them."""
    total = int(duration * FPS)
    serif_big = _font(SERIF, 48)
    sans = _font(SANS, 22)
    mono = _font(MONO, 14)
    sans_sm = _font(SANS, 15)

    # Agent layout — vertical flow
    N = len(AGENTS)
    # one reveal per ~(total/(N+1)) frames
    reveal_step = total // (N + 1)

    card_w, card_h = 520, 100
    x = (W - card_w) // 2

    for i in range(total):
        img = _new_frame()
        draw = ImageDraw.Draw(img)

        draw.text((W // 2, 70), "SEVEN AGENTS · ONE CALL",
                   font=_font(MONO, 14), fill=ACCENT, anchor="mm")
        draw.line([(W // 2 - 140, 90), (W // 2 + 140, 90)], fill=(*ACCENT, 120), width=1)

        draw.text((W // 2, 140), "The orchestration", font=serif_big, fill=TEXT, anchor="mm")

        active_idx = min(N - 1, i // reveal_step)

        # start y
        start_y = 220
        gap = 110

        # draw nodes
        for idx, agent in enumerate(AGENTS):
            y = start_y + idx * gap
            reveal_i = i - idx * reveal_step
            if reveal_i < 0:
                continue
            fade = min(1.0, reveal_i / (reveal_step * 0.6))

            # node colors
            is_active = (idx == active_idx and i < (N) * reveal_step)
            is_done = (idx < active_idx)
            if is_active:
                border = ACCENT
                border_width = 3
                glow_radius = 40
                _blur_glow(img, (W // 2, y + card_h // 2), 90, ACCENT)
            elif is_done:
                border = SUCCESS
                border_width = 1
            else:
                border = (*MUTED, 60)
                border_width = 1
                if fade < 1:
                    # ramp border alpha
                    border = (*MUTED, int(60 * fade))

            draw.rounded_rectangle([x, y, x + card_w, y + card_h],
                                     radius=14,
                                     fill=(*SURFACE, int(200 * fade)),
                                     outline=border, width=border_width)

            # pulse dot
            dot_cx = x + 30
            dot_cy = y + card_h // 2
            if is_active:
                # bigger pulsing
                dot_color = ACCENT
                pulse_r = 10 + int(4 * ((i % 18) / 18.0))
                draw.ellipse([dot_cx - pulse_r, dot_cy - pulse_r, dot_cx + pulse_r, dot_cy + pulse_r],
                             outline=(*ACCENT, 80), width=2)
                draw.ellipse([dot_cx - 6, dot_cy - 6, dot_cx + 6, dot_cy + 6], fill=dot_color)
            elif is_done:
                draw.ellipse([dot_cx - 5, dot_cy - 5, dot_cx + 5, dot_cy + 5], fill=SUCCESS)
            else:
                dot_color = (*DIM, int(200 * fade))
                draw.ellipse([dot_cx - 5, dot_cy - 5, dot_cx + 5, dot_cy + 5], fill=dot_color)

            # name
            name_color = TEXT if fade > 0.7 else tuple(int(c * fade) for c in TEXT)
            draw.text((x + 70, y + 28), agent["name"],
                       font=sans, fill=name_color + (255,) if len(name_color)==3 else name_color,
                       anchor="lt")
            # role
            draw.text((x + 70, y + 58), agent["role"],
                       font=sans_sm, fill=tuple(int(c * fade) for c in MUTED),
                       anchor="lt")
            # model chip right
            model = agent["model"]
            mb = draw.textbbox((0, 0), model, font=mono)
            mw = mb[2] - mb[0]
            chip_x1 = x + card_w - 20
            chip_x0 = chip_x1 - mw - 20
            chip_y0 = y + card_h // 2 - 14
            chip_y1 = y + card_h // 2 + 14
            draw.rounded_rectangle([chip_x0, chip_y0, chip_x1, chip_y1],
                                     radius=14,
                                     fill=(*ACCENT, int(20 * fade)),
                                     outline=(*ACCENT_DIM, int(120 * fade)))
            draw.text((chip_x1 - 10, y + card_h // 2), model,
                       font=mono, fill=tuple(int(c * fade) for c in ACCENT),
                       anchor="rm")

            # connector line to next
            if idx < N - 1:
                cy1 = y + card_h
                cy2 = start_y + (idx + 1) * gap
                # fade line
                line_alpha = int(80 * fade)
                draw.line([(W // 2, cy1), (W // 2, cy2)],
                           fill=(*ACCENT, line_alpha), width=1)

        _save_frame(img, "04_orchestra", i)


# ============================================================
# SCENE 5: Airplane video as a content insert (with overlay)
# ============================================================
def scene_video_insert(src_mp4: Path, duration_hint: float):
    """We'll stitch the actual Conduit video directly — no frame rendering here."""
    # This scene is not re-rendered; we use the source mp4 directly.
    pass


# ============================================================
# SCENE 6: Outro
# ============================================================
def scene_outro(duration: float = 5.0):
    total = int(duration * FPS)
    serif_hero = _font(SERIF, 160)
    mono_big = _font(MONO, 36)
    sans = _font(SANS, 24)

    for i in range(total):
        img = _new_frame()
        draw = ImageDraw.Draw(img)

        # ambient glow center
        _blur_glow(img, (W // 2, H // 2), 400, ACCENT)

        # tag line
        fade_in = min(1.0, i / (FPS * 0.6))
        col_text = tuple(int(c * fade_in) for c in MUTED)
        col_accent = tuple(int(c * fade_in) for c in ACCENT)
        draw.text((W // 2, H // 2 - 220), "CALL IT. DIRECT IT.",
                   font=_font(MONO, 18), fill=col_accent, anchor="mm")
        draw.line([(W // 2 - 120, H // 2 - 195), (W // 2 + 120, H // 2 - 195)],
                   fill=(*col_accent, 200), width=1)

        # brand
        draw.text((W // 2, H // 2 - 70), "Conduit",
                   font=serif_hero,
                   fill=tuple(int(c * fade_in) for c in TEXT), anchor="mm")

        # tagline
        draw.text((W // 2, H // 2 + 80), "the AI you direct on the phone",
                   font=_font(SERIF, 34),
                   fill=col_text, anchor="mm")

        # phone number pill
        phone_str = "+1 (443) 464-8118"
        tb = draw.textbbox((0, 0), phone_str, font=mono_big)
        ph_w = tb[2] - tb[0]
        pill_y = H // 2 + 180
        pill_pad = 40
        draw.rounded_rectangle([W // 2 - ph_w // 2 - pill_pad, pill_y - 36,
                                 W // 2 + ph_w // 2 + pill_pad, pill_y + 36],
                                 radius=40,
                                 fill=(*SURFACE, 240),
                                 outline=(*col_accent, 220), width=2)
        draw.text((W // 2, pill_y), phone_str, font=mono_big,
                   fill=tuple(int(c * fade_in) for c in TEXT), anchor="mm")

        _save_frame(img, "06_outro", i)


# ============================================================
# Main orchestrator
# ============================================================
def main():
    """
    Build the full demo video.

    Requires state/demo_airplane.json to exist with finalized shots.
    """
    import json

    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Load the airplane session
    session_path = Path("state/demo_airplane.json")
    if not session_path.exists():
        raise RuntimeError("Run the airplane session first via /dev/simulate")
    sess = json.loads(session_path.read_text())
    final_video_path = sess.get("final_video_path")
    if not final_video_path or not Path(final_video_path).exists():
        raise RuntimeError(f"Final video not yet rendered: {final_video_path}")
    print(f"using conduit-generated video: {final_video_path}")

    # Render scenes
    print("\n[1/5] phone call UI…")
    scene_phone_call(duration=5.5)
    _scene_to_mp4("01_phone")

    prompt_short = "Airplane passing by the Eiffel Tower. A child on the ground, eating ice cream, looks up and shouts: yay, look, airplane!"
    print("[2/5] prompt typing…")
    scene_prompt_typing(prompt_short, duration=8.0)
    _scene_to_mp4("02_prompt")

    print("[3/5] director planning…")
    scene_director_planning(sess.get("title", ""), sess.get("shots", []), duration=6.0)
    _scene_to_mp4("03_director")

    print("[4/5] agent orchestration…")
    scene_orchestration(duration=8.0)
    _scene_to_mp4("04_orchestra")

    print("[5/5] outro…")
    scene_outro(duration=5.0)
    _scene_to_mp4("06_outro")

    # Concat: 01 + 02 + 03 + 04 + conduit_output + 06
    concat_items = [
        SCENES_DIR / "01_phone.mp4",
        SCENES_DIR / "02_prompt.mp4",
        SCENES_DIR / "03_director.mp4",
        SCENES_DIR / "04_orchestra.mp4",
        Path(final_video_path).resolve(),
        SCENES_DIR / "06_outro.mp4",
    ]

    # Normalize all to same codec/res via re-encode pass for concat safety
    normalized_dir = OUT_DIR / "_normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[Path] = []
    for i, src in enumerate(concat_items):
        dst = normalized_dir / f"n_{i:02d}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(src),
            "-map", "0:v:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
            "-pix_fmt", "yuv420p", "-r", "30", "-vf", "scale=1920:1080",
            str(dst),
        ], check=True, capture_output=True)
        normalized.append(dst)

    list_txt = OUT_DIR / "concat.txt"
    list_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in normalized))
    final = OUT_DIR / "conduit_demo.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_txt),
        "-c", "copy", str(final),
    ], check=True, capture_output=True)

    print(f"\n✓ DEMO VIDEO: {final}")
    return final


if __name__ == "__main__":
    main()
