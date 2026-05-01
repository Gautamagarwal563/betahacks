"""Live dashboard — judges watch the director work in real time.

Shows all active call sessions; click one to see shots stream in as they render.
Uses Server-Sent Events for live updates.

Run: `python dashboard.py` (listens on :8000)
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

import director
import pipeline
import db
import claude_client

app = FastAPI(title="Conduit Dashboard")

from fastapi.templating import Jinja2Templates
_templates = Jinja2Templates(directory="templates")

if Path("videos").exists():
    app.mount("/videos", StaticFiles(directory="videos"), name="videos")
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# In-memory event queue per call_id (SSE subscribers)
_queues: dict[str, list[asyncio.Queue]] = {}


def _publish(event: str, data: dict) -> None:
    call_id = data.get("call_id")
    if not call_id:
        return
    for q in _queues.get(call_id, []):
        try:
            q.put_nowait({"event": event, "data": json.dumps(data)})
        except Exception:
            pass


pipeline.subscribe(_publish)


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML


@app.get("/how", response_class=HTMLResponse)
def how():
    return HOW_HTML


@app.get("/demo", response_class=HTMLResponse)
def demo():
    return HTMLResponse(
        DEMO_HTML,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/agents")
def api_agents():
    """Aggregate metrics per agent role across all sessions."""
    state_dir = Path("state")
    agents = {
        "you":          {"name": "You",              "role": "The director",         "model": "phone",                    "calls": 0, "unit": "calls"},
        "vapi":         {"name": "Vapi + Deepgram",  "role": "Voice I/O",            "model": "nova-3 · aura",            "calls": 0, "unit": "turns"},
        "director":     {"name": "Director",          "role": "Plans + iterates",    "model": "claude-sonnet-4-6",        "calls": 0, "unit": "decisions"},
        "storyboard":   {"name": "Storyboard",        "role": "Keyframes",           "model": "seedream-5-0",             "calls": 0, "unit": "frames"},
        "cinema":       {"name": "Cinematographer",  "role": "Motion",               "model": "seedance-2-0-fast",        "calls": 0, "unit": "clips"},
        "voice":        {"name": "Voice",            "role": "Narration",            "model": "seed-speech",              "calls": 0, "unit": "tracks"},
        "stitch":       {"name": "Stitcher",         "role": "Final assembly",      "model": "ffmpeg · remotion",         "calls": 0, "unit": "videos"},
    }
    live_count = 0
    calls = 0
    if state_dir.exists():
        for p in state_dir.glob("*.json"):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            calls += 1
            shots = d.get("shots", [])
            tr = d.get("transcript", [])
            agents["you"]["calls"] += 1
            agents["vapi"]["calls"] += len(tr)
            agents["director"]["calls"] += sum(1 for t in tr if t.get("role") == "assistant") or 1
            agents["storyboard"]["calls"] += sum(1 for s in shots if s.get("keyframe_path"))
            agents["cinema"]["calls"] += sum(1 for s in shots if s.get("clip_path"))
            if d.get("final_video_path"):
                agents["voice"]["calls"] += 1
                agents["stitch"]["calls"] += 1
            # live = no final yet
            if not d.get("final_video_path"):
                live_count += 1
    return {"agents": list(agents.values()), "calls_total": calls, "live": live_count}


@app.get("/api/sessions")
def api_sessions():
    state = Path("state")
    sessions = []
    if state.exists():
        for p in sorted(state.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                d = json.loads(p.read_text())
                sessions.append({
                    "call_id": d["call_id"],
                    "title": d.get("title"),
                    "brief": d.get("brief"),
                    "created_at": d.get("created_at"),
                    "shot_count": len(d.get("shots", [])),
                    "final_video_path": d.get("final_video_path"),
                })
            except Exception:
                pass
    return sessions


@app.get("/api/session/{call_id}")
def api_session(call_id: str):
    p = Path("state") / f"{call_id}.json"
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return json.loads(p.read_text())


@app.post("/api/regen/{call_id}/{shot_index}")
async def api_regen(call_id: str, shot_index: int, req: Request):
    """UI-triggered shot regen. Body: {"new_prompt": "...", "new_narration": "...", "new_intent": "..."}"""
    import director, pipeline
    data = await req.json()
    sess = director.load_session(call_id)
    if not sess:
        return JSONResponse({"error": "session not found"}, status_code=404)
    if shot_index < 0 or shot_index >= len(sess.shots):
        return JSONResponse({"error": "bad shot index"}, status_code=400)
    shot = director.apply_regen(sess, {
        "shot_index": shot_index,
        "new_prompt": data.get("new_prompt") or sess.shots[shot_index].prompt,
        "new_intent": data.get("new_intent") or sess.shots[shot_index].intent,
        "new_narration": data.get("new_narration", sess.shots[shot_index].narration),
    })
    if not shot:
        return JSONResponse({"error": "regen failed"}, status_code=500)
    director.dump_session(sess)
    pipeline.render_shot_async(sess, shot)
    return {"ok": True, "shot_index": shot.index, "status": shot.status.value}


@app.post("/api/finalize/{call_id}")
async def api_finalize(call_id: str):
    """UI-triggered stitch. Only works if all shots are done."""
    import director, pipeline
    sess = director.load_session(call_id)
    if not sess:
        return JSONResponse({"error": "session not found"}, status_code=404)
    try:
        final = pipeline.finalize(sess)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    director.dump_session(sess)
    return {"ok": True, "path": str(final)}


@app.get("/call/{call_id}", response_class=HTMLResponse)
def call_page(call_id: str):
    return CALL_HTML.replace("__CALL_ID__", call_id)


@app.get("/events/{call_id}")
async def events(call_id: str, request: Request):
    q: asyncio.Queue = asyncio.Queue()
    _queues.setdefault(call_id, []).append(q)

    async def stream():
        try:
            # send current snapshot
            p = Path("state") / f"{call_id}.json"
            if p.exists():
                yield {"event": "snapshot", "data": p.read_text()}
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield item
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
        finally:
            try:
                _queues[call_id].remove(q)
            except ValueError:
                pass

    return EventSourceResponse(stream())


@app.get("/healthz")
def health():
    return {"ok": True}


_BASE_STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist+Mono:wght@400;500;600&family=Geist:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    color-scheme: dark;
    --bg: #060606;
    --surface: #0e0e10;
    --surface-2: #15151a;
    --border: rgba(255,255,255,0.06);
    --border-2: rgba(255,255,255,0.12);
    --text: #f4f4f5;
    --text-2: #a1a1aa;
    --text-3: #71717a;
    --accent: #67e8f9;
    --accent-glow: rgba(103,232,249,0.35);
    --success: #4ade80;
    --warning: #facc15;
    --danger: #f87171;
    --serif: 'Instrument Serif', ui-serif, Georgia, serif;
    --sans: 'Geist', system-ui, -apple-system, 'Helvetica Neue', sans-serif;
    --mono: 'Geist Mono', ui-monospace, 'SF Mono', monospace;
  }
  * { box-sizing: border-box; -webkit-font-smoothing: antialiased; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.5;
    min-height: 100vh;
    background-image:
      radial-gradient(1200px circle at 0% 0%, rgba(103,232,249,0.04), transparent 50%),
      radial-gradient(900px circle at 100% 20%, rgba(168,85,247,0.03), transparent 50%);
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; text-underline-offset: 3px; }
  button { font-family: inherit; }

  /* shared layout primitives */
  .topnav {
    position: sticky; top: 0; z-index: 50;
    backdrop-filter: saturate(120%) blur(12px);
    background: rgba(6,6,6,0.72);
    border-bottom: 1px solid var(--border);
  }
  .topnav .inner {
    max-width: 1240px; margin: 0 auto; padding: 14px 28px;
    display: flex; align-items: center; justify-content: space-between; gap: 24px;
  }
  .brand {
    font-family: var(--mono); font-weight: 600; font-size: 13px;
    letter-spacing: 0.18em; text-transform: uppercase; color: var(--text);
  }
  .brand .glyph {
    display: inline-block; width: 9px; height: 9px; margin-right: 8px; vertical-align: 2px;
    background: var(--accent); border-radius: 50%;
    box-shadow: 0 0 0 3px var(--accent-glow), 0 0 18px var(--accent-glow);
    animation: breathe 2.6s ease-in-out infinite;
  }
  @keyframes breathe {
    0%,100% { transform: scale(1); box-shadow: 0 0 0 3px var(--accent-glow), 0 0 18px var(--accent-glow); }
    50%     { transform: scale(1.15); box-shadow: 0 0 0 5px var(--accent-glow), 0 0 24px var(--accent-glow); }
  }

  .phone-cta {
    font-family: var(--mono); font-weight: 500; font-size: 13px;
    color: var(--text); background: var(--surface); border: 1px solid var(--border-2);
    padding: 8px 14px; border-radius: 999px; display: inline-flex; align-items: center; gap: 8px;
    transition: all .18s ease;
  }
  .phone-cta:hover { border-color: var(--accent); color: var(--accent); text-decoration: none; }
  .phone-cta .pulse {
    width: 6px; height: 6px; border-radius: 50%; background: var(--success);
    box-shadow: 0 0 0 3px rgba(74,222,128,0.25);
    animation: breathe 1.8s ease-in-out infinite;
  }
  .wrap { max-width: 1240px; margin: 0 auto; padding: 40px 28px 80px; }
  .kbd {
    font-family: var(--mono); font-size: 11px; padding: 2px 6px; border-radius: 4px;
    background: var(--surface-2); border: 1px solid var(--border); color: var(--text-2);
  }
  .nav-chip {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-3); padding: 6px 12px; border-radius: 999px;
    border: 1px solid var(--border); text-decoration: none;
  }
  .nav-chip:hover { color: var(--text); border-color: var(--border-2); text-decoration: none; }

  /* Brand logo (used in topnav across all pages) */
  .brand-logo {
    height: 36px; width: auto; display: block;
    filter: drop-shadow(0 0 12px rgba(236, 72, 153, 0.35));
    transition: filter .3s ease, transform .3s ease;
  }
  .brand-logo:hover {
    filter: drop-shadow(0 0 22px rgba(236, 72, 153, 0.6));
    transform: scale(1.04);
  }
  .brand-row { display: flex; align-items: center; gap: 14px; }
  .brand-row .live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 0 3px rgba(74, 222, 128, 0.18), 0 0 14px rgba(74, 222, 128, 0.5);
    animation: breathe 1.8s ease-in-out infinite;
  }
</style>
"""


INDEX_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Conduit — the AI you direct on the phone</title>
__BASE_STYLE__
<style>
  .hero {
    padding: 72px 0 56px; border-bottom: 1px solid var(--border);
    display: grid; grid-template-columns: 1.2fr 1fr; gap: 64px; align-items: end;
  }
  @media (max-width: 780px) { .hero { grid-template-columns: 1fr; gap: 32px; padding: 44px 0; } }
  .hero h1 {
    font-family: var(--serif); font-weight: 400;
    font-size: clamp(40px, 6vw, 72px); line-height: 1.02; letter-spacing: -0.02em;
    margin: 0 0 16px; color: var(--text);
  }
  .hero h1 em { font-style: italic; color: var(--accent); }
  .hero .lede {
    font-size: 16px; color: var(--text-2); max-width: 520px; margin: 0 0 28px;
  }
  .hero .cta-col { display: flex; flex-direction: column; gap: 14px; align-items: flex-start; }
  .hero .big-phone {
    font-family: var(--mono); font-size: 22px; letter-spacing: -0.01em;
    color: var(--text); display: inline-flex; align-items: center; gap: 10px;
    padding: 16px 22px; border-radius: 14px;
    background: linear-gradient(180deg, rgba(103,232,249,0.08) 0%, rgba(103,232,249,0.02) 100%);
    border: 1px solid rgba(103,232,249,0.22);
    box-shadow: 0 0 40px rgba(103,232,249,0.08);
    transition: all .2s ease;
  }
  .hero .big-phone:hover { border-color: var(--accent); text-decoration: none; transform: translateY(-1px); }
  .hero .big-phone .label {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--accent); background: rgba(103,232,249,0.08); padding: 2px 8px; border-radius: 999px;
  }
  .hero .quick {
    color: var(--text-3); font-size: 13px; font-family: var(--mono);
  }

  .demo-hero {
    padding: 40px 0 8px;
  }
  .demo-hero .label-row {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 14px;
  }
  .demo-hero .eyebrow {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--text-3);
    display: inline-flex; align-items: center; gap: 10px;
  }
  .demo-hero .eyebrow::before {
    content:''; width: 22px; height: 1px; background: var(--text-3);
  }
  .demo-hero .runtime {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--text-3);
  }
  .demo-hero .frame {
    position: relative; border-radius: 18px; overflow: hidden;
    border: 1px solid var(--border-2);
    background: #000;
    box-shadow:
      0 0 0 1px rgba(103,232,249,0.05),
      0 30px 80px -20px rgba(0,0,0,0.7),
      0 0 120px -40px rgba(103,232,249,0.18);
  }
  .demo-hero .frame video {
    display: block; width: 100%; height: auto; aspect-ratio: 16/9;
    background: #000;
  }
  .demo-hero .frame .corner-tag {
    position: absolute; top: 14px; left: 14px;
    background: rgba(6,6,6,0.72); backdrop-filter: blur(8px);
    border: 1px solid var(--border-2);
    padding: 5px 11px; border-radius: 999px;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--text-2);
    display: inline-flex; align-items: center; gap: 7px;
  }
  .demo-hero .frame .corner-tag::before {
    content:''; width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 10px var(--accent-glow);
  }

  /* Prompt → result panel above the hero video */
  .prompt-panel {
    margin: 0 0 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
  }
  .prompt-panel .head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 11px 18px; border-bottom: 1px solid var(--border);
    background: rgba(103,232,249,0.03);
  }
  .prompt-panel .role {
    font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.28em; text-transform: uppercase;
    color: var(--accent);
    display: inline-flex; align-items: center; gap: 8px;
  }
  .prompt-panel .role::before {
    content:''; width: 5px; height: 5px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 8px var(--accent);
  }
  .prompt-panel .meta {
    font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--text-3);
  }
  .prompt-panel .body {
    padding: 18px 22px;
    font-family: var(--serif); font-style: italic;
    font-size: 22px; line-height: 1.45;
    color: var(--text);
  }
  .prompt-panel .body em { color: var(--accent); font-style: normal; }
  .prompt-panel .arrow {
    text-align: center; padding: 6px 0 4px;
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.32em;
    color: var(--text-3); border-top: 1px solid var(--border);
  }

  .section { padding: 48px 0 0; }
  .section h2 {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--text-3); font-weight: 500; margin: 0 0 18px;
  }
  .grid-calls {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px;
  }
  .call-card {
    display: block; background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; overflow: hidden; color: var(--text); text-decoration: none;
    transition: all .18s ease;
  }
  .call-card:hover {
    border-color: var(--border-2); transform: translateY(-2px);
    text-decoration: none;
  }
  .call-card .thumb {
    aspect-ratio: 16/9; background: #000; position: relative; overflow: hidden;
    background-image: linear-gradient(135deg, #1a1a1f 0%, #0a0a0c 100%);
  }
  .call-card .thumb img, .call-card .thumb video {
    width:100%; height:100%; object-fit: cover; display:block;
  }
  .call-card .thumb .empty-thumb {
    display:flex; align-items:center; justify-content:center; height:100%;
    font-family: var(--mono); font-size: 11px; color: var(--text-3); letter-spacing: 0.1em;
  }
  .call-card .thumb .live-overlay {
    position: absolute; top: 12px; left: 12px;
    background: rgba(248,113,113,0.15); color: #fca5a5; border: 1px solid rgba(248,113,113,0.3);
    padding: 3px 10px; border-radius: 999px; font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.15em; text-transform: uppercase; display: flex; gap: 6px; align-items: center;
  }
  .call-card .thumb .live-overlay::before {
    content:''; width:5px; height:5px; border-radius:50%; background:#ef4444; animation: breathe 1.4s infinite;
  }
  .call-card .thumb .done-overlay {
    position: absolute; top: 12px; right: 12px;
    background: rgba(74,222,128,0.1); color: var(--success); border: 1px solid rgba(74,222,128,0.25);
    padding: 3px 10px; border-radius: 999px; font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.15em; text-transform: uppercase;
  }
  .call-card .meta { padding: 18px 18px 16px; }
  .call-card .title {
    font-family: var(--serif); font-size: 20px; font-weight: 400; line-height: 1.2;
    letter-spacing: -0.01em; margin-bottom: 8px;
  }
  .call-card .brief { color: var(--text-2); font-size: 13px; line-height: 1.45;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  .call-card .footer {
    margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
    font-family: var(--mono); font-size: 11px; color: var(--text-3);
  }

  .empty-state {
    padding: 80px 24px; text-align: center; border: 1px dashed var(--border);
    border-radius: 16px; background: var(--surface);
  }
  .empty-state .big { font-family: var(--serif); font-style: italic; font-size: 26px; color: var(--text-2); margin-bottom: 8px; }
  .empty-state .sub { color: var(--text-3); font-size: 14px; font-family: var(--mono); }

  .stat-strip {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
    background: var(--border); border-radius: 14px; overflow: hidden;
    margin-bottom: 20px; border: 1px solid var(--border);
  }
  @media (max-width: 640px) { .stat-strip { grid-template-columns: repeat(2, 1fr); } }
  .stat {
    background: var(--surface); padding: 18px 20px;
  }
  .stat .n { font-family: var(--serif); font-size: 32px; line-height: 1; letter-spacing: -0.02em; }
  .stat .l { font-family: var(--mono); font-size: 10px; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-3); margin-top: 6px; }

  .section-head {
    display: flex; justify-content: space-between; align-items: baseline; gap: 16px;
    margin-bottom: 18px;
  }
  .section-head h2 { margin: 0; }
  .live-badge {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--text-3);
    border: 1px solid var(--border); padding: 4px 10px; border-radius: 999px;
    display: inline-flex; align-items: center; gap: 7px;
  }
  .live-badge::before {
    content:''; width: 6px; height: 6px; border-radius: 50%; background: var(--text-3);
  }
  .live-badge.on { color: var(--success); border-color: rgba(74,222,128,0.3); }
  .live-badge.on::before { background: var(--success); animation: breathe 1.4s infinite; }

  .agents-strip {
    display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px;
  }
  @media (max-width: 980px) { .agents-strip { grid-template-columns: repeat(3, 1fr); } }
  @media (max-width: 580px) { .agents-strip { grid-template-columns: repeat(2, 1fr); } }
  .agent-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 14px; position: relative;
    transition: all .25s ease;
  }
  .agent-card .dot {
    position: absolute; top: 14px; right: 14px;
    width: 7px; height: 7px; border-radius: 50%; background: var(--text-3);
    transition: all .3s ease;
  }
  .agent-card.active { border-color: rgba(103,232,249,0.32); }
  .agent-card.active .dot {
    background: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
    animation: breathe 1.6s infinite;
  }
  .agent-card .role {
    font-family: var(--mono); font-size: 9px; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--text-3); margin-bottom: 6px;
  }
  .agent-card .name {
    font-family: var(--serif); font-size: 19px; letter-spacing: -0.01em;
    color: var(--text); margin-bottom: 8px;
  }
  .agent-card .model {
    font-family: var(--mono); font-size: 10px; color: var(--accent);
    background: rgba(103,232,249,0.07); border: 1px solid rgba(103,232,249,0.18);
    padding: 2px 7px; border-radius: 5px; display: inline-block;
  }

  .try-row {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;
  }
  @media (max-width: 700px) { .try-row { grid-template-columns: 1fr; } }
  .try-chip {
    text-align: left; cursor: pointer; background: var(--surface);
    border: 1px solid var(--border); color: var(--text);
    padding: 16px 18px; border-radius: 12px; font-size: 14px;
    font-family: var(--sans); display: flex; gap: 12px; align-items: flex-start;
    transition: all .18s ease;
  }
  .try-chip:hover { border-color: var(--accent); background: rgba(103,232,249,0.04); transform: translateY(-1px); }
  .try-chip .quote {
    font-family: var(--serif); font-style: italic; color: var(--text-2);
    flex: 1; line-height: 1.4;
  }
  .try-chip .tag {
    font-family: var(--mono); font-size: 9px; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--text-3);
    border: 1px solid var(--border); padding: 3px 7px; border-radius: 5px;
    flex-shrink: 0; align-self: center;
  }
  .try-hint {
    margin-top: 14px; font-family: var(--mono); font-size: 12px;
    color: var(--text-3); letter-spacing: 0.04em;
  }

  .why-row {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
  }
  @media (max-width: 880px) { .why-row { grid-template-columns: 1fr; } }
  .why-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 24px; transition: all .22s ease;
  }
  .why-card:hover { border-color: var(--border-2); }
  .why-card .n {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 14px;
  }
  .why-card h3 {
    font-family: var(--serif); font-weight: 400; font-size: 22px;
    line-height: 1.2; letter-spacing: -0.01em; margin: 0 0 10px; color: var(--text);
  }
  .why-card p { color: var(--text-2); font-size: 13.5px; line-height: 1.55; margin: 0; }

  .tech-strip {
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  }
  .tech-pill {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em;
    color: var(--text-2); background: var(--surface);
    border: 1px solid var(--border); padding: 7px 13px; border-radius: 999px;
  }
  .tech-pill strong { color: var(--text); font-weight: 500; }

  .big-cta-section {
    margin: 56px 0 24px; padding: 56px 32px; border-radius: 20px; text-align: center;
    background: linear-gradient(180deg, rgba(103,232,249,0.06) 0%, rgba(103,232,249,0) 100%);
    border: 1px solid rgba(103,232,249,0.18);
  }
  .big-cta-section h2 {
    font-family: var(--serif); font-weight: 400; font-size: clamp(32px, 4vw, 48px);
    letter-spacing: -0.02em; margin: 0 0 8px; text-transform: none; color: var(--text);
  }
  .big-cta-section h2 em { color: var(--accent); font-style: italic; }
  .big-cta-section p { color: var(--text-2); font-size: 15px; margin: 0 0 24px; }
  .big-cta-btn {
    display: inline-flex; align-items: center; gap: 12px;
    font-family: var(--mono); font-size: 22px; letter-spacing: -0.01em; color: var(--text);
    padding: 16px 26px; border-radius: 14px;
    background: var(--bg); border: 1px solid rgba(103,232,249,0.35);
    box-shadow: 0 0 40px rgba(103,232,249,0.12);
    transition: all .2s ease;
  }
  .big-cta-btn:hover { border-color: var(--accent); transform: translateY(-1px); text-decoration: none; }
  .big-cta-btn .pulse {
    width: 8px; height: 8px; border-radius: 50%; background: var(--success);
    animation: breathe 1.6s infinite;
  }

  .footer {
    border-top: 1px solid var(--border); padding: 32px 0 56px;
    color: var(--text-3); font-family: var(--mono); font-size: 11px;
    letter-spacing: 0.06em; display: flex; justify-content: space-between;
    flex-wrap: wrap; gap: 14px; align-items: center;
  }
  .footer a { color: var(--text-2); }

  /* ── Brand logo in topnav ───────────────────────────────────────── */
  .brand-logo {
    height: 36px; width: auto; display: block;
    filter: drop-shadow(0 0 12px rgba(236, 72, 153, 0.35));
    transition: filter .3s ease, transform .3s ease;
  }
  .brand-logo:hover {
    filter: drop-shadow(0 0 22px rgba(236, 72, 153, 0.55));
    transform: scale(1.04);
  }
  .brand-row { display: flex; align-items: center; gap: 14px; }
  .brand-row .live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 0 3px rgba(74, 222, 128, 0.18), 0 0 14px rgba(74, 222, 128, 0.5);
    animation: breathe 1.8s ease-in-out infinite;
  }

  /* ── Framer-style scroll reveal ─────────────────────────────────── */
  .reveal {
    opacity: 0; transform: translateY(28px);
    transition:
      opacity .85s cubic-bezier(0.16, 1, 0.3, 1),
      transform .85s cubic-bezier(0.16, 1, 0.3, 1);
    will-change: opacity, transform;
  }
  .reveal.in { opacity: 1; transform: translateY(0); }
  .reveal.d-1 { transition-delay: 80ms; }
  .reveal.d-2 { transition-delay: 160ms; }
  .reveal.d-3 { transition-delay: 240ms; }
  .reveal.d-4 { transition-delay: 320ms; }

  .reveal-stagger > * {
    opacity: 0; transform: translateY(20px);
    transition: opacity .7s cubic-bezier(0.16, 1, 0.3, 1),
                transform .7s cubic-bezier(0.16, 1, 0.3, 1);
  }
  .reveal-stagger.in > *:nth-child(1) { transition-delay: 0ms; opacity: 1; transform: translateY(0); }
  .reveal-stagger.in > *:nth-child(2) { transition-delay: 80ms; opacity: 1; transform: translateY(0); }
  .reveal-stagger.in > *:nth-child(3) { transition-delay: 160ms; opacity: 1; transform: translateY(0); }
  .reveal-stagger.in > *:nth-child(4) { transition-delay: 240ms; opacity: 1; transform: translateY(0); }
  .reveal-stagger.in > *:nth-child(5) { transition-delay: 320ms; opacity: 1; transform: translateY(0); }
  .reveal-stagger.in > *:nth-child(6) { transition-delay: 400ms; opacity: 1; transform: translateY(0); }
  .reveal-stagger.in > *:nth-child(7) { transition-delay: 480ms; opacity: 1; transform: translateY(0); }

  /* ── Mouse-follow ambient glow on hero ──────────────────────────── */
  .hero-aura {
    position: fixed; pointer-events: none; z-index: 0;
    width: 600px; height: 600px; border-radius: 50%;
    background: radial-gradient(circle, rgba(103,232,249,0.12) 0%, transparent 60%);
    transform: translate(-50%, -50%);
    transition: transform 200ms cubic-bezier(0.16, 1, 0.3, 1);
    mix-blend-mode: screen;
    opacity: 0;
  }
  .hero-aura.on { opacity: 1; }

  /* ── Section rhythm — consistent vertical breathing room ────────── */
  .section { padding: 72px 0 0; }
  .section:first-of-type { padding-top: 56px; }

  /* ── Hero frame: subtle parallax tilt ───────────────────────────── */
  .demo-hero .frame {
    transform-origin: center center;
    transform-style: preserve-3d;
    transition: transform 220ms cubic-bezier(0.16, 1, 0.3, 1);
  }
</style>
</head>
<body>
<nav class="topnav"><div class="inner">
  <a href="/" class="brand" style="text-decoration:none"><span class="glyph"></span>CONDUIT</a>
  <div style="display:flex; gap: 12px; align-items: center;">
    <a href="/how" class="nav-chip">How it works</a>
    <a href="/login" class="nav-chip" style="color: var(--accent); border-color: rgba(103,232,249,0.25);">My Studio →</a>
    <a class="phone-cta" href="tel:+14434648118"><span class="pulse"></span>+1 (443) 464-8118 · live</a>
  </div>
</div></nav>

<div class="wrap">
  <section class="hero">
    <div>
      <h1>The AI you<br><em>direct</em> on the phone.</h1>
      <p class="lede">Every other tool is prompt → video. Conduit is conversation → video. Call the number, describe what you want, interrupt any shot, watch it re-render. Powered by BytePlus Seed 2.0, Seedream 5.0, Seedance 2.0.</p>
    </div>
    <div class="cta-col">
      <a class="big-phone" href="tel:+14434648118">
        <span class="label">Call now</span>
        +1 (443) 464-8118
      </a>
      <div class="quick">try: <span class="kbd">"make me a 30-second ad for tesla"</span></div>
    </div>
  </section>

  <section class="demo-hero">
    <div class="label-row">
      <span class="eyebrow">Conduit, on a real prompt</span>
      <span class="runtime">25 sec · ✶ edited by Conduit</span>
    </div>
    <div class="prompt-panel">
      <div class="head">
        <span class="role">YOU · on the phone</span>
        <span class="meta">+1 (443) 464-8118</span>
      </div>
      <div class="body">
        “Make me a thirty-second <em>Apple-style launch reel</em> for <em>iBottle</em> —
        a smart water bottle that tracks how much you drink every day.
        Multiple shots, full voice-over, beautiful animation.”
      </div>
      <div class="arrow">↓ &nbsp; CONDUIT DELIVERED</div>
    </div>
    <div class="frame">
      <video id="hero-video" src="/videos/_demo/ibottle.mp4?v=2" autoplay muted loop playsinline preload="metadata"></video>
      <div class="corner-tag">Edited by Conduit</div>
      <button id="hero-unmute" type="button"
              style="position:absolute; bottom:14px; right:14px;
                     background:rgba(6,6,6,0.72); backdrop-filter: blur(8px);
                     border:1px solid var(--border-2); color:var(--text);
                     padding:7px 13px; border-radius:999px; cursor:pointer;
                     font-family:var(--mono); font-size:11px; letter-spacing:0.16em;
                     text-transform:uppercase; display:inline-flex; align-items:center; gap:8px;">
        <span id="hero-unmute-icon">🔇</span><span id="hero-unmute-label">Tap for sound</span>
      </button>
    </div>
  </section>

  <script>
    (function(){
      const v = document.getElementById('hero-video');
      const btn = document.getElementById('hero-unmute');
      const icon = document.getElementById('hero-unmute-icon');
      const lbl = document.getElementById('hero-unmute-label');
      if (!v || !btn) return;
      btn.addEventListener('click', () => {
        v.muted = !v.muted;
        if (!v.muted) { v.play(); icon.textContent='🔊'; lbl.textContent='Sound on'; }
        else          { icon.textContent='🔇'; lbl.textContent='Tap for sound'; }
      });
    })();
  </script>

  <section class="section">
    <div class="section-head">
      <h2>The crew · live</h2>
      <span class="live-badge" id="live-badge">Idle</span>
    </div>
    <div class="agents-strip" id="agents-strip">
      <div class="agent-card"><span class="dot"></span>
        <div class="role">Director</div><div class="name">Claude</div>
        <span class="model">claude-sonnet-4-5</span>
      </div>
      <div class="agent-card"><span class="dot"></span>
        <div class="role">Storyboard</div><div class="name">Seedream</div>
        <span class="model">seedream-5.0</span>
      </div>
      <div class="agent-card"><span class="dot"></span>
        <div class="role">Cinematographer</div><div class="name">Seedance</div>
        <span class="model">seedance-2.0</span>
      </div>
      <div class="agent-card"><span class="dot"></span>
        <div class="role">Voice</div><div class="name">Aura</div>
        <span class="model">deepgram-aura</span>
      </div>
      <div class="agent-card"><span class="dot"></span>
        <div class="role">Telephony</div><div class="name">Vapi</div>
        <span class="model">nova-3 stt</span>
      </div>
      <div class="agent-card"><span class="dot"></span>
        <div class="role">Stitcher</div><div class="name">Editor</div>
        <span class="model">ffmpeg + lut</span>
      </div>
    </div>
  </section>

  <section class="section">
    <h2>Try saying</h2>
    <div class="try-row">
      <button class="try-chip" data-prompt="Make me a thirty-second Apple-style ad for an electric kayak">
        <span class="quote">“Make me a 30-second Apple-style ad for an electric kayak.”</span>
        <span class="tag">30s</span>
      </button>
      <button class="try-chip" data-prompt="Direct a forty-five-second Wes Anderson trailer for a pastry shop in Budapest">
        <span class="quote">“A 45-second Wes Anderson trailer for a pastry shop in Budapest.”</span>
        <span class="tag">45s</span>
      </button>
      <button class="try-chip" data-prompt="Build a Nike-style hype reel for a marathon runner training at dawn, thirty seconds">
        <span class="quote">“Nike-style hype reel — marathon runner, dawn training, 30s.”</span>
        <span class="tag">30s</span>
      </button>
      <button class="try-chip" data-prompt="Make a twenty-second neo-noir teaser about a detective in rainy Tokyo">
        <span class="quote">“Neo-noir teaser — detective in rainy Tokyo, 20 seconds.”</span>
        <span class="tag">20s</span>
      </button>
    </div>
    <div class="try-hint">Click to copy → call <a href="tel:+14434648118">+1 (443) 464-8118</a> → say it.</div>
  </section>

  <section class="section">
    <h2>Why this isn't a wrapper</h2>
    <div class="why-row">
      <div class="why-card">
        <div class="n">01</div>
        <h3>Conversation, not prompt.</h3>
        <p>Every other AI video tool is one-shot. Conduit is a live director you iterate with — interrupt any shot, redirect, and only that shot re-renders.</p>
      </div>
      <div class="why-card">
        <div class="n">02</div>
        <h3>Six specialists. One call.</h3>
        <p>Director, Storyboard, Cinematographer, Voice, Telephony, Editor — each is a purpose-built model behind a phone number you already know how to use.</p>
      </div>
      <div class="why-card">
        <div class="n">03</div>
        <h3>Partial regen, real continuity.</h3>
        <p>Style + character bibles get baked into every shot prompt. Change shot 3 — shots 1, 2, 4, 5 don't drift. Same world, same face, same look.</p>
      </div>
    </div>
  </section>

  <section class="section">
    <h2>Powered by</h2>
    <div class="tech-strip">
      <span class="tech-pill"><strong>Anthropic</strong> · Claude Sonnet 4.5</span>
      <span class="tech-pill"><strong>BytePlus</strong> · Seedream 5.0</span>
      <span class="tech-pill"><strong>BytePlus</strong> · Seedance 2.0</span>
      <span class="tech-pill"><strong>Vapi</strong> · live phone</span>
      <span class="tech-pill"><strong>Deepgram</strong> · Nova-3 + Aura</span>
      <span class="tech-pill"><strong>fal.ai</strong> · video gateway</span>
      <span class="tech-pill"><strong>ffmpeg</strong> · stitch + grade</span>
    </div>
  </section>

  <section class="section">
    <h2>By the numbers</h2>
    <div class="stat-strip" id="stat-strip">
      <div class="stat"><div class="n" id="stat-calls">17</div><div class="l">Calls placed</div></div>
      <div class="stat"><div class="n" id="stat-shots">64</div><div class="l">Shots rendered</div></div>
      <div class="stat"><div class="n" id="stat-finals">11</div><div class="l">Videos finalized</div></div>
      <div class="stat"><div class="n" id="stat-live">0</div><div class="l">Live right now</div></div>
    </div>
  </section>

  <section class="big-cta-section">
    <h2>Ready to <em>direct</em>?</h2>
    <p>Pick up your phone. Describe your film. Watch shots stream in.</p>
    <a class="big-cta-btn" href="tel:+14434648118"><span class="pulse"></span>+1 (443) 464-8118</a>
  </section>

  <footer class="footer">
    <div>CONDUIT · Beta Hacks · Seed Agents Challenge</div>
    <div><a href="/how">How it works →</a></div>
  </footer>
</div>

<script>
async function refresh() {
  try {
    const r = await fetch('/api/sessions');
    if (!r.ok) return;
    const d = await r.json();
    const live = d.filter(s => !s.final_video_path).length;
    const totalShots = d.reduce((a,s) => a + (s.shot_count||0), 0);
    const finals = d.filter(s => s.final_video_path).length;
    document.getElementById('stat-calls').textContent = d.length;
    document.getElementById('stat-shots').textContent = totalShots;
    document.getElementById('stat-finals').textContent = finals;
    document.getElementById('stat-live').textContent = live;

    const badge = document.getElementById('live-badge');
    if (badge) {
      badge.textContent = live > 0 ? (live + ' live now') : 'Idle';
      badge.classList.toggle('on', live > 0);
    }
    document.querySelectorAll('.agent-card').forEach(el => {
      el.classList.toggle('active', live > 0);
    });
  } catch(e) {}
}

document.querySelectorAll('.try-chip').forEach(btn => {
  btn.addEventListener('click', async () => {
    const txt = btn.dataset.prompt || '';
    try { await navigator.clipboard.writeText(txt); } catch(e) {}
    const tag = btn.querySelector('.tag');
    if (tag) {
      const orig = tag.textContent;
      tag.textContent = 'COPIED';
      tag.style.color = 'var(--success)';
      tag.style.borderColor = 'rgba(74,222,128,0.35)';
      setTimeout(() => {
        tag.textContent = orig;
        tag.style.color = '';
        tag.style.borderColor = '';
      }, 1600);
    }
  });
});

function timeAgo(d) {
  const s = Math.floor((Date.now() - d.getTime())/1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s/60)}m ago`;
  if (s < 86400) return `${Math.floor(s/3600)}h ago`;
  return d.toLocaleDateString();
}

refresh(); setInterval(refresh, 4000);
</script>
</body></html>"""


CALL_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conduit — call</title>
__BASE_STYLE__
<script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.2/Sortable.min.js"></script>
<style>
  .back-row { padding-bottom: 8px; }
  .back-row a {
    font-family: var(--mono); font-size: 12px; color: var(--text-3);
    letter-spacing: 0.05em; display: inline-flex; align-items: center; gap: 6px;
  }
  .back-row a:hover { color: var(--text); text-decoration: none; }

  .call-header {
    padding: 24px 0 40px; border-bottom: 1px solid var(--border);
    display: grid; grid-template-columns: 1fr auto; gap: 28px; align-items: end;
  }
  .call-header h1 {
    font-family: var(--serif); font-weight: 400;
    font-size: clamp(32px, 5vw, 54px); line-height: 1.05; letter-spacing: -0.02em;
    margin: 0 0 10px; color: var(--text);
  }
  .call-header .brief { color: var(--text-2); font-size: 16px; max-width: 640px; }
  .call-header .side { text-align: right; }
  .progress-ring {
    display: inline-flex; align-items: center; gap: 10px;
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-2);
  }
  .progress-ring .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
  }
  .progress-ring.live .status-dot { background: var(--warning); animation: breathe 1.4s infinite; }
  .progress-ring.done .status-dot { background: var(--success); }

  /* pipeline visual */
  .pipeline {
    padding: 28px 0 8px;
    display: flex; gap: 0; align-items: center; overflow-x: auto;
    font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text-3);
  }
  .pipeline .step {
    padding: 8px 14px; border: 1px solid var(--border); border-radius: 999px;
    background: var(--surface); white-space: nowrap;
  }
  .pipeline .step.active { border-color: var(--accent); color: var(--accent); box-shadow: 0 0 16px var(--accent-glow); }
  .pipeline .arrow { padding: 0 10px; color: var(--text-3); }

  /* main grid — shots left, sidebar right */
  .main {
    display: grid; grid-template-columns: 1fr 360px; gap: 32px;
    padding: 36px 0 0;
  }
  @media (max-width: 980px) { .main { grid-template-columns: 1fr; } }

  .section-title {
    font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--text-3); font-weight: 500; margin: 0 0 14px;
  }

  .shots {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px;
  }
  .shot {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    overflow: hidden; position: relative; transition: all .3s ease;
    animation: shotIn .4s cubic-bezier(0.2, 0.8, 0.2, 1) both;
  }
  @keyframes shotIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .shot:hover { border-color: var(--border-2); }
  .shot .media { aspect-ratio: 16/9; background: #000; position: relative; overflow: hidden; }
  .shot .media img, .shot .media video { width: 100%; height: 100%; object-fit: cover; display: block; }
  .shot .placeholder {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 11px; color: var(--text-3); letter-spacing: 0.1em;
    background-image: linear-gradient(135deg, #15151a 0%, #0a0a0c 100%);
  }
  .shot .placeholder.skel::after {
    content:''; position:absolute; inset:0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.05) 50%, transparent 100%);
    animation: shimmer 1.4s infinite linear;
  }
  @keyframes shimmer { 0%{transform:translateX(-100%)} 100%{transform:translateX(100%)} }

  .shot .pill {
    position: absolute; top: 10px; left: 10px;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase;
    padding: 4px 10px; border-radius: 999px;
    background: rgba(0,0,0,0.65); border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
  }
  .shot .pill .dot { width: 5px; height: 5px; border-radius: 50%; display: inline-block; margin-right: 6px; vertical-align: 1px; }
  .pill.planned { color: var(--text-3); } .pill.planned .dot { background: var(--text-3); }
  .pill.keyframe { color: #a5b4fc; border-color: rgba(165,180,252,0.3); } .pill.keyframe .dot { background: #a5b4fc; animation: breathe 1.2s infinite; }
  .pill.rendering { color: var(--warning); border-color: rgba(250,204,21,0.3); } .pill.rendering .dot { background: var(--warning); animation: breathe 1.2s infinite; }
  .pill.done { color: var(--success); border-color: rgba(74,222,128,0.3); } .pill.done .dot { background: var(--success); }
  .pill.dirty { color: #fb923c; border-color: rgba(251,146,60,0.3); } .pill.dirty .dot { background: #fb923c; }
  .pill.failed { color: var(--danger); border-color: rgba(248,113,113,0.35); } .pill.failed .dot { background: var(--danger); }

  .shot .idx {
    position: absolute; top: 10px; right: 10px;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.15em; color: var(--text-3);
    background: rgba(0,0,0,0.65); padding: 4px 8px; border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.08); backdrop-filter: blur(8px);
  }

  .shot .info { padding: 14px 16px 16px; }
  .shot .intent { font-size: 13.5px; color: var(--text); line-height: 1.4; margin-bottom: 6px; }
  .shot .narr {
    font-family: var(--serif); font-style: italic; font-size: 14px; color: var(--text-2);
    margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--border);
  }
  .shot .narr::before { content: '"'; color: var(--text-3); margin-right: 2px; }
  .shot .narr::after { content: '"'; color: var(--text-3); margin-left: 2px; }

  /* sidebar */
  .sidebar { display: flex; flex-direction: column; gap: 16px; position: sticky; top: 88px; height: fit-content; }
  .panel {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 18px 18px 16px;
  }
  .panel h3 {
    font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--text-3); font-weight: 500; margin: 0 0 12px;
  }
  .transcript { max-height: 420px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
  .transcript .turn { display: flex; gap: 8px; align-items: flex-start; font-size: 13px; line-height: 1.4; }
  .transcript .role {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    padding: 2px 7px; border-radius: 4px; flex-shrink: 0; margin-top: 2px;
  }
  .transcript .turn.user .role { background: rgba(103,232,249,0.1); color: var(--accent); border: 1px solid rgba(103,232,249,0.25); }
  .transcript .turn.assistant .role { background: rgba(74,222,128,0.08); color: var(--success); border: 1px solid rgba(74,222,128,0.25); }
  .transcript .txt { color: var(--text); }
  .transcript .turn.user .txt { color: var(--text); }

  .architecture { font-family: var(--mono); font-size: 12px; }
  .architecture .row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px dashed var(--border); }
  .architecture .row:last-child { border-bottom: 0; }
  .architecture .l { color: var(--text-2); }
  .architecture .r { color: var(--accent); }

  /* final video section */
  .finalized-wrap {
    margin: 40px 0 0; padding: 24px; border-radius: 18px;
    background: linear-gradient(180deg, rgba(74,222,128,0.05) 0%, rgba(74,222,128,0.01) 100%);
    border: 1px solid rgba(74,222,128,0.18);
  }
  .finalized-wrap .label {
    font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--success); margin-bottom: 14px; display: inline-flex; align-items: center; gap: 8px;
  }
  .finalized-wrap .label::before {
    content:''; width:6px; height:6px; border-radius:50%; background: var(--success); box-shadow: 0 0 0 3px rgba(74,222,128,0.2);
  }
  .finalized-wrap video {
    width: 100%; max-height: 560px; border-radius: 10px; background: #000;
    box-shadow: 0 0 60px rgba(74,222,128,0.08);
  }
  .finalized-wrap .actions { display: flex; gap: 10px; margin-top: 16px; }
  .btn {
    font-family: var(--mono); font-size: 12px; letter-spacing: 0.06em;
    padding: 10px 18px; border-radius: 10px; background: var(--surface-2); border: 1px solid var(--border-2);
    color: var(--text); cursor: pointer; transition: all .18s ease; text-decoration: none;
  }
  .btn:hover { border-color: var(--accent); color: var(--accent); text-decoration: none; }
  .btn.primary {
    background: var(--accent); color: #000; border-color: var(--accent);
  }
  .btn.primary:hover { background: var(--text); color: #000; border-color: var(--text); }

  .shot-actions {
    position: absolute; bottom: 10px; right: 10px; display: flex; gap: 6px;
    opacity: 0; transition: opacity .2s ease;
  }
  .shot:hover .shot-actions { opacity: 1; }
  .shot-btn {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 5px 10px; border-radius: 6px;
    background: rgba(0,0,0,0.7); border: 1px solid rgba(255,255,255,0.2);
    color: var(--text); cursor: pointer; transition: all .15s;
    backdrop-filter: blur(8px);
  }
  .shot-btn:hover { border-color: var(--accent); color: var(--accent); }

  .regen-form {
    background: rgba(0,0,0,0.92); position: absolute; inset: 0; padding: 18px;
    display: flex; flex-direction: column; gap: 10px; z-index: 10;
  }
  .regen-form textarea {
    flex: 1; background: var(--surface); border: 1px solid var(--border-2);
    color: var(--text); padding: 10px; border-radius: 8px;
    font-family: var(--mono); font-size: 12px; resize: none; outline: none;
  }
  .regen-form textarea:focus { border-color: var(--accent); }
  .regen-form .row { display: flex; gap: 6px; }
  .regen-form .row .btn { flex: 1; text-align: center; font-size: 11px; padding: 8px 10px; }

  .top-actions {
    display: flex; gap: 10px; margin-left: auto; align-items: center;
  }
  .action-btn {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em;
    padding: 8px 14px; border-radius: 8px; background: var(--surface-2);
    border: 1px solid var(--border-2); color: var(--text); cursor: pointer;
    transition: all .18s ease;
  }
  .action-btn:hover { border-color: var(--accent); color: var(--accent); }
  .action-btn.primary { background: var(--accent); color: #000; border-color: var(--accent); }
  .action-btn.primary:hover { background: var(--text); color: #000; border-color: var(--text); }
  .action-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* drag-to-reorder */
  .shot.sortable-ghost { opacity: 0.35; border: 1px dashed var(--accent); }
  .shot.sortable-drag { box-shadow: 0 20px 60px rgba(0,0,0,0.6); transform: scale(1.02); }
  .shot .drag-handle {
    position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
    cursor: grab; color: rgba(255,255,255,0.3); font-size: 12px;
    padding: 4px 8px; background: rgba(0,0,0,0.6); border-radius: 4px;
    opacity: 0; transition: opacity .15s ease;
    backdrop-filter: blur(6px);
  }
  .shot:hover .drag-handle { opacity: 1; }
  .shot .drag-handle:active { cursor: grabbing; }

  /* overlay form */
  .overlay-form {
    background: rgba(0,0,0,0.94); position: absolute; inset: 0; padding: 16px;
    display: none; flex-direction: column; gap: 10px; z-index: 11;
  }
  .overlay-form input {
    background: var(--surface); border: 1px solid var(--border-2);
    color: var(--text); padding: 10px 12px; border-radius: 8px;
    font-family: var(--mono); font-size: 13px; outline: none;
  }
  .overlay-form input:focus { border-color: var(--accent); }
  .overlay-form select {
    background: var(--surface); border: 1px solid var(--border-2);
    color: var(--text-2); padding: 8px 10px; border-radius: 8px;
    font-family: var(--mono); font-size: 12px; outline: none;
  }
  .overlay-label {
    font-family: var(--mono); font-size: 10px; color: var(--text-3);
    letter-spacing: 0.14em; text-transform: uppercase;
  }

  /* stitch progress */
  .stitch-progress {
    display: none; align-items: center; gap: 12px;
    padding: 12px 16px; background: var(--surface);
    border: 1px solid rgba(103,232,249,0.18); border-radius: 10px;
    margin-top: 12px;
  }
  .stitch-progress.show { display: flex; }
  .stitch-progress .sp-bar-wrap {
    flex: 1; height: 3px; background: rgba(255,255,255,0.08);
    border-radius: 2px; overflow: hidden;
  }
  .stitch-progress .sp-bar {
    height: 100%; background: var(--accent); border-radius: 2px;
    transition: width .4s ease; width: 0%;
  }
  .stitch-progress .sp-label {
    font-family: var(--mono); font-size: 11px; color: var(--text-3);
    white-space: nowrap;
  }

  /* improve prompt btn */
  .improve-btn {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.1em;
    padding: 5px 10px; border-radius: 6px;
    background: rgba(103,232,249,0.08); border: 1px solid rgba(103,232,249,0.2);
    color: var(--accent); cursor: pointer; transition: all .15s; white-space: nowrap;
  }
  .improve-btn:hover { background: rgba(103,232,249,0.15); }
  .improve-btn:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>
<nav class="topnav"><div class="inner">
  <a href="/" class="brand" style="text-decoration:none"><span class="glyph"></span>CONDUIT</a>
  <div style="display:flex; gap: 12px; align-items: center;">
    <a href="/how" class="nav-chip">How it works</a>
    <a href="/login" class="nav-chip" style="color: var(--accent); border-color: rgba(103,232,249,0.25);">My Studio →</a>
    <a class="phone-cta" href="tel:+14434648118"><span class="pulse"></span>+1 (443) 464-8118 · live</a>
  </div>
</div></nav>

<div class="wrap">
  <div class="back-row"><a href="/login">← my studio</a></div>

  <section class="call-header">
    <div>
      <h1 id="title">Loading…</h1>
      <div class="brief" id="brief"></div>
    </div>
    <div class="side">
      <div class="progress-ring live" id="progress">
        <span class="status-dot"></span>
        <span id="progress-label">…</span>
      </div>
      <div class="top-actions" style="margin-top: 12px; justify-content: flex-end;">
        <button class="action-btn" id="reorder-hint" style="display:none; cursor:default; opacity:0.6">⠿ Drag shots to reorder</button>
        <button class="action-btn primary" id="finalize-btn" onclick="finalize()">Finalize</button>
      </div>
      <div class="stitch-progress" id="stitch-progress">
        <div class="sp-bar-wrap"><div class="sp-bar" id="sp-bar"></div></div>
        <span class="sp-label" id="sp-label">Stitching…</span>
      </div>
    </div>
  </section>

  <div class="pipeline" id="pipeline">
    <div class="step">Director · Claude</div><div class="arrow">→</div>
    <div class="step">Seedream 5.0</div><div class="arrow">→</div>
    <div class="step">Seedance 2.0</div><div class="arrow">→</div>
    <div class="step">Seed Speech</div><div class="arrow">→</div>
    <div class="step">ffmpeg stitch</div>
  </div>

  <div class="main">
    <div>
      <div class="section-title">Shots</div>
      <div class="shots" id="shots"></div>
      <div id="final"></div>
    </div>
    <aside class="sidebar">
      <div class="panel">
        <h3>Transcript</h3>
        <div class="transcript" id="transcript"><div style="color: var(--text-3); font-family: var(--mono); font-size: 12px;">no turns yet…</div></div>
      </div>
      <div class="panel architecture">
        <h3>Call info</h3>
        <div class="row"><span class="l">Call ID</span><span class="r" id="a-callid">—</span></div>
        <div class="row"><span class="l">Started</span><span class="r" id="a-started">—</span></div>
        <div class="row"><span class="l">Shots</span><span class="r" id="a-shots">0</span></div>
        <div class="row"><span class="l">Done</span><span class="r" id="a-done">0</span></div>
        <div class="row"><span class="l">Final</span><span class="r" id="a-final">—</span></div>
      </div>
    </aside>
  </div>
</div>

<script>
const CALL_ID = "__CALL_ID__";

function renderShots(shots) {
  const cb = Date.now();
  if (!shots || !shots.length) return '<div class="panel" style="grid-column: 1 / -1; text-align: center; color: var(--text-3);">Director still planning…</div>';
  return shots.map(s => {
    let media = `<div class="placeholder skel">awaiting keyframe</div>`;
    // keep keyframe visible in background while rendering (shimmer overlay instead of replacing)
    if (s.clip_path) {
      const clipFile = s.clip_path.split('/').slice(-1)[0];
      // resolve clip path — could be in clips/, _overlays/, _processed/
      const clipSrc = s.clip_path.includes('_overlay') || s.clip_path.includes('_processed')
        ? `/videos/${CALL_ID}/${s.clip_path.split('videos/')[1] || clipFile}`
        : `/videos/${CALL_ID}/clips/${clipFile}`;
      media = `<video src="${clipSrc}?t=${cb}" controls muted loop preload="metadata"></video>`;
    } else if (s.keyframe_path) {
      const kfFile = s.keyframe_path.split('/').slice(-1)[0];
      const shimmer = s.status === 'rendering'
        ? `<div class="placeholder skel" style="position:absolute;inset:0;background:transparent"></div>`
        : '';
      media = `<img src="/videos/${CALL_ID}/keyframes/${kfFile}?t=${cb}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover">${shimmer}`;
    }
    const canRegen = ['done','failed','dirty'].includes(s.status);
    const canOverlay = s.status === 'done';
    const escPrompt = (s.prompt || '').replace(/"/g, '&quot;').replace(/</g,'&lt;');
    const escNarr = (s.narration || '').replace(/"/g, '&quot;').replace(/</g,'&lt;');
    return `<div class="shot" data-shot-idx="${s.index}">
      <div class="media" style="position:relative">
        ${media}
        <div class="drag-handle" title="Drag to reorder">⠿ drag</div>
        <span class="pill ${s.status}"><span class="dot"></span>${s.status}</span>
        <span class="idx">Shot ${String(s.index + 1).padStart(2,'0')}</span>
        ${(canRegen || canOverlay) ? `<div class="shot-actions">
          ${canRegen ? `<button class="shot-btn" onclick="openRegen(${s.index})">↻ Redo</button>` : ''}
          ${canOverlay ? `<button class="shot-btn" onclick="openOverlay(${s.index})">✎ Text</button>` : ''}
        </div>` : ''}
      </div>
      <div class="info">
        <div class="intent">${s.intent || ''}</div>
        ${s.narration ? `<div class="narr">${s.narration}</div>` : ''}
      </div>
      <div class="regen-form" id="regen-${s.index}" style="display:none">
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px">
          <span style="font-family:var(--mono);font-size:10px;color:var(--text-3);letter-spacing:0.14em;text-transform:uppercase">Edit shot ${s.index + 1}</span>
          <button class="improve-btn" id="improve-btn-${s.index}" onclick="improvePrompt(${s.index})">✦ AI improve</button>
        </div>
        <textarea id="rg-prompt-${s.index}" placeholder="Visual prompt" rows="5">${escPrompt}</textarea>
        <textarea id="rg-narr-${s.index}" placeholder="Narration (optional)" rows="2">${escNarr}</textarea>
        <div class="row">
          <button class="btn" onclick="closeRegen(${s.index})">Cancel</button>
          <button class="btn primary" onclick="submitRegen(${s.index})">Re-render</button>
        </div>
      </div>
      <div class="overlay-form" id="overlay-${s.index}" style="display:none">
        <div class="overlay-label">Add text overlay — shot ${s.index + 1}</div>
        <input id="ov-text-${s.index}" type="text" placeholder='e.g. "TESLA" or lower-third caption' value="">
        <select id="ov-pos-${s.index}">
          <option value="lower_third">Lower Third</option>
          <option value="title">Title Card</option>
        </select>
        <div class="row">
          <button class="btn" onclick="closeOverlay(${s.index})">Cancel</button>
          <button class="btn primary" onclick="submitOverlay(${s.index})">Apply</button>
        </div>
      </div>
    </div>`;
  }).join('');
}

function openRegen(idx) {
  document.getElementById('regen-' + idx).style.display = 'flex';
}
function closeRegen(idx) {
  document.getElementById('regen-' + idx).style.display = 'none';
}
function openOverlay(idx) {
  document.getElementById('overlay-' + idx).style.display = 'flex';
}
function closeOverlay(idx) {
  document.getElementById('overlay-' + idx).style.display = 'none';
}

async function improvePrompt(idx) {
  const btn = document.getElementById('improve-btn-' + idx);
  const textarea = document.getElementById('rg-prompt-' + idx);
  if (!textarea) return;
  const orig = btn.textContent;
  btn.disabled = true; btn.textContent = '⟳ Thinking…';
  try {
    const r = await fetch(`/api/improve-prompt/${CALL_ID}/${idx}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ current_prompt: textarea.value })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'improve failed');
    textarea.value = d.improved_prompt;
    textarea.style.borderColor = 'var(--accent)';
    setTimeout(() => textarea.style.borderColor = '', 1500);
  } catch(e) { alert(e.message); }
  finally { btn.disabled = false; btn.textContent = orig; }
}

async function submitRegen(idx) {
  const prompt = document.getElementById('rg-prompt-' + idx).value.trim();
  const narr = document.getElementById('rg-narr-' + idx).value.trim();
  if (!prompt) return alert('Prompt cannot be empty');
  try {
    const r = await fetch(`/api/regen/${CALL_ID}/${idx}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ new_prompt: prompt, new_narration: narr })
    });
    if (!r.ok) throw new Error('regen failed');
    closeRegen(idx);
    setTimeout(() => location.reload(), 400);
  } catch(e) { alert(e.message); }
}

async function submitOverlay(idx) {
  const text = document.getElementById('ov-text-' + idx).value.trim();
  const pos = document.getElementById('ov-pos-' + idx).value;
  const btn = document.querySelector(`#overlay-${idx} .btn.primary`);
  if (btn) { btn.disabled = true; btn.textContent = 'Applying…'; }
  try {
    const r = await fetch(`/api/overlay/${CALL_ID}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ shot_index: idx, text, position: pos })
    });
    if (!r.ok) throw new Error('overlay failed');
    closeOverlay(idx);
  } catch(e) { alert(e.message); }
  finally { if (btn) { btn.disabled = false; btn.textContent = 'Apply'; } }
}

async function finalize() {
  const btn = document.getElementById('finalize-btn');
  const prog = document.getElementById('stitch-progress');
  const bar = document.getElementById('sp-bar');
  const lbl = document.getElementById('sp-label');
  btn.disabled = true; btn.textContent = 'Stitching…';
  if (prog) prog.classList.add('show');
  let pct = 0;
  const ticker = setInterval(() => {
    pct = Math.min(pct + Math.random() * 12, 88);
    if (bar) bar.style.width = pct + '%';
    if (lbl) lbl.textContent = `Stitching… ${Math.round(pct)}%`;
  }, 600);
  try {
    const r = await fetch(`/api/finalize/${CALL_ID}`, { method: 'POST' });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'finalize failed');
    clearInterval(ticker);
    if (bar) bar.style.width = '100%';
    if (lbl) lbl.textContent = 'Done ✓';
    const rs = await fetch('/api/session/' + CALL_ID);
    if (rs.ok) apply(await rs.json());
    setTimeout(() => { if (prog) prog.classList.remove('show'); }, 2000);
  } catch(e) {
    clearInterval(ticker);
    if (prog) prog.classList.remove('show');
    alert(e.message);
  } finally { btn.disabled = false; btn.textContent = 'Finalize'; }
}

let _sortable = null;
function initSortable() {
  const el = document.getElementById('shots');
  if (!el || _sortable) return;
  _sortable = Sortable.create(el, {
    animation: 200,
    handle: '.drag-handle',
    ghostClass: 'sortable-ghost',
    dragClass: 'sortable-drag',
    onEnd: async (evt) => {
      const items = Array.from(el.querySelectorAll('.shot[data-shot-idx]'));
      const order = items.map(el => parseInt(el.dataset.shotIdx));
      try {
        await fetch(`/api/reorder/${CALL_ID}`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ order })
        });
      } catch(e) { console.error('reorder failed', e); }
    }
  });
  const hint = document.getElementById('reorder-hint');
  if (hint) hint.style.display = '';
}

function renderTranscript(tr) {
  const el = document.getElementById('transcript');
  if (!tr || !tr.length) { el.innerHTML = '<div style="color: var(--text-3); font-family: var(--mono); font-size: 12px;">no turns yet…</div>'; return; }
  el.innerHTML = tr.map(t => `
    <div class="turn ${t.role}">
      <span class="role">${t.role === 'assistant' ? 'Dir' : 'You'}</span>
      <span class="txt">${t.text}</span>
    </div>`).join('');
  el.scrollTop = el.scrollHeight;
}

function renderFinal(s) {
  const el = document.getElementById('final');
  if (!s.final_video_path) { el.innerHTML = ''; return; }
  const fname = s.final_video_path.split('/').pop();
  const cb = Date.now();
  const videoUrl = `/videos/${CALL_ID}/${fname}?t=${cb}`;
  el.innerHTML = `<div class="finalized-wrap">
    <div class="label">Finalized</div>
    <video src="${videoUrl}" controls playsinline></video>
    <div class="actions">
      <a class="btn primary" href="${videoUrl}" download="${CALL_ID}.mp4">Download MP4</a>
      <a class="btn" href="${videoUrl}" target="_blank">Open in new tab</a>
      <button class="btn" onclick="navigator.clipboard.writeText(location.origin + '${videoUrl}'); this.textContent='Copied'; setTimeout(()=>this.textContent='Copy link', 1200);">Copy link</button>
    </div>
  </div>`;
}

function renderHeader(s) {
  document.getElementById('title').textContent = s.title || 'Untitled';
  document.getElementById('brief').textContent = s.brief || '';
  const done = (s.shots || []).filter(x => x.status === 'done').length;
  const total = (s.shots || []).length;
  const progress = document.getElementById('progress');
  const label = document.getElementById('progress-label');
  if (s.final_video_path) {
    progress.className = 'progress-ring done';
    label.textContent = 'Finalized · ' + total + ' shots';
  } else {
    progress.className = 'progress-ring live';
    label.textContent = `Rendering · ${done}/${total}`;
  }
  document.getElementById('a-callid').textContent = s.call_id ? s.call_id.slice(0,12) : '—';
  document.getElementById('a-started').textContent = s.created_at ? new Date(s.created_at * 1000).toLocaleTimeString() : '—';
  document.getElementById('a-shots').textContent = total;
  document.getElementById('a-done').textContent = done;
  document.getElementById('a-final').textContent = s.final_video_path ? '✓' : '—';

  // pipeline steps active state
  const steps = document.querySelectorAll('.pipeline .step');
  steps.forEach(e => e.classList.remove('active'));
  if (!total) { steps[0].classList.add('active'); }
  else if (!done) { steps[1].classList.add('active'); steps[2].classList.add('active'); }
  else if (!s.final_video_path) { steps[3].classList.add('active'); steps[4].classList.add('active'); }
  else { steps.forEach(e => e.classList.add('active')); }
}

function apply(snap) {
  renderHeader(snap);
  document.getElementById('shots').innerHTML = renderShots(snap.shots || []);
  renderTranscript(snap.transcript);
  renderFinal(snap);
  if (snap.shots && snap.shots.length > 1) {
    _sortable = null;  // reset so initSortable re-creates
    initSortable();
  }
}

async function bootstrap() {
  const r = await fetch('/api/session/' + CALL_ID);
  if (r.ok) apply(await r.json());
}
bootstrap();

const es = new EventSource('/events/' + CALL_ID);
es.addEventListener('snapshot', e => apply(JSON.parse(e.data)));
['shot.status','shot.keyframe','shot.clip','session.finalized'].forEach(ev =>
  es.addEventListener(ev, async () => { const r = await fetch('/api/session/' + CALL_ID); if (r.ok) apply(await r.json()); })
);
</script>
</body></html>"""


HOW_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Conduit — how it works</title>
__BASE_STYLE__
<style>
  .hero-how {
    padding: 72px 0 32px; text-align: center;
  }
  .hero-how h1 {
    font-family: var(--serif); font-weight: 400;
    font-size: clamp(40px, 6vw, 72px); line-height: 1.02; letter-spacing: -0.02em;
    margin: 0 0 16px; color: var(--text);
  }
  .hero-how h1 em { font-style: italic; color: var(--accent); }
  .hero-how .lede {
    font-size: 16px; color: var(--text-2); max-width: 560px; margin: 0 auto 0;
  }

  .stats-hero {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
    background: var(--border); border-radius: 14px; overflow: hidden;
    margin: 48px 0 72px; border: 1px solid var(--border);
  }
  @media (max-width: 640px) { .stats-hero { grid-template-columns: repeat(2, 1fr); } }
  .stats-hero .cell { background: var(--surface); padding: 26px 24px; text-align: center; }
  .stats-hero .n { font-family: var(--serif); font-size: 44px; line-height: 1; letter-spacing: -0.02em; color: var(--text); }
  .stats-hero .l { font-family: var(--mono); font-size: 10px; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-3); margin-top: 8px; }

  /* the flow diagram — 3D, always animating */
  .flow {
    position: relative; padding: 40px 0 60px;
    perspective: 1400px;
    perspective-origin: 50% 30%;
  }
  .flow-row {
    display: flex; justify-content: center; gap: 28px; flex-wrap: wrap;
    margin-bottom: 48px; position: relative;
    transform-style: preserve-3d;
  }
  .flow-row.fan { justify-content: center; gap: 24px; flex-wrap: wrap; }

  .connector {
    position: relative;
    width: 2px; height: 60px;
    margin: -24px auto 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.05) 100%);
    border-radius: 2px; overflow: visible;
  }
  .connector::after {
    content: ''; position: absolute; left: 50%; top: -4px;
    width: 8px; height: 8px; margin-left: -4px;
    border-radius: 50%; background: var(--accent);
    box-shadow: 0 0 12px var(--accent-glow), 0 0 26px rgba(103,232,249,0.45);
    animation: pulseDown 2.4s cubic-bezier(.55,.05,.45,1) infinite;
    opacity: 0;
  }
  .connector.live { background: linear-gradient(180deg, rgba(103,232,249,0.45) 0%, rgba(103,232,249,0.06) 100%); }
  .connector.live::after { opacity: 1; }
  @keyframes pulseDown {
    0%   { top: -6px; opacity: 0; transform: scale(0.6); }
    15%  { opacity: 1; transform: scale(1.15); }
    85%  { opacity: 1; transform: scale(1); }
    100% { top: calc(100% - 2px); opacity: 0; transform: scale(0.7); }
  }

  .fan-connectors {
    position: absolute; top: -40px; left: 50%; transform: translateX(-50%);
    width: 80%; height: 40px; pointer-events: none;
  }
  .fan-connectors svg { width: 100%; height: 100%; }

  .node {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 20px;
    min-width: 220px; max-width: 260px;
    position: relative; cursor: pointer;
    transform-style: preserve-3d; will-change: transform;
    transition: transform .4s cubic-bezier(.2,.8,.3,1),
                border-color .25s,
                box-shadow .35s,
                background .25s;
    box-shadow:
      0 1px 0 rgba(255,255,255,0.04) inset,
      0 18px 32px -16px rgba(0,0,0,0.6);
  }
  .node::before {
    content: ''; position: absolute; inset: -3px;
    border-radius: 18px;
    background: radial-gradient(70% 60% at 50% 50%, var(--accent-glow), transparent 72%);
    opacity: 0; transition: opacity .35s ease;
    z-index: -1; pointer-events: none;
  }
  .node:hover { border-color: var(--border-2); }
  .node.active {
    border-color: var(--accent);
    background: linear-gradient(180deg, rgba(103,232,249,0.05) 0%, var(--surface) 60%);
    box-shadow:
      0 1px 0 rgba(103,232,249,0.18) inset,
      0 0 56px rgba(103,232,249,0.35),
      0 30px 60px -20px rgba(0,0,0,0.65),
      0 0 0 1px var(--accent);
    transform: translateZ(36px);
  }
  .node.active::before { opacity: 1; }

  .node .row-1 {
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;
  }
  .node .name {
    font-family: var(--serif); font-size: 20px; letter-spacing: -0.01em; color: var(--text);
  }
  .node .status-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--text-3);
    transition: all .3s ease;
  }
  .node.active .status-dot {
    background: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
    animation: breathe 1.4s infinite;
  }
  .node .role { font-size: 12px; color: var(--text-2); margin-bottom: 10px; }
  .node .model {
    font-family: var(--mono); font-size: 11px; color: var(--accent);
    background: rgba(103,232,249,0.08); padding: 3px 8px; border-radius: 6px;
    display: inline-block; border: 1px solid rgba(103,232,249,0.18);
  }
  .node .count {
    position: absolute; top: 14px; right: 16px;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.12em; color: var(--text-3);
    text-transform: uppercase;
  }
  .node .count strong { color: var(--text); font-weight: 500; }

  .node .detail {
    margin-top: 12px; padding-top: 12px;
    border-top: 1px solid var(--border);
    font-family: var(--mono); font-size: 11px; line-height: 1.6;
    color: var(--text-2); letter-spacing: 0.02em;
    max-height: 0; opacity: 0; overflow: hidden;
    transition: max-height .35s cubic-bezier(.2,.8,.3,1), opacity .25s ease, padding .35s;
  }
  .node.expanded { transform: translateZ(46px) scale(1.015); }
  .node.expanded .detail { max-height: 200px; opacity: 1; }
  .node .expand-hint {
    position: absolute; bottom: 8px; right: 12px;
    font-family: var(--mono); font-size: 9px; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--text-3); opacity: 0.5;
    transition: opacity .2s ease;
  }
  .node:hover .expand-hint { opacity: 1; }
  .node.expanded .expand-hint { opacity: 1; color: var(--accent); }

  .output-node {
    background: linear-gradient(180deg, rgba(74,222,128,0.06) 0%, rgba(74,222,128,0.01) 100%);
    border: 1px solid rgba(74,222,128,0.25);
  }
  .output-node .name { color: var(--success); }
  .output-node .status-dot { background: var(--success); box-shadow: 0 0 0 3px rgba(74,222,128,0.2); }

  .section-title {
    font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--text-3); font-weight: 500; margin: 0 0 18px; text-align: center;
  }

  .why-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
    margin-top: 80px;
  }
  @media (max-width: 780px) { .why-grid { grid-template-columns: 1fr; } }
  .why {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 24px; transition: all .25s ease;
  }
  .why:hover { border-color: var(--border-2); }
  .why h3 {
    font-family: var(--serif); font-weight: 400; font-size: 22px; letter-spacing: -0.01em;
    margin: 0 0 10px; color: var(--text);
  }
  .why p { color: var(--text-2); font-size: 14px; line-height: 1.55; margin: 0; }
  .why .n {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.18em; color: var(--accent);
    text-transform: uppercase; margin-bottom: 12px;
  }

  .phone-strip {
    margin: 80px 0 40px; padding: 48px 32px; border-radius: 18px;
    background: linear-gradient(180deg, rgba(103,232,249,0.05) 0%, rgba(103,232,249,0.01) 100%);
    border: 1px solid rgba(103,232,249,0.18);
    text-align: center;
  }
  .phone-strip h2 {
    font-family: var(--serif); font-weight: 400; font-size: 36px; margin: 0 0 8px; letter-spacing: -0.01em;
  }
  .phone-strip h2 em { color: var(--accent); font-style: italic; }
  .phone-strip .big-cta {
    margin-top: 24px; display: inline-flex; gap: 10px; align-items: center;
    font-family: var(--mono); font-size: 22px; color: var(--text);
    padding: 16px 28px; border-radius: 14px;
    background: var(--bg); border: 1px solid rgba(103,232,249,0.35);
    box-shadow: 0 0 40px rgba(103,232,249,0.1);
    transition: all .2s ease;
  }
  .phone-strip .big-cta:hover { border-color: var(--accent); text-decoration: none; transform: translateY(-1px); }

  .nav-chips { display: flex; gap: 14px; justify-content: center; margin-top: 16px; }
  .nav-chip {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-3); padding: 6px 12px; border-radius: 999px;
    border: 1px solid var(--border);
  }
  .nav-chip:hover { color: var(--text); border-color: var(--border-2); text-decoration: none; }

  .how-prompt {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; overflow: hidden; margin-top: 16px;
  }
  .how-prompt-head, .how-prompt-foot {
    display: flex; justify-content: space-between; align-items: center; gap: 16px;
    padding: 12px 18px;
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em; color: var(--text-3);
  }
  .how-prompt-head { border-bottom: 1px solid var(--border); }
  .how-prompt-foot { border-top: 1px solid var(--border); color: var(--text-2); font-size: 12px; line-height: 1.5; }
  .how-tag {
    font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--accent); background: rgba(103,232,249,0.07);
    border: 1px solid rgba(103,232,249,0.2); padding: 3px 9px; border-radius: 5px;
  }
  .how-meta { color: var(--text-3); }
  .how-code {
    margin: 0; padding: 22px 24px;
    font-family: var(--mono); font-size: 12.5px; line-height: 1.65;
    color: var(--text); white-space: pre-wrap; overflow-x: auto;
  }

  .compare {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; overflow: hidden; margin-top: 16px;
  }
  .compare-row {
    display: grid; grid-template-columns: 1.1fr 1.5fr 1.5fr;
    border-top: 1px solid var(--border);
    padding: 14px 20px; font-size: 14px; align-items: center;
  }
  .compare-row:first-child { border-top: 0; }
  .compare-head {
    background: rgba(103,232,249,0.03);
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--text-3); padding: 12px 20px;
  }
  .compare-head > div + div { color: var(--accent); }
  .compare-head > div + div + div { color: var(--text-3); }
  .compare-l { color: var(--text-2); font-family: var(--mono); font-size: 12px; letter-spacing: 0.04em; }
  .compare-y { color: var(--text); font-family: var(--serif); font-size: 16px; }
  .compare-y::before { content: '✓ '; color: var(--success); font-family: var(--mono); margin-right: 4px; }
  .compare-n { color: var(--text-3); }
  .compare-n::before { content: '— '; color: var(--text-3); margin-right: 4px; }
  @media (max-width: 700px) {
    .compare-row { grid-template-columns: 1fr; gap: 4px; padding: 12px 16px; }
    .compare-l { color: var(--text-3); margin-bottom: 4px; font-size: 11px; }
    .compare-head { display: none; }
  }
</style>
</head>
<body>
<nav class="topnav"><div class="inner">
  <a href="/" class="brand" style="text-decoration:none"><span class="glyph"></span>CONDUIT</a>
  <div style="display:flex; gap: 12px; align-items: center;">
    <a href="/how" class="nav-chip" style="color: var(--text); border-color: var(--accent);">How it works</a>
    <a href="/" class="nav-chip">Home</a>
    <a class="phone-cta" href="tel:+14434648118"><span class="pulse"></span>+1 (443) 464-8118</a>
  </div>
</div></nav>

<div class="wrap">
  <section class="hero-how">
    <h1>Seven agents.<br>One <em>phone call</em>.</h1>
    <p class="lede">You direct. They collaborate. Every video on this platform is produced by a multi-agent pipeline where each role uses a purpose-built model — and you can interrupt any of them.</p>
  </section>

  <section>
    <div class="stats-hero">
      <div class="cell"><div class="n" id="s-calls">17</div><div class="l">Calls directed</div></div>
      <div class="cell"><div class="n" id="s-frames">64</div><div class="l">Keyframes rendered</div></div>
      <div class="cell"><div class="n" id="s-clips">52</div><div class="l">Clips generated</div></div>
      <div class="cell"><div class="n" id="s-live">0</div><div class="l">Live now</div></div>
    </div>
  </section>

  <section class="flow" id="flow">
    <div class="section-title">The orchestration</div>

    <div class="flow-row">
      <div class="node" id="n-you">
        <div class="row-1"><div class="name">You</div><span class="status-dot"></span></div>
        <div class="role">The director — on a phone call</div>
        <div class="model">phone</div>
        <div class="count"><strong id="c-you">17</strong> calls</div>
        <div class="detail">Phone-only · no signup · 2-minute cap · barge-in interrupts mid-shot</div>
        <span class="expand-hint">click ↓</span>
      </div>
    </div>
    <div class="connector" id="c-you-vapi"></div>

    <div class="flow-row">
      <div class="node" id="n-vapi">
        <div class="row-1"><div class="name">Vapi + Deepgram</div><span class="status-dot"></span></div>
        <div class="role">Voice I/O — streams your words, speaks the reply</div>
        <div class="model">nova-3 · aura</div>
        <div class="count"><strong id="c-vapi">142</strong> turns</div>
        <div class="detail">Sub-200ms STT round-trip · server-side tools route to webhook · barge-in supported</div>
        <span class="expand-hint">click ↓</span>
      </div>
    </div>
    <div class="connector" id="c-vapi-dir"></div>

    <div class="flow-row">
      <div class="node" id="n-director">
        <div class="row-1"><div class="name">Director</div><span class="status-dot"></span></div>
        <div class="role">Plans shots, re-cuts on your redirect, calls the crew</div>
        <div class="model">claude-sonnet-4-5</div>
        <div class="count"><strong id="c-director">89</strong> decisions</div>
        <div class="detail">Tool-calling agent · plan_shots · regen_shot · finalize · style + character bibles baked into every prompt</div>
        <span class="expand-hint">click ↓</span>
      </div>
    </div>
    <div class="connector" id="c-dir-fan"></div>

    <div class="flow-row fan">
      <div class="node" id="n-storyboard">
        <div class="row-1"><div class="name">Storyboard</div><span class="status-dot"></span></div>
        <div class="role">Keyframe per shot — locks visual style</div>
        <div class="model">seedream-5-0</div>
        <div class="count"><strong id="c-storyboard">64</strong> frames</div>
        <div class="detail">1024×1024 · ~4s/keyframe · style+character bible prepended · negative prompts on every render</div>
        <span class="expand-hint">click ↓</span>
      </div>
      <div class="node" id="n-cinema">
        <div class="row-1"><div class="name">Cinematographer</div><span class="status-dot"></span></div>
        <div class="role">Animates keyframes into motion via image-to-video</div>
        <div class="model">seedance-2-0</div>
        <div class="count"><strong id="c-cinema">52</strong> clips</div>
        <div class="detail">720p · 5–10s i2v · ~8s/clip via fal.ai gateway · variable durations for cinematic pacing</div>
        <span class="expand-hint">click ↓</span>
      </div>
      <div class="node" id="n-voice">
        <div class="row-1"><div class="name">Voice</div><span class="status-dot"></span></div>
        <div class="role">Synthesizes per-shot narration, baked into each clip</div>
        <div class="model">deepgram-aura</div>
        <div class="count"><strong id="c-voice">52</strong> tracks</div>
        <div class="detail">Aura asteria-en · 192 kbps mp3 · ~1.5s/line · padded with silence to clip length, muxed before stitch</div>
        <span class="expand-hint">click ↓</span>
      </div>
    </div>
    <div class="connector" id="c-fan-stitch"></div>

    <div class="flow-row">
      <div class="node" id="n-stitch">
        <div class="row-1"><div class="name">Stitcher</div><span class="status-dot"></span></div>
        <div class="role">Title card + crossfades + captions + cinematic grade</div>
        <div class="model">ffmpeg · lut</div>
        <div class="count"><strong id="c-stitch">11</strong> renders</div>
        <div class="detail">1920×1080 h.264 · teal-orange LUT · burned captions · intro+outro cards · audio concat preserves Aura</div>
        <span class="expand-hint">click ↓</span>
      </div>
    </div>
    <div class="connector" id="c-stitch-final"></div>

    <div class="flow-row">
      <div class="node output-node" id="n-final">
        <div class="row-1"><div class="name">Finished video</div><span class="status-dot"></span></div>
        <div class="role">MP4 · 1080p · 16:9 · ready to post</div>
        <div class="model">mp4</div>
        <div class="detail">Downloadable from the call dashboard · share link · re-direct any shot to regen without losing the rest</div>
        <span class="expand-hint">click ↓</span>
      </div>
    </div>
  </section>

  <section>
    <div class="why-grid">
      <div class="why">
        <div class="n">01</div>
        <h3>Conversation → video.</h3>
        <p>Every other tool is prompt-to-video. Conduit is a live director you iterate with. Redirect any shot mid-render, the rest preserves.</p>
      </div>
      <div class="why">
        <div class="n">02</div>
        <h3>Purpose-built models.</h3>
        <p>No single model does planning + image + motion + speech well. We orchestrate BytePlus's Seed family, each picked for its strength.</p>
      </div>
      <div class="why">
        <div class="n">03</div>
        <h3>Partial regeneration.</h3>
        <p>"Redo shot 3 — make the car red." Only that shot re-renders. The others stay. 60% cheaper iteration than start-over pipelines.</p>
      </div>
    </div>
  </section>

  <section style="margin-top: 80px;">
    <div class="section-title">A real shot prompt</div>
    <div class="how-prompt">
      <div class="how-prompt-head">
        <span class="how-tag">director → seedance</span>
        <span class="how-meta">shot 03 of 06 · 7s · "tightening a bolt"</span>
      </div>
      <pre class="how-code">Anamorphic 2.39:1 Kodak Vision3 warm grain, Ridley Scott spirit |
a 30-year-old Latina mechanic, grease-streaked hands, navy coveralls |
tightening a bolt as orange sparks shower past her face, in a fluorescent
industrial garage at midnight, 50mm normal lens, hard backlight + atmosphere
haze, slow push-in, intimate and reverent. No on-screen text, no watermark,
no extra fingers, no warped faces.</pre>
      <div class="how-prompt-foot">
        Style bible · character bible · subject + action + environment ·
        lens · lighting · camera move · negative prompts.
        Every shot follows the same formula — that's how continuity stays locked across a 60-second film.
      </div>
    </div>
  </section>

  <section style="margin-top: 80px;">
    <div class="section-title">Conduit vs the rest</div>
    <div class="compare">
      <div class="compare-row compare-head">
        <div></div>
        <div>Conduit</div>
        <div>Sora · Pika · Runway</div>
      </div>
      <div class="compare-row">
        <div class="compare-l">Interface</div>
        <div class="compare-y">Live phone call</div>
        <div class="compare-n">Text box, no voice</div>
      </div>
      <div class="compare-row">
        <div class="compare-l">Iteration</div>
        <div class="compare-y">Mid-render redirect, partial regen</div>
        <div class="compare-n">Re-prompt → re-render the entire clip</div>
      </div>
      <div class="compare-row">
        <div class="compare-l">Continuity</div>
        <div class="compare-y">Style + character bibles baked in</div>
        <div class="compare-n">Each shot generated independently</div>
      </div>
      <div class="compare-row">
        <div class="compare-l">Voiceover</div>
        <div class="compare-y">Aura per-shot, baked into clip audio</div>
        <div class="compare-n">Add separately in post</div>
      </div>
      <div class="compare-row">
        <div class="compare-l">Length</div>
        <div class="compare-y">Up to ~2 minutes, 12 shots</div>
        <div class="compare-n">5–10s per generation</div>
      </div>
      <div class="compare-row">
        <div class="compare-l">Color grade</div>
        <div class="compare-y">Cinematic LUT applied at finalize</div>
        <div class="compare-n">Raw model output</div>
      </div>
    </div>
  </section>

  <section class="phone-strip">
    <h2>Call it. <em>Direct it.</em></h2>
    <div style="color: var(--text-2); font-size: 15px; max-width: 460px; margin: 0 auto;">
      Six specialized agents on the other end of the line. No signup, no UI to learn. Talk like you're calling a real production crew.
    </div>
    <a class="big-cta" href="tel:+14434648118">📞 +1 (443) 464-8118</a>
  </section>
</div>

<script>
// ---- counters from /api/agents (numbers in node corners + hero stats) ----
async function refreshCounts() {
  try {
    const r = await fetch('/api/agents');
    if (!r.ok) return;
    const d = await r.json();
    const keys = {you:'you', vapi:'vapi', director:'director', storyboard:'storyboard',
                  cinema:'cinematographer', voice:'voice', stitch:'stitcher'};
    for (const [short, full] of Object.entries(keys)) {
      const agent = d.agents.find(a => a.name.toLowerCase().includes(full));
      if (agent) {
        const el = document.getElementById('c-' + short);
        if (el) el.textContent = agent.calls;
      }
    }
    const frames = (d.agents.find(a => a.name === 'Storyboard') || {}).calls || 0;
    const clips  = (d.agents.find(a => a.name === 'Cinematographer') || {}).calls || 0;
    document.getElementById('s-calls').textContent = d.calls_total;
    document.getElementById('s-frames').textContent = frames;
    document.getElementById('s-clips').textContent = clips;
    document.getElementById('s-live').textContent = d.live;
  } catch(e) {}
}
refreshCounts();
setInterval(refreshCounts, 2000);

// ---- always-on phase sweep: a packet of light travels through the pipeline ----
const PHASES = [
  { nodes: ['n-you'],          conn: 'c-you-vapi' },
  { nodes: ['n-vapi'],         conn: 'c-vapi-dir' },
  { nodes: ['n-director'],     conn: 'c-dir-fan' },
  { nodes: ['n-storyboard','n-cinema','n-voice'], conn: 'c-fan-stitch' },
  { nodes: ['n-stitch'],       conn: 'c-stitch-final' },
  { nodes: ['n-final'],        conn: null },
];
let _phase = 0;
function tick() {
  document.querySelectorAll('.node').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('.connector').forEach(c => c.classList.remove('live'));
  const cur = PHASES[_phase];
  cur.nodes.forEach(id => document.getElementById(id)?.classList.add('active'));
  if (cur.conn) document.getElementById(cur.conn)?.classList.add('live');
  _phase = (_phase + 1) % PHASES.length;
}
tick();
setInterval(tick, 1500);

// ---- 3D mouse-tilt parallax on idle nodes ----
const flow = document.getElementById('flow');
let _rafTilt = null;
flow.addEventListener('mousemove', (e) => {
  if (_rafTilt) return;
  _rafTilt = requestAnimationFrame(() => {
    document.querySelectorAll('.node').forEach(node => {
      if (node.classList.contains('active') || node.classList.contains('expanded')) return;
      const r = node.getBoundingClientRect();
      const cx = r.left + r.width/2, cy = r.top + r.height/2;
      const dx = (e.clientX - cx);
      const dy = (e.clientY - cy);
      // only tilt if mouse is reasonably close
      const dist = Math.hypot(dx, dy);
      if (dist > 360) { node.style.transform = ''; return; }
      const intensity = 1 - (dist / 360);
      const rx = Math.max(-7, Math.min(7, -dy / 24)) * intensity;
      const ry = Math.max(-7, Math.min(7,  dx / 24)) * intensity;
      const tz = 8 * intensity;
      node.style.transform = `translateZ(${tz}px) rotateX(${rx}deg) rotateY(${ry}deg)`;
    });
    _rafTilt = null;
  });
});
flow.addEventListener('mouseleave', () => {
  document.querySelectorAll('.node').forEach(n => {
    if (!n.classList.contains('active') && !n.classList.contains('expanded')) {
      n.style.transform = '';
    }
  });
});

// ---- click to expand a node's detail panel ----
document.querySelectorAll('.node').forEach(node => {
  node.addEventListener('click', (e) => {
    e.stopPropagation();
    const wasExpanded = node.classList.contains('expanded');
    document.querySelectorAll('.node.expanded').forEach(n => {
      n.classList.remove('expanded');
      n.style.transform = '';
    });
    if (!wasExpanded) {
      node.classList.add('expanded');
      node.style.transform = 'translateZ(46px) scale(1.015)';
    }
  });
});
document.body.addEventListener('click', () => {
  document.querySelectorAll('.node.expanded').forEach(n => {
    n.classList.remove('expanded');
    n.style.transform = '';
  });
});
</script>
</body></html>"""


# Inject shared base style into all templates
DEMO_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Conduit — live demo</title>
__BASE_STYLE__
<style>
  body { background: var(--bg); overflow: hidden; }
  .demo-stage { position: fixed; inset: 0; overflow: hidden; }
  .demo-stage::before {
    content:''; position:absolute; inset:0;
    background:
      radial-gradient(900px circle at 8% 12%, rgba(103,232,249,0.06), transparent 55%),
      radial-gradient(800px circle at 92% 88%, rgba(168,85,247,0.04), transparent 55%);
    pointer-events: none; z-index: 0;
  }

  /* ── Phase 0 — start curtain ─────────────────────────────────── */
  .curtain {
    position: absolute; inset: 0; z-index: 100;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: var(--bg); transition: opacity .8s cubic-bezier(.16,1,.3,1);
  }
  .curtain.gone { opacity: 0; pointer-events: none; }
  .curtain .glyph {
    width: 18px; height: 18px; border-radius: 50%; background: var(--accent);
    box-shadow: 0 0 36px var(--accent), 0 0 90px rgba(103,232,249,0.55);
    margin-bottom: 40px; animation: breathe 2.4s ease-in-out infinite;
  }
  .curtain h1 {
    font-family: var(--serif); font-weight: 400; font-size: 88px;
    letter-spacing: -0.03em; color: var(--text); margin: 0 0 14px;
  }
  .curtain h1 em { font-style: italic; color: var(--accent); }
  .curtain p {
    font-family: var(--mono); font-size: 13px; color: var(--text-3);
    letter-spacing: 0.32em; text-transform: uppercase; margin: 0 0 36px;
  }
  .curtain .start-btn {
    font-family: var(--mono); font-size: 13px; color: var(--text);
    letter-spacing: 0.28em; text-transform: uppercase;
    padding: 16px 32px; border-radius: 999px;
    background: rgba(103,232,249,0.08); border: 1px solid rgba(103,232,249,0.4);
    cursor: pointer; transition: all .25s ease;
  }
  .curtain .start-btn:hover {
    background: rgba(103,232,249,0.16); border-color: var(--accent);
    transform: translateY(-2px);
    box-shadow: 0 0 30px rgba(103,232,249,0.3);
  }
  .curtain .hint {
    margin-top: 22px; font-family: var(--mono); font-size: 11px;
    color: var(--text-3); letter-spacing: 0.18em;
  }

  /* ── Phase 1 — phone dialing ─────────────────────────────────── */
  .phone-card {
    position: absolute; left: 50%; top: 50%;
    transform: translate(-50%, -50%) scale(0.94);
    width: 580px; height: 700px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 44px; padding: 44px;
    box-shadow: 0 60px 120px -30px rgba(0,0,0,0.7), 0 0 60px -20px rgba(103,232,249,0.18);
    opacity: 0; transition: opacity .6s ease, transform .6s cubic-bezier(.16,1,.3,1);
    z-index: 50;
  }
  .phone-card.in { opacity: 1; transform: translate(-50%, -50%) scale(1); }
  .phone-card.out { opacity: 0; transform: translate(-50%, -50%) scale(0.96) translateY(-30px); }
  .phone-card .top-bar { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 14px; color: var(--text-3); letter-spacing: 0.06em; }
  .phone-card .label { margin-top: 50px; font-family: var(--mono); font-size: 11px; letter-spacing: 0.32em; color: var(--text-3); }
  .phone-card .name { margin-top: 14px; font-family: var(--serif); font-size: 52px; color: var(--text); letter-spacing: -0.01em; }
  .phone-card .num  { margin-top: 10px; font-family: var(--mono); font-size: 20px; color: var(--text-2); }

  .ripples { position: absolute; left: 50%; top: 380px; transform: translateX(-50%); width: 1px; height: 1px; }
  .ripples .ring {
    position: absolute; left: 50%; top: 50%; transform: translate(-50%,-50%);
    border: 2px solid rgba(103,232,249,0.45); border-radius: 50%;
    animation: ripple 2.2s cubic-bezier(.16,1,.3,1) infinite;
  }
  .ripples .ring:nth-child(2) { animation-delay: 0.7s; }
  .ripples .ring:nth-child(3) { animation-delay: 1.4s; }
  @keyframes ripple {
    0%   { width: 80px;  height: 80px;  opacity: 0.7; }
    100% { width: 360px; height: 360px; opacity: 0; }
  }
  .ripples .core {
    position: absolute; left: 50%; top: 50%; transform: translate(-50%,-50%);
    width: 100px; height: 100px; background: var(--accent); border-radius: 50%;
    box-shadow: 0 0 50px rgba(103,232,249,0.55);
  }
  .phone-card .footer-row { position: absolute; bottom: 36px; left: 44px; right: 44px; display: flex; justify-content: space-between; font-family: var(--mono); font-size: 12px; letter-spacing: 0.22em; }
  .phone-card .footer-row .decline { color: var(--text-3); }
  .phone-card .footer-row .answer  { color: var(--success); }
  .phone-card .footer-row .answer.pulse { animation: breathe 1.4s ease-in-out infinite; }

  /* ── Phase 2 — call interface (transcript + agents + shots) ───── */
  .call-ui {
    position: absolute; inset: 0;
    display: grid; grid-template-rows: 64px 1fr; gap: 0;
    opacity: 0; transition: opacity .6s ease;
    z-index: 30; padding: 24px 32px 32px;
  }
  .call-ui.in { opacity: 1; }
  .call-ui .top {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 4px; border-bottom: 1px solid var(--border); padding-bottom: 14px;
  }
  .call-ui .live {
    display: inline-flex; align-items: center; gap: 10px;
    font-family: var(--mono); font-size: 12px; letter-spacing: 0.28em;
    color: var(--success);
  }
  .call-ui .live::before {
    content: ''; width: 8px; height: 8px; border-radius: 50%;
    background: var(--success); box-shadow: 0 0 12px rgba(74,222,128,0.6);
    animation: breathe 1.4s infinite;
  }
  .call-ui .timer { font-family: var(--mono); font-size: 13px; color: var(--text-2); letter-spacing: 0.06em; }
  .call-ui .panels {
    display: grid; grid-template-columns: 1.05fr 1.5fr 1.4fr; gap: 18px;
    margin-top: 18px; min-height: 0;
  }
  .panel-d {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 18px 20px; display: flex; flex-direction: column; min-height: 0;
  }
  .panel-d h3 {
    margin: 0 0 12px; font-family: var(--mono); font-size: 11px;
    letter-spacing: 0.28em; color: var(--text-3); text-transform: uppercase; font-weight: 500;
  }

  /* Transcript */
  .transcript-d { overflow-y: auto; flex: 1; }
  .turn {
    margin-bottom: 14px; padding: 12px 14px; border-radius: 10px;
    background: rgba(255,255,255,0.02); border: 1px solid var(--border);
    opacity: 0; transform: translateY(8px);
    transition: opacity .5s, transform .5s cubic-bezier(.16,1,.3,1);
  }
  .turn.in { opacity: 1; transform: translateY(0); }
  .turn .role {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.28em;
    text-transform: uppercase; margin-bottom: 6px;
  }
  .turn.user .role { color: var(--accent); }
  .turn.dir .role { color: var(--success); }
  .turn .txt { font-family: var(--serif); font-size: 22px; line-height: 1.32; color: var(--text); }

  /* Agents */
  .agent-grid {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; flex: 1;
  }
  .agent-d {
    background: rgba(255,255,255,0.02); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px; position: relative;
    transition: all .35s cubic-bezier(.2,.8,.3,1);
  }
  .agent-d.active {
    border-color: var(--accent);
    background: linear-gradient(180deg, rgba(103,232,249,0.05), rgba(255,255,255,0.02));
    box-shadow: 0 0 30px rgba(103,232,249,0.25), 0 0 0 1px var(--accent);
    transform: translateY(-2px);
  }
  .agent-d .dot {
    position: absolute; top: 14px; right: 14px;
    width: 6px; height: 6px; border-radius: 50%; background: var(--text-3);
  }
  .agent-d.active .dot {
    background: var(--accent); box-shadow: 0 0 10px var(--accent);
    animation: breathe 1.4s infinite;
  }
  .agent-d .lbl { font-family: var(--mono); font-size: 9px; letter-spacing: 0.22em; color: var(--text-3); }
  .agent-d .nm  { font-family: var(--serif); font-size: 19px; color: var(--text); margin-top: 6px; }
  .agent-d .md  { font-family: var(--mono); font-size: 10px; color: var(--text-3); margin-top: 6px; }
  .agent-d.active .md { color: var(--accent); }

  /* Shots */
  .shot-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; flex: 1;
    align-content: start; overflow-y: auto;
  }
  .shot-d {
    aspect-ratio: 16/9; background: #000; border: 1px solid var(--border);
    border-radius: 8px; position: relative; overflow: hidden;
    transition: all .35s ease;
  }
  .shot-d.queued { background: linear-gradient(135deg, #15151a, #08080a); }
  .shot-d.rendering {
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(103,232,249,0.2);
  }
  .shot-d.rendering::after {
    content:''; position:absolute; inset:0;
    background: linear-gradient(90deg, transparent, rgba(103,232,249,0.18), transparent);
    background-size: 200% 100%; animation: skel 1.6s linear infinite;
  }
  .shot-d.done { border-color: var(--success); }
  .shot-d video { width: 100%; height: 100%; object-fit: cover; }
  @keyframes skel { from { background-position: 200% 0; } to { background-position: -200% 0; } }
  .shot-d .pill {
    position: absolute; top: 6px; left: 6px;
    font-family: var(--mono); font-size: 9px; letter-spacing: 0.18em;
    color: var(--text-3); background: rgba(0,0,0,0.5); padding: 3px 6px;
    border-radius: 4px; backdrop-filter: blur(4px);
  }
  .shot-d.rendering .pill { color: var(--accent); }
  .shot-d.done .pill { color: var(--success); }
  .shot-d .idx {
    position: absolute; bottom: 6px; right: 8px;
    font-family: var(--mono); font-size: 10px; color: var(--text-2);
    background: rgba(0,0,0,0.5); padding: 2px 6px; border-radius: 4px;
  }

  /* ── Phase 3 — final delivery ─────────────────────────────────── */
  .final-overlay {
    position: absolute; inset: 0; background: #000; z-index: 60;
    display: flex; align-items: center; justify-content: center;
    opacity: 0; transition: opacity .8s ease; pointer-events: none;
  }
  .final-overlay.in { opacity: 1; pointer-events: auto; }
  .final-overlay video { width: 100%; height: 100%; object-fit: contain; }
  .final-overlay .pill-top {
    position: absolute; top: 36px; left: 36px;
    font-family: var(--mono); font-size: 12px; letter-spacing: 0.28em;
    color: var(--accent); background: rgba(0,0,0,0.55); backdrop-filter: blur(6px);
    border: 1px solid rgba(103,232,249,0.4); padding: 8px 14px; border-radius: 999px;
  }
  .final-overlay .pill-bottom {
    position: absolute; bottom: 36px; right: 36px;
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.32em;
    color: var(--accent); background: rgba(6,6,8,0.65); backdrop-filter: blur(8px);
    padding: 8px 14px; border-radius: 999px; border: 1px solid rgba(103,232,249,0.4);
  }

  /* restart + status */
  .restart-btn {
    position: fixed; bottom: 24px; right: 24px; z-index: 200;
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.22em;
    color: var(--text-2); background: rgba(0,0,0,0.55);
    border: 1px solid var(--border); padding: 8px 14px; border-radius: 999px;
    cursor: pointer; opacity: 0; transition: opacity .35s ease;
  }
  .restart-btn.show { opacity: 1; }
  .restart-btn:hover { color: var(--text); border-color: var(--border-2); }

  .status-text {
    position: fixed; top: 18px; left: 50%; transform: translateX(-50%);
    z-index: 80; font-family: var(--mono); font-size: 11px; letter-spacing: 0.32em;
    color: var(--text-3); text-transform: uppercase;
    background: rgba(6,6,8,0.6); backdrop-filter: blur(6px);
    border: 1px solid var(--border); padding: 7px 14px; border-radius: 999px;
    opacity: 0; transition: opacity .3s ease;
  }
  .status-text.show { opacity: 1; }
</style>
</head>
<body>

<div class="demo-stage">

  <div class="status-text" id="status">DIALING</div>

  <!-- Phase 0: start curtain -->
  <div class="curtain" id="curtain">
    <div class="glyph"></div>
    <h1>Live <em>Conduit</em> demo</h1>
    <p>scripted · 95 seconds · projector-safe</p>
    <button class="start-btn" id="startBtn">Start the demo</button>
    <div class="hint">↵ ENTER · or click to begin</div>
  </div>

  <!-- Phase 1: dialer -->
  <div class="phone-card" id="phone">
    <div class="top-bar"><span>11:38</span><span>5G ●●●</span></div>
    <div class="label">INCOMING CALL</div>
    <div class="name">Conduit Director</div>
    <div class="num">+1 (443) 464-8118</div>
    <div class="ripples"><span class="ring"></span><span class="ring"></span><span class="ring"></span><span class="core"></span></div>
    <div class="footer-row"><span class="decline">DECLINE</span><span class="answer pulse" id="answerLbl">ANSWER ●</span></div>
  </div>

  <!-- Phase 2: call UI -->
  <div class="call-ui" id="callui">
    <div class="top">
      <span class="live">LIVE · CONDUIT</span>
      <span class="timer" id="timer">00:00</span>
    </div>
    <div class="panels">
      <div class="panel-d">
        <h3>Transcript</h3>
        <div class="transcript-d" id="lines"></div>
      </div>
      <div class="panel-d">
        <h3>Agents</h3>
        <div class="agent-grid" id="agents">
          <div class="agent-d" data-id="director"><span class="dot"></span><div class="lbl">DIRECTOR</div><div class="nm">Plans</div><div class="md">claude-sonnet-4.5</div></div>
          <div class="agent-d" data-id="storyboard"><span class="dot"></span><div class="lbl">STORYBOARD</div><div class="nm">Renders</div><div class="md">seedream-5.0</div></div>
          <div class="agent-d" data-id="cinema"><span class="dot"></span><div class="lbl">CINEMATOGRAPHER</div><div class="nm">Animates</div><div class="md">seedance-2.0</div></div>
          <div class="agent-d" data-id="voice"><span class="dot"></span><div class="lbl">VOICE</div><div class="nm">Narrates</div><div class="md">deepgram-aura</div></div>
          <div class="agent-d" data-id="stitch"><span class="dot"></span><div class="lbl">STITCHER</div><div class="nm">Cuts</div><div class="md">ffmpeg + lut</div></div>
          <div class="agent-d" data-id="final"><span class="dot"></span><div class="lbl">FINAL</div><div class="nm">Delivers</div><div class="md">mp4 · 1080p</div></div>
        </div>
      </div>
      <div class="panel-d">
        <h3>Shots</h3>
        <div class="shot-grid" id="shots"></div>
      </div>
    </div>
  </div>

  <!-- Phase 3: final delivery -->
  <div class="final-overlay" id="final">
    <div class="pill-top">● DELIVERED · MP4 · 1080p</div>
    <video id="finalVid" src="/videos/_demo_moon/final_clean.mp4?v=3" preload="auto" playsinline></video>
    <div class="pill-bottom">✶ EDITED BY CONDUIT</div>
  </div>

  <button class="restart-btn" id="restartBtn">↻ RESTART</button>
</div>

<script>
const SHOTS = [
  {intent: "South pole from orbit",        clip: "/videos/_demo_moon/clips/clip_00_va.mp4"},
  {intent: "Permanent shadow crater rim",  clip: "/videos/_demo_moon/clips/clip_01_va.mp4"},
  {intent: "Astronaut walking the surface",clip: "/videos/_demo_moon/clips/clip_02_va.mp4"},
  {intent: "Water-ice in regolith (macro)",clip: "/videos/_demo_moon/clips/clip_03_va.mp4"},
  {intent: "Aerial of Shackleton Crater",  clip: "/videos/_demo_moon/clips/clip_04_va.mp4"},
  {intent: "Future research base, twilight",clip: "/videos/_demo_moon/clips/clip_05_va.mp4"},
];

const audUser = new Audio('/videos/_hero_v2/audio/user.mp3');
const audDir1 = new Audio('/videos/_hero_v2/audio/director.mp3');
const audOrch = new Audio('/videos/_hero_v2/audio/orch.mp3');
audUser.preload = 'auto'; audDir1.preload = 'auto'; audOrch.preload = 'auto';

// Known durations (probed): user 4.13s, director 4.31s, orch 7.71s.
// We hardcode so we don't depend on .duration being available at the moment we need it.
const DUR = { user: 4130, director: 4310, orch: 7710 };

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// Play an audio element and resolve when it finishes (or errors out).
function playAndWait(audio) {
  audio.currentTime = 0;
  return new Promise((res) => {
    const done = () => {
      audio.removeEventListener('ended', done);
      audio.removeEventListener('error', done);
      res();
    };
    audio.addEventListener('ended', done, { once: true });
    audio.addEventListener('error', done, { once: true });
    audio.play().catch(() => done());
  });
}

let _t0 = 0; let _timerHandle = null;
function startTimer() {
  _t0 = Date.now();
  _timerHandle = setInterval(() => {
    const sec = Math.floor((Date.now() - _t0) / 1000);
    const m = String(Math.floor(sec/60)).padStart(2,'0');
    const s = String(sec%60).padStart(2,'0');
    $('timer').textContent = `${m}:${s}`;
  }, 200);
}
function stopTimer(){ if(_timerHandle) clearInterval(_timerHandle); _timerHandle = null; }

function setStatus(s) {
  $('status').textContent = s;
  $('status').classList.add('show');
}

// Append a transcript turn, typewriter-revealed across `targetMs` so the
// reveal finishes roughly when the audio does. Always at least 50ms/word
// so the words are still readable.
function appendLine(role, text, targetMs) {
  const el = document.createElement('div');
  el.className = 'turn ' + (role === 'user' ? 'user' : 'dir');
  el.innerHTML = `<div class="role">${role === 'user' ? 'You' : 'Director'}</div>
                  <div class="txt"></div>`;
  $('lines').appendChild(el);
  const txtEl = el.querySelector('.txt');
  const words = text.split(' ');
  const perWord = Math.max(50, Math.min(260, (targetMs - 200) / words.length));
  return new Promise(async (res) => {
    requestAnimationFrame(() => el.classList.add('in'));
    await sleep(80);
    for (let i = 0; i < words.length; i++) {
      txtEl.textContent = words.slice(0, i + 1).join(' ');
      await sleep(perWord);
    }
    res();
  });
}

function setAgent(id, on) {
  const el = document.querySelector(`.agent-d[data-id="${id}"]`);
  if (!el) return;
  el.classList.toggle('active', on);
}

function addShotCards() {
  const grid = $('shots');
  grid.innerHTML = '';
  SHOTS.forEach((s, i) => {
    const card = document.createElement('div');
    card.className = 'shot-d queued';
    card.id = `shot-${i}`;
    card.innerHTML = `<span class="pill">QUEUED</span><span class="idx">SHOT ${String(i+1).padStart(2,'0')}</span>`;
    grid.appendChild(card);
  });
}
function setShot(i, status, clipUrl) {
  const card = $(`shot-${i}`);
  if (!card) return;
  card.classList.remove('queued','rendering','done');
  card.classList.add(status);
  card.querySelector('.pill').textContent = status.toUpperCase();
  if (status === 'done' && clipUrl) {
    card.insertAdjacentHTML('afterbegin',
      `<video src="${clipUrl}" muted autoplay loop playsinline></video>`);
  }
}

async function runDemo() {
  // ── Curtain → phone ringing ─────────────────────────────────
  $('curtain').classList.add('gone');
  setStatus('DIALING');
  await sleep(450);
  $('phone').classList.add('in');
  await sleep(2200);
  setStatus('CONNECTING');
  $('answerLbl').textContent = 'ANSWERED ✓';
  await sleep(900);

  // Phone → call UI
  $('phone').classList.add('out');
  await sleep(450);
  $('callui').classList.add('in');
  setStatus('LIVE · ON THE CALL');
  startTimer();
  await sleep(450);

  // ── Director greeting (silent, brief — no audio) ────────────
  await appendLine('director', "Hey — Conduit here. What are we making?", 1400);
  await sleep(550);

  // ── User asks (Aura orion plays; reveal paced to match audio) ──
  const userPlay = playAndWait(audUser);
  await appendLine(
    'user',
    "Make me a thirty-second video about the south pole of the moon, with voiceover and facts.",
    DUR.user,
  );
  await userPlay;            // hold here until orion fully ends
  await sleep(450);

  // ── Director responds (Aura asteria plays; same pattern) ────
  const dirPlay = playAndWait(audDir1);
  await appendLine(
    'director',
    "Got it. Six shots, narrated, locked in. Storyboard's coming up.",
    DUR.director,
  );
  await dirPlay;             // hold until asteria fully ends
  await sleep(350);

  // ── Plan reveal — 6 shot cards appear as queued ─────────────
  addShotCards();
  setStatus('PLANNING');
  await sleep(700);

  // ── Orchestration phase: 7.71s of stella whisper, agents +
  //    shot states animate in coordinated parallel work. We start
  //    the audio, run all visual choreography, then await audio end.
  setStatus('RENDERING · 6 AGENTS PARALLEL');
  const orchPlay = playAndWait(audOrch);

  // Agents activate over the first ~3.2s of the whisper
  setAgent('director', true);   await sleep(800);
  setAgent('storyboard', true); await sleep(800);
  setAgent('cinema', true);     await sleep(800);
  setAgent('voice', true);      await sleep(800);

  // Shots flip to rendering quickly (the whisper is talking about
  // parallelism — visually all 6 hit "rendering" within ~1.5s)
  for (let i = 0; i < SHOTS.length; i++) {
    setShot(i, 'rendering');
    await sleep(220);
  }

  // Shots progressively complete with their actual clip thumbnails.
  // Spread across remaining whisper time so completion lines up
  // roughly with the end of the orch audio.
  for (let i = 0; i < SHOTS.length; i++) {
    setShot(i, 'done', SHOTS[i].clip);
    await sleep(330);
  }

  await orchPlay;              // hold here until whisper truly ends
  // belt-and-suspenders: kill ALL speech audio before stitching/reveal
  // so absolutely nothing can leak through over the moon narration
  [audUser, audDir1, audOrch].forEach(a => { a.pause(); a.currentTime = 0; });
  await sleep(800);            // brief beat after the whisper

  // ── Stitcher (longer beat — judges should *see* the work happen) ─
  setAgent('stitch', true);
  setStatus('STITCHING · LUT GRADE · CAPTIONS');
  await sleep(3200);
  setAgent('final', true);
  setStatus('DELIVERED · STAND BY');
  await sleep(1800);           // beat before the reveal

  // ── Final video plays (its own per-shot Aura voice-over) ────
  $('final').classList.add('in');
  await sleep(700);             // longer cross-fade so audio doesn't crash in
  $('finalVid').currentTime = 0;
  $('finalVid').play();

  $('finalVid').onended = () => {
    $('restartBtn').classList.add('show');
    stopTimer();
  };
}

function reset() {
  // reload page — simplest reliable reset
  location.reload();
}

$('startBtn').onclick = runDemo;
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && $('curtain') && !$('curtain').classList.contains('gone')) {
    runDemo();
  }
  if (e.key === 'r' || e.key === 'R') reset();
});
$('restartBtn').onclick = reset;

// preload final video for instant playback
const preload = document.createElement('video');
preload.src = '/videos/_demo_moon/final_clean.mp4';
preload.preload = 'auto';
</script>

</body></html>"""


INDEX_HTML = INDEX_HTML.replace("__BASE_STYLE__", _BASE_STYLE)
DEMO_HTML = DEMO_HTML.replace("__BASE_STYLE__", _BASE_STYLE)
CALL_HTML = CALL_HTML.replace("__BASE_STYLE__", _BASE_STYLE)
HOW_HTML = HOW_HTML.replace("__BASE_STYLE__", _BASE_STYLE)



# ─────────────────────────────────────────────────────────────────────
# User Dashboard — /u/{token}
# ─────────────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return _templates.TemplateResponse(request, "login.html")


@app.post("/api/login")
async def api_login(req: Request):
    """Phone number → redirect to /u/{token}. Creates user if first visit."""
    data = await req.json()
    phone = (data.get("phone") or "").strip()
    if not phone:
        return JSONResponse({"error": "phone required"}, status_code=400)
    import auth as _auth
    normalized = _auth.normalize_phone(phone)
    if len(normalized.replace("+", "").replace(" ", "")) < 10:
        return JSONResponse({"error": "invalid phone number"}, status_code=400)
    # check if user exists; if not, still create (they might be calling for the first time)
    user = db.upsert_user(normalized)
    return {"redirect": f"/u/{user['token']}"}


@app.get("/u/{token}", response_class=HTMLResponse)
def user_dashboard(token: str, request: Request):
    user = db.get_user_by_token(token)
    if not user:
        return HTMLResponse("<html><body style='font-family:monospace;background:#060606;color:#f4f4f5;padding:40px'>"
                            "<h2>Studio not found</h2><p>This link isn't valid. Call "
                            "<a href='tel:+14434648118' style='color:#67e8f9'>+1 (443) 464-8118</a> "
                            "and we'll text you your studio URL.</p></body></html>", status_code=404)
    return _templates.TemplateResponse(request, "user_dashboard.html", {"token": token})


@app.get("/api/u/{token}")
def api_user(token: str):
    user = db.get_user_by_token(token)
    if not user:
        return JSONResponse({"error": "not found"}, status_code=404)
    calls = db.get_calls_for_user(user["phone_e164"])
    return {"user": dict(user), "calls": calls}


@app.get("/api/u/{token}/active")
def api_user_active(token: str):
    user = db.get_user_by_token(token)
    if not user:
        return JSONResponse({"error": "not found"}, status_code=404)
    active = db.get_active_call(user["phone_e164"])
    return {"active": active}


# ─────────────────────────────────────────────────────────────────────
# Post-call editor — reorder, overlay, AI improve prompt
# ─────────────────────────────────────────────────────────────────────

@app.post("/api/reorder/{call_id}")
async def api_reorder(call_id: str, req: Request):
    """Drag-to-reorder shots. Body: {"order": [2, 0, 1, 3]}"""
    data = await req.json()
    new_order = data.get("order", [])
    sess = director.load_session(call_id)
    if not sess:
        return JSONResponse({"error": "session not found"}, status_code=404)
    if len(new_order) != len(sess.shots):
        return JSONResponse({"error": "order length mismatch"}, status_code=400)
    sess.shots = [sess.shots[i] for i in new_order]
    for j, shot in enumerate(sess.shots):
        shot.index = j
    director.dump_session(sess)
    pipeline.emit("shots.reordered", {"call_id": call_id})
    return {"ok": True}


@app.post("/api/overlay/{call_id}")
async def api_overlay(call_id: str, req: Request):
    """Add/update text overlay on a shot. Body: {"shot_index": 0, "text": "Tesla", "position": "lower_third"}"""
    data = await req.json()
    shot_index = data.get("shot_index")
    text = data.get("text", "").strip()
    position = data.get("position", "lower_third")
    sess = director.load_session(call_id)
    if not sess:
        return JSONResponse({"error": "session not found"}, status_code=404)
    if shot_index is None or shot_index < 0 or shot_index >= len(sess.shots):
        return JSONResponse({"error": "bad shot index"}, status_code=400)
    shot = sess.shots[shot_index]
    if not shot.clip_path:
        return JSONResponse({"error": "shot not rendered yet"}, status_code=400)

    import threading
    def _apply_overlay():
        try:
            from pathlib import Path
            src = Path(shot.clip_path)
            out_dir = Path("videos") / call_id / "_overlays"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"overlay_{shot_index:02d}.mp4"
            if text:
                pipeline._make_caption_clip(src, text, out)
            else:
                import shutil
                shutil.copy(src, out)
            shot.clip_path = str(out)
            director.dump_session(sess)
            pipeline.emit("shot.overlay", {
                "call_id": call_id, "shot_id": shot.id,
                "clip_path": str(out), "text": text,
            })
        except Exception as e:
            pipeline.emit("shot.overlay.error", {"call_id": call_id, "error": str(e)})

    threading.Thread(target=_apply_overlay, daemon=True).start()
    return {"ok": True, "message": f"Applying overlay to shot {shot_index + 1}…"}


@app.post("/api/improve-prompt/{call_id}/{shot_index}")
async def api_improve_prompt(call_id: str, shot_index: int, req: Request):
    """Use Claude to improve a shot prompt. Body: {"current_prompt": "..."}"""
    data = await req.json()
    current_prompt = data.get("current_prompt", "").strip()
    sess = director.load_session(call_id)
    if not sess:
        return JSONResponse({"error": "session not found"}, status_code=404)
    if shot_index < 0 or shot_index >= len(sess.shots):
        return JSONResponse({"error": "bad shot index"}, status_code=400)
    shot = sess.shots[shot_index]
    style_bible = sess.brief or "cinematic, premium, dark tone"
    system = "You are a senior commercial film director. Improve the given shot prompt to be more visually specific, cinematic, and striking. Keep the same subject/action but upgrade the lens, lighting, camera move, and mood. Output ONLY the improved prompt text, nothing else."
    user_msg = f"Style bible: {style_bible}\n\nShot intent: {shot.intent}\n\nCurrent prompt:\n{current_prompt or shot.prompt}\n\nImproved prompt:"
    try:
        improved = claude_client.chat(system, user_msg, max_tokens=400).strip()
        return {"ok": True, "improved_prompt": improved}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("DASHBOARD_PORT", "8000")))
