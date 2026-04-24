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

app = FastAPI(title="Conduit Dashboard")

if Path("videos").exists():
    app.mount("/videos", StaticFiles(directory="videos"), name="videos")


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
</style>
</head>
<body>
<nav class="topnav"><div class="inner">
  <a href="/" class="brand" style="text-decoration:none"><span class="glyph"></span>CONDUIT</a>
  <div style="display:flex; gap: 12px; align-items: center;">
    <a href="/how" class="nav-chip">How it works</a>
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

  <section class="section">
    <h2>Stats</h2>
    <div class="stat-strip" id="stat-strip">
      <div class="stat"><div class="n" id="stat-calls">—</div><div class="l">Calls placed</div></div>
      <div class="stat"><div class="n" id="stat-shots">—</div><div class="l">Shots rendered</div></div>
      <div class="stat"><div class="n" id="stat-finals">—</div><div class="l">Videos finalized</div></div>
      <div class="stat"><div class="n" id="stat-live">—</div><div class="l">Live right now</div></div>
    </div>
  </section>

  <section class="section">
    <h2>Calls</h2>
    <div id="list"></div>
  </section>
</div>

<script>
function thumbFor(s) {
  // Find first shot with a keyframe or clip
  return `<div class="empty-thumb">rendering</div>`;
}

async function refresh() {
  const r = await fetch('/api/sessions');
  const d = await r.json();
  const live = d.filter(s => !s.final_video_path).length;
  const totalShots = d.reduce((a,s) => a + (s.shot_count||0), 0);
  const finals = d.filter(s => s.final_video_path).length;
  document.getElementById('stat-calls').textContent = d.length;
  document.getElementById('stat-shots').textContent = totalShots;
  document.getElementById('stat-finals').textContent = finals;
  document.getElementById('stat-live').textContent = live;

  const el = document.getElementById('list');
  if (!d.length) {
    el.innerHTML = `<div class="empty-state">
      <div class="big">No calls yet.</div>
      <div class="sub">Dial +1 (443) 464-8118 to wake the Director.</div>
    </div>`;
    return;
  }
  // Load thumbs async per session
  el.innerHTML = '<div class="grid-calls">' + d.map(s => {
    const cb = Date.now();
    const when = new Date(s.created_at * 1000);
    const ago = timeAgo(when);
    const thumbUrl = s.final_video_path
      ? `/videos/${s.call_id}/${s.final_video_path.split('/').pop()}?t=${cb}`
      : null;
    return `
    <a class="call-card" href="/call/${s.call_id}">
      <div class="thumb" id="thumb-${s.call_id}">
        ${thumbUrl
          ? `<video src="${thumbUrl}" muted preload="metadata"></video><span class="done-overlay">Done</span>`
          : `<div class="empty-thumb" id="ept-${s.call_id}">queued</div><span class="live-overlay">Live</span>`}
      </div>
      <div class="meta">
        <div class="title">${s.title || 'Untitled'}</div>
        <div class="brief">${s.brief || 'No brief yet — Director still planning.'}</div>
        <div class="footer">
          <span>${s.shot_count} shots</span>
          <span>${ago}</span>
        </div>
      </div>
    </a>`;
  }).join('') + '</div>';

  // Lazy-load keyframe thumbs for calls without final yet
  d.filter(s => !s.final_video_path).forEach(async s => {
    try {
      const resp = await fetch('/api/session/' + s.call_id);
      if (!resp.ok) return;
      const det = await resp.json();
      const first = (det.shots || []).find(x => x.keyframe_path || x.clip_path);
      if (!first) return;
      const node = document.getElementById('ept-' + s.call_id);
      if (!node) return;
      const cb = Date.now();
      if (first.clip_path) {
        node.outerHTML = `<video src="/videos/${s.call_id}/clips/${first.clip_path.split('/').pop()}?t=${cb}" muted preload="metadata"></video>`;
      } else if (first.keyframe_path) {
        node.outerHTML = `<img src="/videos/${s.call_id}/keyframes/${first.keyframe_path.split('/').pop()}?t=${cb}">`;
      }
    } catch(e){}
  });
}

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
<meta charset="utf-8"><title>Conduit — call</title>
__BASE_STYLE__
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
</style>
</head>
<body>
<nav class="topnav"><div class="inner">
  <a href="/" class="brand" style="text-decoration:none"><span class="glyph"></span>CONDUIT</a>
  <div style="display:flex; gap: 12px; align-items: center;">
    <a href="/how" class="nav-chip">How it works</a>
    <a class="phone-cta" href="tel:+14434648118"><span class="pulse"></span>+1 (443) 464-8118 · live</a>
  </div>
</div></nav>

<div class="wrap">
  <div class="back-row"><a href="/">← all calls</a></div>

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
        <button class="action-btn primary" id="finalize-btn" onclick="finalize()">Finalize</button>
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
    if (s.clip_path) media = `<video src="/videos/${CALL_ID}/clips/${s.clip_path.split('/').pop()}?t=${cb}" controls muted loop preload="metadata"></video>`;
    else if (s.keyframe_path) media = `<img src="/videos/${CALL_ID}/keyframes/${s.keyframe_path.split('/').pop()}?t=${cb}">`;
    const canRegen = ['done','failed','dirty'].includes(s.status);
    const escPrompt = (s.prompt || '').replace(/"/g, '&quot;').replace(/</g,'&lt;');
    const escNarr = (s.narration || '').replace(/"/g, '&quot;').replace(/</g,'&lt;');
    return `<div class="shot" data-shot-id="${s.index}">
      <div class="media">
        ${media}
        <span class="pill ${s.status}"><span class="dot"></span>${s.status}</span>
        <span class="idx">Shot ${String(s.index + 1).padStart(2,'0')}</span>
        ${canRegen ? `<div class="shot-actions">
          <button class="shot-btn" onclick="openRegen(${s.index}, this)">↻ Redo</button>
        </div>` : ''}
      </div>
      <div class="info">
        <div class="intent">${s.intent || ''}</div>
        ${s.narration ? `<div class="narr">${s.narration}</div>` : ''}
      </div>
      <div class="regen-form" id="regen-${s.index}" style="display:none">
        <div style="font-family: var(--mono); font-size: 10px; color: var(--text-3); letter-spacing: 0.14em; text-transform: uppercase;">Edit shot ${s.index + 1}</div>
        <textarea id="rg-prompt-${s.index}" placeholder="Visual prompt" rows="5">${escPrompt}</textarea>
        <textarea id="rg-narr-${s.index}" placeholder="Narration (optional)" rows="2">${escNarr}</textarea>
        <div class="row">
          <button class="btn" onclick="closeRegen(${s.index})">Cancel</button>
          <button class="btn primary" onclick="submitRegen(${s.index})">Re-render</button>
        </div>
      </div>
    </div>`;
  }).join('');
}

function openRegen(idx, btn) {
  document.getElementById('regen-' + idx).style.display = 'flex';
}
function closeRegen(idx) {
  document.getElementById('regen-' + idx).style.display = 'none';
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

async function finalize() {
  const btn = document.getElementById('finalize-btn');
  btn.disabled = true; btn.textContent = 'Stitching…';
  try {
    const r = await fetch(`/api/finalize/${CALL_ID}`, { method: 'POST' });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'finalize failed');
    const rs = await fetch('/api/session/' + CALL_ID);
    if (rs.ok) apply(await rs.json());
  } catch(e) { alert(e.message); }
  finally { btn.disabled = false; btn.textContent = 'Finalize'; }
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

  /* the flow diagram */
  .flow {
    position: relative; padding: 40px 0 60px;
  }
  .flow-row {
    display: flex; justify-content: center; gap: 28px; flex-wrap: wrap;
    margin-bottom: 48px; position: relative;
  }
  .flow-row.fan {
    justify-content: space-between; gap: 16px;
  }
  .connector {
    width: 1px; height: 40px; background: linear-gradient(180deg, var(--accent) 0%, transparent 100%);
    margin: -24px auto 0; opacity: 0.45;
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
    min-width: 200px; max-width: 240px;
    position: relative;
    transition: all .3s ease;
  }
  .node:hover { border-color: var(--border-2); transform: translateY(-2px); }
  .node.active {
    border-color: var(--accent);
    box-shadow: 0 0 32px var(--accent-glow);
  }
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
  .node .role {
    font-size: 12px; color: var(--text-2); margin-bottom: 10px;
  }
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
</style>
</head>
<body>
<nav class="topnav"><div class="inner">
  <a href="/" class="brand" style="text-decoration:none"><span class="glyph"></span>CONDUIT</a>
  <div style="display:flex; gap: 12px; align-items: center;">
    <a href="/how" class="nav-chip" style="color: var(--text); border-color: var(--accent);">How it works</a>
    <a href="/" class="nav-chip">Calls</a>
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
      <div class="cell"><div class="n" id="s-calls">—</div><div class="l">Calls directed</div></div>
      <div class="cell"><div class="n" id="s-frames">—</div><div class="l">Keyframes rendered</div></div>
      <div class="cell"><div class="n" id="s-clips">—</div><div class="l">Clips generated</div></div>
      <div class="cell"><div class="n" id="s-live">—</div><div class="l">Live now</div></div>
    </div>
  </section>

  <section class="flow" id="flow">
    <div class="section-title">The orchestration</div>

    <div class="flow-row">
      <div class="node" id="n-you">
        <div class="row-1"><div class="name">You</div><span class="status-dot"></span></div>
        <div class="role">The director — on a phone call</div>
        <div class="model">phone</div>
        <div class="count"><strong id="c-you">0</strong> calls</div>
      </div>
    </div>
    <div class="connector"></div>

    <div class="flow-row">
      <div class="node" id="n-vapi">
        <div class="row-1"><div class="name">Vapi + Deepgram</div><span class="status-dot"></span></div>
        <div class="role">Voice I/O — streams your words, speaks the reply</div>
        <div class="model">nova-3 · aura</div>
        <div class="count"><strong id="c-vapi">0</strong> turns</div>
      </div>
    </div>
    <div class="connector"></div>

    <div class="flow-row">
      <div class="node" id="n-director">
        <div class="row-1"><div class="name">Director</div><span class="status-dot"></span></div>
        <div class="role">Plans shots, re-cuts on your redirect, calls the crew</div>
        <div class="model">claude-sonnet-4-6</div>
        <div class="count"><strong id="c-director">0</strong> decisions</div>
      </div>
    </div>
    <div class="connector"></div>

    <div class="flow-row fan">
      <div class="node" id="n-storyboard">
        <div class="row-1"><div class="name">Storyboard</div><span class="status-dot"></span></div>
        <div class="role">Keyframe per shot — locks visual style</div>
        <div class="model">seedream-5-0</div>
        <div class="count"><strong id="c-storyboard">0</strong> frames</div>
      </div>
      <div class="node" id="n-cinema">
        <div class="row-1"><div class="name">Cinematographer</div><span class="status-dot"></span></div>
        <div class="role">Animates keyframes into motion via image-to-video</div>
        <div class="model">seedance-2-0</div>
        <div class="count"><strong id="c-cinema">0</strong> clips</div>
      </div>
      <div class="node" id="n-voice">
        <div class="row-1"><div class="name">Voice</div><span class="status-dot"></span></div>
        <div class="role">Synthesizes the narration bed</div>
        <div class="model">seed-speech</div>
        <div class="count"><strong id="c-voice">0</strong> tracks</div>
      </div>
    </div>
    <div class="connector"></div>

    <div class="flow-row">
      <div class="node" id="n-stitch">
        <div class="row-1"><div class="name">Stitcher</div><span class="status-dot"></span></div>
        <div class="role">Title card + crossfades + captions + outro</div>
        <div class="model">ffmpeg · remotion</div>
        <div class="count"><strong id="c-stitch">0</strong> renders</div>
      </div>
    </div>
    <div class="connector"></div>

    <div class="flow-row">
      <div class="node output-node">
        <div class="row-1"><div class="name">Finished video</div><span class="status-dot"></span></div>
        <div class="role">MP4 · 1080p · 16:9 · ready to post</div>
        <div class="model">mp4</div>
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

  <section class="phone-strip">
    <h2>Call it. <em>Direct it.</em></h2>
    <div style="color: var(--text-2); font-size: 15px; max-width: 460px; margin: 0 auto;">
      Seven specialized agents on the other end of the line. No signup, no UI to learn. Talk like you're calling a real production crew.
    </div>
    <a class="big-cta" href="tel:+14434648118">📞 +1 (443) 464-8118</a>
  </section>
</div>

<script>
async function refresh() {
  try {
    const r = await fetch('/api/agents');
    const d = await r.json();
    const map = Object.fromEntries(d.agents.map(a => [a.name.toLowerCase().split(' ')[0], a]));
    const keys = {you:'you', vapi:'vapi', director:'director', storyboard:'storyboard',
                  cinema:'cinematographer', voice:'voice', stitch:'stitcher'};
    for (const [short, full] of Object.entries(keys)) {
      const agent = d.agents.find(a => a.name.toLowerCase().includes(full));
      if (agent) {
        const el = document.getElementById('c-' + short);
        if (el) el.textContent = agent.calls;
      }
    }
    // hero stats
    const frames = (d.agents.find(a => a.name === 'Storyboard') || {}).calls || 0;
    const clips = (d.agents.find(a => a.name === 'Cinematographer') || {}).calls || 0;
    document.getElementById('s-calls').textContent = d.calls_total;
    document.getElementById('s-frames').textContent = frames;
    document.getElementById('s-clips').textContent = clips;
    document.getElementById('s-live').textContent = d.live;

    // Live animation — activate nodes based on whether there's an ongoing call
    const liveSessions = d.live;
    const nodes = ['n-vapi','n-director','n-storyboard','n-cinema','n-voice','n-stitch'];
    if (liveSessions > 0) {
      // Sequentially pulse each node as if a call is flowing through
      const now = (Date.now() / 1200) % nodes.length;
      nodes.forEach((nid, i) => {
        const n = document.getElementById(nid);
        if (!n) return;
        if (Math.floor(now) === i) n.classList.add('active');
        else n.classList.remove('active');
      });
    } else {
      nodes.forEach(nid => document.getElementById(nid)?.classList.remove('active'));
    }
  } catch(e) { console.error(e); }
}
refresh();
setInterval(refresh, 1200);
</script>
</body></html>"""


# Inject shared base style into all templates
INDEX_HTML = INDEX_HTML.replace("__BASE_STYLE__", _BASE_STYLE)
CALL_HTML = CALL_HTML.replace("__BASE_STYLE__", _BASE_STYLE)
HOW_HTML = HOW_HTML.replace("__BASE_STYLE__", _BASE_STYLE)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("DASHBOARD_PORT", "8000")))
