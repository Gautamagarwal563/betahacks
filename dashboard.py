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


INDEX_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8"><title>Conduit — calls</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font: 14px ui-monospace, Menlo, monospace; background:#0a0a0a; color:#eaeaea }
  .wrap { max-width: 920px; margin: 0 auto; padding: 2rem 1.5rem }
  h1 { margin: 0 0 .25rem; font-size: 1.75rem; letter-spacing: -0.01em }
  .sub { color:#888; margin: 0 0 2rem }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#2ecc71;
         margin-right:6px; vertical-align:middle; animation: pulse 1.4s infinite }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
  .card { display:block; background:#141414; border:1px solid #222; padding:1rem 1.25rem;
          border-radius:8px; margin-bottom:.75rem; color:inherit; text-decoration:none }
  .card:hover { border-color:#444 }
  .card .t { font-weight:500; margin-bottom:2px }
  .card .b { color:#888; font-size:.85rem }
  .card .meta { color:#555; font-size:.75rem; margin-top:4px }
  .empty { padding:3rem 1rem; text-align:center; color:#666 }
</style></head>
<body><div class="wrap">
<h1><span class="dot"></span> Conduit</h1>
<p class="sub">conversational AI video director · calls in progress</p>
<div id="list"><div class="empty">loading…</div></div>
</div>
<script>
async function refresh() {
  const r = await fetch('/api/sessions');
  const d = await r.json();
  const el = document.getElementById('list');
  if (!d.length) { el.innerHTML = '<div class="empty">no calls yet — call the Conduit number to start</div>'; return; }
  el.innerHTML = d.map(s => `
    <a class="card" href="/call/${s.call_id}">
      <div class="t">${s.title || '(untitled)'}</div>
      <div class="b">${s.brief || 'no brief'}</div>
      <div class="meta">${new Date(s.created_at * 1000).toLocaleString()} · ${s.shot_count} shots ${s.final_video_path ? '· ✓ finalized' : ''}</div>
    </a>`).join('');
}
refresh(); setInterval(refresh, 4000);
</script>
</body></html>"""


CALL_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8"><title>Conduit — call</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font: 14px ui-monospace, Menlo, monospace; background:#0a0a0a; color:#eaeaea }
  .wrap { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem }
  h1 { margin:0; font-size:1.5rem }
  .brief { color:#888; margin: .25rem 0 2rem }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem }
  .shot { background:#141414; border:1px solid #222; border-radius:8px; overflow:hidden; position:relative }
  .shot img { width:100%; aspect-ratio:16/9; object-fit:cover; display:block; background:#222 }
  .shot video { width:100%; aspect-ratio:16/9; background:#000; display:block }
  .shot .meta { padding:.75rem 1rem }
  .shot .idx { color:#666; font-size:.75rem }
  .shot .intent { color:#eee; margin: 4px 0 6px; font-size:.9rem }
  .shot .narr { color:#888; font-size:.8rem; font-style:italic }
  .badge { position:absolute; top:8px; right:8px; padding:2px 8px; border-radius:3px; font-size:.65rem;
           background:#1a1a1a; border:1px solid #333 }
  .badge.planned { color:#777 }
  .badge.keyframe { color:#3498db; border-color:#345 }
  .badge.rendering { color:#f39c12; border-color:#543 }
  .badge.done { color:#2ecc71; border-color:#354 }
  .badge.dirty { color:#e67e22; border-color:#642 }
  .badge.failed { color:#e74c3c; border-color:#622 }
  .final { margin-top:2rem; padding:1rem; background:#0f1a0f; border:1px solid #254; border-radius:8px }
  .final video { width:100%; max-height:400px; background:#000; border-radius:4px }
  .final a { color:#6ab7ff }
  .transcript { margin-top:2rem; padding:1rem; background:#141414; border:1px solid #222; border-radius:8px }
  .transcript .t { margin: .25rem 0 }
  .transcript .t.user { color:#6ab7ff }
  .transcript .t.assistant { color:#2ecc71 }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#2ecc71;
         margin-right:6px; vertical-align:middle; animation: pulse 1.4s infinite }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
  a.back { color:#6ab7ff; text-decoration:none; font-size:.85rem }
</style></head>
<body><div class="wrap">
<a class="back" href="/">← all calls</a>
<h1 style="margin-top:1rem"><span class="dot"></span> <span id="title">loading…</span></h1>
<div class="brief" id="brief"></div>
<div class="grid" id="shots"></div>
<div id="final"></div>
<div class="transcript" id="transcript" style="display:none"><div style="color:#888; margin-bottom:.5rem">transcript</div></div>
</div>
<script>
const CALL_ID = "__CALL_ID__";

function renderShots(shots) {
  return shots.map(s => {
    let media = '<img src="" alt="">';
    if (s.clip_path) media = `<video src="/videos/${CALL_ID}/clips/${s.clip_path.split('/').pop()}" controls muted></video>`;
    else if (s.keyframe_path) media = `<img src="/videos/${CALL_ID}/keyframes/${s.keyframe_path.split('/').pop()}">`;
    return `<div class="shot">
      ${media}
      <span class="badge ${s.status}">${s.status}</span>
      <div class="meta">
        <div class="idx">shot ${s.index + 1}</div>
        <div class="intent">${s.intent || ''}</div>
        ${s.narration ? `<div class="narr">"${s.narration}"</div>` : ''}
      </div>
    </div>`;
  }).join('');
}

function renderTranscript(tr) {
  const el = document.getElementById('transcript');
  if (!tr || !tr.length) { el.style.display='none'; return; }
  el.style.display='block';
  el.innerHTML = '<div style="color:#888; margin-bottom:.5rem">transcript</div>' +
    tr.map(t => `<div class="t ${t.role}"><b>${t.role}:</b> ${t.text}</div>`).join('');
}

function renderFinal(s) {
  const el = document.getElementById('final');
  if (!s.final_video_path) { el.innerHTML=''; return; }
  const fname = s.final_video_path.split('/').pop();
  el.innerHTML = `<div class="final">
    <div style="margin-bottom:.5rem">✓ finalized</div>
    <video src="/videos/${CALL_ID}/${fname}" controls></video>
    <div style="margin-top:.5rem"><a href="/videos/${CALL_ID}/${fname}" download>download</a></div>
  </div>`;
}

function apply(snap) {
  document.getElementById('title').textContent = snap.title || '(untitled)';
  document.getElementById('brief').textContent = snap.brief || '';
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
es.addEventListener('shot.status', async () => { const r=await fetch('/api/session/' + CALL_ID); if(r.ok) apply(await r.json()); });
es.addEventListener('shot.keyframe', async () => { const r=await fetch('/api/session/' + CALL_ID); if(r.ok) apply(await r.json()); });
es.addEventListener('shot.clip', async () => { const r=await fetch('/api/session/' + CALL_ID); if(r.ok) apply(await r.json()); });
es.addEventListener('session.finalized', async () => { const r=await fetch('/api/session/' + CALL_ID); if(r.ok) apply(await r.json()); });
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("DASHBOARD_PORT", "8000")))
