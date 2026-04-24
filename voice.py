"""Vapi webhook server — phone conversation <-> Conduit Director.

Vapi configuration (set via `python vapi_setup.py`):
  - Transcriber: Deepgram (real-time)
  - LLM: Anthropic Claude (Vapi-native provider)
  - System prompt: Director system from director.py
  - Tools: plan_shots, regen_shot, finalize — each hit this webhook
  - Webhook URL: https://<tunnel>/vapi/webhook

Flow:
  User speaks → Vapi STT → Claude (Vapi hosts) decides → tool call → our webhook
  → we run pipeline → return JSON result → Claude speaks back to user

Run: `python voice.py` (listens on :8080)
Expose: `cloudflared tunnel --url http://localhost:8080`
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

import director
import pipeline
from director import Session, Shot, ShotStatus

PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000")

app = FastAPI(title="Conduit Voice")

# keyed by Vapi call_id (string)
sessions: dict[str, Session] = {}


def _get_or_create(call_id: str) -> Session:
    if call_id not in sessions:
        sessions[call_id] = Session(call_id=call_id)
    return sessions[call_id]


# ---------- Vapi tool implementations ----------

def tool_plan_shots(session: Session, args: dict) -> dict:
    """Accept a shot plan, start rendering immediately (fire-and-forget)."""
    title = args.get("title", "Untitled")
    brief = args.get("brief", "")
    raw_shots = args.get("shots", [])
    action = {"title": title, "brief": brief, "shots": raw_shots}
    new_shots = director.apply_plan(session, action)
    pipeline.render_shots(session, new_shots)
    return {
        "ok": True,
        "message": f"Planned {len(new_shots)} shots. Rendering now. "
                   f"Watch at {PUBLIC_URL}/call/{session.call_id}",
        "shot_count": len(new_shots),
        "dashboard_url": f"{PUBLIC_URL}/call/{session.call_id}",
    }


def tool_regen_shot(session: Session, args: dict) -> dict:
    """Re-render one shot with updated intent/prompt."""
    idx = args.get("shot_index")
    if idx is None and args.get("shot_number"):
        idx = int(args["shot_number"]) - 1  # 1-based → 0-based
    action = {
        "shot_index": idx,
        "new_intent": args.get("new_intent", ""),
        "new_prompt": args.get("new_prompt", ""),
        "new_narration": args.get("new_narration", ""),
    }
    shot = director.apply_regen(session, action)
    if not shot:
        return {"ok": False, "error": f"no shot at index {idx}"}
    pipeline.render_shot_async(session, shot)
    return {
        "ok": True,
        "message": f"Re-rendering shot {shot.index + 1}: {shot.intent[:50]}",
        "shot_index": shot.index,
    }


def tool_finalize(session: Session, args: dict) -> dict:
    """Stitch + return final mp4 URL."""
    try:
        final = pipeline.finalize(session)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "message": "Done. Your video is ready.",
        "video_url": f"{PUBLIC_URL}/video/{session.call_id}",
        "path": str(final),
    }


TOOLS = {
    "plan_shots": tool_plan_shots,
    "regen_shot": tool_regen_shot,
    "finalize": tool_finalize,
}


# ---------- Vapi webhook endpoints ----------

@app.post("/vapi/webhook")
async def vapi_webhook(req: Request):
    """Unified Vapi webhook — handles function calls + call lifecycle."""
    body = await req.json()
    msg = body.get("message", {}) or body
    msg_type = msg.get("type") or msg.get("event")

    call = msg.get("call") or {}
    call_id = call.get("id") or msg.get("callId") or "dev-call"
    session = _get_or_create(call_id)

    # Log transcript updates
    if msg_type == "transcript":
        role = msg.get("role", "user")
        text = msg.get("transcript") or msg.get("content", "")
        if text:
            session.say(role, text)
        return {"ok": True}

    # Tool invocations
    if msg_type in ("function-call", "tool-calls", "tool_call"):
        tool_call = msg.get("functionCall") or msg.get("toolCalls", [{}])[0]
        name = tool_call.get("name") or (tool_call.get("function") or {}).get("name")
        raw_args = tool_call.get("parameters") or tool_call.get("arguments") or \
                   (tool_call.get("function") or {}).get("arguments", {})
        if isinstance(raw_args, str):
            import json
            try:
                raw_args = json.loads(raw_args)
            except Exception:
                raw_args = {}
        handler = TOOLS.get(name)
        if not handler:
            return {"result": {"ok": False, "error": f"unknown tool {name}"}}
        result = handler(session, raw_args)
        director.dump_session(session)
        return {"result": result}

    # Call ended — persist
    if msg_type in ("end-of-call-report", "call.ended"):
        director.dump_session(session)
        return {"ok": True}

    # Default
    return {"ok": True, "received": msg_type}


# ---------- Dev helpers ----------

@app.get("/", response_class=HTMLResponse)
def root():
    sess_list = "".join(
        f'<li><a href="/call/{cid}">{cid}</a> — {len(s.shots)} shots — {s.title or "(untitled)"}</li>'
        for cid, s in sessions.items()
    ) or "<li>no sessions yet</li>"
    return f"""<!doctype html><title>Conduit Voice</title>
<body style="font: 14px ui-monospace; background:#0a0a0a; color:#eaeaea; padding:2rem">
<h1>Conduit Voice Webhook</h1>
<p>POST /vapi/webhook — bound for Vapi events</p>
<h3>Active sessions</h3>
<ul>{sess_list}</ul>
</body>"""


@app.get("/call/{call_id}")
def call_detail(call_id: str):
    s = sessions.get(call_id) or director.load_session(call_id)
    if not s:
        return JSONResponse({"error": "not found"}, status_code=404)
    return s.to_dict()


@app.get("/video/{call_id}")
def call_video(call_id: str):
    path = Path("videos") / call_id / "final.mp4"
    if not path.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    from fastapi.responses import FileResponse
    return FileResponse(path, media_type="video/mp4", filename=f"{call_id}.mp4")


@app.post("/dev/simulate")
async def dev_simulate(req: Request):
    """Local test: POST {'text': 'make a tesla ad'} to run the director without Vapi."""
    data = await req.json()
    text = data.get("text", "")
    call_id = data.get("call_id", f"sim_{int(time.time())}")
    session = _get_or_create(call_id)
    session.say("user", text)
    decision = director.decide(session)
    session.say("assistant", decision.get("say", ""))
    results = []
    for action in decision.get("actions", []):
        op = action.get("op")
        if op == "plan":
            r = tool_plan_shots(session, action)
        elif op == "regen":
            r = tool_regen_shot(session, {
                "shot_index": action.get("shot_index"),
                "new_intent": action.get("new_intent", ""),
                "new_prompt": action.get("new_prompt", ""),
                "new_narration": action.get("new_narration", ""),
            })
        elif op == "finalize":
            r = tool_finalize(session, {})
        else:
            r = {"ok": True, "op": op}
        results.append({"op": op, "result": r})
    director.dump_session(session)
    return {"say": decision.get("say"), "actions": results,
            "call_id": call_id, "dashboard_url": f"{PUBLIC_URL}/call/{call_id}"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("WEBHOOK_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
