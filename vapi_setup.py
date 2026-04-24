"""One-time Vapi assistant setup.

Creates (or updates) the Conduit assistant on Vapi with:
- Anthropic Claude as the LLM
- Deepgram as the transcriber
- 3 server-side tools (plan_shots, regen_shot, finalize)
- Our webhook URL

Usage:
  python vapi_setup.py create            # first run
  python vapi_setup.py update <id>       # subsequent updates
  python vapi_setup.py assign <id>       # attach to the phone number
"""

from __future__ import annotations

import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

import httpx

VAPI_API_KEY = os.getenv("VAPI_API_KEY", "")
VAPI_PHONE_NUMBER_ID = os.getenv("VAPI_PHONE_NUMBER_ID", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://example.ngrok.io")  # set to tunnel URL

BASE = "https://api.vapi.ai"


SYSTEM_PROMPT = """\
You are Conduit — a conversational AI film director on a live phone call.

You work with a team of specialized AI agents underneath you (all hidden — user sees one director):
  - Storyboard artist (Seedream 5.0) renders keyframes
  - Cinematographer (Seedance 2.0) animates keyframes
  - Voice (Seed Speech) narrates

You can call three server-side functions to actually produce the video:

1. plan_shots — call ONCE after the user describes what they want. Produce a 5-7 shot plan.
   For each shot: a cinematic visual prompt (subject + action + environment + lighting +
   camera language), optional narration line, and duration (5 seconds default).

2. regen_shot — call when the user wants to change a specific shot they already saw.
   Shot numbers in speech are 1-based; pass 0-based shot_index to the function.
   Only ONE shot per call unless user clearly changed multiple.

3. finalize — call when user says they're happy. Stitches and returns download URL.

RULES:
- Be brief. This is a phone call. 1-2 sentences per turn unless listing shots.
- Cinematic visual prompts always — camera move, lens, lighting, time of day.
- After plan_shots returns, tell the user "watch the shots come in at [dashboard_url]"
  (the function returns a dashboard_url you must speak).
- If user asks for illegal content, real-person deepfakes, NSFW, or real minors: refuse
  politely and propose an alternative.
- You're the director. Speak like one: confident, decisive, visual-first.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plan_shots",
            "description": "Create the initial shot list from the user's brief. Triggers parallel rendering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "5-7 word title"},
                    "brief": {"type": "string", "description": "One-line summary"},
                    "shots": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "intent": {"type": "string", "description": "Plain-language intent"},
                                "prompt": {"type": "string", "description": "Cinematic visual prompt"},
                                "narration": {"type": "string", "description": "Voiceover line, or empty"},
                                "duration": {"type": "number", "description": "Seconds (default 5)"}
                            },
                            "required": ["intent", "prompt"]
                        },
                        "minItems": 3,
                        "maxItems": 10
                    }
                },
                "required": ["title", "brief", "shots"]
            }
        },
        "server": {"url": f"{PUBLIC_URL}/vapi/webhook"}
    },
    {
        "type": "function",
        "function": {
            "name": "regen_shot",
            "description": "Re-render one shot with updated intent and prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shot_index": {"type": "integer", "description": "0-based shot index"},
                    "new_intent": {"type": "string"},
                    "new_prompt": {"type": "string"},
                    "new_narration": {"type": "string"}
                },
                "required": ["shot_index", "new_prompt"]
            }
        },
        "server": {"url": f"{PUBLIC_URL}/vapi/webhook"}
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": "Stitch final video and return download URL.",
            "parameters": {"type": "object", "properties": {}}
        },
        "server": {"url": f"{PUBLIC_URL}/vapi/webhook"}
    }
]


ASSISTANT_BODY = {
    "name": "Conduit Director",
    "firstMessage": "Hey — Conduit here. I'm your AI director. What are we making?",
    "model": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "systemPrompt": SYSTEM_PROMPT,
        "tools": TOOLS,
        "temperature": 0.6,
        "maxTokens": 800
    },
    "voice": {
        "provider": "11labs",
        "voiceId": "rachel"   # calm-pro default; override in dashboard if needed
    },
    "transcriber": {
        "provider": "deepgram",
        "model": "nova-3",
        "language": "en"
    },
    "endCallFunctionEnabled": True,
    "recordingEnabled": True,
    "server": {"url": f"{PUBLIC_URL}/vapi/webhook"}
}


def _headers() -> dict:
    if not VAPI_API_KEY:
        sys.exit("VAPI_API_KEY not set in .env")
    return {"Authorization": f"Bearer {VAPI_API_KEY}", "Content-Type": "application/json"}


def create() -> str:
    r = httpx.post(f"{BASE}/assistant", headers=_headers(), json=ASSISTANT_BODY, timeout=30)
    r.raise_for_status()
    aid = r.json()["id"]
    print(f"created assistant {aid}")
    print(f"next: python vapi_setup.py assign {aid}")
    return aid


def update(assistant_id: str) -> None:
    r = httpx.patch(f"{BASE}/assistant/{assistant_id}", headers=_headers(),
                    json=ASSISTANT_BODY, timeout=30)
    r.raise_for_status()
    print(f"updated assistant {assistant_id}")


def assign(assistant_id: str) -> None:
    """Attach the assistant to the configured phone number so calls route to it."""
    if not VAPI_PHONE_NUMBER_ID:
        sys.exit("VAPI_PHONE_NUMBER_ID not set")
    r = httpx.patch(
        f"{BASE}/phone-number/{VAPI_PHONE_NUMBER_ID}",
        headers=_headers(),
        json={"assistantId": assistant_id},
        timeout=30,
    )
    r.raise_for_status()
    info = r.json()
    num = info.get("number") or info.get("twilioPhoneNumber") or "(number)"
    print(f"assigned — calls to {num} now hit assistant {assistant_id}")


def list_assistants() -> None:
    r = httpx.get(f"{BASE}/assistant", headers=_headers(), timeout=30)
    r.raise_for_status()
    for a in r.json():
        print(f"  {a['id']}  name={a.get('name')}  createdAt={a.get('createdAt')}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "create":
        create()
    elif cmd == "update":
        update(sys.argv[2])
    elif cmd == "assign":
        assign(sys.argv[2])
    elif cmd == "list":
        list_assistants()
    else:
        print("usage: python vapi_setup.py [create | update <id> | assign <id> | list]")
