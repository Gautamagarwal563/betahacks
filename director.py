"""The Director — per-call state machine + Claude brain.

Each call is a session. The director holds:
  - transcript (running log of user + director utterances)
  - shots (ordered list of Shot objects with status: planned | rendering | done | dirty)
  - pipeline events (emitted to dashboard via SSE)

Claude receives the full session state + last user utterance → emits a JSON decision:
  { "say": "<what to speak>", "actions": [ {"op": "plan"|"regen"|"finalize"|"noop", ...} ] }

Keeping tool-use simple (JSON response) rather than Anthropic tool-use schema —
saves round-trips, easier to debug, fits inside hackathon time budget.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import claude_client


class ShotStatus(str, Enum):
    PLANNED = "planned"
    KEYFRAME = "keyframe"
    RENDERING = "rendering"
    DONE = "done"
    DIRTY = "dirty"       # user asked to change, needs regen
    FAILED = "failed"


@dataclass
class Shot:
    id: str
    index: int
    intent: str                    # human-language description ("red Tesla drifting at sunset")
    prompt: str                    # Seedream/Seedance visual prompt
    narration: str                 # voiceover line (or "" for no narration)
    duration: float
    status: ShotStatus = ShotStatus.PLANNED
    keyframe_url: Optional[str] = None
    keyframe_path: Optional[str] = None
    clip_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Session:
    call_id: str
    created_at: float = field(default_factory=time.time)
    transcript: list[dict] = field(default_factory=list)   # [{role, text, ts}]
    shots: list[Shot] = field(default_factory=list)
    final_video_path: Optional[str] = None
    brief: Optional[str] = None                            # one-line summary
    title: Optional[str] = None

    def say(self, role: str, text: str) -> None:
        self.transcript.append({"role": role, "text": text, "ts": time.time()})

    def shot_by_id(self, sid: str) -> Optional[Shot]:
        return next((s for s in self.shots if s.id == sid), None)

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "created_at": self.created_at,
            "brief": self.brief,
            "title": self.title,
            "final_video_path": self.final_video_path,
            "transcript": self.transcript,
            "shots": [asdict(s) for s in self.shots],
        }


# ---------- Claude brain ----------

SYSTEM = """\
You are Conduit — a conversational AI film director on a live phone call.

You plan, direct, and iterate on a short video while talking to the user.
You work with a team of specialized AI agents underneath you:
  - Storyboard artist (Seedream 5.0) renders keyframes
  - Cinematographer (Seedance 2.0) animates keyframes into motion
  - Voice (Seed Speech) narrates

At every turn, given the current call state, respond with ONE JSON object:
{
  "say": "<1-2 short sentences you'll say back to the user, conversational, under 25 words>",
  "actions": [ <zero or more action objects> ]
}

Action types (emit at most ONE per turn that changes state, plus 'noop' otherwise):

1. PLAN — user has described what they want; produce the initial shot list.
   Match length to what the user asked for. If they don't specify, default to 30-45s.
     - 15s ad: 3-4 shots × 5s
     - 30s ad: 5-7 shots × 5s
     - 60s ad: 8-10 shots × 5-7s
     - 90s+ ad: 10-12 shots × 7-10s
   Max 12 shots. Max 10s per shot.
   { "op": "plan",
     "title": "<5-7 word title>",
     "brief": "<one-line summary>",
     "shots": [
       { "intent": "<plain language>",
         "prompt": "<cinematic visual prompt for seedream>",
         "narration": "<voiceover line or empty>",
         "duration": <5-10>
       },
       ... 3-12 shots ...
     ]
   }

2. REGEN — user wants to change a specific shot. Shot indices are 1-based in your 'say'
   (shot 1, shot 2…) but you must emit the 0-based shot_index in the action.
   { "op": "regen",
     "shot_index": <int>,
     "new_intent": "...",
     "new_prompt": "<updated seedream prompt>",
     "new_narration": "<or empty to keep>"
   }

3. FINALIZE — user is happy; stitch and deliver.
   { "op": "finalize" }

4. NOOP — you're just talking, no state change yet.
   { "op": "noop" }

RULES:
- Be brief on the phone. This is a real call.
- If PLAN: match shot count to requested length (rules above). Every prompt must
  be renderable (subject + action + environment + mood). No abstract concepts.
- If REGEN: only change ONE shot per turn unless user clearly asked to change multiple.
- NEVER include markdown fences or prose outside the JSON. Response is JSON only.
- Cinematic, specific visual prompts always. Name camera moves, lenses, lighting,
  time of day.
- If user asks for anything illegal, impersonating a real person in a defamatory way,
  sexual content, or real minors — refuse gently, propose an alternative, and return
  {"op":"noop"}.
"""


def _render_state_prompt(session: Session) -> str:
    """Build the user-turn prompt: full state + last utterance."""
    shots_view = [
        {
            "index": s.index,
            "intent": s.intent,
            "status": s.status.value,
            "narration": s.narration,
        }
        for s in session.shots
    ]
    last_user = next(
        (t["text"] for t in reversed(session.transcript) if t["role"] == "user"),
        ""
    )
    # keep only last ~8 turns in transcript (tokens)
    convo = session.transcript[-8:]
    return json.dumps({
        "brief": session.brief,
        "title": session.title,
        "shots": shots_view,
        "last_transcript": convo,
        "last_user_utterance": last_user,
    }, indent=2)


def decide(session: Session) -> dict:
    """Call Claude. Returns {'say', 'actions'}."""
    user_prompt = _render_state_prompt(session)
    raw = claude_client.chat(SYSTEM, user_prompt, max_tokens=1500)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw.strip())


# ---------- state mutations ----------

def apply_plan(session: Session, action: dict) -> list[Shot]:
    session.title = action.get("title") or session.title
    session.brief = action.get("brief") or session.brief
    new_shots: list[Shot] = []
    for i, s in enumerate(action.get("shots", [])):
        shot = Shot(
            id=f"sh_{uuid.uuid4().hex[:8]}",
            index=i,
            intent=s["intent"],
            prompt=s["prompt"],
            narration=s.get("narration", ""),
            duration=float(s.get("duration", 5.0)),
            status=ShotStatus.PLANNED,
        )
        session.shots.append(shot)
        new_shots.append(shot)
    return new_shots


def apply_regen(session: Session, action: dict) -> Optional[Shot]:
    idx = action.get("shot_index")
    if idx is None or idx < 0 or idx >= len(session.shots):
        return None
    shot = session.shots[idx]
    shot.intent = action.get("new_intent", shot.intent)
    shot.prompt = action.get("new_prompt", shot.prompt)
    if action.get("new_narration"):
        shot.narration = action["new_narration"]
    shot.status = ShotStatus.DIRTY
    shot.keyframe_url = None
    shot.keyframe_path = None
    shot.clip_path = None
    shot.error = None
    return shot


# ---------- persistence ----------

def dump_session(session: Session, root: Path = Path("state")) -> Path:
    root.mkdir(exist_ok=True)
    p = root / f"{session.call_id}.json"
    p.write_text(json.dumps(session.to_dict(), indent=2, default=str))
    return p


def load_session(call_id: str, root: Path = Path("state")) -> Optional[Session]:
    p = root / f"{call_id}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    sess = Session(
        call_id=d["call_id"],
        created_at=d["created_at"],
        brief=d.get("brief"),
        title=d.get("title"),
        final_video_path=d.get("final_video_path"),
        transcript=d.get("transcript", []),
    )
    for sd in d.get("shots", []):
        sess.shots.append(Shot(
            id=sd["id"], index=sd["index"], intent=sd["intent"],
            prompt=sd["prompt"], narration=sd.get("narration", ""),
            duration=sd.get("duration", 5.0),
            status=ShotStatus(sd.get("status", "planned")),
            keyframe_url=sd.get("keyframe_url"),
            keyframe_path=sd.get("keyframe_path"),
            clip_path=sd.get("clip_path"),
            error=sd.get("error"),
        ))
    return sess


if __name__ == "__main__":
    # smoke test — simulate a one-shot call without a phone
    sess = Session(call_id=f"test_{int(time.time())}")
    sess.say("user", "Director, make me a Super Bowl ad for Tesla. 30 seconds. Make it feel like a movie trailer.")
    d = decide(sess)
    print("director says:", d["say"])
    for a in d.get("actions", []):
        print("  action:", a.get("op"))
        if a["op"] == "plan":
            shots = apply_plan(sess, a)
            for s in shots:
                print(f"    {s.index}: {s.intent}")
                print(f"       prompt: {s.prompt[:100]}...")
                print(f"       narration: {s.narration}")
    dump_session(sess)
    print(f"\nsession saved to state/{sess.call_id}.json")
