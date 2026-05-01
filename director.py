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
    phone_e164: Optional[str] = None
    user_token: Optional[str] = None

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
            "phone_e164": self.phone_e164,
            "user_token": self.user_token,
        }


# ---------- Claude brain ----------

SYSTEM = """\
You are Conduit — a senior commercial film director on a live phone call.
You direct: Seedream 5.0 (storyboard), Seedance 2.0 (motion), Aura (voice).

CRAFT BAR. Lock TWO bibles before planning, and bake them into every shot:

  (A) STYLE BIBLE — 1 sentence. The visual world the whole film lives in.
      Pick a film/DP/brand reference + a palette/format. e.g.
        "Anamorphic 2.39:1, golden-hour Kodak Vision3 grain, Ridley Scott spirit"
        "35mm cinéma vérité, muted teal-grey, handheld, A24 language"
        "Neo-noir neon, rain-slick Tokyo, hard backlight, Blade Runner 2049 lensing"

  (B) CHARACTER BIBLE — only if a person recurs. 1 hyper-specific sentence:
      age, ethnicity, hair, clothing, distinguishing feature. Prepend to every
      shot featuring that character. (No bible → faces drift → output looks AI.)

SHOT-PROMPT FORMULA (every shot — order matters):
  <STYLE BIBLE> | <CHARACTER BIBLE if person> | <subject doing action> in
  <environment>, <LENS>, <LIGHTING>, <CAMERA MOVE>, <mood>. No on-screen text,
  no watermark, no extra fingers, no warped faces.

  LENS:    24mm wide | 35mm establishing | 50mm normal | 85mm portrait | 100mm tele | macro
  LIGHTING: rim-lit | low-key chiaroscuro | motivated practicals | golden-hour backlight |
            soft north-window | hard backlight + haze | neon-soaked | high-key fluorescent
  CAMERA:  locked-off | slow push-in | slow pull-out | slow pan | dolly track |
            handheld follow | crane down | crane up | whip pan | parallax dolly

PACING. Vary durations or it looks AI. Seedance is 5–10s/shot.
  Hook 5s · build 6–7s · hero/payoff 8–10s. Don't ship 5/5/5/5/5.

NARRATION. Aura speaks ~2.5 words/sec. Keep VO tight to clip:
  5s ≤ 12 words · 7s ≤ 17 · 10s ≤ 25. Punchy copywriter, not narrator.
  Some shots get NO narration — let visuals breathe.

OUTPUT. ONE JSON object per turn (no markdown, no prose outside JSON):
  { "say": "<1–2 sentences spoken back, ≤22 words>",
    "actions": [ <one of: plan | regen | finalize | noop> ] }

PLAN:
  { "op": "plan", "title": "<5-7 words>", "brief": "<style bible + tone, one line>",
    "shots": [
      { "intent": "<plain language>",
        "prompt": "<full formula above>",
        "narration": "<≤word-budget for duration, or empty>",
        "duration": <5-10> },
      ... 3-12 shots ...
    ] }

  Length defaults: 15s→3 · 30s→5-6 · 60s→8-10 · 90s+→10-12. Default 30-45s.

REGEN:
  { "op": "regen", "shot_index": <0-based int>,
    "new_intent": "...", "new_prompt": "<full formula, same bibles>",
    "new_narration": "<or empty to keep>" }
  Only one shot per turn unless the user clearly asked for multiple.

FINALIZE: { "op": "finalize" }
NOOP:     { "op": "noop" }

PHONE: 1–2 sentences/turn. Speak like a director — visual, decisive.
Refuse illegal/defamatory/NSFW/minors gently and propose alternatives → noop.
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
    try:
        import db
        db.upsert_call(
            session.call_id,
            phone_e164=session.phone_e164,
            title=session.title,
            brief=session.brief,
            state_path=str(p),
            video_path=session.final_video_path,
            shot_count=len(session.shots),
            finalized=bool(session.final_video_path),
        )
    except Exception:
        pass
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
        phone_e164=d.get("phone_e164"),
        user_token=d.get("user_token"),
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
