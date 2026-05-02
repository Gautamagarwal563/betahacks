"""Medical director — takes a doctor's input, produces a patient education video plan.

Input: procedure, patient context, language, tone
Output: JSON shot plan with patient-friendly narration + Seedream/Seedance prompts
"""

from __future__ import annotations
import json
import claude_client

STYLE_BIBLE = (
    "Clean medical illustration, friendly Pixar-style 3D render, "
    "soft blue-white-white palette, warm professional lighting, "
    "approachable and reassuring, no blood, no scary imagery, "
    "high detail anatomical accuracy"
)

SYSTEM = """\
You are a medical education video director. You create short, clear, reassuring
patient education videos. Your goal: a patient who just received a diagnosis or
procedure recommendation should feel informed, calm, and empowered after watching.

RULES:
- Plain language only. Maximum 8th grade reading level. Zero medical jargon unless
  immediately explained in simple terms.
- Warm, human, reassuring tone. Never clinical or cold.
- Each narration line must fit the shot duration (2.5 words/second):
  8s = max 20 words · 10s = max 25 words · 12s = max 30 words
- Always structure: Hook → Problem → Solution → Steps → Recovery → Next step
- Generate 5–7 shots. Total video 60–90 seconds.
- Style bible prepended to EVERY visual prompt verbatim.
- If a language other than English is specified, write ALL narration in that language.
  Keep shot prompts and intent in English.

VISUAL PROMPT FORMULA (every shot):
<STYLE BIBLE> | <subject + action> in <environment>, <lighting>, <camera move>, reassuring mood.
Camera options: slow push-in · slow pull-out · slow pan · gentle dolly · locked-off
Lighting options: soft warm key light · soft north-window · blue-white clinical soft box

OUTPUT: pure JSON only — no markdown, no prose outside JSON:
{
  "title": "<procedure name, plain language, ≤6 words>",
  "summary": "<one sentence for doctor to read before sending>",
  "language": "<language used for narration>",
  "shots": [
    {
      "index": 0,
      "intent": "<plain English description of this shot>",
      "prompt": "<full visual prompt with style bible prepended>",
      "narration": "<patient-friendly narration, word-count within duration budget>",
      "duration": <8|10|12>
    }
  ]
}
"""


def generate_plan(
    procedure: str,
    patient_context: str,
    language: str = "English",
    tone: str = "reassuring",
) -> dict:
    """Call Claude to generate a full shot plan for this patient education video."""
    user_msg = json.dumps({
        "procedure": procedure,
        "patient_context": patient_context,
        "language": language,
        "tone": tone,
        "style_bible": STYLE_BIBLE,
    }, indent=2)

    return claude_client.chat_json(SYSTEM, user_msg, max_tokens=3000)
