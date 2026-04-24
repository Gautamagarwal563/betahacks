# Conduit — the first AI you can *direct* on the phone

> Every other AI video tool is prompt → video.
> **Conduit is conversation → video.**

You call Conduit. You talk. A team of AI agents — director, cinematographer, storyboard artist, VFX, voice — plans, generates, and stitches a finished video while you're still on the call. Interrupt any time: "No, make her older. Pan instead of zoom. Re-shoot shot 3 in a kitchen." Only the shots you changed re-render. The rest are preserved.

Built for **Beta Hacks: Seed Agents Challenge** · Track 5 (Most Creative / Multimodal).

## The demo

You call the Conduit number. You say:

> *"Director, make me a Super Bowl ad for Tesla."*

90 seconds later you're watching the ad render, shot by shot, on a dashboard URL. You redirect mid-call ("make the car red, not silver"), watch that single shot regenerate, and hang up with an mp4 in your hand.

## Why this isn't a wrapper

| Layer | Role | Seed model |
|---|---|---|
| Director | Conversation + shot planning + state | Claude (orchestration) |
| Script doctor | Hook + beat + voiceover pass | **Seed 2.0** |
| Storyboard artist | Keyframe per shot | **Seedream 5.0** |
| Cinematographer | I2V with camera controls | **Seedance 2.0** |
| Voice | Narration + character dialogue | **Seed Speech** |
| Presenter | Talking-head sign-off, if requested | **OmniHuman** |
| Critic | VLM grades each shot, triggers regen on low score | **Seed 2.0 VLM** |

Every video is produced by a closed agentic loop: plan → render → critique → regen where needed → finalize. The user drops in at any point and redirects.

## Stack

- **Voice**: Vapi (phone call interface)
- **STT**: Deepgram (real-time)
- **Orchestration LLM**: Anthropic Claude (hackathon rules allow)
- **All video/image/voice generation**: BytePlus Seed Agent Family
- **Stitching**: ffmpeg
- **Live UI**: FastAPI + Server-Sent Events
- **Hosting**: Vercel (dashboard) + cloudflared tunnel (dev webhooks)

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # fill in keys
python voice.py          # starts Vapi webhook server on :8080
python dashboard.py      # live UI on :8000
cloudflared tunnel --url http://localhost:8080   # public URL for Vapi
```

## Structure

```
conduit/
├── byteplus.py      # Seedance 2.0, Seedream 5.0, Seed 2.0 client
├── claude_client.py # Anthropic wrapper (director brain)
├── director.py      # state machine: shots, dirty flags, planning
├── voice.py         # Vapi webhook — phone call <-> director
├── pipeline.py      # parallel Seedream→Seedance, partial regen
├── dashboard.py     # live UI with SSE for streaming shots
├── state.py         # per-call session persistence
└── videos/<call_id>/...
```
