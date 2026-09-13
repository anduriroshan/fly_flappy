"""FastAPI app for the interactive fly-brain platform.

Serves the web UI, the traced-morphology brain data, and a WebSocket that
streams a live game+activation feed from the trained connectome.

Run:  python -m scripts.serve   (or: uvicorn server.app:app)

Config via env vars (all optional):
    FLY_CHECKPOINT   path to the PPO checkpoint  (default: runs/checkpoints/ppo_connectome_full.zip)
    FLY_PROFILE      config profile              (default: full)
    FLY_MAX_NEURONS  spotlight size              (default: 400)
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.brain import BrainSession

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
BRAIN_JSON = ROOT / "runs" / "web" / "brain.json"

CHECKPOINT = os.environ.get("FLY_CHECKPOINT", "runs/checkpoints/ppo_connectome_full.zip")
PROFILE = os.environ.get("FLY_PROFILE", "full")
MAX_NEURONS = int(os.environ.get("FLY_MAX_NEURONS", "2000"))
TARGET_FPS = float(os.environ.get("FLY_FPS", "30"))

app = FastAPI(title="fly-flappy")

_session: BrainSession | None = None
_brain: dict | None = None


@app.on_event("startup")
def _startup() -> None:
    global _session, _brain
    ckpt = CHECKPOINT if Path(CHECKPOINT).is_absolute() else str(ROOT / CHECKPOINT)
    _session = BrainSession(ckpt, profile=PROFILE, max_neurons=MAX_NEURONS)

    cached = json.loads(BRAIN_JSON.read_text()) if BRAIN_JSON.exists() else None
    if cached is not None and cached.get("n_spotlight") == MAX_NEURONS:
        print(f"[app] using cached brain morphology -> {BRAIN_JSON}")
        _brain = cached
    else:
        if cached is not None:
            print(f"[app] cached brain.json was built for {cached.get('n_spotlight')} neurons, "
                  f"but FLY_MAX_NEURONS={MAX_NEURONS} now — rebuilding...")
        else:
            print("[app] building brain morphology (first run, fetches skeletons)...")
        _brain = _session.build_brain_json(out_json=BRAIN_JSON)
    print(f"[app] ready — open http://127.0.0.1:8000")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/brain")
def api_brain() -> JSONResponse:
    if _brain is None:
        return JSONResponse({"error": "brain not built yet"}, status_code=503)
    return JSONResponse(_brain)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    assert _session is not None
    state = {"playing": False}
    loop = asyncio.get_event_loop()

    async def reader() -> None:
        while True:
            msg = await ws.receive_json()
            cmd = msg.get("cmd")
            if cmd == "start":
                _session.reset()
                state["playing"] = True
            elif cmd == "pause":
                state["playing"] = False
            elif cmd == "reset":
                _session.reset()

    reader_task = asyncio.create_task(reader())
    frame_dt = 1.0 / TARGET_FPS
    try:
        while True:
            if not state["playing"]:
                await asyncio.sleep(0.05)
                continue
            frame = await loop.run_in_executor(None, _session.step)
            await ws.send_json(frame)
            if frame["done"]:
                await asyncio.sleep(0.9)          # let the crash register
                _session.reset()
            await asyncio.sleep(frame_dt)
    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()


# Static assets (main.js, style.css). Mounted last so it doesn't shadow routes.
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
