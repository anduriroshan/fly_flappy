"""Launch the interactive fly-brain web platform.

    python -m scripts.serve

Loads the trained checkpoint, (on first run) fetches the spotlight neurons'
traced skeletons from the public MCNS bucket, and serves the live game +
connectome-activation UI at http://127.0.0.1:8000.

CPU-only; no GPU and no cloud instance required.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    print(f"[serve] starting fly-flappy web platform on http://{host}:{port}")
    print("[serve] first run fetches ~400 skeletons (cached afterwards) — give it a minute")
    uvicorn.run("server.app:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
