"""Pre-build the traced-morphology brain.json used by the web platform.

Optional — the server builds this automatically on first run. Run it ahead
of time if you'd rather not wait during the first page load:

    python -m scripts.prepare_brain --checkpoint runs/checkpoints/ppo_connectome_full.zip

Fetches the spotlight neurons' skeletons from the public MCNS GCS bucket
(no auth) and writes runs/web/brain.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.brain import BrainSession


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/checkpoints/ppo_connectome_full.zip")
    ap.add_argument("--profile", default="full")
    ap.add_argument("--max-neurons", type=int, default=400)
    ap.add_argument("--out", default="runs/web/brain.json")
    args = ap.parse_args()

    session = BrainSession(args.checkpoint, profile=args.profile,
                           max_neurons=args.max_neurons)
    session.build_brain_json(out_json=args.out)
    print("[prepare-brain] done.")


if __name__ == "__main__":
    main()
