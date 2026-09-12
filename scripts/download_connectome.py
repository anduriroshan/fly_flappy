"""Fetch the FlyWire adult brain connectome from the Codex data release.

The Codex team publishes annual snapshots of the full adult female
Drosophila brain connectome. The URLs below are placeholders — Codex
rotates the exact filenames occasionally, so verify against
https://codex.flywire.ai/api/download before running on a fresh box.

Usage:
    python -m scripts.download_connectome --dest data/connectome/raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# NOTE: update these to the current release before running for real.
CODEX_URLS = {
    "neurons.csv":     "https://codex.flywire.ai/api/download/neurons.csv",
    "connections.csv": "https://codex.flywire.ai/api/download/connections.csv",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/connectome/raw")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    import urllib.request

    for name, url in CODEX_URLS.items():
        out = dest / name
        if out.exists():
            print(f"[skip] {out} already exists")
            continue
        print(f"[download] {url} -> {out}")
        try:
            urllib.request.urlretrieve(url, out)
        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  Manual fallback: log in at https://codex.flywire.ai, download {name}, drop into {dest}/")

    print("\nDone. Set `connectome.source: flywire` in config/config.yaml to use this data.")


if __name__ == "__main__":
    main()
