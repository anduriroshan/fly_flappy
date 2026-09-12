"""CLI entrypoint: `python -m scripts.train --profile full`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--profile", default=None, choices=[None, "smoke", "full"])
    args = ap.parse_args()
    train(config_path=args.config, profile=args.profile)


if __name__ == "__main__":
    main()
