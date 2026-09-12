"""Rollout a trained checkpoint and record the dashboard video.

Usage:
    python -m scripts.evaluate --checkpoint runs/checkpoints/ppo_connectome_full.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO

from src.env import make_env
from src.telemetry import Dashboard, VideoRecorder
from src.utils import load_config, resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="runs/videos/eval.mp4")
    args = ap.parse_args()

    cfg = load_config(args.config, profile=args.profile)
    device = resolve_device(cfg.device)
    env = make_env(cfg.env["id"])()
    model = PPO.load(args.checkpoint, device=device)

    positions = model.policy.connectome.neuron_positions.detach().cpu().numpy()
    dashboard = Dashboard(
        panel_h=cfg.telemetry["panel_height"],
        panel_w=cfg.telemetry["panel_width"],
        heatmap_neurons=cfg.telemetry["heatmap_neurons"],
        positions=positions,
        rotate_speed=cfg.telemetry.get("brain_rotate_speed", 0.02),
    )
    with VideoRecorder(args.out, fps=cfg.telemetry["fps"]) as rec:
        total_step = 0
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
                ep_reward += float(reward)
                total_step += 1

                cache = getattr(model.policy, "neural_state_cache", None)
                neural = cache[0].detach().cpu().numpy() if cache is not None else None
                frame = dashboard.compose(
                    game_frame=env.last_frame,
                    neural_state=neural,
                    reward=float(reward),
                    step=total_step,
                    action=int(action),
                    extra={"episode": ep + 1, "ep_reward": f"{ep_reward:+.2f}"},
                )
                rec.write(frame)
            print(f"episode {ep + 1}  reward={ep_reward:+.2f}")
    print(f"wrote {rec.frames_written} frames → {args.out}")


if __name__ == "__main__":
    main()
