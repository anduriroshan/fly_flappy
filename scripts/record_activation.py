"""Record a connectome policy's activation over a rollout, for the offline
morphology render.

    python -m scripts.record_activation --profile full \
        --checkpoint runs/checkpoints/ppo_connectome_full.zip \
        --out runs/morphology/activation.npz --steps 900 --max-neurons 400

If --checkpoint is omitted, an untrained policy is built from config (useful
for exercising the render pipeline before a model is trained). The connectome
source must be `flywire` for the downstream neuprint fetch to find real
skeletons — synthetic graphs have no real bodyIds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO

from src.connectome import load_connectome, build_neuron_map
from src.env import make_env
from src.morphology import record_activation
from src.rl.policy import ConnectomeActorCriticPolicy
from src.utils import load_config, resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--profile", default="full")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="runs/morphology/activation.npz")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--max-neurons", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config, profile=args.profile)
    device = resolve_device(cfg.device)

    print(f"[record] loading connectome (source={cfg.connectome_source})...")
    spec = load_connectome(
        source=cfg.connectome_source,
        n_neurons=cfg.n_neurons,
        synapses_per_neuron=cfg.synapses_per_neuron,
        raw_dir=cfg.connectome["raw_dir"],
        cache_path=cfg.cache_path,
        seed=cfg.connectome["seed"],
    )
    neuron_map = build_neuron_map(
        spec,
        sensory_fraction=cfg.connectome["sensory_fraction"],
        motor_fraction=cfg.connectome["motor_fraction"],
        seed=cfg.connectome["seed"],
    )
    if spec.node_ids is None:
        print("[record] WARNING: connectome has no real bodyIds (synthetic source). "
              "The neuprint fetch step will not find skeletons.")

    env = make_env(cfg.env["id"], observation_mode=cfg.env.get("observation_mode", "simple"))()

    if args.checkpoint:
        print(f"[record] loading checkpoint {args.checkpoint}")
        model = PPO.load(args.checkpoint, device=device)
    else:
        print("[record] no checkpoint -> building untrained policy from config")
        model = PPO(
            policy=ConnectomeActorCriticPolicy,
            env=make_env(cfg.env["id"],
                         observation_mode=cfg.env.get("observation_mode", "simple"))(),
            policy_kwargs=dict(connectome_spec=spec, neuron_map=neuron_map, model_cfg=cfg.model),
            device=device,
        )

    print(f"[record] rolling out {args.steps} steps, spotlight <= {args.max_neurons} neurons...")
    rec = record_activation(
        model, env, spec, neuron_map,
        n_steps=args.steps, max_neurons=args.max_neurons, seed=args.seed,
    )
    out = rec.save(args.out)
    print(f"[record] saved {rec.n_neurons} neurons x {rec.n_steps} steps -> {out}")
    print(f"[record]   roles: {dict(zip(*np.unique(rec.role, return_counts=True)))}")
    print(f"[record]   reward sum over rollout: {rec.rewards.sum():+.1f}")


if __name__ == "__main__":
    main()
