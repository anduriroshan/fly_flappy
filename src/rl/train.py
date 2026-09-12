"""Training entry — wired end-to-end from config → connectome → PPO."""
from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

from ..connectome import load_connectome, build_neuron_map
from ..env import make_vec_env
from ..utils import load_config, resolve_device
from ..utils.device import describe as describe_device
from .callbacks import DashboardCallback
from .policy import ConnectomeActorCriticPolicy


def train(config_path: str = "config/config.yaml", profile: str | None = None) -> Path:
    cfg = load_config(config_path, profile=profile)
    device = resolve_device(cfg.device)
    print(f"[fly-flappy] profile={cfg.profile}  device={describe_device(device)}")

    # ---- 1. Connectome ----
    print("[fly-flappy] loading connectome...")
    spec = load_connectome(
        source=cfg.connectome["source"],
        n_neurons=cfg.n_neurons,
        synapses_per_neuron=cfg.synapses_per_neuron,
        raw_dir=cfg.connectome["raw_dir"],
        cache_path=cfg.connectome["cache_path"],
        seed=cfg.connectome["seed"],
    )
    print(f"  → {spec.n_neurons} neurons, {spec.weights.size} synapses, source={spec.source}")

    neuron_map = build_neuron_map(
        spec,
        sensory_fraction=cfg.connectome["sensory_fraction"],
        motor_fraction=cfg.connectome["motor_fraction"],
        seed=cfg.connectome["seed"],
    )
    print(f"  → {neuron_map.n_sensory} sensory neurons, {neuron_map.n_motor} motor neurons")

    # ---- 2. Env ----
    vec_env = make_vec_env(cfg.env["id"], n_envs=cfg.n_envs)

    # ---- 3. Policy ----
    policy_kwargs = dict(
        connectome_spec=spec,
        neuron_map=neuron_map,
        model_cfg=cfg.model,
    )
    model = PPO(
        policy=ConnectomeActorCriticPolicy,
        env=vec_env,
        learning_rate=cfg.training["learning_rate"],
        n_steps=cfg.training["n_steps"],
        batch_size=cfg.training["batch_size"],
        gamma=cfg.training["gamma"],
        gae_lambda=cfg.training["gae_lambda"],
        ent_coef=cfg.training["ent_coef"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=cfg.training["tensorboard_dir"],
        device=device,
        verbose=1,
    )

    callbacks = []
    if cfg.telemetry["enabled"]:
        video_path = Path(cfg.telemetry["video_dir"]) / f"train_{cfg.profile}.mp4"
        callbacks.append(DashboardCallback(
            video_path=video_path,
            fps=cfg.telemetry["fps"],
            frame_stride=cfg.telemetry["frame_stride"],
            panel_h=cfg.telemetry["panel_height"],
            panel_w=cfg.telemetry["panel_width"],
            heatmap_neurons=cfg.telemetry["heatmap_neurons"],
            positions=spec.positions,
            rotate_speed=cfg.telemetry.get("brain_rotate_speed", 0.02),
            verbose=1,
        ))

    print(f"[fly-flappy] starting training for {cfg.total_timesteps} steps")
    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks)

    ckpt_dir = Path(cfg.training["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"ppo_connectome_{cfg.profile}.zip"
    model.save(out)
    print(f"[fly-flappy] saved model → {out}")
    return out
