"""Training entry — wired end-to-end from config → connectome → PPO.

Produces a single artifact: the PPO checkpoint (`runs/checkpoints/*.zip`).
That checkpoint is everything the web platform needs to serve live play —
all trained synapse weights, real bodyIds, anatomical positions, and
sensory/motor index sets are baked into it via ConnectomeNet's persistent
buffers (see src/connectome/model.py).
"""
from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

from ..connectome import load_connectome, build_neuron_map
from ..env import make_vec_env
from ..utils import load_config, resolve_device
from ..utils.device import describe as describe_device
from .policy import ConnectomeActorCriticPolicy


def _tensorboard_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("tensorboard") is not None


def train(config_path: str = "config/config.yaml", profile: str | None = None) -> Path:
    cfg = load_config(config_path, profile=profile)
    device = resolve_device(cfg.device)
    print(f"[fly-flappy] profile={cfg.profile}  device={describe_device(device)}")

    # ---- 1. Connectome ----
    print("[fly-flappy] loading connectome...")
    spec = load_connectome(
        source=cfg.connectome_source,
        n_neurons=cfg.n_neurons,
        synapses_per_neuron=cfg.synapses_per_neuron,
        raw_dir=cfg.connectome["raw_dir"],
        cache_path=cfg.cache_path,
        seed=cfg.connectome["seed"],
    )
    print(f"  -> {spec.n_neurons} neurons, {spec.weights.size} synapses, source={spec.source}")

    neuron_map = build_neuron_map(
        spec,
        sensory_fraction=cfg.connectome["sensory_fraction"],
        motor_fraction=cfg.connectome["motor_fraction"],
        seed=cfg.connectome["seed"],
    )
    print(f"  -> {neuron_map.n_sensory} sensory neurons, {neuron_map.n_motor} motor neurons")

    # ---- 2. Env ----
    vec_env = make_vec_env(cfg.env["id"], n_envs=cfg.n_envs,
                           observation_mode=cfg.env.get("observation_mode", "simple"))

    # ---- 3. Policy ----
    policy_kwargs = dict(
        connectome_spec=spec,
        neuron_map=neuron_map,
        model_cfg=cfg.model,
    )
    # TensorBoard logging is optional — SB3 hard-errors if tensorboard_log is
    # set but the package isn't installed, so only enable it when tensorboard
    # actually imports.
    tb_dir = None
    if _tensorboard_available():
        tb_dir = cfg.training["tensorboard_dir"]
    else:
        print("[fly-flappy] tensorboard not installed - training without TB logs "
              "(pip install tensorboard to enable)")

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
        tensorboard_log=tb_dir,
        device=device,
        verbose=1,
    )

    ckpt_dir = Path(cfg.training["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"ppo_connectome_{cfg.profile}.zip"

    print(f"[fly-flappy] starting training for {cfg.total_timesteps} steps")
    try:
        model.learn(total_timesteps=cfg.total_timesteps)
    finally:
        # Ctrl+C (or any crash) during learn() skips SB3's own end-of-training
        # save dispatch — without this, an interrupted run would lose the
        # checkpoint entirely. Save unconditionally.
        model.save(out)
        print(f"[fly-flappy] saved model -> {out}")
    return out
