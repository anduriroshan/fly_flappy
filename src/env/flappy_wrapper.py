"""Flappy Bird gym wrapper helpers.

Two observation modes:
    - simple: 12-dim vector (default gym obs). Cheap and clean; used everywhere.
    - pixels: RGB frames. Only needed if we want the connectome to "see" the
      screen. Requires `render_mode='rgb_array'` and a lot more compute.

The wrapper also stores the last rendered RGB frame under `env.last_frame`
so the telemetry dashboard can grab it without a second render() call.
"""
from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np

# Registers the FlappyBird envs.
import flappy_bird_gymnasium  # noqa: F401


class RenderCache(gym.Wrapper):
    """Cache the RGB frame on each step so the dashboard can read it cheaply."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.last_frame: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._refresh_frame()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._refresh_frame()
        return obs, reward, terminated, truncated, info

    def _refresh_frame(self):
        try:
            self.last_frame = self.env.render()
        except Exception:
            # Env might not support render() in the current mode — silently skip
            # so training still proceeds without the dashboard.
            self.last_frame = None


def make_env(env_id: str = "FlappyBird-v0",
             render_mode: str = "rgb_array",
             seed: int | None = None) -> Callable[[], gym.Env]:
    """Factory suitable for SB3 vec-env constructors."""
    def _thunk() -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = RenderCache(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _thunk


def make_vec_env(env_id: str, n_envs: int, seed: int = 0, render_mode: str = "rgb_array"):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    thunks = [make_env(env_id, render_mode, seed=seed + i) for i in range(n_envs)]
    # Sub-processes for 2+ envs; a single dummy env for the smoke profile.
    return SubprocVecEnv(thunks) if n_envs > 1 else DummyVecEnv(thunks)
