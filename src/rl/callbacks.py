"""SB3 callback that drives the telemetry dashboard from inside training."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ..telemetry import Dashboard, VideoRecorder


class DashboardCallback(BaseCallback):
    """Every `frame_stride` env steps, render a composite frame to the mp4.

    Reads the game frame from `env.last_frame` (populated by our RenderCache
    wrapper) and the neural state from `policy.neural_state_cache` (populated
    by ConnectomeActorCriticPolicy.forward).
    """

    def __init__(self, video_path: str | Path, fps: int = 30,
                 frame_stride: int = 1, panel_h: int = 512, panel_w: int = 512,
                 heatmap_neurons: int = 20000, positions: Optional[np.ndarray] = None,
                 rotate_speed: float = 0.02, verbose: int = 0):
        super().__init__(verbose)
        self.video_path = Path(video_path)
        self.frame_stride = int(frame_stride)
        self.recorder = VideoRecorder(self.video_path, fps=fps)
        self.dashboard = Dashboard(panel_h=panel_h, panel_w=panel_w,
                                   heatmap_neurons=heatmap_neurons,
                                   positions=positions, rotate_speed=rotate_speed)
        self._last_action: int = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.frame_stride != 0:
            return True

        vec_env = self.training_env
        # Grab env-0 (works for both DummyVecEnv and SubprocVecEnv).
        try:
            frame = vec_env.env_method("__getattribute__", "last_frame", indices=[0])[0]
        except Exception:
            frame = None

        neural_state: Optional[np.ndarray] = None
        cache = getattr(self.model.policy, "neural_state_cache", None)
        if cache is not None:
            neural_state = cache[0].detach().cpu().numpy()

        actions = self.locals.get("actions")
        if actions is not None:
            self._last_action = int(np.asarray(actions).flatten()[0])
        rewards = self.locals.get("rewards")
        reward = float(np.asarray(rewards).flatten()[0]) if rewards is not None else 0.0

        composite = self.dashboard.compose(
            game_frame=frame,
            neural_state=neural_state,
            reward=reward,
            step=self.num_timesteps,
            action=self._last_action,
            extra={
                "device": str(self.model.device),
                "envs": vec_env.num_envs,
            },
        )
        self.recorder.write(composite)
        return True

    def _on_training_end(self) -> None:
        self.recorder.close()
        if self.verbose:
            print(f"[dashboard] wrote {self.recorder.frames_written} frames → {self.video_path}")
