"""Three-panel training dashboard rendered with OpenCV + NumPy.

    | game frame | neural heatmap | telemetry text |

Everything is pure NumPy arrays stitched with np.hstack so we can pipe the
composite frame directly into VideoRecorder.write(). No GUI required —
runs unmodified under Xvfb on a headless GPU box.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Deque, Optional
from collections import deque

import cv2
import numpy as np


@dataclass
class Dashboard:
    panel_h: int = 512
    panel_w: int = 512
    heatmap_neurons: int = 1024
    # Colour map applied to per-neuron activations.
    colormap: int = cv2.COLORMAP_INFERNO
    # Reward history for the tiny sparkline in the telemetry panel.
    reward_history: Deque[float] = field(default_factory=lambda: deque(maxlen=120))

    def compose(
        self,
        game_frame: Optional[np.ndarray],
        neural_state: Optional[np.ndarray],
        reward: float,
        step: int,
        action: int,
        extra: Optional[dict] = None,
    ) -> np.ndarray:
        """Return a (H, W*3, 3) uint8 BGR composite ready for the mp4 writer."""
        self.reward_history.append(float(reward))

        game_panel = self._game_panel(game_frame)
        heat_panel = self._heatmap_panel(neural_state)
        tel_panel = self._telemetry_panel(reward, step, action, extra or {})
        return np.hstack([game_panel, heat_panel, tel_panel])

    # ---------- individual panels ----------
    def _game_panel(self, frame: Optional[np.ndarray]) -> np.ndarray:
        if frame is None:
            panel = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
            cv2.putText(panel, "no frame", (20, self.panel_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            return panel
        # gymnasium returns RGB; OpenCV wants BGR.
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (self.panel_w, self.panel_h), interpolation=cv2.INTER_NEAREST)

    def _heatmap_panel(self, state: Optional[np.ndarray]) -> np.ndarray:
        if state is None:
            panel = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
            cv2.putText(panel, "no neural state", (20, self.panel_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            return panel

        # Take the first `heatmap_neurons` cells and reshape to a square grid.
        k = min(self.heatmap_neurons, state.size)
        side = int(math.ceil(math.sqrt(k)))
        padded = np.zeros(side * side, dtype=np.float32)
        padded[:k] = state.flatten()[:k]
        grid = padded.reshape(side, side)

        # Normalise activations to 0..255 for the colour map.
        g = grid - grid.min()
        rng = g.max() if g.max() > 1e-6 else 1.0
        norm = (g / rng * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(norm, self.colormap)
        return cv2.resize(colored, (self.panel_w, self.panel_h), interpolation=cv2.INTER_NEAREST)

    def _telemetry_panel(self, reward: float, step: int, action: int, extra: dict) -> np.ndarray:
        panel = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
        # Header
        cv2.putText(panel, "FLY-FLAPPY // LIVE", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        cv2.line(panel, (16, 44), (self.panel_w - 16, 44), (60, 60, 60), 1)

        lines = [
            f"step:    {step:>10d}",
            f"reward:  {reward:>+10.3f}",
            f"action:  {action}  ({'FLAP' if action == 1 else 'FALL'})",
        ]
        for k, v in extra.items():
            lines.append(f"{k}: {v}")

        y = 80
        for line in lines:
            cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (240, 240, 240), 1, lineType=cv2.LINE_AA)
            y += 26

        # Reward sparkline
        self._sparkline(panel, list(self.reward_history), y0=y + 20)
        return panel

    def _sparkline(self, panel: np.ndarray, series: list[float], y0: int) -> None:
        if len(series) < 2:
            return
        h = 90
        w = self.panel_w - 32
        cv2.rectangle(panel, (16, y0), (16 + w, y0 + h), (40, 40, 40), 1)
        cv2.putText(panel, "reward (last 120 steps)", (16, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

        arr = np.array(series, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        rng = max(hi - lo, 1e-6)
        xs = np.linspace(0, w - 1, arr.size).astype(int) + 16
        ys = y0 + h - ((arr - lo) / rng * (h - 2)).astype(int) - 1
        for (x1, y1), (x2, y2) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
            cv2.line(panel, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 120), 1)
