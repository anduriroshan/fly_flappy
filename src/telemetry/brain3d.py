"""Rotating 3D point-cloud renderer for live neuron activation.

Renders each neuron as a point at its (real or fabricated) anatomical
position, brightness-coded by that step's activation, with a slow auto-
rotating camera so the brain silhouette reads clearly on video. Pure
numpy + OpenCV — no OpenGL/EGL context needed, so it runs unmodified
under Xvfb-less headless containers.

Perf note: everything is vectorised (matrix rotation, a single scatter via
`np.maximum.at`, one Gaussian blur for the glow halo). Even at the full
~139k-neuron connectome this renders in low single-digit milliseconds.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class Brain3DRenderer:
    def __init__(
        self,
        positions: np.ndarray,
        panel_h: int = 512,
        panel_w: int = 512,
        colormap: int = cv2.COLORMAP_INFERNO,
        max_points: int = 50_000,
        rotate_speed: float = 0.02,
        pitch: float = 0.35,
        seed: int = 0,
    ) -> None:
        self.panel_h = panel_h
        self.panel_w = panel_w
        self.colormap = colormap
        self.rotate_speed = rotate_speed
        self.pitch = pitch
        self.theta = 0.0

        n = positions.shape[0]
        if n > max_points:
            rng = np.random.default_rng(seed)
            self._idx = rng.choice(n, max_points, replace=False)
        else:
            self._idx = np.arange(n)

        pts = positions[self._idx].astype(np.float32)
        center = pts.mean(axis=0)
        centered = pts - center
        scale = np.linalg.norm(centered, axis=1).max() or 1.0
        self._positions = centered / scale  # roughly within a unit ball

        # Dot footprint scales with panel size so points stay visible
        # whether we're rendering a 128px test panel or a 512px+ real one.
        self._dilate_kernel = np.ones((max(1, panel_h // 170),) * 2, dtype=np.uint8)

    # ---------- public API ----------
    def render(self, activation: Optional[np.ndarray]) -> np.ndarray:
        """Return a (panel_h, panel_w, 3) uint8 BGR frame and advance rotation."""
        px, py, depth = self._project(self._positions, self.theta)
        self.theta += self.rotate_speed

        brightness = self._brightness(activation, depth)

        canvas = np.zeros((self.panel_h, self.panel_w), dtype=np.float32)
        in_bounds = (px >= 0) & (px < self.panel_w) & (py >= 0) & (py < self.panel_h)
        np.maximum.at(canvas, (py[in_bounds], px[in_bounds]), brightness[in_bounds])

        # Fatten single-pixel scatter points into visible dots, then add a
        # soft blurred halo on top for the "glow" look.
        canvas = cv2.dilate(canvas, self._dilate_kernel)
        halo = cv2.GaussianBlur(canvas, (0, 0), sigmaX=3.0) * 0.65
        combined = np.maximum(canvas, halo)
        canvas_u8 = np.clip(combined, 0, 255).astype(np.uint8)

        frame = cv2.applyColorMap(canvas_u8, self.colormap)
        self._draw_silhouette(frame, px[in_bounds], py[in_bounds])
        return frame

    # ---------- internals ----------
    def _project(self, pts: np.ndarray, theta: float):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_y = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]], dtype=np.float32)
        cos_p, sin_p = np.cos(self.pitch), np.sin(self.pitch)
        R_x = np.array([[1, 0, 0], [0, cos_p, -sin_p], [0, sin_p, cos_p]], dtype=np.float32)
        rotated = pts @ (R_x @ R_y).T

        cam_dist = 2.6
        focal = 2.0
        zc = rotated[:, 2] + cam_dist
        zc = np.clip(zc, 0.5, None)
        xp = focal * rotated[:, 0] / zc
        yp = focal * rotated[:, 1] / zc

        px = ((xp * 0.5 + 0.5) * self.panel_w).astype(np.int32)
        py = ((1.0 - (yp * 0.5 + 0.5)) * self.panel_h).astype(np.int32)
        return px, py, zc

    def _brightness(self, activation: Optional[np.ndarray], depth: np.ndarray) -> np.ndarray:
        n = self._positions.shape[0]
        base_glow = 45.0
        if activation is not None:
            a = np.asarray(activation).flatten()[self._idx]
            g = a - a.min()
            rng = g.max() if g.max() > 1e-6 else 1.0
            act_norm = g / rng
        else:
            act_norm = np.zeros(n, dtype=np.float32)

        d_min, d_max = depth.min(), depth.max()
        depth_norm = (d_max - depth) / (d_max - d_min + 1e-6)   # 1 = nearest
        depth_factor = 0.5 + 0.5 * depth_norm

        brightness = (base_glow + act_norm * 205.0) * depth_factor
        return np.clip(brightness, 0, 255).astype(np.float32)

    def _draw_silhouette(self, frame: np.ndarray, px: np.ndarray, py: np.ndarray) -> None:
        """Faint outline of the brain's current silhouette (2D hull of the
        projected points). Recomputed every frame so it stays correct as the
        camera rotates — cheap even at tens of thousands of points."""
        if px.size < 3:
            return
        pts = np.stack([px, py], axis=1).astype(np.int32)
        hull = cv2.convexHull(pts)
        overlay = frame.copy()
        cv2.polylines(overlay, [hull], isClosed=True, color=(90, 90, 90),
                      thickness=1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)
