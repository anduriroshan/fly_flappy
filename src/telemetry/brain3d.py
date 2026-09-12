"""Rotating 3D point-cloud renderer for live neuron activation.

Renders each neuron as a point at its (real or fabricated) anatomical
position, brightness-coded by that step's activation, with a slow auto-
rotating camera so the brain silhouette reads clearly on video. Two extra
layers on top of the raw point cloud:

  - a soft translucent shell per hemisphere (2D hull of that side's
    projected points, filled at low alpha) standing in for a real brain
    mesh surface, which we don't have data for.
  - live synaptic "wires" between the most active neurons and their
    strongest connectome partners, colored by excitatory/inhibitory sign —
    a cheap approximation of the traced-pathway look in connectome
    viewers, built entirely from the connectivity data we already load
    (no morphology/skeleton data required).

Pure numpy + OpenCV — no OpenGL/EGL context needed, so it runs unmodified
under Xvfb-less headless containers.

Perf note: everything but the pathway loop is vectorised (matrix rotation,
a single scatter via `np.maximum.at`, one Gaussian blur for the glow halo).
The pathway overlay only touches a bounded handful of "hot" neurons per
frame, so it stays cheap even at the full ~166k-neuron connectome.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import sparse


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
        edges: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        top_active: int = 25,
        top_partners: int = 3,
    ) -> None:
        self.panel_h = panel_h
        self.panel_w = panel_w
        self.colormap = colormap
        self.rotate_speed = rotate_speed
        self.pitch = pitch
        self.theta = 0.0
        self.top_active = top_active
        self.top_partners = top_partners

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

        # Hemisphere split (by x-sign, pre-rotation) so the translucent
        # shell renders as two lobes instead of one blob-shaped hull.
        self._side_masks = [self._positions[:, 0] < 0, self._positions[:, 0] >= 0]

        # Dot footprint scales with panel size so points stay visible
        # whether we're rendering a 128px test panel or a 512px+ real one.
        self._dilate_kernel = np.ones((max(1, panel_h // 170),) * 2, dtype=np.uint8)

        self._edge_csr: Optional[sparse.csr_matrix] = None
        if edges is not None:
            self._edge_csr = self._build_local_edges(edges, n)

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
        self._draw_shell(frame, px, py, in_bounds)
        if self._edge_csr is not None and activation is not None:
            self._draw_pathways(frame, px, py, in_bounds, activation)
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

    def _draw_shell(self, frame: np.ndarray, px: np.ndarray, py: np.ndarray,
                     in_bounds: np.ndarray) -> None:
        """Soft translucent hull per hemisphere — a cheap stand-in for a
        real brain mesh surface. Recomputed every frame from the current
        projection so it stays correct as the camera rotates."""
        overlay = frame.copy()
        any_drawn = False
        for side_mask in self._side_masks:
            mask = side_mask & in_bounds
            if mask.sum() < 3:
                continue
            pts = np.stack([px[mask], py[mask]], axis=1).astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillPoly(overlay, [hull], color=(85, 70, 60))
            cv2.polylines(overlay, [hull], isClosed=True, color=(160, 145, 130),
                          thickness=1, lineType=cv2.LINE_AA)
            any_drawn = True
        if any_drawn:
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)

    def _build_local_edges(self, edges, n_full: int) -> Optional[sparse.csr_matrix]:
        rows, cols, weights = edges
        full_to_local = -np.ones(n_full, dtype=np.int64)
        full_to_local[self._idx] = np.arange(self._idx.size)
        lr, lc = full_to_local[rows], full_to_local[cols]
        keep = (lr >= 0) & (lc >= 0)
        if not keep.any():
            return None
        return sparse.csr_matrix(
            (weights[keep], (lr[keep], lc[keep])),
            shape=(self._idx.size, self._idx.size),
        )

    def _draw_pathways(self, frame: np.ndarray, px: np.ndarray, py: np.ndarray,
                        in_bounds: np.ndarray, activation: np.ndarray) -> None:
        """Draw synaptic 'wires' from the currently most-active neurons to
        their strongest partners — excitatory in warm orange, inhibitory in
        cool magenta-blue, brightness scaled by source activation."""
        a = np.asarray(activation).flatten()[self._idx]
        if a.size == 0:
            return
        k = min(self.top_active, a.size)
        hot = np.argpartition(-np.abs(a), k - 1)[:k]

        csr = self._edge_csr
        a_absmax = np.abs(a).max() or 1.0
        for i in hot:
            if not in_bounds[i]:
                continue
            start, end = csr.indptr[i], csr.indptr[i + 1]
            if start == end:
                continue
            partner_cols = csr.indices[start:end]
            partner_w = csr.data[start:end]
            m = min(self.top_partners, partner_w.size)
            top = np.argpartition(-np.abs(partner_w), m - 1)[:m] if partner_w.size > m else np.arange(partner_w.size)

            strength = min(1.0, abs(a[i]) / a_absmax)
            for t in top:
                j = partner_cols[t]
                if not in_bounds[j]:
                    continue
                excitatory = partner_w[t] > 0
                base = (60, 140, 255) if excitatory else (200, 60, 130)   # BGR: orange / magenta
                color = tuple(int(c * (0.35 + 0.65 * strength)) for c in base)
                cv2.line(frame, (int(px[i]), int(py[i])), (int(px[j]), int(py[j])),
                         color, 1, cv2.LINE_AA)
