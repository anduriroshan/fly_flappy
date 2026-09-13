"""Live connectome session for the interactive web platform.

Loads a trained checkpoint and drives Flappy Bird one frame at a time,
exposing on every step:
  - the real rendered game frame (JPEG),
  - the genuine per-neuron membrane state for a spotlight subset (the same
    values the RL policy computes internally — not a replay),
  - the chosen action, reward and score.

Everything comes from the checkpoint itself: the connectome bakes in real
bodyIds, anatomical positions and the sensory/motor index sets (see
ConnectomeNet's persistent buffers), so no CSVs and no GPU are needed here.

The traced-neuron morphology for the spotlight neurons is fetched once from
Janelia's public GCS bucket (no auth) and cached as a compact JSON the
browser loads into a Three.js scene.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from stable_baselines3 import PPO

from src.env import make_env
from src.utils import load_config, resolve_device

ROLE_COLORS = {
    "motor":   (1.00, 0.45, 0.12),   # warm orange
    "sensory": (0.15, 0.75, 1.00),   # cyan
    "other":   (0.62, 0.35, 1.00),   # violet
}


class BrainSession:
    def __init__(
        self,
        checkpoint: str,
        profile: str = "full",
        config: str = "config/config.yaml",
        max_neurons: int = 400,
        seed: int = 0,
    ) -> None:
        cfg = load_config(config, profile=profile)
        self.device = resolve_device(cfg.device)
        self.seed = seed

        print(f"[brain] loading checkpoint {checkpoint} ...")
        self.model = PPO.load(checkpoint, device=self.device)
        self.connectome = self.model.policy.connectome
        self.n_neurons = int(self.connectome.n_neurons)

        self.env = make_env(
            cfg.env["id"],
            observation_mode=cfg.env.get("observation_mode", "simple"),
        )()

        # Choose a spotlight subset to visualise, balanced across roles so the
        # brain shows sensory (input), motor (output) AND interneuron
        # populations lighting up — not just one monochrome cluster.
        self.spotlight_idx, self.roles = self._balanced_spotlight(
            motor=self.connectome.motor_idx.detach().cpu().numpy(),
            sensory=self.connectome.sensory_idx.detach().cpu().numpy(),
            max_neurons=max_neurons,
            seed=seed,
        )
        ids = self.connectome.neuron_ids.detach().cpu().numpy()
        self.node_ids = ids[self.spotlight_idx]
        self.positions = (
            self.connectome.neuron_positions.detach().cpu().numpy()[self.spotlight_idx]
        )
        self.has_real_ids = bool((self.node_ids >= 0).any())
        print(f"[brain] spotlight: {self.spotlight_idx.size} neurons "
              f"({(self.roles == 'motor').sum()} motor, "
              f"{(self.roles == 'sensory').sum()} sensory), "
              f"real bodyIds: {self.has_real_ids}")

        self._build_fast_path()
        self.act_scale = 1.0
        self._reset_state()
        self._warmup()

    def _build_fast_path(self) -> None:
        """Cache a CSR copy of the (frozen, eval-mode) synapse matrix.

        ConnectomeNet.forward() goes through sparse_ops.sparse_synapse_drive,
        which rebuilds a COO sparse tensor from the raw indices/values on
        every call — necessary during training (custom autograd boundary,
        batched over 16 envs) but pure overhead at inference, where the
        weights never change and we step one observation at a time. Measured
        136 ms/step that way vs 12 ms/step here (11x) — the difference
        between 6 fps and smooth 30fps live play. Verified numerically
        identical to the real forward pass (logit/state diffs ~1e-6, pure
        float accumulation-order noise) before switching the live loop to
        this path. Purely a server-side inference optimization: does not
        touch src/connectome/ (training/gradient path) at all.
        """
        c = self.connectome
        coo = torch.sparse_coo_tensor(
            c._syn_indices.detach(), c._syn_values.detach(),
            (self.n_neurons, self.n_neurons),
        ).coalesce()
        self._csr = coo.to_sparse_csr()

    def _balanced_spotlight(self, motor, sensory, max_neurons, seed):
        """~30% motor / 35% sensory / 35% interneuron, capped at max_neurons.

        Note on visible L/R asymmetry: the render looks lopsided because our
        20,000-neuron training subgraph was built by BFS/snowball sampling
        from a single seed neuron, biasing the pool toward whichever
        hemisphere the seed sat in. Verified empirically (arbor-vertex
        skew 0.21 on x, 0.32 on z). This CANNOT be fixed at the display
        layer — you can't invent trained neurons on the missing side that
        were never sampled. The real fix is retraining with bilateral
        snowball seeds; see BRAIN_VISUALIZATION_FAQ.md for the full story.
        """
        rng = np.random.default_rng(seed)
        motor = np.asarray(motor, dtype=np.int64)
        sensory = np.asarray(sensory, dtype=np.int64)

        n_motor = min(motor.size, int(round(max_neurons * 0.30)))
        n_sensory = min(sensory.size, int(round(max_neurons * 0.35)))
        m = rng.choice(motor, n_motor, replace=False) if n_motor else np.array([], np.int64)
        s = rng.choice(sensory, n_sensory, replace=False) if n_sensory else np.array([], np.int64)

        used = np.union1d(m, s)
        remaining = max_neurons - used.size
        io_all = np.union1d(motor, sensory)
        pool = np.setdiff1d(np.arange(self.n_neurons), io_all)
        o = (rng.choice(pool, min(remaining, pool.size), replace=False)
             if remaining > 0 and pool.size else np.array([], np.int64))

        idx = np.concatenate([m, s, o]).astype(np.int64)
        roles = np.array(
            ["motor"] * m.size + ["sensory"] * s.size + ["other"] * o.size,
            dtype=object,
        )
        return idx, roles

    # ------------------------------------------------------------------ play
    def _reset_state(self) -> None:
        self.obs, _ = self.env.reset(seed=self.seed)
        self.ep_reward = 0.0
        self.step_count = 0

    def reset(self) -> None:
        self._reset_state()

    def _forward(self):
        """One connectome forward pass -> (action, spotlight membrane state).

        Reimplements ConnectomeNet.forward's exact math using the cached CSR
        matrix (see _build_fast_path) instead of calling self.connectome
        directly — same trained weights, same computation, ~11x faster at
        batch size 1. See _build_fast_path's docstring for the equivalence
        check that justified this.
        """
        c = self.connectome
        obs_t = torch.as_tensor(
            np.asarray(self.obs), dtype=torch.float32, device=self.device
        ).unsqueeze(0).to(dtype=c.sensory_proj.weight.dtype)

        with torch.no_grad():
            input_current = torch.zeros(1, self.n_neurons, device=self.device, dtype=obs_t.dtype)
            sensory_signal = c.sensory_proj(obs_t)
            input_current.index_add_(1, c.sensory_idx, sensory_signal)

            v = torch.zeros(1, self.n_neurons, device=self.device, dtype=obs_t.dtype)
            for _ in range(c.sim_steps):
                drive = torch.sparse.mm(self._csr, v.t()).t()
                v = (1.0 - c.leak) * v + c.act_fn(drive + input_current)

            motor_activity = v.index_select(1, c.motor_idx)
            logits = c.action_head(motor_activity)

        action = int(torch.argmax(logits, dim=-1).item())   # deterministic play
        spot = v[0, self.spotlight_idx].abs().detach().cpu().numpy()
        return action, spot

    def _warmup(self, steps: int = 200) -> None:
        """Establish a stable brightness scale so the brain doesn't flicker
        from per-frame min/max normalisation."""
        mags = []
        for _ in range(steps):
            action, spot = self._forward()
            mags.append(spot)
            self.obs, _r, term, trunc, _i = self.env.step(action)
            if term or trunc:
                self.obs, _ = self.env.reset()
        stacked = np.concatenate(mags) if mags else np.array([1.0])
        self.act_scale = float(np.percentile(stacked, 99)) or 1.0
        self._reset_state()
        print(f"[brain] activation scale (p99) = {self.act_scale:.4f}")

    def step(self) -> dict:
        action, spot = self._forward()
        norm = np.clip(spot / self.act_scale, 0.0, 1.0)

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        self.ep_reward += float(reward)
        self.step_count += 1

        score = 0
        if isinstance(info, dict):
            score = int(info.get("score", 0) or 0)

        return {
            "type": "frame",
            "img": self._encode_frame(self.env.last_frame),
            "act": [round(float(x), 3) for x in norm],
            "action": action,
            "reward": round(float(reward), 3),
            "score": score,
            "step": self.step_count,
            "done": done,
            "ep_reward": round(self.ep_reward, 2),
        }

    @staticmethod
    def _encode_frame(frame) -> str | None:
        if frame is None:
            return None
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")

    # ------------------------------------------------------------ morphology
    def build_brain_json(
        self,
        swc_dir: str | Path = "runs/morphology/skeletons",
        out_json: str | Path = "runs/web/brain.json",
        max_edges_per_neuron: int = 600,
    ) -> dict:
        """Fetch (once, cached) the spotlight neurons' traced skeletons and
        emit a compact JSON the browser renders as glowing 3D lines."""
        from src.morphology.gcs_fetch import fetch_skeletons_swc

        swc_dir = Path(swc_dir)
        if not self.has_real_ids:
            raise RuntimeError(
                "Checkpoint has no real bodyIds (synthetic connectome) — "
                "cannot fetch traced morphology."
            )

        real_ids = [int(b) for b in self.node_ids if int(b) >= 0]
        print(f"[brain] fetching {len(real_ids)} skeletons (public bucket, cached) ...")
        fetch_skeletons_swc(real_ids, swc_dir)

        # First pass: load skeletons + collect all coords for a shared frame.
        loaded: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        all_coords = []
        for bid in real_ids:
            p = swc_dir / f"{bid}.swc"
            if not p.exists():
                continue
            coords, parents = _load_swc(p)
            if coords.size == 0:
                continue
            loaded[bid] = (coords, parents)
            all_coords.append(coords)
        if not all_coords:
            raise RuntimeError(f"No usable SWC skeletons found in {swc_dir}")

        stacked = np.concatenate(all_coords, axis=0)
        center = stacked.mean(axis=0)
        scale = float(np.linalg.norm(stacked - center, axis=1).max()) or 1.0

        neurons = []
        for k, bid in enumerate(self.node_ids):
            bid = int(bid)
            if bid not in loaded:
                continue
            coords, parents = loaded[bid]
            v = ((coords - center) / scale).astype(np.float32)
            edges = [(int(par), row) for row, par in enumerate(parents) if par >= 0]
            if len(edges) > max_edges_per_neuron:
                stride = math.ceil(len(edges) / max_edges_per_neuron)
                edges = edges[::stride]
            role = str(self.roles[k])
            verts_flat = [round(float(x), 4) for x in v.reshape(-1)]
            edges_flat = [i for e in edges for i in e]
            neurons.append({
                "id": bid,
                "role": role,
                "act_index": int(k),            # index into the streamed `act` array
                "color": ROLE_COLORS.get(role, ROLE_COLORS["other"]),
                "verts": verts_flat,
                "edges": edges_flat,
            })

        payload = {
            "n_spotlight": int(self.spotlight_idx.size),
            "n_rendered": len(neurons),
            "dataset": "male-cns:v1.0",
            "neurons": neurons,
        }
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload))
        total_edges = sum(len(n["edges"]) // 2 for n in neurons)
        print(f"[brain] brain.json: {len(neurons)} neurons, "
              f"{total_edges} segments -> {out_json}")
        return payload


def _load_swc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords Nx3, parents N) with parents as row indices (-1 = root)."""
    coords, parents, id_to_row = [], [], {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            nid = int(parts[0])
            x, y, z = map(float, parts[2:5])
            par = int(parts[6])
            id_to_row[nid] = len(coords)
            coords.append((x, y, z))
            parents.append(par)
    if not coords:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64)
    par_rows = np.array([id_to_row.get(p, -1) for p in parents], dtype=np.int64)
    return coords, par_rows
