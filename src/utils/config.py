"""Config loading with profile flattening.

The YAML file has two profiles — `smoke` and `full`. `ProfileConfig`
resolves the profile-specific values for the fields that carry per-profile
overrides (budgets, n_envs, total_timesteps) so callers do not need to
remember which fields are dict-shaped.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ProfileConfig:
    profile: str
    device: str
    connectome: Dict[str, Any]
    model: Dict[str, Any]
    env: Dict[str, Any]
    training: Dict[str, Any]
    telemetry: Dict[str, Any]

    @property
    def connectome_source(self) -> str:
        return self.connectome["sources"][self.profile]

    @property
    def cache_path(self) -> Path:
        # Keyed by profile + source + neuron count so smoke/full,
        # synthetic/flywire, AND different neuron budgets under the same
        # profile+source never collide on the same cache file (two configs
        # sharing profile="full" but different n_neurons — e.g. a scratch
        # test config — would otherwise silently overwrite each other's
        # cached graph).
        cache_dir = Path(self.connectome["cache_dir"])
        return cache_dir / f"{self.profile}_{self.connectome_source}_{self.n_neurons}.npz"

    @property
    def n_neurons(self) -> int:
        return int(self.connectome["budgets"][self.profile]["neurons"])

    @property
    def synapses_per_neuron(self) -> int:
        return int(self.connectome["budgets"][self.profile]["synapses_per_neuron"])

    @property
    def n_envs(self) -> int:
        return int(self.env["n_envs"][self.profile])

    @property
    def total_timesteps(self) -> int:
        return int(self.training["total_timesteps"][self.profile])


def load_config(path: str | Path = "config/config.yaml", profile: str | None = None) -> ProfileConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ProfileConfig(
        profile=profile or raw.get("profile", "smoke"),
        device=raw.get("device", "auto"),
        connectome=raw["connectome"],
        model=raw["model"],
        env=raw["env"],
        training=raw["training"],
        telemetry=raw["telemetry"],
    )
