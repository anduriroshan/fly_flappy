"""Custom stable-baselines3 policy that plugs the connectome into PPO.

The policy replaces SB3's stock MLP feature extractor with our ConnectomeNet.
Because SB3 expects `forward()` to return (actions, values, log_prob), we
sample from a Categorical over the connectome's motor logits.

The `neural_state_cache` attribute exposes the most recent per-neuron
membrane vector so the telemetry dashboard can visualise it live.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution

from ..connectome.loader import ConnectomeSpec
from ..connectome.mapping import NeuronMap
from ..connectome.model import ConnectomeNet


class ConnectomeActorCriticPolicy(ActorCriticPolicy):
    """SB3 policy whose brain is a biological connectome."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        connectome_spec: ConnectomeSpec,
        neuron_map: NeuronMap,
        model_cfg: Dict[str, Any],
        **kwargs,
    ):
        # Stash pre-super so _build (called from parent __init__) can use them.
        self._connectome_spec = connectome_spec
        self._neuron_map = neuron_map
        self._model_cfg = model_cfg
        # Shared feature extractor is disabled — the connectome IS the network.
        kwargs.setdefault("net_arch", [])
        kwargs.setdefault("share_features_extractor", False)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.neural_state_cache: Optional[torch.Tensor] = None

    # SB3 calls _build() from its constructor; we override to swap in ConnectomeNet.
    def _build(self, lr_schedule) -> None:
        # Do a minimal parent build to populate distribution + feature extractor,
        # then throw away the auto-built MLPs and replace them with ours.
        super()._build(lr_schedule)
        obs_dim = int(self.observation_space.shape[0])
        action_dim = int(self.action_space.n)
        self.connectome = ConnectomeNet(
            spec=self._connectome_spec,
            neuron_map=self._neuron_map,
            obs_dim=obs_dim,
            action_dim=action_dim,
            sim_steps=self._model_cfg.get("sim_steps", 4),
            leak=self._model_cfg.get("leak", 0.15),
            activation=self._model_cfg.get("activation", "tanh"),
            train_synapses=self._model_cfg.get("train_synapses", True),
        )
        # Neutralise the stock heads so parameter count reporting stays clean.
        self.mlp_extractor = _Identity()
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        # Re-init the optimizer over the *new* parameter set.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    # --------- SB3 hooks: forward / evaluate_actions / predict_values ----------
    def forward(
        self, obs: torch.Tensor, deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, state = self.connectome(obs, return_neural_state=True)
        self.neural_state_cache = state.detach()
        dist = CategoricalDistribution(action_dim=logits.shape[-1])
        dist.proba_distribution(action_logits=logits)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        logits, values, _ = self.connectome(obs)
        dist = CategoricalDistribution(action_dim=logits.shape[-1])
        dist.proba_distribution(action_logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs):
        _, values, _ = self.connectome(obs)
        return values


class _Identity(nn.Module):
    def forward(self, x): return x
    latent_dim_pi = 0
    latent_dim_vf = 0
