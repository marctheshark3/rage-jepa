"""A small action-conditioned joint embedding predictive architecture.

This is a state-vector JEPA trained from random weights, not a pretrained
I-JEPA/V-JEPA image model. Bodies have separate observation/action adapters and
share one latent dynamics predictor. Future targets come from a frozen moving
average encoder; variance and covariance penalties discourage collapse.

Architecture references: https://arxiv.org/abs/2301.08243 (I-JEPA),
https://arxiv.org/abs/2506.09985 (action-conditioned V-JEPA 2), and
https://arxiv.org/abs/2105.04906 (VICReg). This implementation is independent.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class JEPAConfig:
    latent_dim: int = 32
    hidden_dim: int = 128
    ema_decay: float = 0.99
    variance_weight: float = 1.0
    covariance_weight: float = 0.01

    def __post_init__(self) -> None:
        for name in ("latent_dim", "hidden_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.ema_decay, (int, float)) or not math.isfinite(self.ema_decay):
            raise ValueError("ema_decay must be finite and in [0, 1)")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        for name in ("variance_weight", "covariance_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _mlp(nin: int, hidden: int, nout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(nin, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, nout),
    )


class MultiEnvJEPA(nn.Module):
    """One latent world model with adapters for heterogeneous environments.

    ``env_dims`` maps each environment ID to ``(observation_dim, action_dim)``.
    Inputs must be finite floating tensors matching the model's dtype/device;
    observations should already be normalized using training-only statistics.
    Actions represent normalized commands, not motor-specific bus units.
    """

    def __init__(
        self,
        env_dims: dict[str, tuple[int, int]],
        config: JEPAConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(env_dims, dict) or not env_dims:
            raise ValueError("env_dims must be a nonempty dictionary")
        self.config = config if config is not None else JEPAConfig()
        if not isinstance(self.config, JEPAConfig):
            raise TypeError("config must be a JEPAConfig")
        self.env_dims: dict[str, tuple[int, int]] = {}
        for env_id, dims in env_dims.items():
            if not isinstance(env_id, str) or not env_id:
                raise ValueError("environment IDs must be nonempty strings")
            if not isinstance(dims, (tuple, list)) or len(dims) != 2:
                raise ValueError(f"{env_id}: dimensions must be (observation_dim, action_dim)")
            if any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in dims):
                raise ValueError(f"{env_id}: dimensions must be positive integers")
            self.env_dims[env_id] = (dims[0], dims[1])
        # Stable internal keys also support IDs containing '.' (ModuleDict does not).
        self._env_keys = {env_id: f"body_{i}" for i, env_id in enumerate(self.env_dims)}
        self._env_indices = {env_id: i for i, env_id in enumerate(self.env_dims)}
        latent, hidden = self.config.latent_dim, self.config.hidden_dim
        self.encoders = nn.ModuleDict({
            self._env_keys[env_id]: _mlp(obs_dim, hidden, latent)
            for env_id, (obs_dim, _action_dim) in self.env_dims.items()
        })
        self.action_projections = nn.ModuleDict({
            self._env_keys[env_id]: nn.Sequential(nn.Linear(action_dim, latent), nn.SiLU())
            for env_id, (_obs_dim, action_dim) in self.env_dims.items()
        })
        self.env_embeddings = nn.Embedding(len(self.env_dims), latent)
        nn.init.normal_(self.env_embeddings.weight, std=0.02)
        self.predictor = _mlp(latent * 3, hidden, latent)
        self.target_encoders = deepcopy(self.encoders)
        self.target_encoders.requires_grad_(False)
        self.target_encoders.eval()

    def train(self, mode: bool = True) -> MultiEnvJEPA:
        """The EMA target remains frozen/evaluating while online modules train."""
        super().train(mode)
        self.target_encoders.eval()
        return self

    def _key(self, env_id: str) -> str:
        if env_id not in self._env_keys:
            raise ValueError(f"unknown environment {env_id!r}; expected one of {tuple(self.env_dims)}")
        return self._env_keys[env_id]

    def _validate_tensor(self, name: str, value: torch.Tensor, ndim: int) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != ndim or any(size == 0 for size in value.shape):
            raise ValueError(f"{name} must be a nonempty {ndim}-dimensional tensor")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating dtype")
        parameter = self.env_embeddings.weight
        if value.dtype != parameter.dtype:
            raise ValueError(f"{name} dtype {value.dtype} differs from model dtype {parameter.dtype}")
        if value.device != parameter.device:
            raise ValueError(f"{name} device {value.device} differs from model device {parameter.device}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains nonfinite values")

    def _validate_observations(self, env_id: str, observations: torch.Tensor, ndim: int) -> str:
        key = self._key(env_id)
        self._validate_tensor("observations", observations, ndim)
        obs_dim, _ = self.env_dims[env_id]
        if observations.shape[-1] != obs_dim:
            raise ValueError(f"{env_id}: observations must end in dimension {obs_dim}")
        return key

    def _validate_actions(self, env_id: str, actions: torch.Tensor, batch: int) -> None:
        self._validate_tensor("actions", actions, 3)
        action_dim = self.env_dims[env_id][1]
        if actions.shape[0] != batch or actions.shape[2] != action_dim:
            raise ValueError(f"{env_id}: actions must have shape [batch={batch}, horizon, {action_dim}]")

    def encode(self, env_id: str, observations: torch.Tensor, target: bool = False) -> torch.Tensor:
        """Encode ``[batch, observation_dim]`` into ``[batch, latent_dim]``."""
        key = self._validate_observations(env_id, observations, 2)
        if target:
            with torch.no_grad():
                return self.target_encoders[key](observations)
        return self.encoders[key](observations)

    def _rollout_latent(self, env_id: str, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        key = self._env_keys[env_id]
        embedding = self.env_embeddings.weight[self._env_indices[env_id]].expand(latent.shape[0], -1)
        projected_actions = self.action_projections[key](actions)
        predictions = []
        for action in projected_actions.unbind(dim=1):
            # Predict changes in latent state autoregressively, never teacher-force futures.
            latent = latent + self.predictor(torch.cat((latent, action, embedding), dim=-1))
            predictions.append(latent)
        return torch.stack(predictions, dim=1)

    def rollout(self, env_id: str, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict ``[batch, horizon, latent_dim]`` from one observation and actions."""
        key = self._validate_observations(env_id, observations, 2)
        self._validate_actions(env_id, actions, observations.shape[0])
        return self._rollout_latent(env_id, self.encoders[key](observations), actions)

    def loss(self, env_id: str, observations: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Multi-step latent prediction loss, with per-body collapse regularization.

        Observations contain ``[batch, horizon + 1, observation_dim]``; action t
        moves observation t to observation t+1. Targets have no gradient. The
        online encoder regularizer covers both initial and future observations.
        ``latent_std`` is a detached diagnostic, not an optimization objective.
        """
        key = self._validate_observations(env_id, observations, 3)
        self._validate_actions(env_id, actions, observations.shape[0])
        batch, frames, obs_dim = observations.shape
        if frames != actions.shape[1] + 1:
            raise ValueError("observations must contain exactly horizon + 1 frames")
        online = self.encoders[key](observations.reshape(batch * frames, obs_dim))
        online_sequence = online.reshape(batch, frames, self.config.latent_dim)
        predictions = self._rollout_latent(env_id, online_sequence[:, 0], actions)
        with torch.no_grad():
            future = observations[:, 1:].reshape(-1, obs_dim)
            targets = self.target_encoders[key](future).reshape_as(predictions)
        prediction = F.mse_loss(predictions, targets)
        # Population statistics remain defined even for the smallest two-frame window.
        centered = online - online.mean(dim=0)
        variance_per_dimension = centered.square().mean(dim=0)
        std = torch.sqrt(variance_per_dimension + 1e-4)
        variance = F.relu(1.0 - std).mean()
        covariance_matrix = centered.T @ centered / (online.shape[0] - 1)
        off_diagonal = covariance_matrix - torch.diag_embed(covariance_matrix.diagonal())
        covariance = off_diagonal.square().sum() / self.config.latent_dim
        total = (
            prediction
            + self.config.variance_weight * variance
            + self.config.covariance_weight * covariance
        )
        return {
            "total": total,
            "prediction": prediction,
            "variance": variance,
            "covariance": covariance,
            "latent_std": variance_per_dimension.sqrt().mean().detach(),
        }

    @torch.no_grad()
    def update_target(self) -> None:
        """Apply EMA after an optimizer step, including floating/integer buffers."""
        decay = self.config.ema_decay
        for name, online in self.encoders.named_parameters():
            self.target_encoders.get_parameter(name).mul_(decay).add_(online, alpha=1.0 - decay)
        for name, online in self.encoders.named_buffers():
            target = self.target_encoders.get_buffer(name)
            if online.is_floating_point():
                target.mul_(decay).add_(online, alpha=1.0 - decay)
            else:
                target.copy_(online)

    @torch.no_grad()
    def plan(
        self,
        env_id: str,
        observations: torch.Tensor,
        goals: torch.Tensor,
        *,
        horizon: int = 12,
        candidates: int = 256,
        iterations: int = 4,
        elite_fraction: float = 0.1,
        seed: int = 0,
        action_penalty: float = 1e-4,
    ) -> torch.Tensor:
        """CEM latent-goal planning; return bounded first actions ``[batch, A]``.

        Goals use the frozen target encoder. Candidates are scored by terminal
        latent squared distance plus mean squared action effort. A local RNG
        gives repeatable results without changing global torch RNG state. This
        returns a simulation command, not a hardware control authorization.
        """
        key = self._validate_observations(env_id, observations, 2)
        self._validate_tensor("goals", goals, 2)
        if goals.shape != observations.shape:
            raise ValueError("goals must have the same [batch, observation_dim] shape as observations")
        for name, value in (("horizon", horizon), ("candidates", candidates), ("iterations", iterations)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(elite_fraction, (int, float)) or not math.isfinite(elite_fraction) or not 0 < elite_fraction <= 1:
            raise ValueError("elite_fraction must be finite and in (0, 1]")
        if not isinstance(action_penalty, (int, float)) or not math.isfinite(action_penalty) or action_penalty < 0:
            raise ValueError("action_penalty must be finite and nonnegative")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        batch, action_dim = observations.shape[0], self.env_dims[env_id][1]
        generator = torch.Generator(device=observations.device).manual_seed(seed)
        initial = self.encoders[key](observations)
        initial = initial[:, None].expand(batch, candidates, -1).reshape(batch * candidates, -1)
        goal_latent = self.target_encoders[key](goals)
        mean = observations.new_zeros(batch, horizon, action_dim)
        std = torch.ones_like(mean)
        best_sequence = mean.clone()
        best_cost = observations.new_full((batch,), float("inf"))
        n_elite = max(1, math.ceil(candidates * elite_fraction))
        rows = torch.arange(batch, device=observations.device)
        for _ in range(iterations):
            noise = torch.randn(
                (batch, candidates, horizon, action_dim),
                generator=generator,
                device=observations.device,
                dtype=observations.dtype,
            )
            sequences = (mean[:, None] + std[:, None] * noise).clamp(-1.0, 1.0)
            sequences[:, 0] = best_sequence
            if candidates > 1:
                sequences[:, 1] = mean
            predictions = self._rollout_latent(env_id, initial, sequences.reshape(batch * candidates, horizon, action_dim))
            terminal = predictions[:, -1].reshape(batch, candidates, self.config.latent_dim)
            costs = (terminal - goal_latent[:, None]).square().mean(dim=-1)
            costs = costs + action_penalty * sequences.square().mean(dim=(-1, -2))
            if not bool(torch.isfinite(costs).all()):
                raise ValueError("model produced nonfinite planning costs")
            cost, index = costs.min(dim=1)
            better = cost < best_cost
            best_sequence[better] = sequences[rows, index][better]
            best_cost = torch.minimum(best_cost, cost)
            elite_index = costs.topk(n_elite, dim=1, largest=False).indices
            elite = sequences.gather(1, elite_index[..., None, None].expand(-1, -1, horizon, action_dim))
            mean = elite.mean(dim=1)
            std = elite.std(dim=1, correction=0).clamp_min(0.05)
        return best_sequence[:, 0].clamp(-1.0, 1.0)
