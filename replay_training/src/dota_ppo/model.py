"""Actor-critic network for a masked discrete Dota action space."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.body = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, observations: Tensor, action_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        hidden = self.body(observations)
        logits = self.actor(hidden)
        if action_mask is not None:
            if action_mask.shape != logits.shape:
                raise ValueError(f"action_mask must have shape {tuple(logits.shape)}, got {tuple(action_mask.shape)}")
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return logits, self.critic(hidden).squeeze(-1)

    def distribution(self, observations: Tensor, action_mask: Tensor | None = None) -> tuple[Categorical, Tensor]:
        logits, values = self(observations, action_mask)
        return Categorical(logits=logits), values

    @torch.no_grad()
    def act(self, observations: Tensor, action_mask: Tensor | None = None, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        distribution, values = self.distribution(observations, action_mask)
        actions = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), values
