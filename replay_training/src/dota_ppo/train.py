"""CUDA behavior cloning and mathematically on-policy PPO training loops."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .data import Rollouts, Trajectories
from .model import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    epochs: int = 10
    minibatch_size: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device")
    return torch.device(requested if requested != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _checkpoint(path: Path, model: ActorCritic, optimizer: torch.optim.Optimizer | None, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                "observation_dim": model.observation_dim, "action_dim": model.action_dim, "metadata": metadata}, path)


def load_model(path: Path, device: torch.device) -> ActorCritic:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(int(payload["observation_dim"]), int(payload["action_dim"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def behavior_clone(data: Trajectories, action_dim: int, device: torch.device, output: Path, *, epochs: int = 30, batch_size: int = 4096, learning_rate: float = 3e-4, seed: int = 7) -> dict[str, float]:
    """Initialize a policy from replay-inferred labels before real PPO collection."""
    data.validate(action_dim)
    _seed(seed)
    model = ActorCritic(data.observations.shape[1], action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    observations = torch.as_tensor(data.observations, device=device)
    actions = torch.as_tensor(data.actions, device=device)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        for indexes in torch.randperm(len(observations), device=device).split(batch_size):
            logits, _ = model(observations[indexes])
            loss = F.cross_entropy(logits, actions[indexes])
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
    _checkpoint(output, model, optimizer, {"algorithm": "behavior_cloning", "source": data.source, "epochs": epochs})
    return {"loss": float(np.mean(losses[-max(1, len(losses)//10):])), "steps": float(len(data.observations)), "device": str(device)}


def _gae(rewards: Tensor, dones: Tensor, values: Tensor, gamma: float, gae_lambda: float) -> tuple[Tensor, Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros((), device=rewards.device)
    for index in range(len(rewards) - 1, -1, -1):
        next_value = torch.zeros((), device=rewards.device) if dones[index] else values[index + 1] if index + 1 < len(values) else torch.zeros((), device=rewards.device)
        delta = rewards[index] + gamma * next_value * (~dones[index]).float() - values[index]
        last_advantage = delta + gamma * gae_lambda * (~dones[index]).float() * last_advantage
        advantages[index] = last_advantage
    return advantages, advantages + values


def _ppo_update_for_source(rollouts: Rollouts, model: ActorCritic, device: torch.device, config: PPOConfig, source: str) -> dict[str, float]:
    """Perform one mathematically on-policy PPO update for one declared environment."""
    rollouts.validate(model.action_dim)
    if rollouts.source != source:
        raise ValueError(f"PPO expected on-policy {source!r} rollouts, got {rollouts.source!r}")
    if rollouts.action_masks is None:
        raise ValueError("PPO requires recorded action masks for every sampled action")
    if not bool(rollouts.dones[-1]):
        raise ValueError("PPO rollout ends mid-episode; finish the local batch before training")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    obs = torch.as_tensor(rollouts.observations, device=device)
    actions = torch.as_tensor(rollouts.actions, device=device)
    old_log_probs = torch.as_tensor(rollouts.old_log_probs, device=device)
    old_values = torch.as_tensor(rollouts.old_values, device=device)
    rewards = torch.as_tensor(rollouts.rewards, device=device)
    dones = torch.as_tensor(rollouts.dones, device=device)
    masks = torch.as_tensor(rollouts.action_masks, device=device) if rollouts.action_masks is not None else None
    advantages, returns = _gae(rewards, dones, old_values, config.gamma, config.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    stats: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
    for _ in range(config.epochs):
        for indexes in torch.randperm(len(obs), device=device).split(config.minibatch_size):
            dist, values = model.distribution(obs[indexes], masks[indexes] if masks is not None else None)
            log_probs = dist.log_prob(actions[indexes])
            log_ratio = log_probs - old_log_probs[indexes]
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(ratio * advantages[indexes], torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * advantages[indexes]).mean()
            value_loss = F.mse_loss(values, returns[indexes])
            entropy = dist.entropy().mean()
            loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm); optimizer.step()
            stats["policy_loss"].append(float(policy_loss.detach().cpu())); stats["value_loss"].append(float(value_loss.detach().cpu()))
            stats["entropy"].append(float(entropy.detach().cpu())); stats["approx_kl"].append(float(((-log_ratio).mean()).detach().cpu()))
    return {key: float(np.mean(value)) for key, value in stats.items()}


def ppo_update(rollouts: Rollouts, model: ActorCritic, device: torch.device, config: PPOConfig) -> dict[str, float]:
    """Train only from exact real-client local-lobby PPO trajectories."""
    if rollouts.source != "local_instrumented_lobby":
        raise ValueError("PPO accepts only exact on-policy local-lobby rollouts; replay-inferred data must use `bc` first")
    return _ppo_update_for_source(rollouts, model, device, config, "local_instrumented_lobby")


def ppo_update_headless_lane(rollouts: Rollouts, model: ActorCritic, device: torch.device, config: PPOConfig) -> dict[str, float]:
    """PPO for the explicitly separate approximate headless lane simulator."""
    return _ppo_update_for_source(rollouts, model, device, config, "headless_lane_simulator")


def train_ppo(rollouts: Rollouts, checkpoint_in: Path, checkpoint_out: Path, device: torch.device, config: PPOConfig) -> dict[str, float]:
    model = load_model(checkpoint_in, device)
    metrics = ppo_update(rollouts, model, device, config)
    _checkpoint(checkpoint_out, model, None, {"algorithm": "ppo", "rollout_source": rollouts.source, "config": asdict(config), **metrics})
    return metrics
