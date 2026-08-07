"""Fast, vectorized lane approximation for PPO *pretraining* only.

This is not a Dota simulator and it must never be saved as or mixed with a
``local_instrumented_lobby`` rollout.  It exists to cheaply teach the policy
the observation/action geometry (approach, range, timing, and survival) before
the much slower real-client collection stage.
"""

from __future__ import annotations

import time
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .actions import ACTION_DIM, ACTION_IDS
from .data import Rollouts
from .model import ActorCritic
from .observations import OBSERVATION_DIM, OBSERVATION_VERSION, health_bar_fraction
from .train import PPOConfig, _checkpoint, load_model, ppo_update_headless_lane


SOURCE = "headless_lane_simulator"


@dataclass(frozen=True)
class HeadlessLaneConfig:
    """Explicit approximation settings, optionally measured in a local lobby."""

    tick_seconds: float = 0.25
    hero_move_speed: float = 440.0
    hero_attack_range: float = 500.0
    attack_range_buffer: float = 50.0
    hero_attack_damage: float = 0.075
    hero_attack_cooldown: float = 0.75
    creep_passive_damage: float = 0.017
    hero_damage_near_creep: float = 0.007
    last_hit_health: float = 0.12

    @classmethod
    def from_calibration(cls, path: Path) -> "HeadlessLaneConfig":
        """Load only supported measured fields from a local calibration report."""
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("source") != "local_lane_calibration":
            raise ValueError("expected a local lane calibration report")
        measured = report.get("measured_lane_config")
        if not isinstance(measured, dict):
            raise ValueError("calibration report has no measured_lane_config object")
        allowed = {
            field: float(measured[field])
            for field in cls.__dataclass_fields__
            if field in measured and np.isfinite(float(measured[field])) and float(measured[field]) > 0
        }
        return cls(**allowed)


class HeadlessLane:
    """Independent vectorized one-creep lane drills with no rendering or IPC."""

    def __init__(self, environments: int, *, seed: int = 7, horizon: int = 96,
                 config: HeadlessLaneConfig | None = None) -> None:
        if environments < 1 or horizon < 2:
            raise ValueError("environments must be positive and horizon must be at least two")
        self.n, self.horizon = environments, horizon
        self.config = config or HeadlessLaneConfig()
        self.rng = np.random.default_rng(seed)
        self.step_index = np.zeros(environments, dtype=np.int32)
        self.hero_x = np.zeros(environments, dtype=np.float32)
        self.hero_y = np.zeros(environments, dtype=np.float32)
        self.hero_hp = np.ones(environments, dtype=np.float32)
        self.creep_x = np.zeros(environments, dtype=np.float32)
        self.creep_y = np.zeros(environments, dtype=np.float32)
        self.creep_hp = np.ones(environments, dtype=np.float32)
        self.previous_creep_hp = np.ones(environments, dtype=np.float32)
        self.ally_x = np.zeros(environments, dtype=np.float32)
        self.ally_y = np.zeros(environments, dtype=np.float32)
        self.ally_hp = np.ones(environments, dtype=np.float32)
        self.allied_pressure = np.zeros(environments, dtype=np.float32)
        self.ally_count = np.zeros(environments, dtype=np.float32)
        self.enemy_count = np.zeros(environments, dtype=np.float32)
        self.attack_ready_at = np.zeros(environments, dtype=np.float32)
        self.time = np.zeros(environments, dtype=np.float32)
        self.reset(np.ones(environments, dtype=bool))

    def reset(self, indexes: np.ndarray | None = None) -> None:
        mask = np.ones(self.n, dtype=bool) if indexes is None else indexes
        count = int(mask.sum())
        if not count:
            return
        angle = self.rng.uniform(-np.pi, np.pi, count)
        distance = self.rng.uniform(180.0, 820.0, count)
        self.hero_x[mask] = 0; self.hero_y[mask] = 0; self.hero_hp[mask] = 1.0
        self.creep_x[mask] = np.cos(angle) * distance
        self.creep_y[mask] = np.sin(angle) * distance
        self.creep_hp[mask] = self.rng.uniform(0.18, 1.0, count)
        self.previous_creep_hp[mask] = self.creep_hp[mask]
        ally_angle = self.rng.uniform(-np.pi, np.pi, count)
        ally_distance = self.rng.uniform(50.0, 260.0, count)
        self.ally_x[mask] = self.creep_x[mask] + np.cos(ally_angle) * ally_distance
        self.ally_y[mask] = self.creep_y[mask] + np.sin(ally_angle) * ally_distance
        self.ally_hp[mask] = self.rng.uniform(0.2, 1.0, count)
        self.allied_pressure[mask] = self.rng.integers(0, 5, count) / 4.0
        self.ally_count[mask] = self.rng.integers(1, 5, count) / 4.0
        self.enemy_count[mask] = self.rng.integers(1, 5, count) / 4.0
        self.step_index[mask] = 0
        self.attack_ready_at[mask] = 0.0
        self.time[mask] = 0.0

    def observation(self) -> np.ndarray:
        dx, dy = self.creep_x - self.hero_x, self.creep_y - self.hero_y
        distance = np.hypot(dx, dy)
        ally_dx, ally_dy = self.ally_x - self.hero_x, self.ally_y - self.hero_y
        ally_distance = np.hypot(ally_dx, ally_dy)
        result = np.zeros((self.n, OBSERVATION_DIM), dtype=np.float32)
        # Base fields are a stable schema; the suffix mirrors the live
        # health-bar and allied-pressure context used by the deployed bridge.
        result[:, 4] = self.step_index / float(self.horizon)
        result[:, 8] = self.hero_hp
        lane = result[:, 10:]
        lane[:, 0] = np.clip(dx / 800.0, -1.0, 1.0)
        lane[:, 1] = np.clip(dy / 800.0, -1.0, 1.0)
        lane[:, 2] = np.clip(distance / 800.0, 0.0, 1.0)
        lane[:, 3] = health_bar_fraction(self.creep_hp)
        lane[:, 4] = np.clip((self.previous_creep_hp - self.creep_hp) / self.config.tick_seconds, -1.0, 1.0)
        lane[:, 5] = self.config.hero_attack_damage
        lane[:, 6] = distance <= self.config.hero_attack_range
        lane[:, 7] = self.allied_pressure
        lane[:, 8] = np.clip(ally_dx / 800.0, -1.0, 1.0)
        lane[:, 9] = np.clip(ally_dy / 800.0, -1.0, 1.0)
        lane[:, 10] = np.clip(ally_distance / 800.0, 0.0, 1.0)
        lane[:, 11] = health_bar_fraction(self.ally_hp)
        lane[:, 12] = self.enemy_count
        lane[:, 13] = self.ally_count
        lane[:, 14] = np.clip((self.attack_ready_at - self.time) / self.config.hero_attack_cooldown, 0.0, 1.0)
        return result

    def action_masks(self) -> np.ndarray:
        masks = np.zeros((self.n, ACTION_DIM), dtype=bool)
        masks[:, :10] = True  # idle, eight movement directions, attack
        masks[:, ACTION_IDS["attack"]] = self.observation()[:, 16] > 0
        return masks

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n,):
            raise ValueError(f"actions must have shape ({self.n},)")
        before_distance = np.hypot(self.creep_x - self.hero_x, self.creep_y - self.hero_y)
        self.previous_creep_hp = self.creep_hp.copy()
        directions = np.array(((0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)), dtype=np.float32)
        moving = (actions >= 1) & (actions <= 8)
        move_distance = self.config.hero_move_speed * self.config.tick_seconds
        self.hero_x[moving] += directions[actions[moving], 0] * move_distance
        self.hero_y[moving] += directions[actions[moving], 1] * move_distance
        distance = np.hypot(self.creep_x - self.hero_x, self.creep_y - self.hero_y)
        attack = actions == ACTION_IDS["attack"]
        in_range = distance <= self.config.hero_attack_range + self.config.attack_range_buffer
        ready = self.creep_hp <= self.config.last_hit_health
        can_attack = attack & in_range & (self.time >= self.attack_ready_at)
        last_hit = can_attack & ready
        valid_hit = can_attack & ~ready
        self.creep_hp[valid_hit] -= self.config.hero_attack_damage
        self.attack_ready_at[can_attack] = self.time[can_attack] + self.config.hero_attack_cooldown
        self.creep_hp -= self.config.creep_passive_damage * (1.0 + self.allied_pressure)
        near_enemy = distance <= self.config.hero_attack_range + self.config.attack_range_buffer
        self.hero_hp[near_enemy] -= self.config.hero_damage_near_creep * self.enemy_count[near_enemy]
        progress = np.clip((before_distance - distance) / 800.0, -0.03, 0.03)
        rewards = progress.astype(np.float32) - np.where(attack & ~in_range, 0.02, 0.0)
        rewards += last_hit.astype(np.float32)
        self.step_index += 1
        self.time += self.config.tick_seconds
        dead = self.hero_hp <= 0.0
        # A creep killed by allied pressure ends the drill without a last-hit
        # reward. Leaving it alive here would let the policy repeatedly attack
        # a dead target in the approximation.
        creep_dead = self.creep_hp <= 0.0
        done = dead | last_hit | creep_dead | (self.step_index >= self.horizon)
        rewards[dead] -= 2.0
        self.reset(done)
        return self.observation(), rewards.astype(np.float32), done, self.action_masks()


class TorchHeadlessLane:
    """CUDA vectorized form of :class:`HeadlessLane` for offline pretraining.

    It intentionally mirrors the same approximate dynamics and 25-feature
    contract. Keeping this separate from the NumPy class preserves a small,
    inspectable CPU reference for tests while avoiding CPU/GPU transfers at
    every simulated step during large CUDA pretraining runs.
    """

    def __init__(self, environments: int, device: torch.device, *, seed: int = 7,
                 horizon: int = 96, config: HeadlessLaneConfig | None = None) -> None:
        if environments < 1 or horizon < 2:
            raise ValueError("environments must be positive and horizon must be at least two")
        self.n, self.device, self.horizon = environments, device, horizon
        self.config = config or HeadlessLaneConfig()
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.directions = torch.tensor(
            ((0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)),
            dtype=torch.float32, device=device,
        )
        self.step_index = torch.zeros(environments, dtype=torch.int32, device=device)
        self.hero_x = torch.zeros(environments, device=device); self.hero_y = torch.zeros(environments, device=device)
        self.hero_hp = torch.ones(environments, device=device)
        self.creep_x = torch.zeros(environments, device=device); self.creep_y = torch.zeros(environments, device=device)
        self.creep_hp = torch.ones(environments, device=device); self.previous_creep_hp = torch.ones(environments, device=device)
        self.ally_x = torch.zeros(environments, device=device); self.ally_y = torch.zeros(environments, device=device)
        self.ally_hp = torch.ones(environments, device=device); self.allied_pressure = torch.zeros(environments, device=device)
        self.ally_count = torch.zeros(environments, device=device); self.enemy_count = torch.zeros(environments, device=device)
        self.attack_ready_at = torch.zeros(environments, device=device); self.time = torch.zeros(environments, device=device)
        self.reset(torch.ones(environments, dtype=torch.bool, device=device))

    def _uniform(self, low: float, high: float) -> torch.Tensor:
        return torch.rand(self.n, generator=self.generator, device=self.device) * (high - low) + low

    def _integer_fraction(self) -> torch.Tensor:
        return torch.randint(0, 5, (self.n,), generator=self.generator, device=self.device, dtype=torch.int64).float() / 4.0

    @staticmethod
    def _replace(previous: torch.Tensor, mask: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, current, previous)

    def reset(self, mask: torch.Tensor) -> None:
        """Reset only finished lanes, without synchronizing CUDA to count them."""
        if mask.dtype != torch.bool or mask.shape != (self.n,):
            raise ValueError(f"reset mask must have shape ({self.n},)")
        angle = self._uniform(-float(np.pi), float(np.pi))
        distance = self._uniform(180.0, 820.0)
        creep_x, creep_y = torch.cos(angle) * distance, torch.sin(angle) * distance
        creep_hp = self._uniform(0.18, 1.0)
        ally_angle, ally_distance = self._uniform(-float(np.pi), float(np.pi)), self._uniform(50.0, 260.0)
        self.hero_x = self._replace(self.hero_x, mask, torch.zeros_like(self.hero_x))
        self.hero_y = self._replace(self.hero_y, mask, torch.zeros_like(self.hero_y))
        self.hero_hp = self._replace(self.hero_hp, mask, torch.ones_like(self.hero_hp))
        self.creep_x = self._replace(self.creep_x, mask, creep_x); self.creep_y = self._replace(self.creep_y, mask, creep_y)
        self.creep_hp = self._replace(self.creep_hp, mask, creep_hp)
        self.previous_creep_hp = self._replace(self.previous_creep_hp, mask, creep_hp)
        self.ally_x = self._replace(self.ally_x, mask, creep_x + torch.cos(ally_angle) * ally_distance)
        self.ally_y = self._replace(self.ally_y, mask, creep_y + torch.sin(ally_angle) * ally_distance)
        self.ally_hp = self._replace(self.ally_hp, mask, self._uniform(0.2, 1.0))
        self.allied_pressure = self._replace(self.allied_pressure, mask, self._integer_fraction())
        self.ally_count = self._replace(self.ally_count, mask, self._integer_fraction())
        self.enemy_count = self._replace(self.enemy_count, mask, self._integer_fraction())
        self.step_index = torch.where(mask, torch.zeros_like(self.step_index), self.step_index)
        self.attack_ready_at = self._replace(self.attack_ready_at, mask, torch.zeros_like(self.attack_ready_at))
        self.time = self._replace(self.time, mask, torch.zeros_like(self.time))

    def observation(self) -> torch.Tensor:
        dx, dy = self.creep_x - self.hero_x, self.creep_y - self.hero_y
        distance = torch.hypot(dx, dy)
        ally_dx, ally_dy = self.ally_x - self.hero_x, self.ally_y - self.hero_y
        result = torch.zeros((self.n, OBSERVATION_DIM), dtype=torch.float32, device=self.device)
        result[:, 4] = self.step_index.float() / float(self.horizon); result[:, 8] = self.hero_hp
        lane = result[:, 10:]
        lane[:, 0] = torch.clamp(dx / 800.0, -1.0, 1.0); lane[:, 1] = torch.clamp(dy / 800.0, -1.0, 1.0)
        lane[:, 2] = torch.clamp(distance / 800.0, 0.0, 1.0)
        lane[:, 3] = torch.floor(torch.clamp(self.creep_hp, 0.0, 1.0) * 20.0 + 0.5) / 20.0
        lane[:, 4] = torch.clamp((self.previous_creep_hp - self.creep_hp) / self.config.tick_seconds, -1.0, 1.0)
        lane[:, 5] = self.config.hero_attack_damage; lane[:, 6] = (distance <= self.config.hero_attack_range).float()
        lane[:, 7] = self.allied_pressure; lane[:, 8] = torch.clamp(ally_dx / 800.0, -1.0, 1.0)
        lane[:, 9] = torch.clamp(ally_dy / 800.0, -1.0, 1.0); lane[:, 10] = torch.clamp(torch.hypot(ally_dx, ally_dy) / 800.0, 0.0, 1.0)
        lane[:, 11] = torch.floor(torch.clamp(self.ally_hp, 0.0, 1.0) * 20.0 + 0.5) / 20.0
        lane[:, 12] = self.enemy_count; lane[:, 13] = self.ally_count
        lane[:, 14] = torch.clamp((self.attack_ready_at - self.time) / self.config.hero_attack_cooldown, 0.0, 1.0)
        return result

    def action_masks(self) -> torch.Tensor:
        masks = torch.zeros((self.n, ACTION_DIM), dtype=torch.bool, device=self.device)
        masks[:, :10] = True
        masks[:, ACTION_IDS["attack"]] = torch.hypot(self.creep_x - self.hero_x, self.creep_y - self.hero_y) <= self.config.hero_attack_range
        return masks

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if actions.shape != (self.n,):
            raise ValueError(f"actions must have shape ({self.n},)")
        before_distance = torch.hypot(self.creep_x - self.hero_x, self.creep_y - self.hero_y)
        self.previous_creep_hp = self.creep_hp.clone()
        moving = (actions >= 1) & (actions <= 8)
        direction_indexes = torch.clamp(actions, 0, 8)
        direction = self.directions[direction_indexes]
        move_distance = self.config.hero_move_speed * self.config.tick_seconds
        self.hero_x = self.hero_x + direction[:, 0] * move_distance * moving.float()
        self.hero_y = self.hero_y + direction[:, 1] * move_distance * moving.float()
        distance = torch.hypot(self.creep_x - self.hero_x, self.creep_y - self.hero_y)
        attack = actions == ACTION_IDS["attack"]
        in_range = distance <= self.config.hero_attack_range + self.config.attack_range_buffer
        ready = self.creep_hp <= self.config.last_hit_health
        can_attack = attack & in_range & (self.time >= self.attack_ready_at)
        last_hit = can_attack & ready
        valid_hit = can_attack & ~ready
        self.creep_hp = self.creep_hp - valid_hit.float() * self.config.hero_attack_damage
        self.attack_ready_at = torch.where(can_attack, self.time + self.config.hero_attack_cooldown, self.attack_ready_at)
        self.creep_hp = self.creep_hp - self.config.creep_passive_damage * (1.0 + self.allied_pressure)
        near_enemy = distance <= self.config.hero_attack_range + self.config.attack_range_buffer
        self.hero_hp = self.hero_hp - near_enemy.float() * self.config.hero_damage_near_creep * self.enemy_count
        rewards = torch.clamp((before_distance - distance) / 800.0, -0.03, 0.03)
        rewards = rewards - (attack & ~in_range).float() * 0.02 + last_hit.float()
        self.step_index += 1; self.time += self.config.tick_seconds
        dead, creep_dead = self.hero_hp <= 0.0, self.creep_hp <= 0.0
        done = dead | last_hit | creep_dead | (self.step_index >= self.horizon)
        rewards = rewards - dead.float() * 2.0
        self.reset(done)
        return self.observation(), rewards, done, self.action_masks()


def collect_headless_rollout(model: ActorCritic, environments: int, horizon: int, *, seed: int = 7,
                             config: HeadlessLaneConfig | None = None) -> Rollouts:
    """Sample an on-policy rollout from the current policy in this simulator."""
    device = next(model.parameters()).device
    if device.type == "cuda":
        return _collect_torch_headless_rollout(model, environments, horizon, device=device, seed=seed, config=config)
    env = HeadlessLane(environments, seed=seed, horizon=horizon, config=config)
    observations: list[np.ndarray] = []; actions: list[np.ndarray] = []; rewards: list[np.ndarray] = []
    dones: list[np.ndarray] = []; log_probs: list[np.ndarray] = []; values: list[np.ndarray] = []; masks: list[np.ndarray] = []
    model.eval()
    for _ in range(horizon):
        obs, mask = env.observation(), env.action_masks()
        with torch.no_grad():
            action, log_prob, value = model.act(torch.as_tensor(obs, device=device), torch.as_tensor(mask, device=device))
        _next, reward, done, _next_mask = env.step(action.cpu().numpy())
        observations.append(obs); actions.append(action.cpu().numpy()); rewards.append(reward); dones.append(done)
        log_probs.append(log_prob.cpu().numpy()); values.append(value.cpu().numpy()); masks.append(mask)
    # A vectorized collection horizon is itself an episode boundary for every lane.
    dones[-1] = np.ones(environments, dtype=bool)
    # Keep every environment's time series contiguous. GAE is defined along a
    # trajectory; the former time-major flattening mixed values from different
    # environments into one trajectory.
    def env_major(values: list[np.ndarray]) -> np.ndarray:
        stacked = np.stack(values, axis=0)
        return np.swapaxes(stacked, 0, 1).reshape((-1, *stacked.shape[2:]))
    rollout = Rollouts(
        env_major(observations), env_major(actions), env_major(rewards), env_major(dones), SOURCE,
        env_major(log_probs), env_major(values), env_major(masks), None,
    )
    rollout.validate(ACTION_DIM)
    return rollout


def _collect_torch_headless_rollout(model: ActorCritic, environments: int, horizon: int, *, device: torch.device,
                                    seed: int, config: HeadlessLaneConfig | None) -> Rollouts:
    """Collect CUDA simulation data without a per-step NumPy/CPU round trip."""
    env = TorchHeadlessLane(environments, device, seed=seed, horizon=horizon, config=config)
    observations: list[torch.Tensor] = []; actions: list[torch.Tensor] = []; rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []; log_probs: list[torch.Tensor] = []; values: list[torch.Tensor] = []; masks: list[torch.Tensor] = []
    model.eval()
    for _ in range(horizon):
        obs, mask = env.observation(), env.action_masks()
        with torch.no_grad():
            action, log_prob, value = model.act(obs, mask)
        _next, reward, done, _next_mask = env.step(action)
        observations.append(obs); actions.append(action); rewards.append(reward); dones.append(done)
        log_probs.append(log_prob); values.append(value); masks.append(mask)
    dones[-1] = torch.ones(environments, dtype=torch.bool, device=device)

    def env_major(items: list[torch.Tensor]) -> np.ndarray:
        stacked = torch.stack(items, dim=0)
        dimensions = (1, 0, *range(2, stacked.ndim))
        return stacked.permute(dimensions).reshape((-1, *stacked.shape[2:])).cpu().numpy()

    rollout = Rollouts(
        env_major(observations), env_major(actions), env_major(rewards), env_major(dones), SOURCE,
        env_major(log_probs), env_major(values), env_major(masks), None,
    )
    rollout.validate(ACTION_DIM)
    return rollout


def train_headless_lane(checkpoint_in: Path, checkpoint_out: Path, device: torch.device, *, updates: int = 8,
                        environments: int = 1024, horizon: int = 96, epochs: int = 4, seed: int = 7,
                        calibration_report: Path | None = None, minibatch_size: int = 16_384) -> dict[str, object]:
    """Fast PPO pretraining; output remains unverified until local-Dota evaluation."""
    if updates < 1 or minibatch_size < 1:
        raise ValueError("updates and minibatch_size must be positive")
    model = load_model(checkpoint_in, device)
    if model.observation_dim != OBSERVATION_DIM or model.action_dim != ACTION_DIM:
        raise ValueError("headless lane trainer requires the current lane-v3 observation/action contract")
    started = time.perf_counter(); metrics: list[dict[str, float]] = []
    # A large minibatch is safe for this tiny dense policy and eliminates most
    # CUDA kernel-launch overhead during offline vectorized pretraining. The
    # real local-Dota PPO command intentionally keeps its separate default.
    config = PPOConfig(epochs=epochs, minibatch_size=minibatch_size)
    lane_config = (HeadlessLaneConfig.from_calibration(calibration_report)
                   if calibration_report is not None else HeadlessLaneConfig())
    for update in range(updates):
        rollout = collect_headless_rollout(model, environments, horizon, seed=seed + update, config=lane_config)
        metrics.append(ppo_update_headless_lane(rollout, model, device, config))
    elapsed = time.perf_counter() - started
    samples = updates * environments * horizon
    _checkpoint(checkpoint_out, model, None, {
        "algorithm": "headless_lane_simulator_ppo", "source": SOURCE, "updates": updates,
        "environments": environments, "horizon": horizon, "epochs": epochs,
        "minibatch_size": minibatch_size,
        "real_dota_verified": False, "metrics": metrics[-1],
        "observation_version": OBSERVATION_VERSION,
        "simulation_backend": "torch_cuda" if device.type == "cuda" else "numpy_cpu",
        "lane_config": asdict(lane_config),
        "calibration_report": str(calibration_report) if calibration_report is not None else None,
    })
    return {"source": SOURCE, "samples": samples, "updates": updates, "seconds": elapsed,
            "samples_per_second": samples / max(elapsed, 1e-9), "metrics": metrics[-1],
            "minibatch_size": minibatch_size,
            "lane_config": asdict(lane_config), "output": str(checkpoint_out)}
