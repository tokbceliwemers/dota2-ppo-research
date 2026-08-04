"""Vectorized render-free lane approximation.

The order model is informed by static server.dll evidence: movement updates
position first, attack requires range plus a buffer, and moving out of range
cancels a queued attack. Numeric settings are intentionally explicit calibration
parameters rather than proprietary game constants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTION_DIM = 24
OBSERVATION_DIM = 18
ATTACK_ACTION = 9
_DIRECTIONS = np.array(((0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)), dtype=np.float32)


@dataclass(frozen=True)
class LaneConfig:
    """Explicit approximation parameters; calibrate from local rollout data."""

    tick_seconds: float = 0.25
    horizon_steps: int = 96
    hero_move_speed: float = 300.0
    hero_attack_range: float = 500.0
    attack_range_buffer: float = 24.0
    hero_attack_damage: float = 0.14
    hero_attack_cooldown: float = 0.75
    creep_passive_damage: float = 0.018
    hero_damage_near_creep: float = 0.008
    last_hit_health: float = 0.12

    @classmethod
    def from_calibration(cls, path: Path) -> "LaneConfig":
        """Apply only directly measured values from a local calibration report."""
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("source") != "local_lane_calibration":
            raise ValueError("expected a local lane calibration report")
        measured = report.get("measured_lane_config")
        if not isinstance(measured, dict):
            raise ValueError("calibration report has no measured_lane_config object")
        allowed = {field: measured[field] for field in cls.__dataclass_fields__ if field in measured}
        return cls(**allowed)


class LaneSimulator:
    """Many independent deterministic single-creep lane drills in NumPy."""

    source = "headless_lane_simulator"

    def __init__(self, environments: int, config: LaneConfig = LaneConfig(), *, seed: int = 7) -> None:
        if environments < 1:
            raise ValueError("environments must be positive")
        self.n, self.config = environments, config
        self.rng = np.random.default_rng(seed)
        self.hero_xy = np.zeros((environments, 2), np.float32)
        self.creep_xy = np.zeros((environments, 2), np.float32)
        self.hero_hp = np.ones(environments, np.float32)
        self.creep_hp = np.ones(environments, np.float32)
        self.attack_ready_at = np.zeros(environments, np.float32)
        self.time = np.zeros(environments, np.float32)
        self.steps = np.zeros(environments, np.int32)
        self.enemy_count = np.zeros(environments, np.float32)
        self.ally_count = np.zeros(environments, np.float32)
        self.reset()

    def reset(self, where: np.ndarray | None = None) -> np.ndarray:
        mask = np.ones(self.n, bool) if where is None else np.asarray(where, bool)
        count = int(mask.sum())
        if not count:
            return self.observation()
        angle = self.rng.uniform(-np.pi, np.pi, count)
        distance = self.rng.uniform(180.0, 820.0, count)
        self.hero_xy[mask] = 0
        self.creep_xy[mask, 0] = np.cos(angle) * distance
        self.creep_xy[mask, 1] = np.sin(angle) * distance
        self.hero_hp[mask] = 1.0
        self.creep_hp[mask] = self.rng.uniform(0.18, 1.0, count)
        self.attack_ready_at[mask] = 0.0
        self.time[mask] = 0.0
        self.steps[mask] = 0
        self.enemy_count[mask] = self.rng.integers(1, 5, count) / 4.0
        self.ally_count[mask] = self.rng.integers(1, 5, count) / 4.0
        return self.observation()

    def _distance(self) -> np.ndarray:
        return np.linalg.norm(self.creep_xy - self.hero_xy, axis=1)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros((self.n, ACTION_DIM), dtype=bool)
        mask[:, :10] = True
        in_range = self._distance() <= self.config.hero_attack_range + self.config.attack_range_buffer
        mask[:, ATTACK_ACTION] = in_range
        return mask

    def observation(self) -> np.ndarray:
        delta = self.creep_xy - self.hero_xy
        distance = self._distance()
        result = np.zeros((self.n, OBSERVATION_DIM), np.float32)
        result[:, 4] = self.steps / float(self.config.horizon_steps)
        result[:, 8] = self.hero_hp
        lane = result[:, 10:]
        lane[:, :2] = np.clip(delta / 800.0, -1.0, 1.0)
        lane[:, 2] = np.clip(distance / 800.0, 0.0, 1.0)
        lane[:, 3] = self.creep_hp
        lane[:, 4] = distance <= self.config.hero_attack_range
        lane[:, 5] = self.creep_hp <= self.config.last_hit_health
        lane[:, 6] = self.enemy_count
        lane[:, 7] = self.ally_count
        return result

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Apply one authoritative-tick approximation and automatically reset terminals."""
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n,) or actions.min() < 0 or actions.max() >= ACTION_DIM:
            raise ValueError(f"actions must be {self.n} valid action IDs")
        before = self._distance()
        moving = (actions >= 1) & (actions <= 8)
        self.hero_xy[moving] += _DIRECTIONS[actions[moving]] * (self.config.hero_move_speed * self.config.tick_seconds)
        distance = self._distance()
        in_buffer = distance <= self.config.hero_attack_range + self.config.attack_range_buffer
        attack = actions == ATTACK_ACTION
        can_attack = attack & in_buffer & (self.time >= self.attack_ready_at)
        self.creep_hp[can_attack] -= self.config.hero_attack_damage
        self.attack_ready_at[can_attack] = self.time[can_attack] + self.config.hero_attack_cooldown
        self.creep_hp -= self.config.creep_passive_damage * (1.0 + self.ally_count)
        near = distance <= self.config.hero_attack_range + self.config.attack_range_buffer
        self.hero_hp[near] -= self.config.hero_damage_near_creep * self.enemy_count[near]
        last_hit = can_attack & (self.creep_hp <= 0.0)
        dead = self.hero_hp <= 0.0
        self.steps += 1
        self.time += self.config.tick_seconds
        timeout = self.steps >= self.config.horizon_steps
        done = last_hit | dead | timeout
        reward = np.clip((before - distance) / 800.0, -0.03, 0.03).astype(np.float32)
        reward[attack & ~in_buffer] -= 0.02
        reward[last_hit] += 1.0
        reward[dead] -= 2.0
        info = {"last_hit": last_hit.copy(), "hero_dead": dead.copy(), "action_mask": self.action_mask()}
        self.reset(done)
        return self.observation(), reward, done, info
