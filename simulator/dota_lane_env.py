"""Gymnasium terrain-aware lane environment for *offline* policy pretraining.

It consumes the checked-in heightmap and exposes a normal single-agent
Gymnasium API.  This is a deliberately inspectable approximation, not a Dota
server or a source of local-Dota PPO rollout data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from terrain import TerrainHeightmap


ACTION_DIM = 24
ATTACK_ACTION = 9
OBSERVATION_DIM = 25
HEALTH_BAR_SEGMENTS = 20
_DIRECTIONS = np.array(
    ((0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)),
    dtype=np.float32,
)
_DIRECTIONS[1:] /= np.linalg.norm(_DIRECTIONS[1:], axis=1, keepdims=True)


@dataclass(frozen=True)
class TerrainLaneConfig:
    """Explicit parameters; only measured fields should be calibrated later."""

    tick_seconds: float = 0.2666664
    horizon_steps: int = 300
    hero_move_speed: float = 305.0
    hero_attack_range: float = 525.0
    attack_range_buffer: float = 48.0
    hero_attack_damage: float = 0.08545
    hero_attack_cooldown: float = 0.75
    creep_max_health: float = 550.0
    allied_creep_damage_per_tick: float = 7.0
    enemy_creep_damage_per_tick: float = 1.5
    max_ground_slope: float = 1.25
    spawn_margin: float = 1_400.0
    wave_spacing: float = 120.0


def _health_bar_fraction(values: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.floor(np.clip(values, 0.0, 1.0) * HEALTH_BAR_SEGMENTS + 0.5) / HEALTH_BAR_SEGMENTS


class DotaTerrainLaneEnv(gym.Env[np.ndarray, int]):
    """One Shadow Fiend-style last-hit episode over exported physical ground.

    The observation has 25 float values.  It deliberately exposes a quantized
    target health bar and recent loss rate, not exact target hit points.  The
    full target state remains internal to the environment.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}
    source = "terrain_headless_gymnasium"
    observation_version = "terrain_lane_v1"

    def __init__(self, data_directory: Path | str = Path(__file__).with_name("data"),
                 config: TerrainLaneConfig = TerrainLaneConfig(), render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("only non-graphical ansi rendering is supported")
        self.terrain = TerrainHeightmap.from_data_directory(Path(data_directory))
        self.config, self.render_mode = config, render_mode
        self.action_space = spaces.Discrete(ACTION_DIM)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self.hero_xy = np.zeros(2, dtype=np.float32)
        self.hero_hp = np.float32(1.0)
        self.enemy_xy = np.zeros((4, 2), dtype=np.float32)
        self.enemy_hp = np.zeros(4, dtype=np.float32)
        self.ally_xy = np.zeros((4, 2), dtype=np.float32)
        self.ally_hp = np.zeros(4, dtype=np.float32)
        self.previous_target_hp = np.float32(1.0)
        self.attack_ready_at = np.float32(0.0)
        self.elapsed, self.step_count = np.float32(0.0), 0
        self._target_index = 0
        self._spawn_cells = self._build_spawn_cells()

    def _build_spawn_cells(self) -> np.ndarray:
        margin_cells = int(np.ceil(self.config.spawn_margin / self.terrain.cell_size))
        rows, cols = self.terrain.normalized_heights.shape
        y, x = np.mgrid[margin_cells:rows - margin_cells, margin_cells:cols - margin_cells]
        points = np.stack((
            self.terrain.map_min + x.ravel() * self.terrain.cell_size,
            self.terrain.map_min + y.ravel() * self.terrain.cell_size,
        ), axis=1).astype(np.float32)
        acceptable = self.terrain.slope(points) <= self.config.max_ground_slope
        candidates = points[acceptable]
        if len(candidates) == 0:
            raise ValueError("heightmap has no terrain below max_ground_slope")
        return candidates

    def _sample_wave_center(self) -> np.ndarray:
        return self._spawn_cells[int(self.np_random.integers(len(self._spawn_cells)))].copy()

    def _target(self) -> int:
        alive = self.enemy_hp > 0.0
        if not alive.any():
            return 0
        distances = np.linalg.norm(self.enemy_xy - self.hero_xy, axis=1)
        scores = np.where(alive, self.enemy_hp * 0.25 + distances / 900.0, np.inf)
        return int(np.argmin(scores))

    def _target_distance(self, target: int) -> float:
        return float(np.linalg.norm(self.enemy_xy[target] - self.hero_xy))

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[:10] = True
        if (self.enemy_hp > 0).any():
            mask[ATTACK_ACTION] = self._target_distance(self._target()) <= self.config.hero_attack_range
        return mask

    def _observation(self) -> np.ndarray:
        target = self._target()
        self._target_index = target
        delta = self.enemy_xy[target] - self.hero_xy
        distance = float(np.linalg.norm(delta))
        alive_enemy = self.enemy_hp > 0.0
        alive_ally = self.ally_hp > 0.0
        nearest_ally = 0
        if alive_ally.any():
            ally_distances = np.linalg.norm(self.ally_xy - self.hero_xy, axis=1)
            nearest_ally = int(np.argmin(np.where(alive_ally, ally_distances, np.inf)))
        ally_delta = self.ally_xy[nearest_ally] - self.hero_xy
        ally_distance = float(np.linalg.norm(ally_delta))
        target_hp = self.enemy_hp[target] / self.config.creep_max_health
        height_span = max(self.terrain.raw_height_span, 1.0)
        terrain_height = float(self.terrain.height(self.hero_xy))
        result = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        # Base: map-relative pose and coarse terrain/context.  No exact enemy HP.
        result[0:2] = self.hero_xy / max(abs(self.terrain.map_min), abs(self.terrain.map_max))
        result[2] = np.clip((terrain_height - self.terrain.raw_height_min) / height_span, 0.0, 1.0)
        result[3] = np.clip(float(self.terrain.slope(self.hero_xy)) / self.config.max_ground_slope, 0.0, 1.0)
        result[4] = self.step_count / float(self.config.horizon_steps)
        result[5] = self.hero_hp
        result[6] = float(alive_enemy.sum()) / 4.0
        result[7] = float(alive_ally.sum()) / 4.0
        result[8] = np.clip((float(self.terrain.height(self.enemy_xy[target])) - terrain_height) / height_span, -1.0, 1.0)
        result[9] = 1.0
        # lane-v3-like visible last-hit context.
        lane = result[10:]
        lane[0:2] = np.clip(delta / 800.0, -1.0, 1.0)
        lane[2] = np.clip(distance / 800.0, 0.0, 1.0)
        lane[3] = _health_bar_fraction(target_hp)
        lane[4] = np.clip((self.previous_target_hp - target_hp) / self.config.tick_seconds, -1.0, 1.0)
        lane[5] = self.config.hero_attack_damage
        lane[6] = float(distance <= self.config.hero_attack_range)
        lane[7] = float(np.count_nonzero(alive_ally & (np.linalg.norm(self.ally_xy - self.enemy_xy[target], axis=1) < 230.0))) / 4.0
        lane[8:10] = np.clip(ally_delta / 800.0, -1.0, 1.0)
        lane[10] = np.clip(ally_distance / 800.0, 0.0, 1.0)
        lane[11] = _health_bar_fraction(self.ally_hp[nearest_ally] / self.config.creep_max_health)
        lane[12] = float(alive_enemy.sum()) / 4.0
        lane[13] = float(alive_ally.sum()) / 4.0
        lane[14] = np.clip((self.attack_ready_at - self.elapsed) / self.config.hero_attack_cooldown, 0.0, 1.0)
        return result

    def _info(self, *, last_hit: bool = False, terrain_blocked: bool = False) -> dict[str, Any]:
        return {
            "action_mask": self.action_mask(),
            "last_hit": last_hit,
            "terrain_blocked": terrain_blocked,
            "target_index": self._target_index,
            "game_seconds": float(self.elapsed),
            "source": self.source,
            "real_dota_verified": False,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        center = self._sample_wave_center()
        direction = self.np_random.normal(size=2).astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        perpendicular = np.array((-direction[1], direction[0]), dtype=np.float32)
        self.hero_xy = center - direction * 500.0
        offsets = np.array((-1.5, -0.5, 0.5, 1.5), dtype=np.float32) * self.config.wave_spacing
        self.enemy_xy = center + np.outer(offsets, perpendicular)
        self.ally_xy = center + direction * 150.0 + np.outer(offsets, perpendicular)
        self.enemy_hp.fill(self.config.creep_max_health)
        self.ally_hp.fill(self.config.creep_max_health)
        self.hero_hp = np.float32(1.0)
        self.previous_target_hp = np.float32(1.0)
        self.attack_ready_at = np.float32(0.0)
        self.elapsed, self.step_count = np.float32(0.0), 0
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {ACTION_DIM})")
        target_before = self._target()
        self.previous_target_hp = np.float32(self.enemy_hp[target_before] / self.config.creep_max_health)
        before_distance = self._target_distance(target_before)
        terrain_blocked = False
        if 1 <= action <= 8:
            proposed = self.hero_xy + _DIRECTIONS[action] * (self.config.hero_move_speed * self.config.tick_seconds)
            if self.terrain.traversable(self.hero_xy, proposed, self.config.max_ground_slope):
                self.hero_xy = proposed.astype(np.float32)
            else:
                terrain_blocked = True
        target = self._target()
        in_buffer = self._target_distance(target) <= self.config.hero_attack_range + self.config.attack_range_buffer
        can_attack = action == ATTACK_ACTION and in_buffer and self.elapsed >= self.attack_ready_at
        hp_before_attack = self.enemy_hp[target]
        if can_attack:
            self.enemy_hp[target] -= self.config.hero_attack_damage * self.config.creep_max_health
            self.attack_ready_at = self.elapsed + self.config.hero_attack_cooldown
        # Allied creeps provide pressure only to their nearest live enemy.
        for ally in np.flatnonzero(self.ally_hp > 0.0):
            candidates = np.flatnonzero(self.enemy_hp > 0.0)
            if len(candidates):
                nearest = candidates[np.argmin(np.linalg.norm(self.enemy_xy[candidates] - self.ally_xy[ally], axis=1))]
                self.enemy_hp[nearest] -= self.config.allied_creep_damage_per_tick
        close_enemies = (self.enemy_hp > 0.0) & (np.linalg.norm(self.enemy_xy - self.hero_xy, axis=1) < 700.0)
        self.hero_hp -= self.config.enemy_creep_damage_per_tick * float(close_enemies.sum()) / self.config.creep_max_health
        self.enemy_hp = np.maximum(self.enemy_hp, 0.0)
        self.ally_hp = np.maximum(self.ally_hp, 0.0)
        last_hit = bool(can_attack and hp_before_attack <= self.config.hero_attack_damage * self.config.creep_max_health)
        after_distance = self._target_distance(self._target()) if (self.enemy_hp > 0.0).any() else before_distance
        reward = float(np.clip((before_distance - after_distance) / 800.0, -0.03, 0.03))
        if action >= 10 or (action == ATTACK_ACTION and not in_buffer):
            reward -= 0.02
        if terrain_blocked:
            reward -= 0.01
        if last_hit:
            reward += 1.0
        self.step_count += 1
        self.elapsed += self.config.tick_seconds
        terminated = bool(self.hero_hp <= 0.0 or not (self.enemy_hp > 0.0).any())
        truncated = bool(self.step_count >= self.config.horizon_steps and not terminated)
        if self.hero_hp <= 0.0:
            reward -= 2.0
        return self._observation(), reward, terminated, truncated, self._info(last_hit=last_hit, terrain_blocked=terrain_blocked)

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        return (f"t={self.elapsed:.2f}s hero_hp={self.hero_hp:.3f} "
                f"enemy_alive={int((self.enemy_hp > 0).sum())} target={self._target_index} "
                f"ground={float(self.terrain.height(self.hero_xy)):.1f}")

