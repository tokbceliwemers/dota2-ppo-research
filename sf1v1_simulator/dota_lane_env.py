"""Gymnasium terrain-aware lane environment for *offline* policy pretraining.

It consumes the checked-in heightmap and exposes a normal single-agent
Gymnasium API.  This is a deliberately inspectable approximation, not a Dota
server or a source of local-Dota PPO rollout data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from grid_nav import GridNavigation
from npc_stats import LaneUnitDefinitions
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

    @classmethod
    def from_calibration(cls, path: Path) -> "TerrainLaneConfig":
        """Adopt only directly measured local-lobby values.

        The calibration report deliberately leaves attack-order buffer and
        projectile/attack-point timing unmeasured. Those remain explicit
        defaults rather than being inferred from an approximate simulator.
        """
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("source") != "local_lane_calibration":
            raise ValueError("expected a local_lane_calibration report")
        measured = report.get("measured_lane_config")
        if not isinstance(measured, dict):
            raise ValueError("calibration report has no measured_lane_config object")
        allowed = {
            name: float(measured[name])
            for name in ("tick_seconds", "hero_move_speed", "hero_attack_range", "hero_attack_damage",
                         "hero_attack_cooldown")
            if name in measured and np.isfinite(float(measured[name])) and float(measured[name]) > 0
        }
        return cls(**allowed)


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
    # Same visible feature layout as the live local-lobby lane-v3 bridge.
    # The *dynamics* remain approximate and its checkpoints must still be
    # validated locally before promotion.
    observation_version = "lane_v3"

    def __init__(self, data_directory: Path | str = Path(__file__).with_name("data"),
                 config: TerrainLaneConfig = TerrainLaneConfig(), render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("only non-graphical ansi rendering is supported")
        self.terrain = TerrainHeightmap.from_data_directory(Path(data_directory))
        self.unit_definitions = LaneUnitDefinitions.from_data_directory(Path(data_directory))
        nav_path = Path(data_directory) / "dota.gnv"
        self.navigation = GridNavigation.from_file(nav_path) if nav_path.exists() else None
        self.config, self.render_mode = config, render_mode
        self.action_space = spaces.Discrete(ACTION_DIM)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self.hero_xy = np.zeros(2, dtype=np.float32)
        self.hero_velocity = np.zeros(2, dtype=np.float32)
        self.hero_hp = np.float32(1.0)
        self.enemy_xy = np.zeros((4, 2), dtype=np.float32)
        self.enemy_hp = np.zeros(4, dtype=np.float32)
        self.enemy_max_health = np.full(4, config.creep_max_health, dtype=np.float32)
        self.enemy_health_regen = np.zeros(4, dtype=np.float32)
        self.enemy_attack_damage = np.zeros(4, dtype=np.float32)
        self.enemy_attack_rate = np.ones(4, dtype=np.float32)
        self.enemy_attack_range = np.zeros(4, dtype=np.float32)
        self.enemy_attack_ready = np.zeros(4, dtype=np.float32)
        self.ally_xy = np.zeros((4, 2), dtype=np.float32)
        self.ally_hp = np.zeros(4, dtype=np.float32)
        self.ally_max_health = np.full(4, config.creep_max_health, dtype=np.float32)
        self.ally_health_regen = np.zeros(4, dtype=np.float32)
        self.ally_attack_damage = np.zeros(4, dtype=np.float32)
        self.ally_attack_rate = np.ones(4, dtype=np.float32)
        self.ally_attack_range = np.zeros(4, dtype=np.float32)
        self.ally_attack_ready = np.zeros(4, dtype=np.float32)
        self.previous_target_hp = np.float32(1.0)
        self.attack_ready_at = np.float32(0.0)
        self.elapsed, self.step_count = np.float32(0.0), 0
        self.gold, self.last_hits, self.xp = np.float32(0.0), 0, np.float32(0.0)
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
        if self.navigation is not None:
            acceptable &= self.navigation.is_walkable(points)
        candidates = points[acceptable]
        if len(candidates) == 0:
            raise ValueError("heightmap has no terrain below max_ground_slope")
        return candidates

    def _sample_wave_center(self) -> np.ndarray:
        return self._spawn_cells[int(self.np_random.integers(len(self._spawn_cells)))].copy()

    def _can_traverse(self, start_xy: np.ndarray, end_xy: np.ndarray) -> bool:
        if not self.terrain.traversable(start_xy, end_xy, self.config.max_ground_slope):
            return False
        return self.navigation is None or self.navigation.traversable(start_xy, end_xy)

    def _spawn_layout(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a wave only where the static GridNav accepts every actor."""
        for _ in range(128):
            center = self._sample_wave_center()
            direction = self.np_random.normal(size=2).astype(np.float32)
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            perpendicular = np.array((-direction[1], direction[0]), dtype=np.float32)
            hero = center - direction * 500.0
            offsets = np.array((-1.5, -0.5, 0.5, 1.5), dtype=np.float32) * self.config.wave_spacing
            enemies = center + np.outer(offsets, perpendicular)
            allies = center + direction * 150.0 + np.outer(offsets, perpendicular)
            points = np.vstack((hero[None, :], enemies, allies))
            nav_ok = self.navigation is None or bool(np.all(self.navigation.is_walkable(points)))
            terrain_ok = bool(np.all(self.terrain.contains(points)))
            if nav_ok and terrain_ok and self._can_traverse(hero, center):
                return hero.astype(np.float32), enemies.astype(np.float32), allies.astype(np.float32), center
        raise RuntimeError("could not place a complete wave on traversable terrain and GridNav")

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
        mask[0] = True  # idle
        if (self.enemy_hp > 0).any():
            # Match the controlled local bridge's curriculum guardrail: explore
            # toward the selected creep and allow an attack order for it.
            delta = self.enemy_xy[self._target()] - self.hero_xy
            mask[1:9] = (_DIRECTIONS[1:] @ delta) >= 0.0
            mask[ATTACK_ACTION] = True
        else:
            mask[1:9] = True
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
        target_hp = self.enemy_hp[target] / max(self.enemy_max_health[target], 1.0)
        result = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        # Base layout mirrors the local lane-v3 bridge. Terrain remains part of
        # the transition function rather than becoming an extra privileged cue.
        result[0:2] = self.hero_xy / 8192.0
        result[2:4] = self.hero_velocity / 550.0
        result[4] = self.elapsed / 3600.0
        result[5] = self.gold / 30000.0
        result[6] = self.last_hits / 400.0
        result[7] = self.xp / 30000.0
        result[8] = 1.0
        result[9] = min(distance / 11585.0, 1.5)
        # lane-v3-like visible last-hit context.
        lane = result[10:]
        lane[0:2] = np.clip(delta / 800.0, -1.0, 1.0)
        lane[2] = np.clip(distance / 800.0, 0.0, 1.5)
        lane[3] = _health_bar_fraction(target_hp)
        lane[4] = np.clip((self.previous_target_hp - target_hp) / self.config.tick_seconds, -1.0, 1.0)
        lane[5] = self.config.hero_attack_damage * self.config.creep_max_health / max(self.enemy_max_health[target], 1.0)
        lane[6] = float(distance <= self.config.hero_attack_range)
        lane[7] = float(np.count_nonzero(alive_ally & (np.linalg.norm(self.ally_xy - self.enemy_xy[target], axis=1) < 230.0))) / 4.0
        lane[8:10] = np.clip(ally_delta / 800.0, -1.0, 1.0)
        lane[10] = np.clip(ally_distance / 800.0, 0.0, 1.5)
        lane[11] = _health_bar_fraction(self.ally_hp[nearest_ally] / max(self.ally_max_health[nearest_ally], 1.0))
        lane[12] = min(float(alive_enemy.sum()) / 4.0, 1.5)
        lane[13] = min(float(alive_ally.sum()) / 4.0, 1.5)
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
            "navigation": "source_gridnav" if self.navigation is not None else "terrain_only",
            "unit_definitions": self.unit_definitions.source,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.hero_xy, self.enemy_xy, self.ally_xy, _center = self._spawn_layout()
        (self.enemy_max_health, self.enemy_health_regen, self.enemy_attack_damage,
         self.enemy_attack_rate, self.enemy_attack_range) = self.unit_definitions.wave()
        (self.ally_max_health, self.ally_health_regen, self.ally_attack_damage,
         self.ally_attack_rate, self.ally_attack_range) = self.unit_definitions.wave()
        self.enemy_hp = self.enemy_max_health.copy()
        self.ally_hp = self.ally_max_health.copy()
        self.enemy_attack_ready.fill(0.0)
        self.ally_attack_ready.fill(0.0)
        self.hero_hp = np.float32(1.0)
        self.hero_velocity.fill(0.0)
        self.previous_target_hp = np.float32(1.0)
        self.attack_ready_at = np.float32(0.0)
        self.elapsed, self.step_count = np.float32(0.0), 0
        self.gold, self.last_hits, self.xp = np.float32(0.0), 0, np.float32(0.0)
        return self._observation(), self._info()

    def _advance_creeps(self, positions: np.ndarray, health: np.ndarray, targets: np.ndarray,
                        target_health: np.ndarray, attack_damage: np.ndarray, attack_rate: np.ndarray,
                        attack_range: np.ndarray, ready_at: np.ndarray) -> None:
        """One deterministic, source-stat-based auto-attack phase.

        Projectile travel and full Dota aggro rules are deliberately omitted;
        the static wave's health, regeneration, movement, range, damage, and
        attack period are all sourced from the exported NPC definitions.
        """
        for index in np.flatnonzero(health > 0.0):
            candidates = np.flatnonzero(target_health > 0.0)
            if not len(candidates):
                return
            deltas = targets[candidates] - positions[index]
            target = int(candidates[np.argmin(np.linalg.norm(deltas, axis=1))])
            distance = float(np.linalg.norm(targets[target] - positions[index]))
            if distance > attack_range[index]:
                direction = (targets[target] - positions[index]) / max(distance, 1e-6)
                proposed = positions[index] + direction * min(325.0 * self.config.tick_seconds, distance - attack_range[index])
                if self._can_traverse(positions[index], proposed):
                    positions[index] = proposed
                distance = float(np.linalg.norm(targets[target] - positions[index]))
            if distance <= attack_range[index] and self.elapsed >= ready_at[index]:
                target_health[target] -= attack_damage[index]
                ready_at[index] = self.elapsed + max(attack_rate[index], 0.01)

    def _creep_combat_phase(self) -> None:
        self.enemy_hp = np.where(self.enemy_hp > 0.0, np.minimum(
            self.enemy_max_health, self.enemy_hp + self.enemy_health_regen * self.config.tick_seconds), 0.0)
        self.ally_hp = np.where(self.ally_hp > 0.0, np.minimum(
            self.ally_max_health, self.ally_hp + self.ally_health_regen * self.config.tick_seconds), 0.0)
        self._advance_creeps(self.ally_xy, self.ally_hp, self.enemy_xy, self.enemy_hp,
                             self.ally_attack_damage, self.ally_attack_rate, self.ally_attack_range,
                             self.ally_attack_ready)
        self.enemy_hp = np.maximum(self.enemy_hp, 0.0)
        self._advance_creeps(self.enemy_xy, self.enemy_hp, self.ally_xy, self.ally_hp,
                             self.enemy_attack_damage, self.enemy_attack_rate, self.enemy_attack_range,
                             self.enemy_attack_ready)
        self.ally_hp = np.maximum(self.ally_hp, 0.0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {ACTION_DIM})")
        target_before = self._target()
        self.previous_target_hp = np.float32(self.enemy_hp[target_before] / max(self.enemy_max_health[target_before], 1.0))
        before_distance = self._target_distance(target_before)
        terrain_blocked = False
        self.hero_velocity.fill(0.0)
        if 1 <= action <= 8:
            proposed = self.hero_xy + _DIRECTIONS[action] * (self.config.hero_move_speed * self.config.tick_seconds)
            if self._can_traverse(self.hero_xy, proposed):
                self.hero_velocity = (proposed - self.hero_xy) / self.config.tick_seconds
                self.hero_xy = proposed.astype(np.float32)
            else:
                terrain_blocked = True
        target = self._target()
        # Only hero movement earns approach shaping. Creep movement belongs to
        # the environment and must not turn an invalid attack into a reward.
        after_hero_distance = self._target_distance(target)
        in_buffer = self._target_distance(target) <= self.config.hero_attack_range + self.config.attack_range_buffer
        can_attack = action == ATTACK_ACTION and in_buffer and self.elapsed >= self.attack_ready_at
        hp_before_attack = self.enemy_hp[target]
        hero_damage = self.config.hero_attack_damage * self.config.creep_max_health
        if can_attack:
            self.enemy_hp[target] -= hero_damage
            self.attack_ready_at = self.elapsed + self.config.hero_attack_cooldown
        self._creep_combat_phase()
        close_enemies = (self.enemy_hp > 0.0) & (np.linalg.norm(self.enemy_xy - self.hero_xy, axis=1) < 700.0)
        self.hero_hp -= self.config.enemy_creep_damage_per_tick * float(close_enemies.sum()) / self.config.creep_max_health
        self.enemy_hp = np.maximum(self.enemy_hp, 0.0)
        self.ally_hp = np.maximum(self.ally_hp, 0.0)
        last_hit = bool(can_attack and hp_before_attack <= hero_damage)
        reward = float(np.clip((before_distance - after_hero_distance) / 800.0, -0.03, 0.03))
        if action >= 10 or (action == ATTACK_ACTION and not in_buffer):
            reward -= 0.02
        if terrain_blocked:
            reward -= 0.01
        if last_hit:
            reward += 1.0
            self.last_hits += 1
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
