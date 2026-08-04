"""Fast synthetic lane demonstrations for a live-Dota PPO warm start."""

from __future__ import annotations

import numpy as np

from .actions import ACTION_IDS, movement_action
from .data import Trajectories, save_trajectories
from .observations import LANE_FEATURE_DIM, OBSERVATION_DIM


def synthetic_lane_expert(steps: int, seed: int = 7) -> Trajectories:
    """Generate a compact approach/last-hit curriculum, not PPO rollouts.

    It intentionally models only the features available in the live lane-v2
    bridge.  Its scripted labels make `attack` visible to the initial policy;
    only later local Dota rollouts determine the true PPO update.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi, steps)
    distance = rng.uniform(40, 800, steps)
    dx, dy = np.cos(angle) * distance, np.sin(angle) * distance
    attack_range = 500.0
    health = rng.uniform(0.02, 1.0, steps)
    in_range = distance <= attack_range
    last_hit_ready = health <= 0.12
    base = np.zeros((steps, 10), dtype=np.float32)
    base[:, 4] = rng.uniform(0, 0.03, steps)  # early-lane time proxy
    base[:, 8] = 1.0
    lane = np.zeros((steps, LANE_FEATURE_DIM), dtype=np.float32)
    lane[:, 0] = dx / 800
    lane[:, 1] = dy / 800
    lane[:, 2] = distance / 800
    lane[:, 3] = health
    lane[:, 4] = in_range.astype(np.float32)
    lane[:, 5] = last_hit_ready.astype(np.float32)
    lane[:, 6] = rng.integers(1, 5, steps) / 4
    lane[:, 7] = rng.integers(1, 5, steps) / 4
    observations = np.concatenate((base, lane), axis=1)
    actions = np.array([movement_action(float(x), float(y), deadzone=10.0) for x, y in zip(dx, dy)], dtype=np.int64)
    actions[in_range & last_hit_ready] = ACTION_IDS["attack"]
    rewards = np.where(actions == ACTION_IDS["attack"], 1.0, 0.02).astype(np.float32)
    dones = np.zeros(steps, dtype=bool)
    dones[::max(1, steps // 32)] = True
    dones[-1] = True
    data = Trajectories(observations, actions, rewards, dones, "synthetic_lane_expert")
    data.validate()
    if data.observations.shape[1] != OBSERVATION_DIM:
        raise AssertionError("synthetic curriculum observation layout drifted")
    return data


def build_synthetic_lane_expert(output: str, steps: int = 100_000, seed: int = 7) -> dict[str, object]:
    data = synthetic_lane_expert(steps, seed)
    from pathlib import Path

    path = Path(output)
    save_trajectories(path, data, {"curriculum": "synthetic_lane_expert_v1", "seed": seed})
    return {"steps": len(data.actions), "attack_labels": int((data.actions == ACTION_IDS["attack"]).sum()), "output": str(path)}
