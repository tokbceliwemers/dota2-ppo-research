"""Shared observation layout for replay bootstrap and live lane PPO."""

from __future__ import annotations

import numpy as np


BASE_OBSERVATION_DIM = 10
LANE_FEATURE_DIM = 15
OBSERVATION_DIM = BASE_OBSERVATION_DIM + LANE_FEATURE_DIM
OBSERVATION_VERSION = "lane_v3"
HEALTH_BAR_SEGMENTS = 20

# Live-only lane suffix. Health is deliberately quantized to the visible health
# bar rather than exposing exact hit points; the policy must combine it with
# health-bar change, allied pressure, and its own attack recovery to time hits.
LANE_FEATURE_NAMES = (
    "target_dx", "target_dy", "target_distance", "target_health_bar",
    "target_health_loss_rate", "hero_damage_fraction", "in_attack_range",
    "allies_attacking_target", "nearest_ally_dx", "nearest_ally_dy",
    "nearest_ally_distance", "nearest_ally_health_bar", "enemy_count",
    "ally_count", "attack_recovery",
)


def health_bar_fraction(values: np.ndarray | float) -> np.ndarray:
    """Quantize health to the 20 visible bars used by the lane contract."""
    health = np.asarray(values, dtype=np.float32)
    return np.floor(np.clip(health, 0.0, 1.0) * HEALTH_BAR_SEGMENTS + 0.5) / HEALTH_BAR_SEGMENTS


def with_lane_features(base: np.ndarray, lane_features: np.ndarray | None = None) -> np.ndarray:
    """Append lane features, zero-filling sources that cannot observe creeps.

    Replay data lacks a reliable per-tick lane unit state.  Zero filling keeps
    it useful for movement bootstrapping while preventing it from pretending to
    contain live creep information.
    """
    values = np.asarray(base, dtype=np.float32)
    if values.shape[-1] != BASE_OBSERVATION_DIM:
        raise ValueError(f"base observation must end with {BASE_OBSERVATION_DIM} features")
    if lane_features is None:
        lane = np.zeros((*values.shape[:-1], LANE_FEATURE_DIM), dtype=np.float32)
    else:
        lane = np.asarray(lane_features, dtype=np.float32)
        if lane.shape != (*values.shape[:-1], LANE_FEATURE_DIM):
            raise ValueError(f"lane features must have shape {(*values.shape[:-1], LANE_FEATURE_DIM)}")
    return np.concatenate((values, lane), axis=-1, dtype=np.float32)
