"""Shared observable layout for the local passive-opponent SF 1v1 curriculum."""

from __future__ import annotations

import numpy as np


BASE_OBSERVATION_DIM = 10
LANE_FEATURE_DIM = 15
PASSIVE_OPPONENT_FEATURE_DIM = 7
OBSERVATION_DIM = BASE_OBSERVATION_DIM + LANE_FEATURE_DIM + PASSIVE_OPPONENT_FEATURE_DIM
OBSERVATION_VERSION = "sf1v1_passive_v1"
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

# All values are visible or directly geometrically inferable from the local
# scene. Health and mana use the same quantized-bar convention as creeps; no
# exact opponent health, cooldown, target, or scripted-intent oracle is sent.
PASSIVE_OPPONENT_FEATURE_NAMES = (
    "opponent_present", "opponent_dx", "opponent_dy", "opponent_distance",
    "opponent_health_bar", "opponent_mana_bar", "opponent_facing_to_hero",
)


def health_bar_fraction(values: np.ndarray | float) -> np.ndarray:
    """Quantize health to the 20 visible bars used by the lane contract."""
    health = np.asarray(values, dtype=np.float32)
    return np.floor(np.clip(health, 0.0, 1.0) * HEALTH_BAR_SEGMENTS + 0.5) / HEALTH_BAR_SEGMENTS


def with_lane_features(base: np.ndarray, lane_features: np.ndarray | None = None,
                       opponent_features: np.ndarray | None = None) -> np.ndarray:
    """Append lane and visible-opponent features.

    Replay data lacks a reliable per-tick lane unit state.  Zero filling keeps
    it useful for movement bootstrapping while preventing it from pretending to
    contain live creep or opponent information.
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
    if opponent_features is None:
        opponent = np.zeros((*values.shape[:-1], PASSIVE_OPPONENT_FEATURE_DIM), dtype=np.float32)
    else:
        opponent = np.asarray(opponent_features, dtype=np.float32)
        if opponent.shape != (*values.shape[:-1], PASSIVE_OPPONENT_FEATURE_DIM):
            raise ValueError(
                f"opponent features must have shape {(*values.shape[:-1], PASSIVE_OPPONENT_FEATURE_DIM)}"
            )
    return np.concatenate((values, lane, opponent), axis=-1, dtype=np.float32)
