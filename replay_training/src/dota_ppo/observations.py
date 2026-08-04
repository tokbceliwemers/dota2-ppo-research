"""Shared observation layout for replay bootstrap and live lane PPO."""

from __future__ import annotations

import numpy as np


BASE_OBSERVATION_DIM = 10
LANE_FEATURE_DIM = 8
OBSERVATION_DIM = BASE_OBSERVATION_DIM + LANE_FEATURE_DIM

# Live-only lane suffix: nearest-creep dx/dy/distance, health fraction,
# in-attack-range flag, last-hit-ready flag, enemy count, ally count.


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
