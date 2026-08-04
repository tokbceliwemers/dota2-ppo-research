"""Dataset formats and safety checks for replay bootstrap and PPO rollouts."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .observations import OBSERVATION_DIM


CURRENT_LOCAL_REWARD_VERSION = "lane_wave_clear_v4_fixed_progression"
CURRENT_LOCAL_OBSERVATION_VERSION = "lane_v3"


def checkpoint_sha256(path: Path) -> str:
    """Stable checkpoint identity used to keep PPO batches on-policy."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Trajectories:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    source: str

    def validate(self, action_dim: int | None = None) -> None:
        n = len(self.observations)
        if self.observations.ndim != 2:
            raise ValueError("observations must be a rank-2 [steps, features] array")
        if any(len(x) != n for x in (self.actions, self.rewards, self.dones)):
            raise ValueError("trajectory arrays must have the same number of steps")
        if n == 0:
            raise ValueError("trajectory dataset is empty")
        if not np.isfinite(self.observations).all() or not np.isfinite(self.rewards).all():
            raise ValueError("trajectory dataset contains non-finite values")
        if not np.issubdtype(self.actions.dtype, np.integer):
            raise ValueError("actions must use an integer dtype")
        if action_dim is not None and (self.actions.min() < 0 or self.actions.max() >= action_dim):
            raise ValueError("actions fall outside the configured action space")


@dataclass(frozen=True)
class Rollouts(Trajectories):
    old_log_probs: np.ndarray
    old_values: np.ndarray
    action_masks: np.ndarray | None = None
    game_times: np.ndarray | None = None

    def validate(self, action_dim: int | None = None) -> None:
        super().validate(action_dim)
        n = len(self.observations)
        if len(self.old_log_probs) != n or len(self.old_values) != n:
            raise ValueError("old_log_probs and old_values must match rollout length")
        if not np.isfinite(self.old_log_probs).all() or not np.isfinite(self.old_values).all():
            raise ValueError("old_log_probs and old_values must be finite")
        if self.action_masks is not None:
            if self.action_masks.ndim != 2 or self.action_masks.shape[0] != n:
                raise ValueError("action_masks must be rank-2 with one row per rollout step")
            if action_dim is not None and self.action_masks.shape[1] != action_dim:
                raise ValueError("action_masks have a different width than the action space")
            if not self.action_masks.any(axis=1).all():
                raise ValueError("every action mask must permit at least one action")
            if np.any(self.actions < 0) or np.any(self.actions >= self.action_masks.shape[1]):
                raise ValueError("actions fall outside the action-mask width")
            if not self.action_masks[np.arange(n), self.actions].all():
                raise ValueError("an action was sampled outside its action mask")
        if self.game_times is not None:
            if len(self.game_times) != n or not np.isfinite(self.game_times).all():
                raise ValueError("game_times must contain one finite value per rollout step")
            if np.any(np.diff(self.game_times) < 0):
                raise ValueError("game_times must be non-decreasing")


def save_trajectories(path: Path, data: Trajectories, metadata: dict[str, object]) -> None:
    data.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=data.observations.astype(np.float32),
        actions=data.actions.astype(np.int64),
        rewards=data.rewards.astype(np.float32),
        dones=data.dones.astype(np.bool_),
        metadata=np.array(json.dumps({"source": data.source, **metadata})),
    )


def save_rollouts(path: Path, data: Rollouts, metadata: dict[str, object]) -> None:
    """Persist exact local-policy transitions in the archive required by PPO."""
    data.validate()
    if data.source != "local_instrumented_lobby":
        raise ValueError("only local instrumented-lobby transitions may be saved as PPO rollouts")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "observations": data.observations.astype(np.float32), "actions": data.actions.astype(np.int64),
        "rewards": data.rewards.astype(np.float32), "dones": data.dones.astype(np.bool_),
        "old_log_probs": data.old_log_probs.astype(np.float32), "old_values": data.old_values.astype(np.float32),
        "metadata": np.array(json.dumps({"source": data.source, **metadata})),
    }
    if data.action_masks is not None:
        arrays["action_masks"] = data.action_masks.astype(np.bool_)
    if data.game_times is not None:
        arrays["game_times"] = data.game_times.astype(np.float32)
    np.savez_compressed(path, **arrays)


def load_trajectories(path: Path) -> Trajectories:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        data = Trajectories(
            archive["observations"].astype(np.float32), archive["actions"].astype(np.int64),
            archive["rewards"].astype(np.float32), archive["dones"].astype(bool), metadata.get("source", "unknown"),
        )
    data.validate()
    return data


def load_rollouts(path: Path) -> Rollouts:
    with np.load(path, allow_pickle=False) as archive:
        needed = {"observations", "actions", "rewards", "dones", "old_log_probs", "old_values", "metadata"}
        missing = needed - set(archive.files)
        if missing:
            raise ValueError(f"{path} is not an on-policy rollout archive; missing {sorted(missing)}")
        metadata = json.loads(str(archive["metadata"].item()))
        masks = archive["action_masks"].astype(bool) if "action_masks" in archive.files else None
        game_times = archive["game_times"].astype(np.float32) if "game_times" in archive.files else None
        data = Rollouts(
            archive["observations"].astype(np.float32), archive["actions"].astype(np.int64),
            archive["rewards"].astype(np.float32), archive["dones"].astype(bool), metadata.get("source", "unknown"),
            archive["old_log_probs"].astype(np.float32), archive["old_values"].astype(np.float32), masks, game_times,
        )
    data.validate()
    return data


def load_rollout_metadata(path: Path) -> dict[str, object]:
    """Read archive provenance without treating metadata as learning data."""
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError(f"{path} is missing rollout metadata")
        metadata = json.loads(str(archive["metadata"].item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} has invalid rollout metadata")
    return metadata


def require_current_local_reward_version(path: Path) -> dict[str, object]:
    """Reject archives from a different local reward/terminal contract."""
    metadata = load_rollout_metadata(path)
    actual = metadata.get("reward_version", "unspecified")
    if actual != CURRENT_LOCAL_REWARD_VERSION:
        raise ValueError(
            f"{path} has reward_version {actual!r}; expected {CURRENT_LOCAL_REWARD_VERSION!r}"
        )
    return metadata


def require_current_local_observation_version(path: Path) -> dict[str, object]:
    """Reject archives collected under a different live observation contract."""
    metadata = load_rollout_metadata(path)
    actual = metadata.get("observation_version", "unspecified")
    if actual != CURRENT_LOCAL_OBSERVATION_VERSION:
        raise ValueError(
            f"{path} has observation_version {actual!r}; expected {CURRENT_LOCAL_OBSERVATION_VERSION!r}"
        )
    return metadata


def require_policy_checkpoint(path: Path, checkpoint: Path) -> dict[str, object]:
    """Require that an archive was sampled by exactly the checkpoint supplied."""
    metadata = load_rollout_metadata(path)
    actual = metadata.get("policy_checkpoint_sha256")
    expected = checkpoint_sha256(checkpoint)
    if actual != expected:
        raise ValueError(f"{path} was sampled by checkpoint {actual!r}, not {checkpoint}")
    return metadata


def merge_rollouts(paths: list[Path], output: Path) -> dict[str, object]:
    """Concatenate compatible exact local-lobby batches for one PPO update."""
    if not paths:
        raise ValueError("supply at least one rollout archive")
    metadata = [require_current_local_reward_version(path) for path in paths]
    for path in paths:
        require_current_local_observation_version(path)
    checkpoint_hashes = {item.get("policy_checkpoint_sha256") for item in metadata}
    if None in checkpoint_hashes or len(checkpoint_hashes) != 1:
        raise ValueError("PPO merge requires archives from one identified policy checkpoint")
    batches = [load_rollouts(path) for path in paths]
    first = batches[0]
    observation_dim = first.observations.shape[1]
    if observation_dim != OBSERVATION_DIM:
        raise ValueError(
            f"{paths[0]} has observation width {observation_dim}; expected current width {OBSERVATION_DIM}"
        )
    mask_shape = None if first.action_masks is None else first.action_masks.shape[1]
    for path, batch in zip(paths, batches):
        if batch.source != "local_instrumented_lobby":
            raise ValueError(f"{path} is not an exact local-lobby rollout")
        if batch.observations.shape[1] != observation_dim:
            raise ValueError(f"{path} has a different observation width")
        current_mask_shape = None if batch.action_masks is None else batch.action_masks.shape[1]
        if current_mask_shape != mask_shape:
            raise ValueError(f"{path} has incompatible action-mask width")
        if not bool(batch.dones[-1]):
            raise ValueError(f"{path} ends mid-episode and cannot be merged with another session")
    merged = Rollouts(
        observations=np.concatenate([batch.observations for batch in batches]),
        actions=np.concatenate([batch.actions for batch in batches]),
        rewards=np.concatenate([batch.rewards for batch in batches]),
        dones=np.concatenate([batch.dones for batch in batches]),
        source="local_instrumented_lobby",
        old_log_probs=np.concatenate([batch.old_log_probs for batch in batches]),
        old_values=np.concatenate([batch.old_values for batch in batches]),
        action_masks=None if mask_shape is None else np.concatenate([batch.action_masks for batch in batches if batch.action_masks is not None]),
        # Batches may have come from independent Dota sessions whose game
        # clocks restart. Preserve exact PPO fields but intentionally omit a
        # misleading cross-session timing sequence; evaluate timing per archive.
        game_times=None,
    )
    save_rollouts(output, merged, {"merged_from": [str(path) for path in paths], "archives": len(paths),
                                   "reward_version": CURRENT_LOCAL_REWARD_VERSION,
                                   "observation_version": CURRENT_LOCAL_OBSERVATION_VERSION,
                                   "policy_checkpoint_sha256": next(iter(checkpoint_hashes))})
    return {"archives": len(paths), "steps": len(merged.observations), "terminal_steps": int(merged.dones.sum()), "output": str(output)}
