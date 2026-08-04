"""Bounded overnight supervisor for exact local PPO rollout batches.

Collection remains a local custom-game responsibility.  This module watches
for completed archives, validates them before use, then evaluates and trains
without requiring an interactive Codex session to remain open.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import (CURRENT_LOCAL_OBSERVATION_VERSION, CURRENT_LOCAL_REWARD_VERSION, Rollouts,
                   checkpoint_sha256, load_rollouts, require_current_local_observation_version,
                   require_current_local_reward_version, merge_rollouts)
from .evaluation import evaluate_rollouts, rollout_metrics
from .observations import OBSERVATION_DIM
from .train import PPOConfig, select_device, train_ppo


@dataclass(frozen=True)
class SupervisorConfig:
    rollout_dir: Path
    checkpoint: Path
    run_dir: Path
    batch_archives: int = 3
    max_batches: int = 3
    max_hours: float = 8.0
    poll_seconds: float = 10.0
    min_archive_age_seconds: float = 2.0
    min_steps: int = 100
    min_decisions_per_game_second: float = 3.0
    max_failures: int = 3
    epochs: int = 10
    device: str = "cuda"
    include_existing: bool = False
    required_reward_version: str = CURRENT_LOCAL_REWARD_VERSION
    required_observation_version: str = CURRENT_LOCAL_OBSERVATION_VERSION
    min_game_seconds: float = 150.0

    def validate(self) -> None:
        if not self.checkpoint.is_file():
            raise ValueError(f"checkpoint does not exist: {self.checkpoint}")
        if self.batch_archives < 1:
            raise ValueError("batch_archives must be positive")
        if self.max_batches != 1:
            raise ValueError("one supervisor run may train exactly one PPO update; restart the policy bridge with the new checkpoint before collecting another on-policy batch")
        if self.max_hours <= 0 or self.poll_seconds < 0:
            raise ValueError("max_hours must be positive and poll_seconds cannot be negative")
        if self.min_steps < 1 or self.max_failures < 1 or self.epochs < 1 or self.min_game_seconds <= 0:
            raise ValueError("min_steps, max_failures, epochs, and min_game_seconds must be positive")
        if not self.required_reward_version or not self.required_observation_version:
            raise ValueError("required reward and observation versions must be non-empty")


def validate_rollout(path: Path, config: SupervisorConfig) -> tuple[Rollouts | None, dict[str, object]]:
    """Return an exact, complete, timing-valid rollout or a rejection reason."""
    try:
        rollout = load_rollouts(path)
        metadata = require_current_local_reward_version(path)
        require_current_local_observation_version(path)
    except (OSError, ValueError, KeyError) as error:
        return None, {"accepted": False, "reason": f"unreadable archive: {error}"}
    if metadata.get("reward_version") != config.required_reward_version:
        return None, {"accepted": False, "reason": f"reward_version is not {config.required_reward_version!r}"}
    if metadata.get("observation_version") != config.required_observation_version:
        return None, {"accepted": False, "reason": f"observation_version is not {config.required_observation_version!r}"}
    if metadata.get("policy_checkpoint_sha256") != checkpoint_sha256(config.checkpoint):
        return None, {"accepted": False, "reason": "archive was not sampled by the configured checkpoint"}
    if rollout.source != "local_instrumented_lobby":
        return None, {"accepted": False, "reason": "not an exact local_instrumented_lobby archive"}
    if rollout.observations.shape[1] != OBSERVATION_DIM:
        return None, {"accepted": False, "reason": f"observation width is not {OBSERVATION_DIM}"}
    if len(rollout.actions) < config.min_steps:
        return None, {"accepted": False, "reason": f"only {len(rollout.actions)} steps; need {config.min_steps}"}
    if rollout.old_log_probs is None or rollout.old_values is None or rollout.action_masks is None:
        return None, {"accepted": False, "reason": "missing exact PPO behavior-policy fields"}
    if not int(rollout.dones.sum()):
        return None, {"accepted": False, "reason": "no completed episode"}
    if not bool(rollout.dones[-1]):
        return None, {"accepted": False, "reason": "archive ends mid-episode"}
    if rollout.game_times is None:
        return None, {"accepted": False, "reason": "missing game-time cadence evidence"}
    if not np.isfinite(rollout.game_times).all():
        return None, {"accepted": False, "reason": "non-finite game times"}
    metrics = rollout_metrics(rollout)
    game_span = metrics["game_time_span"]
    if game_span is None or float(game_span) < config.min_game_seconds:
        return None, {"accepted": False, "reason": f"game-time span {game_span} below {config.min_game_seconds}", "metrics": metrics}
    cadence = metrics["decisions_per_game_second"]
    if cadence is None or float(cadence) < config.min_decisions_per_game_second:
        return None, {"accepted": False, "reason": f"decision cadence {cadence} below {config.min_decisions_per_game_second}", "metrics": metrics}
    if metrics["all_sampled_actions_valid"] is not True:
        return None, {"accepted": False, "reason": "sampled an action outside its action mask", "metrics": metrics}
    return rollout, {"accepted": True, "metrics": metrics}


class OvernightSupervisor:
    """Run bounded evaluation/training cycles as local rollout archives arrive."""

    def __init__(self, config: SupervisorConfig) -> None:
        config.validate()
        self.config = config
        self.audit_path = config.run_dir / "audit.jsonl"
        self.stop_path = config.run_dir / "STOP"
        self.seen: set[Path] = set()
        self.pending: list[Path] = []
        self.failures = 0
        self.current_checkpoint = config.checkpoint
        self.completed_batches = 0

    def _audit(self, event: str, **payload: Any) -> None:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        record = {"time": time.time(), "event": event, **payload}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _new_archives(self) -> list[Path]:
        if not self.config.rollout_dir.exists():
            return []
        now = time.time()
        return [
            path for path in sorted(self.config.rollout_dir.glob("*.npz"))
            if path not in self.seen and now - path.stat().st_mtime >= self.config.min_archive_age_seconds
        ]

    def _process_batch(self, paths: list[Path]) -> None:
        batch_number = self.completed_batches + 1
        batch_dir = self.config.run_dir / "batches"
        report_path = self.config.run_dir / "evaluations" / f"batch_{batch_number:03d}.json"
        merged_path = batch_dir / f"batch_{batch_number:03d}.npz"
        checkpoint_path = self.config.run_dir / "checkpoints" / f"ppo_{batch_number:03d}.pt"
        report = evaluate_rollouts(paths, report_path)
        self._audit("evaluation_completed", batch=batch_number, archives=[str(path) for path in paths], aggregate=report["aggregate"], report=str(report_path))
        merge_rollouts(paths, merged_path)
        metrics = train_ppo(
            load_rollouts(merged_path),
            self.current_checkpoint,
            checkpoint_path,
            select_device(self.config.device),
            PPOConfig(epochs=self.config.epochs),
        )
        self.current_checkpoint = checkpoint_path
        self.completed_batches += 1
        self._audit(
            "ppo_completed", batch=batch_number, merged_rollout=str(merged_path),
            checkpoint=str(checkpoint_path), metrics=metrics,
        )

    def run(self) -> dict[str, object]:
        """Watch for new archives until a configured terminal condition occurs."""
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.config.include_existing and self.config.rollout_dir.exists():
            self.seen.update(self.config.rollout_dir.glob("*.npz"))
        self._audit("supervisor_started", config={key: str(value) if isinstance(value, Path) else value for key, value in asdict(self.config).items()})
        started = time.monotonic()
        reason = "unknown"
        while True:
            if self.stop_path.exists():
                reason = "stop file present"
                break
            if time.monotonic() - started >= self.config.max_hours * 3600:
                reason = "max hours reached"
                break
            if self.completed_batches >= self.config.max_batches:
                reason = "checkpoint updated; restart the policy bridge with the new checkpoint before another PPO batch"
                break
            if self.failures >= self.config.max_failures:
                reason = "max failures reached"
                break
            for path in self._new_archives():
                self.seen.add(path)
                _, result = validate_rollout(path, self.config)
                if result["accepted"]:
                    self.pending.append(path)
                    self._audit("rollout_accepted", rollout=str(path), **result)
                else:
                    self._audit("rollout_rejected", rollout=str(path), **result)
            while len(self.pending) >= self.config.batch_archives and self.completed_batches < self.config.max_batches:
                paths, self.pending = self.pending[: self.config.batch_archives], self.pending[self.config.batch_archives :]
                try:
                    self._process_batch(paths)
                except (OSError, ValueError, RuntimeError) as error:
                    self.failures += 1
                    self._audit("batch_failed", batch=self.completed_batches + 1, archives=[str(path) for path in paths], error=str(error), failures=self.failures)
            if self.config.poll_seconds:
                time.sleep(self.config.poll_seconds)
        result = {
            "reason": reason,
            "completed_batches": self.completed_batches,
            "failures": self.failures,
            "current_checkpoint": str(self.current_checkpoint),
            "audit": str(self.audit_path),
        }
        self._audit("supervisor_stopped", **result)
        return result


def run_supervisor(config: SupervisorConfig) -> dict[str, object]:
    return OvernightSupervisor(config).run()
