"""Comparable metrics for local Dota PPO rollout archives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .actions import ACTION_IDS
from .data import Rollouts, load_rollout_metadata, load_rollouts


def rollout_metrics(rollout: Rollouts) -> dict[str, object]:
    """Summarize one exact rollout without inventing game-state statistics.

    Last-hit and death counts are deliberately named *signals*: this first lane
    reward version emits +1 and -2, while future reward functions may differ.
    """
    attack_id = ACTION_IDS["attack"]
    rewards = rollout.rewards.astype(np.float64)
    terminal_indexes = np.flatnonzero(rollout.dones)
    episode_rewards: list[float] = []
    start = 0
    for end in terminal_indexes:
        episode_rewards.append(float(rewards[start : end + 1].sum()))
        start = int(end) + 1
    attack_allowed = None
    valid_sampled_actions = None
    if rollout.action_masks is not None:
        attack_allowed = int(rollout.action_masks[:, attack_id].sum())
        valid_sampled_actions = bool(rollout.action_masks[np.arange(len(rollout.actions)), rollout.actions].all())
    attacks = int((rollout.actions == attack_id).sum())
    result: dict[str, object] = {
        "steps": len(rollout.actions),
        "completed_episodes": len(terminal_indexes),
        "reward_total": float(rewards.sum()),
        "reward_per_step": float(rewards.mean()),
        "mean_completed_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else None,
        "attack_actions": attacks,
        "attack_allowed_steps": attack_allowed,
        "attack_rate_when_allowed": None if not attack_allowed else float(attacks / attack_allowed),
        "last_hit_reward_signals": int((rewards >= 0.75).sum()),
        "death_reward_signals": int((rewards <= -1.5).sum()),
        "all_sampled_actions_valid": valid_sampled_actions,
    }
    if rollout.game_times is None:
        result.update({"game_time_span": None, "decisions_per_game_second": None,
                       "mean_decision_interval_game_seconds": None, "max_decision_interval_game_seconds": None})
    else:
        times = rollout.game_times.astype(np.float64)
        gaps = np.diff(times)
        span = float(times[-1] - times[0])
        result.update({
            "game_time_span": span,
            "decisions_per_game_second": None if span <= 0 else float(len(times) / span),
            "mean_decision_interval_game_seconds": None if not len(gaps) else float(gaps.mean()),
            "max_decision_interval_game_seconds": None if not len(gaps) else float(gaps.max()),
        })
    return result


def evaluate_rollouts(paths: list[Path], output: Path | None = None) -> dict[str, object]:
    """Report per-archive and aggregate metrics for a comparable evaluation set."""
    if not paths:
        raise ValueError("supply at least one rollout archive")
    reports: list[dict[str, object]] = []
    for path in paths:
        rollout = load_rollouts(path)
        if rollout.source != "local_instrumented_lobby":
            raise ValueError(f"{path} is not a local instrumented-lobby rollout")
        metadata = load_rollout_metadata(path)
        reports.append({"rollout": str(path), "reward_version": str(metadata.get("reward_version", "unspecified")),
                        "policy_checkpoint_sha256": str(metadata.get("policy_checkpoint_sha256", "unspecified")),
                        **rollout_metrics(rollout)})
    total_steps = sum(int(report["steps"]) for report in reports)
    total_reward = sum(float(report["reward_total"]) for report in reports)
    total_allowed = sum(int(report["attack_allowed_steps"] or 0) for report in reports)
    total_attacks = sum(int(report["attack_actions"]) for report in reports)
    completed_rewards = [float(report["mean_completed_episode_reward"]) for report in reports if report["mean_completed_episode_reward"] is not None]
    timed_reports = [report for report in reports if report["game_time_span"] is not None and float(report["game_time_span"]) > 0]
    timed_steps = sum(int(report["steps"]) for report in timed_reports)
    timed_span = sum(float(report["game_time_span"]) for report in timed_reports)
    result = {
        "archives": len(reports),
        "aggregate": {
            "steps": total_steps,
            "completed_episodes": sum(int(report["completed_episodes"]) for report in reports),
            "reward_total": total_reward,
            "reward_per_step": total_reward / total_steps,
            "mean_archive_episode_reward": float(np.mean(completed_rewards)) if completed_rewards else None,
            "attack_actions": total_attacks,
            "attack_allowed_steps": total_allowed,
            "attack_rate_when_allowed": None if not total_allowed else total_attacks / total_allowed,
            "last_hit_reward_signals": sum(int(report["last_hit_reward_signals"]) for report in reports),
            "death_reward_signals": sum(int(report["death_reward_signals"]) for report in reports),
            "all_sampled_actions_valid": all(report["all_sampled_actions_valid"] is True for report in reports),
            "timed_archives": len(timed_reports),
            "game_time_span": timed_span if timed_reports else None,
            "decisions_per_game_second": timed_steps / timed_span if timed_span > 0 else None,
            "reward_versions": sorted({str(report["reward_version"]) for report in reports}),
            "policy_checkpoint_sha256s": sorted({str(report["policy_checkpoint_sha256"]) for report in reports}),
        },
        "rollouts": reports,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["output"] = str(output)
    return result
