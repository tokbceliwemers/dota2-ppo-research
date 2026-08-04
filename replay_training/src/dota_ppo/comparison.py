"""Paired, local-only evidence gate for a Stage 2 policy comparison.

This module evaluates exact local-lobby archives.  It neither trains a policy
nor claims that an offline simulator checkpoint is real-Dota verified.
"""

from __future__ import annotations

import json
from pathlib import Path

from .evaluation import evaluate_rollouts


def _quality_failures(report: dict[str, object], minimum_archives: int,
                      minimum_game_seconds: float) -> list[str]:
    aggregate = report["aggregate"]
    assert isinstance(aggregate, dict)
    failures: list[str] = []
    if int(report["archives"]) < minimum_archives:
        failures.append(f"requires at least {minimum_archives} archives")
    if not bool(aggregate["all_sampled_actions_valid"]):
        failures.append("contains an action outside its action mask")
    if int(aggregate["completed_episodes"]) < minimum_archives:
        failures.append("does not contain at least one completed episode per archive")
    if int(aggregate["timed_archives"]) != int(report["archives"]):
        failures.append("one or more archives lacks game-time cadence evidence")
    if aggregate["decisions_per_game_second"] is None:
        failures.append("has no decision cadence")
    if len(aggregate["reward_versions"]) != 1:
        failures.append("archives use more than one reward_version")
    if len(aggregate["policy_checkpoint_sha256s"]) != 1 or aggregate["policy_checkpoint_sha256s"] == ["unspecified"]:
        failures.append("archives are not all identified as one policy checkpoint")
    short_archives = [
        str(row["rollout"])
        for row in report["rollouts"]
        if row["game_time_span"] is None or float(row["game_time_span"]) < minimum_game_seconds
    ]
    if short_archives:
        failures.append(
            f"incomplete batch below {minimum_game_seconds:g} game seconds: {', '.join(short_archives)}"
        )
    return failures


def compare_rollout_sets(baseline_paths: list[Path], candidate_paths: list[Path], *, minimum_archives: int = 3,
                         minimum_game_seconds: float = 150.0, output: Path | None = None) -> dict[str, object]:
    """Compare fresh, exact local-lobby runs from a frozen baseline and candidate.

    The result is an evidence report, not a promotion.  Candidate improvement
    requires more last-hit signals, no additional death signals, higher reward
    per step, valid actions, completed episodes, and comparable game-time
    decision cadence.
    """
    if minimum_archives < 1 or minimum_game_seconds <= 0:
        raise ValueError("minimum_archives and minimum_game_seconds must be positive")
    baseline = evaluate_rollouts(baseline_paths)
    candidate = evaluate_rollouts(candidate_paths)
    baseline_quality = _quality_failures(baseline, minimum_archives, minimum_game_seconds)
    candidate_quality = _quality_failures(candidate, minimum_archives, minimum_game_seconds)
    base = baseline["aggregate"]
    cand = candidate["aggregate"]
    assert isinstance(base, dict) and isinstance(cand, dict)
    base_cadence = base["decisions_per_game_second"]
    candidate_cadence = cand["decisions_per_game_second"]
    if base["reward_versions"] != cand["reward_versions"]:
        candidate_quality.append("candidate and baseline use different reward_version values")
    cadence_ratio: float | None = None
    if base_cadence is not None and candidate_cadence is not None and float(base_cadence) > 0:
        cadence_ratio = float(candidate_cadence) / float(base_cadence)
        if not 0.80 <= cadence_ratio <= 1.25:
            candidate_quality.append(
                f"candidate cadence ratio {cadence_ratio:.3f} is outside the comparable 0.80..1.25 range"
            )
    else:
        candidate_quality.append("cannot compare decision cadence")
    reward_delta = float(cand["reward_per_step"]) - float(base["reward_per_step"])
    last_hit_delta = int(cand["last_hit_reward_signals"]) - int(base["last_hit_reward_signals"])
    death_delta = int(cand["death_reward_signals"]) - int(base["death_reward_signals"])
    preliminary_improvement = (
        not baseline_quality and not candidate_quality and reward_delta > 0
        and last_hit_delta > 0 and death_delta <= 0
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "source": "paired_local_lobby_evaluation",
        "minimum_archives": minimum_archives,
        "minimum_game_seconds": minimum_game_seconds,
        "baseline": baseline,
        "candidate": candidate,
        "baseline_quality_failures": baseline_quality,
        "candidate_quality_failures": candidate_quality,
        "comparison": {
            "reward_per_step_delta": reward_delta,
            "last_hit_signal_delta": last_hit_delta,
            "death_signal_delta": death_delta,
            "decision_cadence_ratio": cadence_ratio,
            "preliminary_improvement": preliminary_improvement,
            "promotion": "not granted; collect another independent batch before any Stage 2 promotion",
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["output"] = str(output)
    return result
