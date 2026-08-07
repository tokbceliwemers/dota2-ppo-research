"""Convert gem-dota exports into explicitly approximate replay trajectories.

Source 2 replays expose game state but not a trustworthy player command stream.
The labels here are position-transition *approximations*, intended for behavior
cloning only.  They must never be passed to PPO as if they were on-policy data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .actions import movement_action
from .data import Trajectories, save_trajectories
from .observations import with_lane_features


def parse_dem_files(dem_paths: list[Path], parsed_root: Path) -> list[Path]:
    """Parse `.dem` files with gem-dota into one parquet directory per match."""
    from gem import parse_to_parquet

    outputs: list[Path] = []
    for dem in dem_paths:
        target = parsed_root / dem.stem
        positions = target / "positions.parquet"
        if not positions.exists():
            parse_to_parquet(dem, target)
        outputs.append(target)
    return outputs


def _hero_match(series: pd.Series, hero: str) -> pd.Series:
    query = hero.lower().removeprefix("npc_dota_hero_")
    return series.fillna("").str.lower().str.removeprefix("npc_dota_hero_").eq(query)


def _sample_minute_values(minutes: pd.DataFrame, ticks: np.ndarray, column: str) -> np.ndarray:
    if column not in minutes or minutes.empty:
        return np.zeros(len(ticks), dtype=np.float32)
    ordered = minutes.sort_values("tick")
    source_ticks = ordered["tick"].to_numpy(dtype=np.int64)
    values = ordered[column].fillna(0).to_numpy(dtype=np.float32)
    idx = np.searchsorted(source_ticks, ticks, side="right") - 1
    return np.where(idx >= 0, values[np.maximum(idx, 0)], 0.0)


def trajectories_from_parquet(match_dirs: list[Path], hero: str, stride: int = 1) -> tuple[Trajectories, dict[str, object]]:
    """Create state/action/reward arrays from gem's positions and minute tables.

    Observation layout: normalized x/y, normalized velocity, time progress,
    net worth, last hits, XP, a bias-like alive proxy, and position magnitude.
    Reward is a small normalized net-worth delta plus forward movement; it is a
    replay-learning heuristic, not the final local-lobby reward function.
    """
    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []
    all_dones: list[np.ndarray] = []
    included: list[str] = []
    for match_dir in match_dirs:
        position_path, minute_path = match_dir / "positions.parquet", match_dir / "players_minute.parquet"
        if not position_path.exists():
            continue
        positions = pd.read_parquet(position_path)
        if positions.empty or "hero_name" not in positions or not _hero_match(positions["hero_name"], hero).any():
            continue
        player_positions = positions[_hero_match(positions["hero_name"], hero)].sort_values("tick").drop_duplicates("tick")
        if len(player_positions) < 2:
            continue
        player_id = int(player_positions.iloc[0]["player_id"])
        player_positions = player_positions.iloc[::max(1, stride)].reset_index(drop=True)
        if len(player_positions) < 2:
            continue
        ticks = player_positions["tick"].to_numpy(dtype=np.int64)
        x = player_positions["x"].fillna(0).to_numpy(dtype=np.float32)
        y = player_positions["y"].fillna(0).to_numpy(dtype=np.float32)
        dt = np.maximum(np.diff(ticks, prepend=ticks[0]), 1).astype(np.float32) / 30.0
        dx, dy = np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])
        minutes = pd.read_parquet(minute_path) if minute_path.exists() else pd.DataFrame()
        if not minutes.empty and "player_id" in minutes:
            minutes = minutes[minutes["player_id"] == player_id]
        net_worth = _sample_minute_values(minutes, ticks, "net_worth")
        last_hits = _sample_minute_values(minutes, ticks, "lh")
        xp = _sample_minute_values(minutes, ticks, "xp")
        duration = max(float(ticks[-1] - ticks[0]), 1.0)
        base_obs = np.stack((x / 8192, y / 8192, dx / dt / 550, dy / dt / 550,
                        (ticks - ticks[0]) / duration, net_worth / 30000, last_hits / 400,
                        xp / 30000, np.ones_like(x), np.hypot(x, y) / 11585), axis=1).astype(np.float32)
        obs = with_lane_features(base_obs)
        actions = np.array([movement_action(float(a), float(b)) for a, b in zip(np.diff(x), np.diff(y))], dtype=np.int64)
        # A transition uses state[t] and action inferred from position[t] -> position[t+1].
        rewards = (np.diff(net_worth) / 1000 + np.hypot(np.diff(x), np.diff(y)) / 20000).astype(np.float32)
        dones = np.zeros(len(actions), dtype=bool)
        dones[-1] = True
        all_obs.append(obs[:-1]); all_actions.append(actions); all_rewards.append(rewards); all_dones.append(dones)
        included.append(match_dir.name)
    if not all_obs:
        raise ValueError(f"No usable {hero} position trajectories found under supplied parsed match directories")
    data = Trajectories(np.concatenate(all_obs), np.concatenate(all_actions), np.concatenate(all_rewards), np.concatenate(all_dones), "gem_replay_approximate")
    data.validate()
    return data, {"hero": hero, "matches": included, "observation_dim": int(data.observations.shape[1]), "label_warning": "Actions are inferred from positions; use only for behavior cloning."}


def build_replay_dataset(match_dirs: list[Path], output: Path, hero: str, stride: int = 1) -> dict[str, object]:
    data, metadata = trajectories_from_parquet(match_dirs, hero, stride)
    save_trajectories(output, data, metadata)
    (output.with_suffix(".json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "steps": len(data.observations), "output": str(output)}
