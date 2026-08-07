"""Convert controlled-lobby human order logs to canonical training labels."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from .actions import movement_action
from .controls import canonicalize_input
from .data import Trajectories, save_trajectories
from .observations import with_lane_features


def load_layout(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in payload.items()):
        raise ValueError("layout JSON must be an object mapping semantic labels to keys")
    return payload


def canonicalize_jsonl(source: Path, output: Path, layout_path: Path | None = None) -> dict[str, object]:
    """Write canonical labels to Parquet, preserving raw input for audit only."""
    layout = load_layout(layout_path)
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                label = canonicalize_input(record, layout)
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                raise ValueError(f"invalid input log at {source}:{line_number}: {error}") from error
            rows.append(asdict(label))
    if not rows:
        raise ValueError(f"{source} contains no input records")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["tick", "player_id"]).to_parquet(output, index=False)
    return {"records": len(rows), "output": str(output), "canonical_layout": str(layout_path) if layout_path else "default"}


def _minute_value(minutes: pd.DataFrame, tick: int, name: str) -> float:
    if minutes.empty or name not in minutes:
        return 0.0
    earlier = minutes[minutes["tick"] <= tick]
    return float(earlier.iloc[-1][name]) if not earlier.empty else 0.0


def join_orders_to_states(parsed_match: Path, labels: Path, output: Path, *, max_tick_gap: int = 30) -> dict[str, object]:
    """Align local semantic orders to parsed state and make a BC training archive.

    ``tick`` and ``player_id`` must use the same local-lobby clock and player
    IDs in both files. Orders beyond ``max_tick_gap`` from a sampled position
    are rejected rather than silently creating an incorrect demonstration.
    """
    if max_tick_gap < 0:
        raise ValueError("max_tick_gap must be non-negative")
    positions = pd.read_parquet(parsed_match / "positions.parquet")
    minute_path = parsed_match / "players_minute.parquet"
    minutes = pd.read_parquet(minute_path) if minute_path.exists() else pd.DataFrame()
    order_rows = pd.read_parquet(labels).sort_values(["player_id", "tick"])
    required = {"tick", "player_id", "order", "action_id", "target_x", "target_y"}
    missing = required - set(order_rows.columns)
    if missing:
        raise ValueError(f"label parquet is missing {sorted(missing)}")
    observations: list[np.ndarray] = []
    actions: list[int] = []
    unmatched = 0
    for player_id, group in order_rows.groupby("player_id", sort=False):
        player_positions = positions[positions["player_id"] == player_id].sort_values("tick").drop_duplicates("tick").reset_index(drop=True)
        if player_positions.empty:
            unmatched += len(group); continue
        player_minutes = minutes[minutes["player_id"] == player_id].sort_values("tick") if not minutes.empty and "player_id" in minutes else pd.DataFrame()
        ticks = player_positions["tick"].to_numpy(dtype=np.int64)
        xs = player_positions["x"].fillna(0).to_numpy(dtype=np.float32)
        ys = player_positions["y"].fillna(0).to_numpy(dtype=np.float32)
        duration = max(float(ticks[-1] - ticks[0]), 1.0)
        for row in group.itertuples(index=False):
            order_tick = int(row.tick)
            index = int(np.abs(ticks - order_tick).argmin())
            if abs(int(ticks[index]) - order_tick) > max_tick_gap:
                unmatched += 1; continue
            action_id = None if pd.isna(row.action_id) else int(row.action_id)
            if row.order == "move":
                if pd.isna(row.target_x) or pd.isna(row.target_y):
                    unmatched += 1; continue
                action_id = movement_action(float(row.target_x) - float(xs[index]), float(row.target_y) - float(ys[index]), deadzone=1.0)
            if action_id is None:
                unmatched += 1; continue
            previous = max(index - 1, 0)
            dt = max(float(ticks[index] - ticks[previous]) / 30.0, 1 / 30)
            net_worth = _minute_value(player_minutes, int(ticks[index]), "net_worth")
            last_hits = _minute_value(player_minutes, int(ticks[index]), "lh")
            xp = _minute_value(player_minutes, int(ticks[index]), "xp")
            base_observation = np.array((xs[index] / 8192, ys[index] / 8192, (xs[index] - xs[previous]) / dt / 550,
                                    (ys[index] - ys[previous]) / dt / 550, (ticks[index] - ticks[0]) / duration,
                                    net_worth / 30000, last_hits / 400, xp / 30000, 1.0,
                                    np.hypot(xs[index], ys[index]) / 11585), dtype=np.float32)
            observations.append(with_lane_features(base_observation)); actions.append(action_id)
    if not observations:
        raise ValueError("no human orders could be aligned to parsed states; check tick/player_id clocks and max_tick_gap")
    data = Trajectories(np.stack(observations), np.asarray(actions, dtype=np.int64), np.zeros(len(actions), dtype=np.float32),
                        np.array([False] * (len(actions) - 1) + [True]), "human_local_lobby_orders")
    save_trajectories(output, data, {"parsed_match": str(parsed_match), "labels": str(labels), "max_tick_gap": max_tick_gap, "unmatched_orders": unmatched})
    return {"steps": len(actions), "unmatched_orders": unmatched, "output": str(output)}
