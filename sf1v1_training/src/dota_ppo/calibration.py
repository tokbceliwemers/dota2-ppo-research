"""Read-only analysis of live local-lobby lane calibration telemetry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

SCHEMA_VERSION = 1
SOURCE = "local_lane_calibration"
NUMERIC_FIELDS = {
    "game_time", "target_distance", "attack_range", "attack_damage", "hero_move_speed",
    "target_health", "target_max_health", "hero_health", "hero_max_health", "reward",
    "enemy_count", "ally_count",
}
OPTIONAL_NUMERIC_FIELDS = {"hero_attack_cooldown", "attack_recovery", "in_attack_range"}


def validate_event(event: dict[str, object]) -> dict[str, object]:
    if event.get("schema_version") != SCHEMA_VERSION or event.get("event") != "transition":
        raise ValueError("unsupported calibration event schema")
    if not isinstance(event.get("action_name"), str):
        raise ValueError("calibration event requires action_name")
    for key in NUMERIC_FIELDS:
        if key not in event:
            raise ValueError(f"calibration event is missing {key}")
        value = float(event[key])
        if not np.isfinite(value):
            raise ValueError(f"calibration event has non-finite {key}")
    for key in OPTIONAL_NUMERIC_FIELDS:
        if key in event and not np.isfinite(float(event[key])):
            raise ValueError(f"calibration event has non-finite {key}")
    if not isinstance(event.get("last_hit"), bool) or not isinstance(event.get("hero_dead"), bool):
        raise ValueError("calibration event requires last_hit and hero_dead booleans")
    return event


def load_events(paths: Iterable[Path]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("event must be an object")
                    events.append(validate_event(event))
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    if not events:
        raise ValueError("no calibration events were found")
    return events


def _median(values: list[float]) -> float | None:
    return None if not values else float(np.median(np.asarray(values, dtype=np.float64)))


def _normalized_lane_count(value: object) -> float:
    """Map a live 0..4 creep count to the simulator's 0.25..1.0 feature."""
    return float(np.clip(float(value) / 4.0, 0.0, 1.0))


def analyze_events(paths: Iterable[Path], output: Path | None = None) -> dict[str, object]:
    events = load_events(paths)
    ordered = sorted(events, key=lambda row: float(row["game_time"]))
    gaps = [float(right["game_time"]) - float(left["game_time"]) for left, right in zip(ordered, ordered[1:])]
    gaps = [gap for gap in gaps if 0.01 <= gap <= 2.0]
    measured_tick = _median(gaps)
    targeted = [row for row in events if int(row.get("target_entindex", -1)) >= 0 and float(row["target_max_health"]) > 1]
    attacks = [row for row in targeted if row["action_name"] == "attack"]
    last_hits = [row for row in targeted if bool(row["last_hit"])]
    deaths = [row for row in events if bool(row["hero_dead"])]
    # Values are exported in the unit system of simulator/LaneConfig.  A
    # parameter appears only when a direct live observation supports it.
    max_health = [float(row["target_max_health"]) for row in targeted]
    damage_fractions = [float(row["attack_damage"]) / float(row["target_max_health"])
                        for row in attacks if float(row["target_max_health"]) > 0]
    last_hit_fractions = [float(row["target_health"]) / float(row["target_max_health"])
                          for row in last_hits if float(row["target_max_health"]) > 0]
    attack_cooldowns = [float(row["hero_attack_cooldown"]) for row in events
                        if "hero_attack_cooldown" in row and float(row["hero_attack_cooldown"]) > 0]
    creep_damage_rates: list[float] = []
    hero_damage_rates: list[float] = []
    for left, right in zip(ordered, ordered[1:]):
        delta_time = float(right["game_time"]) - float(left["game_time"])
        same_target = int(left.get("target_entindex", -1)) >= 0 and left.get("target_entindex") == right.get("target_entindex")
        if not same_target or not 0.01 <= delta_time <= 2.0:
            continue
        if left["action_name"] != "attack" and float(left["target_max_health"]) > 1:
            creep_loss = (float(left["target_health"]) - float(right["target_health"])) / float(left["target_max_health"])
            if 0 < creep_loss < 0.5:
                # The simulator models this as base_damage * (1 + ally_count),
                # where ally_count is normalized from Dota's 0..4 raw count.
                scale = 1.0 + _normalized_lane_count(left["ally_count"])
                creep_damage_rates.append(creep_loss * (measured_tick or delta_time) / delta_time / scale)
        if float(left["target_distance"]) <= float(left["attack_range"]) and float(left["hero_max_health"]) > 0:
            hero_loss = (float(left["hero_health"]) - float(right["hero_health"])) / float(left["hero_max_health"])
            if 0 < hero_loss < 0.5:
                scale = max(0.25, _normalized_lane_count(left["enemy_count"]))
                hero_damage_rates.append(hero_loss * (measured_tick or delta_time) / delta_time / scale)
    measured = {
        "tick_seconds": measured_tick,
        "hero_move_speed": _median([float(row["hero_move_speed"]) for row in events]),
        "hero_attack_range": _median([float(row["attack_range"]) for row in events if float(row["attack_range"]) > 0]),
        "hero_attack_damage": _median(damage_fractions),
        "last_hit_health": _median(last_hit_fractions),
        "creep_passive_damage": _median(creep_damage_rates),
        "hero_damage_near_creep": _median(hero_damage_rates),
        "hero_attack_cooldown": _median(attack_cooldowns),
    }
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "events": len(events),
        "targeted_events": len(targeted),
        "attack_actions": len(attacks),
        "last_hits": len(last_hits),
        "hero_deaths": len(deaths),
        "median_target_max_health": _median(max_health),
        "measured_lane_config": {key: value for key, value in measured.items() if value is not None},
        "unmeasured_lane_config": [key for key in ("attack_range_buffer", "hero_attack_cooldown") if key not in measured or measured[key] is None],
        "note": "Measured values calibrate an approximation only; this report is not an exact Dota server specification.",
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["output"] = str(output)
    return report
