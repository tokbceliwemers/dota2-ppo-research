"""Small, explicit loader for exported Source 2 lane-unit definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class UnitStats:
    health: float
    health_regen: float
    attack_damage_min: float
    attack_damage_max: float
    attack_rate: float
    attack_range: float
    movement_speed: float

    @property
    def mean_attack_damage(self) -> float:
        return (self.attack_damage_min + self.attack_damage_max) / 2.0


@dataclass(frozen=True)
class LaneUnitDefinitions:
    """The four static lane-unit definitions needed by the narrow curriculum."""

    melee: UnitStats
    ranged: UnitStats
    source: str

    @classmethod
    def from_data_directory(cls, directory: Path) -> "LaneUnitDefinitions":
        path = directory / "npc_data.json"
        if not path.exists():
            return cls(_fallback_melee(), _fallback_ranged(), "builtin_fallback")
        payload = json.loads(path.read_text(encoding="utf-8"))
        units = payload["units"]["npc_units.txt"]["DOTAUnits"]
        return cls(
            _unit_stats(units["npc_dota_creep_goodguys_melee"]),
            _unit_stats(units["npc_dota_creep_goodguys_ranged"]),
            "source2_npc_units",
        )

    def wave(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return 3 melee + 1 ranged per-unit simulation arrays."""
        kinds = (self.melee, self.melee, self.melee, self.ranged)
        return tuple(np.asarray([getattr(kind, field) for kind in kinds], dtype=np.float32) for field in (
            "health", "health_regen", "mean_attack_damage", "attack_rate", "attack_range",
        ))


def _unit_stats(raw: dict[str, object]) -> UnitStats:
    def number(name: str) -> float:
        value = raw.get(name)
        if value is None:
            raise ValueError(f"lane unit is missing {name}")
        return float(value)
    return UnitStats(number("StatusHealth"), number("StatusHealthRegen"),
                     number("AttackDamageMin"), number("AttackDamageMax"),
                     number("AttackRate"), number("AttackRange"), number("MovementSpeed"))


def _fallback_melee() -> UnitStats:
    return UnitStats(550.0, 0.5, 19.0, 23.0, 1.0, 100.0, 325.0)


def _fallback_ranged() -> UnitStats:
    return UnitStats(300.0, 2.0, 21.0, 26.0, 1.0, 500.0, 325.0)
