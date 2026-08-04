"""Canonical Dota control labels.

Models learn an in-game *order*, not a volunteer's personal hotkey.  Raw input
is retained only as audit data; ``use_item`` in slot 1 is always labelled ``Z``
under this project's deployment layout, whether the player pressed I, a mouse
button, or used a custom binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import ACTION_IDS


# 1-based slots match the player-facing Dota UI and input-log convention.
DEFAULT_DOTA_LAYOUT: dict[str, str] = {
    "ability_1": "Q", "ability_2": "W", "ability_3": "E", "ability_4": "R",
    "ability_5": "D", "ability_6": "F",
    "item_1": "Z", "item_2": "X", "item_3": "C", "item_4": "V",
    "item_5": "B", "item_6": "N",
    "attack": "A", "stop": "S", "hold": "H", "move": "MOUSE_RIGHT",
}

_SLOTTED_ORDERS = {"cast_ability": "ability", "use_item": "item"}
_SIMPLE_ORDERS = {"move", "attack", "stop", "hold"}


@dataclass(frozen=True)
class CanonicalInput:
    tick: int
    player_id: int
    order: str
    action_label: str
    canonical_key: str
    action_id: int | None
    raw_key: str | None
    target_kind: str | None
    target_x: float | None
    target_y: float | None
    target_entity_id: int | None


def canonicalize_input(record: dict[str, Any], layout: dict[str, str] | None = None) -> CanonicalInput:
    """Validate one instrumented human order and assign its deployment key.

    Required keys are ``tick``, ``player_id``, and ``order``. ``cast_ability``
    requires ``ability_slot`` and ``use_item`` requires ``inventory_slot``;
    both slots are 1-based. The observed ``raw_key`` never changes the label.
    """
    bindings = {**DEFAULT_DOTA_LAYOUT, **(layout or {})}
    try:
        tick, player_id, order = int(record["tick"]), int(record["player_id"]), str(record["order"])
    except KeyError as error:
        raise ValueError(f"missing required input-log field {error.args[0]!r}") from error
    if tick < 0 or player_id < 0:
        raise ValueError("tick and player_id must be non-negative")
    if order in _SLOTTED_ORDERS:
        category = _SLOTTED_ORDERS[order]
        field = "ability_slot" if category == "ability" else "inventory_slot"
        if field not in record:
            raise ValueError(f"{order} requires {field} (1 through 6)")
        slot = int(record[field])
        if not 1 <= slot <= 6:
            raise ValueError(f"{field} must be in 1..6, got {slot}")
        action_label = f"{category}_{slot}"
    elif order in _SIMPLE_ORDERS:
        action_label = order
    else:
        raise ValueError(f"unsupported order {order!r}; record a semantic Dota order, not a key press")
    if action_label not in bindings:
        raise ValueError(f"no deployment binding configured for {action_label}")
    target_kind = str(record["target_kind"]) if record.get("target_kind") is not None else None
    # Move target coordinates are intentionally retained, but turning a target
    # into one of eight policy directions needs the hero's current position.
    # That join happens in trajectory assembly; all other orders have a stable
    # discrete policy action now.
    action_id = ACTION_IDS.get(action_label)
    return CanonicalInput(
        tick=tick, player_id=player_id, order=order, action_label=action_label,
        canonical_key=bindings[action_label], action_id=action_id, raw_key=str(record["raw_key"]) if record.get("raw_key") else None,
        target_kind=target_kind, target_x=float(record["target_x"]) if record.get("target_x") is not None else None,
        target_y=float(record["target_y"]) if record.get("target_y") is not None else None,
        target_entity_id=int(record["target_entity_id"]) if record.get("target_entity_id") is not None else None,
    )
