"""Small, deployable action vocabulary shared by replay labelling and the bot bridge."""

from __future__ import annotations

import math

ACTION_NAMES = (
    "idle",
    "move_north",
    "move_north_east",
    "move_east",
    "move_south_east",
    "move_south",
    "move_south_west",
    "move_west",
    "move_north_west",
    "attack",
    "stop",
    "hold",
    "ability_1",
    "ability_2",
    "ability_3",
    "ability_4",
    "ability_5",
    "ability_6",
    "item_1",
    "item_2",
    "item_3",
    "item_4",
    "item_5",
    "item_6",
)
ACTION_DIM = len(ACTION_NAMES)
ACTION_IDS = {name: index for index, name in enumerate(ACTION_NAMES)}


def movement_action(dx: float, dy: float, deadzone: float = 120.0) -> int:
    """Quantize a replay position displacement into the small movement vocabulary."""
    if math.hypot(dx, dy) < deadzone:
        return 0
    # atan2 is east=0; rotate to the action order above (north, north-east, ...).
    octant = int(round((math.pi / 2 - math.atan2(dy, dx)) / (math.pi / 4))) % 8
    return octant + 1
