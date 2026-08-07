# Local custom-lobby adapter

Copy `scripts/vscripts/` into a **new local custom-game addon** under Dota's
`game/dota_addons/<addon-name>/scripts/vscripts/`, then merge the entry point
into that addon's existing mode. Workshop Tools uses this addon layout for
`addon_game_mode.lua`. Do not replace an existing mode without merging it.

Start the Python service before launching the local lobby:

```powershell
dota-ppo bridge checkpoints\lane_expert_bc_v3.pt --rollouts data\rollouts\batch_0001.npz --human-orders data\human_orders.jsonl
```

The service is bound exclusively to `127.0.0.1:8765`. The Lua reference assumes
the local VScript HTTP API can reach that address and that a JSON library named
`json` exists in your addon. Verify this with `/health` before a long session;
some Dota builds/custom-game configurations restrict VScript HTTP requests.

`RLPPOBridge:Start(hero, player_id)` must be called by the addon after it has
created or selected the bot hero. Its `Think()` method must be registered with
the mode's thinker, and `Finish()` must be called at the local episode end.

The current reference runs the versioned
`lane_wave_clear_v4_fixed_progression` last-hit curriculum: opposing creep
waves spawn near the controlled hero, a cleared enemy wave (or the 75-second
fallback) ends the episode, and the next wave resets the hero to Level 1 with
zero XP and gold. Hero last hits receive reward and death is penalized. It
enables idle, movement, and attack only when an enemy creep is nearby.
Abilities and items remain masked until target-selection heads are added and
verified.

The `ExecuteOrderFilter` logs semantic human move, attack, ability, and item
orders with consent; it cannot observe physical keyboard bindings. `raw_key`
is optional in the human log, while inventory/ability slot orders map to the
fixed layout during `dota-ppo canonicalize-inputs`.
