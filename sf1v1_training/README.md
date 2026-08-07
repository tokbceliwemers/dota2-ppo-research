# Shadow Fiend 1v1 local training

> **Current contract:** use [SF1V1_ROLLOUTS.md](SF1V1_ROLLOUTS.md) for all
> new collection and PPO commands. The lane-v2/v3 sections below are retained
> as historical evidence and are not compatible with the active 1v1 bridge.

This Stage 3 project turns the `.dem` corpus in `../sf1v1_replays` (currently
named `../replay` until the workspace rename completes) into a CUDA PyTorch
bootstrap, then builds and trains an **offline/local custom-lobby** Shadow
Fiend 1v1 curriculum. It deliberately does not target public matchmaking.

## Important training boundary

Valve replay files provide game state, combat, and sampled positions, but not a
reliable player command stream. Consequently, this project infers a small
movement label from position transitions only. That is useful for behavior
cloning, not on-policy PPO. PPO needs the observation, valid-action mask,
sampled action, reward, terminal flag, policy log-probability, and critic value
recorded from the policy that just played the local lobby. `dota-ppo ppo`
rejects replay-derived archives by design.

The discrete policy space includes `idle`, eight movement directions, attack,
stop/hold, ability slots 1–6, and item slots 1–6. Target coordinates/entities
remain separate fields and must be supplied by the local bot bridge; use action
masks until each semantic action can be executed and logged exactly.

## Install

```powershell
Set-Location C:\Users\skaya\Desktop\dota2\sf1v1_training
python -m pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The machine currently has a CUDA-enabled PyTorch build and an RTX 3080, so the
default `--device cuda` will use it.

## Replay bootstrap

```powershell
# Parse the existing real replay corpus with gem-dota (one subdirectory per match).
dota-ppo parse ..\replay\data\raw\shadow_fiend_mid --output data\parsed

# Generate approximate Shadow Fiend movement demonstrations.
dota-ppo dataset data\parsed --hero nevermore --output data\replay_bootstrap.npz

# Train the initial actor-critic on the demonstrations on CUDA.
dota-ppo bc data\replay_bootstrap.npz --output checkpoints\nevermore_bc.pt --epochs 50
```

## Historical fast lane-v3 warm start

The real Dota client remains the source of exact PPO trajectories and runs at
real time. To avoid spending those runs teaching the policy the obvious
approach-and-last-hit pattern, generate a fast **synthetic demonstration** set
and behavior-clone it first. This data is explicitly not accepted by the PPO
command.

```powershell
dota-ppo synthetic-lane-data --output data/synthetic_lane_v3.npz --steps 100000
dota-ppo bc data/synthetic_lane_v3.npz --output checkpoints/lane_expert_bc_v3.pt --epochs 30
```

`lane_expert_bc_v3.pt` uses the retired 25-feature lane-v3 contract: hero features,
nearest-creep geometry, a quantized visible health bar and its recent change,
hero damage/range/recovery, allied-creep pressure, nearest allied-creep
geometry/health, and local creep counts. It deliberately does not contain an
exact last-hit-ready feature. Use it only as the warm start for subsequent
exact local PPO collection.

The dataset sidecar, `data/replay_bootstrap.json`, records the source matches,
feature width, and the approximation warning. Re-run parsing only when needed:
existing `positions.parquet` files are reused.

## Human-control labels: normalize to one layout

Do not train on a player's personal hotkeys. Record the **semantic order** from
a controlled local lobby alongside their raw key for audit, then map every
order to this deployment layout:

| Semantic order | Canonical key |
| --- | --- |
| Ability slots 1–6 | `Q W E R D F` |
| Inventory slots 1–6 | `Z X C V B N` |
| Attack / stop / hold | `A` / `S` / `H` |
| Move | `MOUSE_RIGHT` |

For example, a participant pressing their personal `I` binding to activate an
item in inventory slot 1 becomes `action_label="item_1"`,
`canonical_key="Z"`; `raw_key="I"` remains in the export only for audit.
The export also gives this order a stable `action_id` for the policy action
space. Movement has no fixed `action_id` until its target is joined with the
hero position and quantized to one of the eight movement directions.

Create a JSONL sidecar from your local-lobby order logger (one record per
accepted order):

```json
{"tick":12030,"player_id":1,"raw_key":"I","order":"use_item","inventory_slot":1,"target_kind":"position","target_x":250,"target_y":-700}
{"tick":12045,"player_id":1,"raw_key":"Q","order":"cast_ability","ability_slot":1,"target_kind":"entity","target_entity_id":71}
```

```powershell
dota-ppo canonicalize-inputs data\human_orders.jsonl --output data\human_orders.parquet

# Match the canonical orders to the sampled state from that same local match.
dota-ppo join-human-orders data\parsed\local_match_001 data\human_orders.parquet `
  --output data\human_behavior_cloning.npz

# This dataset can train item/ability labels as well as movement labels.
dota-ppo bc data\human_behavior_cloning.npz --output checkpoints\human_bc.pt --epochs 50
```

The converter rejects bare key presses: a physical key alone does not identify
an in-game order or target. Public `.dem` files also generally do not contain
these human-order records; collect them only with participants' consent in
your own local custom lobbies. Supply `--layout my_layout.json` only when the
bot's deployment bindings intentionally differ from this canonical layout.
`join-human-orders` requires matching `tick` and `player_id` values from the
same local match; it rejects labels more than 30 ticks from a sampled state by
default rather than silently pairing them with the wrong observation.

## Local-lobby PPO collection contract

Instrument a Dota 2 custom-game/bot adapter to write an `.npz` archive with
these arrays after each policy batch:

| Array | Shape | Meaning |
| --- | --- | --- |
| `observations` | `[T, 32]` | 10 hero features, 15 lane features, and 7 visible passive-opponent features exactly as sent to the model |
| `actions` | `[T]` | Actions sampled from the current policy |
| `rewards` | `[T]` | Environment reward (not replay heuristic) |
| `dones` | `[T]` | True when a game/episode ends |
| `old_log_probs` | `[T]` | Log-probability under the behavior policy |
| `old_values` | `[T]` | Value estimate when the action was sampled |
| `action_masks` | `[T, 24]` optional | Valid actions at each step |
| `game_times` | `[T]` optional | Dota game-time when each action was sampled; used to verify collection throughput |
| `metadata` | scalar JSON | Must contain source, reward/observation version, policy checkpoint SHA-256, environment, and `collection_role` |

Use the exact same observation encoder and action mask in the adapter and
trainer. A simple reward could combine team net-worth change, objectives, tower
damage, survival, and a large terminal win/loss component; its final definition
belongs in the local adapter so it reflects what actually occurred.

### Run the local collector

The included loopback bridge owns the action sampling and records the exact
behavior-policy fields required by PPO. Start it before your **local** custom
lobby (the server listens only on `127.0.0.1`):

```powershell
dota-ppo bridge checkpoints\sf1v1_passive_bc_v1.pt `
  --rollouts data\training\batch_{batch:03d}.npz `
  --human-orders data\human_orders.jsonl
```

Use the reference addon in [dota_addon](dota_addon) as the custom-game side.
It calls `/act`, executes the returned action, sends `/transition` with the
observed reward, then calls `/flush` at episode end. The bridge writes a
PPO-compatible archive only after every sampled action has a committed reward.
Run `dota-ppo ppo` on that archive. The adapter now includes a last-hit lane
drill: it enables attack only when an enemy creep is in range, while ability and
item actions remain masked until target-selection heads are implemented.

### Last-hit curriculum and batch collection

The local addon runs repeatable Shadow Fiend lane drills under
`lane_wave_clear_v4_fixed_progression`. It spawns a tracked enemy wave, gives
`+1` for each hero last hit and `-2` for hero death, records every drill
boundary as a PPO terminal state, then resets Shadow Fiend to Level 1, zero XP,
and zero gold before starting a fresh wave. See `PROJECT_PLAN.md` for the
current evidence gate.

For repeatable evaluation, the addon starts a 160-game-second batch timer on
launch and on `rl_ppo_restart`. It prints a 30-second warning and then saves
the rollout. Use `rl_ppo_timer` to print elapsed/remaining time. An early
`rl_ppo_finish` produces an incomplete batch, so it is not valid for the
current 150-game-second comparison gate. Collection remains human-operated;
the bridge does not remotely control Dota.

You can also combine separately collected archives:

```powershell
dota-ppo merge-rollouts data\rollouts\episode_*.npz --output data\rollouts\lane_batch_001.npz
dota-ppo ppo data\rollouts\lane_batch_001.npz `
  --checkpoint checkpoints\movement_bc.pt `
  --output checkpoints\lane_ppo_001.pt
```

```powershell
dota-ppo ppo data\rollouts\batch_0001.npz `
  --checkpoint checkpoints\nevermore_bc.pt `
  --output checkpoints\nevermore_ppo_0001.pt --epochs 10
```

### Reproducible lane evaluation

Do not compare one rollout to another by eye. Keep the hero, map, episode
length, bot setting, and reward version fixed; collect several saved rollouts
for each frozen checkpoint, then summarize each set separately:

```powershell
dota-ppo evaluate-rollouts data/rollouts/baseline_001.npz data/rollouts/baseline_002.npz `
  --output reports/baseline_metrics.json
dota-ppo evaluate-rollouts data/rollouts/lane_ppo_001_001.npz data/rollouts/lane_ppo_001_002.npz `
  --output reports/lane_ppo_001_metrics.json
```

Compare `reward_per_step`, `mean_archive_episode_reward`,
`last_hit_reward_signals`, `death_reward_signals`, and attack metrics. Promote
a checkpoint only when the PPO set improves across repeated local runs, not
because of one unusually good rollout.

New lane-v3 archives also report `decisions_per_game_second` and decision-gap
metrics. At the 0.25 game-second cadence, a healthy collector is close to four
decisions per game-second. Check this after using `rl_ppo_speed_4`: game speed
may reduce wall-clock collection time, but it must not reduce game-time decision
coverage.

## Export and local deployment bridge

```powershell
dota-ppo export checkpoints\nevermore_ppo_0001.pt --output deploy\nevermore_policy.ts
dota-ppo serve deploy\nevermore_policy.ts --device cuda
```

`serve` speaks newline-delimited JSON on stdin/stdout:

```json
{"observation":[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
```

It responds with an action index and name. A Dota 2 Lua bot script cannot load
PyTorch directly, so a local custom-game bridge must obtain the observation,
send it to this process (or a small local HTTP wrapper), and translate the
response into safe bot API calls. Keep that bridge restricted to your local
custom lobbies and log every transition for PPO.
