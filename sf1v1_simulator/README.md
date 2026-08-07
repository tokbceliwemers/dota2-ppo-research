# Approximate Shadow Fiend 1v1 simulator

This Stage 2 folder contains fast, non-rendered **approximations** for the
Shadow Fiend 1v1 curriculum. The existing lane/last-hit environment is a
bootstrap component, not 1v1 evidence. It is independent of the Dota client: it does not
load, patch, inject into, control, or connect to `server.dll`.

## What static IDA analysis contributed

The inspected `server.dll` exposes strings confirming that authoritative lane
orders are range-gated, an attack phase can be interrupted when a target moves
out of range, attack range is modifier-adjusted and buffered, and the server
owns `host_timescale`. Those facts shape this simulator's order of operations.
The numerical settings below are calibration values, not copied server code or
claimed exact game constants.

## Boundaries

- This is for fast PPO pretraining and unit tests only.
- Its `source` is `headless_lane_simulator`, never
  `local_instrumented_lobby`.
- A policy must still be evaluated via the instrumented local Dota addon
  before it is considered a real-Dota checkpoint.

## Run

```powershell
Set-Location C:\Users\skaya\Desktop\dota2\sf1v1_simulator
python -m unittest -v
python benchmark.py --environments 4096 --steps 1000
```

`lane_simulator.py` is the original small vectorized reference.  The new
`dota_lane_env.py` provides the normal Gymnasium API for individual episodes:

```python
from dota_lane_env import DotaTerrainLaneEnv

env = DotaTerrainLaneEnv()
observation, info = env.reset(seed=7)
observation, reward, terminated, truncated, info = env.step(9)  # attack
```

`DotaTerrainLaneEnv` uses `data/dota_heightmap.npy` plus its metadata for
bilinear ground height, slope-gated movement, map bounds, and height-aware
terrain context. When present, it also reads the original static
`data/dota.gnv` GridNav asset and requires its walkability bit for every spawn
and movement segment. It models one 4-creep wave and supplies a 24-action mask;
unsupported ability and item actions remain masked. Its 25-value
`lane_v3` observation matches the local bridge layout, exposes the target's
20-segment health bar and loss rate, and never exact target health.

When `data/npc_data.json` is available, the wave is sourced from the exported
Source 2 NPC definitions: three 550-health melee creeps and one 300-health
ranged creep per side, with their recorded regeneration, attack damage,
attack period, range, and movement speed. Creep combat is deterministic and
non-rendered. Projectile travel, aggro switching, armour, abilities, and other
server modifiers remain deliberately out of scope until measured locally.

The matching observation layout means a terrain-pretrained policy checkpoint
may be tried in the local bridge, but the source remains
`terrain_headless_gymnasium`; it is never local-Dota PPO evidence and must pass
the normal local A/B evaluation before it is kept.

## PPO pretraining

`terrain_ppo.py` collects Gymnasium trajectories and performs masked PPO while
keeping its source separate from local rollouts. For a short smoke run:

```powershell
python terrain_ppo.py `
  --checkpoint ..\sf1v1_training\checkpoints\lane_expert_bc_v3.pt `
  --output ..\sf1v1_training\checkpoints\terrain_gym_candidate.pt `
  --updates 4 --environments 16 --horizon 96 --epochs 4 --device cuda
```

The environment itself is CPU-based because it evaluates the real heightmap
and GridNav. For maximum bulk throughput, the existing CUDA-only
`dota-ppo headless-lane-ppo` approximation remains the faster initializer.
The next required step for either candidate is the normal human-started local
bridge evaluation—not `dota-ppo ppo` on synthetic data.

The heightmap is **not** the original Dota navigation mesh. `dota.gnv` improves
static walkability constraints but is a grid rather than a full server-physics
model, so this still does not claim exact walkability or server physics. The Gym environment has a separate
observation version and its episodes must not be written as
`local_instrumented_lobby` PPO data. Use it for fast offline pretraining, then
validate candidates in the human-started local custom lobby.

## Calibrate from local Dota

Run the loopback bridge with `--calibration data/calibration/lane.jsonl`, then
analyse it with `dota-ppo analyze-calibration ...`. The resulting JSON report
can be loaded without guessing unmeasured values:

```python
from pathlib import Path
from lane_simulator import LaneConfig

config = LaneConfig.from_calibration(Path("lane_calibration.json"))
```
