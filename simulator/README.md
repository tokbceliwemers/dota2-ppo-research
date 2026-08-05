# Deterministic Dota lane simulator

This folder contains fast, non-rendered **approximations** for the Shadow Fiend
lane/last-hit curriculum. It is independent of the Dota client: it does not
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
Set-Location C:\Users\skaya\Desktop\dota2\simulator
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
`terrain_lane_v1` observation exposes the target's 20-segment health bar and
loss rate, never exact target health.

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
