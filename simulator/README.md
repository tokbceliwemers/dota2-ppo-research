# Deterministic Dota lane simulator

This folder contains a fast, non-rendered **approximation** for the Shadow
Fiend lane/last-hit curriculum. It is independent of the Dota client: it does
not load, patch, inject into, control, or connect to `server.dll`.

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

`lane_simulator.py` uses the same 18-value observation contract and 24-action
vocabulary as `replay_training`, while masking unsupported ability/item actions.

## Calibrate from local Dota

Run the loopback bridge with `--calibration data/calibration/lane.jsonl`, then
analyse it with `dota-ppo analyze-calibration ...`. The resulting JSON report
can be loaded without guessing unmeasured values:

```python
from pathlib import Path
from lane_simulator import LaneConfig

config = LaneConfig.from_calibration(Path("lane_calibration.json"))
```
