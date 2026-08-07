# Shadow Fiend 1v1 Research

Local-only research tooling for a Dota 2 Shadow Fiend agent that learns a
controlled 1v1 matchup. It is not a public-match, ranked, or unattended
deployment bot.

## Current status — honest scope

The project is in **Stage 3**. It has a locally instrumented Shadow Fiend lane
foundation and a new `sf1v1_passive_v1` contract that spawns a visible,
passive enemy Shadow Fiend. The active model receives 32 observable features
and produces masked discrete actions through a loopback-only bridge.

It is **not yet a complete 1v1 bot**: the passive opponent does not yet use a
frozen combat policy, the controlled policy cannot yet choose enemy-hero
attacks, and no self-play or public-game deployment exists. Historical lane
rollouts are useful diagnostics, not evidence of 1v1 performance.

## Stages

| Folder | Stage | Purpose |
| --- | --- | --- |
| `replay/` → `sf1v1_replays/` | 1 | Offline replay parsing and behavior bootstrap. The rename completes after the current Codex workspace closes. |
| `sf1v1_simulator/` | 2 | Fast approximate 1v1 initialization only; never real-Dota PPO data. |
| `sf1v1_training/` | 3 | Current: exact local-Dota 1v1 environment, rollout collection, and PPO. |
| `sf1v1_evaluation/` | 4 | Frozen-opponent and consent-based local evaluation. |

The complete curriculum, evidence requirements, and promotion rules are in
[PROJECT_PLAN.md](PROJECT_PLAN.md).

## Safety boundary

- Local `rl_ppo_local` custom games and loopback services only.
- No matchmaking, public lobbies, UI automation, binary modification, protocol
  injection, or remote game-server control.
- Replay, simulator, and exact local-Dota data remain separate sources.

## Quick start

```powershell
Set-Location C:\Users\skaya\Desktop\dota2\sf1v1_training
python -m pip install -e .
pytest -q
```

The lane drill is retained as a regression-tested bootstrap. The next work is
to make combat target-aware, add a frozen scripted enemy SF, and use causal
damage/death/last-hit telemetry before attempting meaningful PPO improvement.

## Data and checkpoint boundary

Git intentionally excludes replay files, local rollouts, checkpoints, logs,
and reports. They can contain large files or local-only material and are not
needed to reproduce the source tree.

- `sf1v1_replays` / current `replay`: behavior-cloning and analysis data only.
- `sf1v1_simulator`: fast approximation and candidate initialization only.
- `sf1v1_training/data/training`: fresh exact local-Dota trajectories eligible
  for PPO after validation.
- `sf1v1_training/data/evaluations`: held-out measurement; never PPO input.

See [sf1v1_training/SF1V1_ROLLOUTS.md](sf1v1_training/SF1V1_ROLLOUTS.md) for
the archive contract and [PROJECT_PLAN.md](PROJECT_PLAN.md) for the required
evidence before a candidate can be promoted.

## Optional local console companion

[tools/vconsole2](tools/vconsole2) contains the versioned, dependency-free
localhost MCP companion used for bounded local collection. It is optional and
does not automate a VConsole window, alter binaries, or connect to public
servers. Its setup file uses placeholder paths; configure it only for your own
local Dota Tools installation.

See [CONTRIBUTING.md](CONTRIBUTING.md) for reproducibility and reporting
guidance.
