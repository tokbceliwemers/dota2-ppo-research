# Dota 2 PPO Research

Experimental, local-only research tooling for a Shadow Fiend reinforcement-learning curriculum in Dota 2.

> **Research status:** this is an in-progress project, currently in the controlled lane last-hit curriculum. It is not a public-match bot, does not support ranked or public matchmaking, and has not reached full-match play.

## What is in this repository

- `replay/` — downloads and records a reproducible corpus of Shadow Fiend replay files, then derives limited behavior-cloning data from replay state.
- `replay_training/` — CUDA/PyTorch behavior cloning, a loopback-only local Dota PPO bridge, PPO rollout validation, local lane calibration, and evaluation tools.
- `simulator/` — a fast non-rendered lane approximation for pretraining only. It is never treated as Dota PPO evidence.
- `PROJECT_PLAN.md` — the staged roadmap, evidence requirements, and safety boundaries.

## Current capabilities

The active Stage 2 lane curriculum runs Shadow Fiend in a human-started, local custom lobby. The bridge records the exact observation, sampled action, reward, terminal flag, action mask, old policy log-probability/value, game time, reward version, and checkpoint identity needed for valid PPO updates.

The current task is to demonstrate repeatable local last-hit improvement against a frozen baseline. It is **not** a claim of a trained full-game Dota agent.

## Boundaries

- Local custom lobbies only; a person starts and observes every Dota session.
- No public matchmaking, ranked queues, UI automation, binary modification, protocol injection, or game-client MCP control.
- Replay-derived labels, headless-simulator data, and exact local-Dota PPO data remain separate.
- Model files, replays, cached data, rollouts, and local evaluation outputs are intentionally not versioned.

## Quick start

Read the roadmap first, then follow the component documentation:

```powershell
git clone https://github.com/<your-account>/dota2-ppo-research.git
Set-Location dota2-ppo-research\replay_training
python -m pip install -e .
pytest -q
```

For Stage 2 local collection and comparison, see [`replay_training/LOCAL_POLICY_COMPARISON.md`](replay_training/LOCAL_POLICY_COMPARISON.md).

## Contributing and reporting problems

Bug reports, reproducible rollout-validation failures, and documentation improvements are welcome. Please include your operating system, Python/PyTorch versions, the exact command, and sanitized console output. Do not upload replay files, model checkpoints, account details, access tokens, or public-match material.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guidance.

**Topics:** #Dota2 #ReinforcementLearning #PPO #PyTorch #GameAI #Research
