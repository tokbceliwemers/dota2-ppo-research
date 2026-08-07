# Overnight PPO supervisor

The supervisor is an always-running **local** process for Stage 2. It does not
need Codex to remain open after it has started.

## What it does

1. Watches `data/rollouts` for new, fully-written `.npz` archives.
2. Rejects archives unless they have exact local PPO fields, completed
   episodes, valid sampled actions, game-time telemetry, enough steps, and the
   configured minimum decision cadence.
3. Waits for a configured number of accepted archives from the same policy.
4. Saves a pre-training evaluation report and an append-only `audit.jsonl`.
5. Merges the accepted archives and runs one CUDA PPO update.
6. Saves the new checkpoint, then exits with a clear reason.

## Why it performs one update per run

PPO rollout data must come from the checkpoint that sampled it. After an
update, the bridge must be restarted with the new checkpoint before collecting
the next batch. The supervisor enforces `--max-batches 1` to prevent silent
off-policy training.

## Start from PowerShell

```powershell
Set-Location C:\Users\skaya\Desktop\dota2\sf1v1_training
dota-ppo supervise `
  --checkpoint checkpoints\lane_expert_bc_v2.pt `
  --rollout-dir data\rollouts `
  --run-dir data\supervisor\overnight `
  --batch-archives 3 `
  --max-hours 8 `
  --min-steps 100 `
  --min-decisions-per-game-second 3 `
  --epochs 10
```

It ignores archives that existed before startup unless `--include-existing` is
given. Create a `STOP` file inside its `run_dir` to ask it to stop without
killing the process.

## Manual boundary

The local game must be started by a person. The supervisor never controls
Dota or VConsole; it only validates completed local archives and trains from
the accepted files.
