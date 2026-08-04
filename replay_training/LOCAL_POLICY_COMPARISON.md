# Local policy comparison

This is the next Stage 2 evidence gate. It compares a frozen baseline with one
candidate through fresh, exact `local_instrumented_lobby` rollouts. It does not
train PPO and does not promote a checkpoint automatically.

The current lane curriculum is `lane_wave_clear_v3_fixed_progression`: clearing the four tracked
enemy creeps terminates the episode and immediately starts a fresh scripted
wave, with Shadow Fiend reset to Level 1, zero XP, and zero gold. This is
intentionally not a full real-Dota 1v1 simulation. Never compare these archives
with older reward versions; the comparison tool rejects mixed reward versions.

## Policies

- Baseline: `checkpoints\lane_expert_bc_v2.pt`
- Candidate: `checkpoints\headless_lane_calibrated.pt` (created from the
  measured lane calibration report)

Collect **three 1x archives for each policy**. Use the identical local
`rl_ppo_local` lane drill, same hero, no `rl_ppo_speed_*` commands, and let a
batch finish normally. Start a new bridge process whenever changing the
checkpoint so each rollout's old log-probabilities are exact for its policy.

Example baseline bridge:

```powershell
Set-Location C:\Users\skaya\Desktop\dota2\replay_training
dota-ppo bridge checkpoints\lane_expert_bc_v2.pt `
  --rollouts data\evaluations\baseline_v3_{batch:03d}.npz --device cuda
```

Run the local match three times, then stop this bridge. Repeat with the
candidate and a separate filename prefix:

```powershell
dota-ppo bridge checkpoints\headless_lane_calibrated.pt `
  --rollouts data\evaluations\candidate_v3_{batch:03d}.npz --device cuda
```

After six completed archives, report rather than train:

```powershell
dota-ppo compare-rollouts `
  data\evaluations\baseline_v3_001.npz data\evaluations\baseline_v3_002.npz data\evaluations\baseline_v3_003.npz `
  --candidate data\evaluations\candidate_v3_001.npz data\evaluations\candidate_v3_002.npz data\evaluations\candidate_v3_003.npz `
  --output reports\lane_v3_candidate_comparison.json
```

Each archive must span at least 150 game seconds (the normal batch is 160), so
an early `rl_ppo_finish` or disconnect cannot silently enter the comparison.
`preliminary_improvement` requires a valid, completed, timing-comparable set
and all three: higher reward per step, more last-hit reward signals, and no
extra death reward signals. It is only preliminary evidence; it neither
completes Stage 2 nor authorizes a real PPO update.
