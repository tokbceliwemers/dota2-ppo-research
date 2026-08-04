# Manual local collection only

The former `dota-ppo autopilot` entry point is retired. It could drive addon
collection through the loopback bridge, which conflicts with this project's
human-operated local-Dota boundary.

Use this workflow instead:

1. Start the `rl_ppo_local` lobby yourself.
2. Start `dota-ppo bridge` with one frozen checkpoint and an explicit rollout
   filename template.
3. Collect complete, normal-speed archives manually.
4. Run `dota-ppo compare-rollouts` for evaluation, or `dota-ppo supervise`
   only after the project plan's Stage 2 gate has been met.

The supervisor accepts only `lane_wave_clear_v3_fixed_progression` archives
from the exact checkpoint it was given, spanning at least 150 game seconds,
ending on a terminal transition, with valid masks and adequate decision
cadence.
