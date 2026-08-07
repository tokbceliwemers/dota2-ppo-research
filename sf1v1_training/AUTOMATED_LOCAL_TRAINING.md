# Local SF 1v1 collection only

The former `dota-ppo autopilot` entry point is retired. It could drive addon
collection through the loopback bridge, which conflicts with this project's
human-operated local-Dota boundary.

Use this workflow instead. See [SF1V1_ROLLOUTS.md](SF1V1_ROLLOUTS.md) for the
archive contract.

1. Start the `rl_ppo_local` lobby yourself.
2. Start `dota-ppo bridge` with one frozen checkpoint and a
   `data\training\...{batch:03d}.npz` template.
3. Collect complete local training archives; use `data\evaluations` only for
   held-out measurement.
4. Run `dota-ppo inspect-rollouts data\training` before PPO. Run
   `dota-ppo compare-rollouts` only on held-out evaluations.

The supervisor accepts only `sf1v1_passive_v1` archives marked
`collection_role: training`, sampled by its exact checkpoint, spanning at
least 150 game seconds, ending on a terminal transition, with valid masks and
adequate decision cadence.
