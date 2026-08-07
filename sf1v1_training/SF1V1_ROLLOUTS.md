# SF 1v1 rollout contract

The active local-Dota contract is `sf1v1_passive_v1`: a 32-value observation
containing the controlled Shadow Fiend, lane/creep context, and only visible
features of a fixed passive enemy Shadow Fiend.  It is not yet a complete 1v1
game or self-play environment.

## Archive roles

| Directory | Role | Can enter `dota-ppo ppo`? |
| --- | --- | --- |
| `data/training` | Fresh, on-policy local collection from one checkpoint | Yes, after validation |
| `data/evaluations` | Held-out local measurements | No |
| `data/bootstrap` | Replay/synthetic behavior-cloning input | No |

The bridge writes `collection_role`, reward/observation versions, policy
SHA-256, environment metadata, action masks, and terminal flags.  PPO rejects
an archive that is missing or mismatching any of those core contracts.

## Why the previous archives are not trainable

The old lane-v3 archives have 25 observations and the retired
`lane_wave_clear_v4_fixed_progression` reward contract.  The current 1v1
bridge requires 32 observations and `sf1v1_passive_v1`.  They are useful
history, but must not be merged into the current PPO batch.  They were also
stored as evaluations, which are intentionally held out.

## Collection sequence

1. Start a bridge with the current checkpoint and a `data/training` template.
2. Start `rl_ppo_local` on `template_map`; the passive opponent must be shown
   by `rl_ppo_opponent`.
3. Let the bounded batch end naturally; do not use an early finish while an
   action is pending.
4. Inspect the output before training:

```powershell
dota-ppo inspect-rollouts data\training
```

5. Only when the inspected archives are `ppo_eligible: true`, merge archives
   sampled by the *same* checkpoint and run one PPO update.  Restart the
   bridge with that newly created checkpoint before collecting another batch.

This is a local/custom-lobby-only research workflow.
