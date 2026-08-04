# Downloaded reference libraries

These repositories are local research references. They are not imported by the
Python package or copied into the deployed Dota addon.

| Folder | Upstream | Pinned commit | License status | Intended use |
| --- | --- | --- | --- | --- |
| `manta` | https://github.com/dotabuff/manta | `0efe7e11c40a4f149f6414b2d162320de34e8446` | MIT (`LICENSE` present) | Inspect modern Source 2 replay entities/events if `gem-dota` lacks fields needed for offline analysis. |
| `dota2py` | https://github.com/andrewsnowden/dota2py | `67637f4b9c160ea90c11b7e81545baf350affa7a` | `LICENSE` present | Historical Web API and replay-parser reference; not a simulator. |
| `redota` | https://github.com/timkurvers/redota | `f51e568e6ef45e32e1a7d21def805bdd7604568b` | `LICENSE.md` present | Web visualisation/icons project; no environment or bot-control runtime. |
| `LastOrder-Dota2` | https://github.com/bilibili/LastOrder-Dota2 | `48f212b6749956c9de1b0dc2d5e5e406f78b62d7` | `LICENSE` present | Historical Shadow Fiend PPO/world-state architecture reference; launches a live Dota process, so not adopted as a runtime. |
| `dotaclient` | https://github.com/TimZaman/dotaclient | `8615b90b7d5b61005f51ba8e73cab206db1d1731` | no root license found | Historical distributed agent that depends on DotaService world-state RPC; its batching and action-mask patterns are references only. |
| `dotaservice` | https://github.com/TimZaman/dotaservice | `733db265f04fa10caab57fa34f7f85861095dc2e` | `LICENSE` present | Historical bot-world-state/Dota-host launcher; documents non-rendered hosting but is not compatible evidence for the current Windows client. |
| `clarity` | https://github.com/skadistats/clarity | `7fb3f1d07564a12efa99194d45cfbf5762ba5910` | `LICENSE` present | Active Java replay parser for entities, combat logs, modifiers, and game events; useful for Stage 1 only. |

Neither repository supplies a high-speed, faithful Dota simulator. The real
client remains the authoritative source for on-policy PPO rollouts. Fast
synthetic pretraining, if added later, must remain separate from real-client
PPO evaluation.

## 2026 headless-lane decision

The requested references were downloaded and audited. `dota2py`, `clarity`,
and `manta` parse offline material; `redota` visualises it; the remaining
three use obsolete world-state/bot interfaces around a running Dota process.
None provides a maintained, faithful, render-free Dota simulator.

The project therefore implements its own small vectorised `headless_lane`
environment in `replay_training/src/dota_ppo/headless_lane.py`. It models only
the current 18-feature lane-v2 contract: movement to a creep, range, health,
last-hit timing, and a survival penalty. It uses all 24 action IDs but masks
unsupported ability/item actions. It is deliberately labelled
`headless_lane_simulator` and its output checkpoints have
`real_dota_verified: false`. They cannot be saved as, merged with, evaluated
as, or PPO-trained as `local_instrumented_lobby` data.

On an RTX 3080, the 2026-08-04 smoke run collected and updated 16,384
simulator transitions in 6.82 seconds (2,401 samples/second). This measures
pretraining throughput only; promotion still requires a new exact local Dota
rollout and a comparable real-lobby evaluation.

## Applied pattern from LuaFun and dotaservice

The local bridge now records `game_times` alongside every exact PPO decision.
`dota-ppo evaluate-rollouts` reports decision coverage per game-second and
decision gaps, making the `host_timescale` speed-up measurable. This follows
their useful timing/observation-contract idea without adopting their obsolete
transport or bot-control code.

See [DOTASERVICE_REVIEW.md](DOTASERVICE_REVIEW.md) for the complete applied,
deferred, and explicitly rejected design decisions.

## Removed reference checkouts

| Folder | Revision | Reason removed |
| --- | --- | --- |
| `Dota2_Bots` | `094168195c4cb13cb2d995e216be54545fe5b14c` | No license file was present, it is not imported, and it is not needed until a future controlled-bot stage. |
| `LuaFun` | `bd0efd8fc2b064d6bf58993e59a6ad4ac6713b39` | Historical 2021 environment; not compatible as a current runtime dependency. Its timing/observation-stride lesson was recorded. |
| `dotaservice` | `733db265f04fa10caab57fa34f7f85861095dc2e` | Historical 2019 gRPC/native bot stack; no current dependency. Its game-time calibration pattern has already been implemented. |
