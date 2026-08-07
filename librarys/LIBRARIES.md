# Research reference library inventory

These are pinned source references, downloaded on 2026-08-05. They are not
Python dependencies, are not vendored into the Dota addon, and are never a
source of PPO transitions. The real local Dota custom lobby remains the sole
authority for `local_instrumented_lobby` data.

## Current references

| Folder | Pinned revision | License | What it can contribute | Decision |
| --- | --- | --- | --- | --- |
| `manta` | `0efe7e11c40a4f149f6414b2d162320de34e8446` | MIT | Source 2 replay entities/events | Stage 1 parser reference only. |
| `clarity` | `7fb3f1d07564a12efa99194d45cfbf5762ba5910` | license present | Java replay parsing, combat logs, modifiers | Stage 1 parser reference only. |
| `dota2py` | `67637f4b9c160ea90c11b7e81545baf350affa7a` | license present | Historical Python replay/Web API patterns | Historical parser reference only. |
| `redota` | `f51e568e6ef45e32e1a7d21def805bdd7604568b` | license present | Web replay visualisation/icons | Not an environment; defer. |
| `LastOrder-Dota2` | `48f212b6749956c9de1b0dc2d5e5e406f78b62d7` | license present | Historical Shadow Fiend world-state/PPO design | Audit observations/rewards only; do not adopt its launcher. |
| `dotaclient` | `8615b90b7d5b61005f51ba8e73cab206db1d1731` | no root license found | Historical distributed-agent batching | Read-only design reference; do not reuse code. |
| `dotaservice` | `733db265f04fa10caab57fa34f7f85861095dc2e` | license present | Historical Dota host/world-state RPC | Timing and observation ideas already considered; transport deferred. |
| `cleanrl` | `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb` | MIT | Compact PPO, GAE, vector-env, logging reference | Immediate audit reference for simulator/PPO tests; no dependency. |
| `gymnasium` | `07eb046aab0bef15b16843eba0f9cafde5f9884d` | MIT | Standard reset/step/termination contract | Use as an interface/testing reference for `headless_lane`; no dependency yet. |
| `torchrl` | `ae421b98d0dba86e5ab0b24917d1e64f376ee6f9` | MIT | PyTorch collectors, tensorized batches, PPO objectives | Audit collector/terminal semantics; dependency deferred to avoid a large rewrite. |
| `imitation` | `e5ef18806c449ca47153b494a02471c5e2ae3a14` | MIT | Behavior cloning and offline-imitation evaluation | Future Stage 1 supervised-order experiments only; never turns `.dem` guesses into PPO. |
| `dotaconstants` | `e7705ee975ebec2a88a59a7b455d4cae5dc69ca1` | MIT | Hero/item/ability constant snapshots | Future Stage 3 action mapping; verify all values against the installed Dota build. |
| `pettingzoo` | `761353e6a9d29cb3c6b88232e50ee036f94b08fd` | MIT | Versioned multi-agent API and parallel-step semantics | Future Stage 3 bot/self-play simulator interface reference. |
| `pydota2_archive` | `f33233ee5393a248e9845bb25ff234bf7ac9ff82` | Apache-2.0 | Historical BotWorldState and training-scenario research | Do not use its transport/protocol path; it is historical and outside this project's local-addon boundary. |

## Research result: how to make training materially faster

There are two valid acceleration layers, and they must keep separate labels.

1. **Fast approximate training.** `headless_lane` can run thousands of
   vectorized transitions on CUDA. CleanRL, Gymnasium, and TorchRL are useful
   references for terminal/truncation handling, batched collection, GAE, and
   reproducible episode statistics. This is where most wall-clock speedup is
   available now. Outputs remain `headless_lane_simulator` and
   `real_dota_verified: false`.
2. **Exact local-Dota validation.** A local Dota addon supplies the actual
   entity state, orders, rewards, and action masks. It is slower because the
   Dota simulation is the environment. No downloaded project was found to be a
   maintained, documented, Windows-compatible, render-free Dota custom-game
   host that preserves the current addon contract.

The historical `pydota2_archive` project explicitly describes high-speed
headless play as needing Valve support. Its BotWorldState route is not adopted:
it is not a supported replacement for the current local VScript addon and
would violate the project's no-protocol-injection boundary.

## Explicitly deferred dependencies

Do not install CleanRL, Gymnasium, TorchRL, imitation, or PettingZoo into this
project merely because their source is present here. Adopting one requires a
small, tested change with a clear benefit. In particular, do not replace the
provenance checks or the local-Dota evaluation gate with an offline library.

## Practical next research candidates

1. Compare the present headless PPO GAE/terminal tests directly with CleanRL's
   PPO reference; adopt only independently testable corrections.
2. Add a Gymnasium-compatible wrapper for the headless simulator if it reduces
   testing friction without changing its labels or promotion status.
3. Before Stage 3, use `dotaconstants` only to generate a checked action map;
   confirm it against the installed game data at that time.
4. Treat any purported Dota headless-server solution as an experiment until it
   can run the unchanged addon, send exact transitions over the loopback
   bridge, and match a manually observed local-lobby rollout.

## Applied PPO/simulator audit (2026-08-05)

The CleanRL/TorchRL review led to four tested changes in
`sf1v1_training`, without importing either package:

1. Headless rollouts are now laid out environment-major before GAE, so values
   and terminals never cross from one simulated lane into another.
2. CUDA pretraining uses a GPU-resident simulator and only transfers a finished
   batch once, while retaining the NumPy implementation as a test reference.
3. A creep killed by allied pressure is a terminal failure in the approximation,
   not an attackable dead target or a false positive last hit.
4. PPO's value-loss scale and monitored KL approximation now follow the
   compact CleanRL PPO reference.

The local RTX 3080 smoke rebuild processed 131,072 approximate transitions in
38.63 seconds (3,393 samples/second). This is a throughput measurement only;
it is not real-Dota performance evidence.
