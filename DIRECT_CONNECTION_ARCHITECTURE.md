# Direct Dota connection architecture

## The two connections are different

| Layer | Responsibility | Replaces the local addon? |
| --- | --- | --- |
| Session adapter (GC/GCS or custom-lobby management) | Create/join a lobby, select participants, read session state | No |
| Gameplay adapter | Read world state, submit semantic hero orders, collect reward and terminal data | Yes, but only after it provides the same exact PPO contract |

A session/GC connection is not an in-game bot API. It cannot by itself provide
the observations, executed actions, rewards, episode boundaries, old
log-probabilities, and old values that PPO requires.

The current local addon is the **gameplay adapter** for Stage 2. It is not a
dead end: its observation/action/reward contract becomes the reference
contract that every later gameplay adapter must satisfy.

## Roadmap

1. **Stage 1 — `replay` (bootstrap)**
   - Use `.dem` state and position samples for approximate behavior cloning.
   - Do not mistake inferred movement for exact human commands or PPO data.

2. **Stage 2 — `replay_training` (current)**
   - Finish a stable local last-hit curriculum.
   - Improve collection throughput only when timing metrics preserve decision
     coverage.
   - Use the local MCP to automate launch, bridge lifecycle, rollout review,
     evaluation, and CUDA PPO jobs.

3. **Stage 3 — `rl_with_bots`**
   - Move to local All Pick against scripted bots.
   - Expand the gameplay adapter gradually: target selection, abilities,
     inventory, objectives, survival, and match outcome rewards.
   - Promote only after repeated evaluation beats a frozen bot baseline.

4. **Stage 4 — `rl_with_humans`**
   - Add the planned GCS/GC session adapter for matchmaking, lobby, draft, and
     match coordination.
   - Keep one fixed hero and the same gameplay adapter contract.
   - Define and validate the gameplay adapter separately; the session adapter
     does not replace gameplay control.
   - Record semantic orders, targets, rewards, and outcomes using the canonical
     deployment layout.

## Immediate next milestone

Collect a fresh **1x timing probe** with the 18-feature lane-v2 bridge, verify
the decision cadence and archive validity, then collect a matched lane-v2
batch for one PPO update and repeatable evaluation. This is the evidence needed
to finish Stage 2 before starting All Pick bots.
