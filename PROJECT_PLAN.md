# Shadow Fiend 1v1 Project Plan

## Goal and safety boundary

Build a research Shadow Fiend agent that learns to play a controlled Dota 2
**1v1 Shadow Fiend** matchup. Every Dota interaction remains restricted to the
local `rl_ppo_local` custom game, the loopback bridge at `127.0.0.1`, and
consenting local participants. The project never uses matchmaking, public
lobbies, UI automation, binary modification, protocol injection, or remote
game servers.

Replay-derived labels, synthetic simulator transitions, and exact local-Dota
rollouts are separate sources. They must never be mixed into one PPO update.

## Folder-stage map

| Stage | Folder | Role |
| --- | --- | --- |
| 1 | `sf1v1_replays` | Offline Shadow Fiend 1v1 replay dataset and behavior bootstrap. |
| 2 | `sf1v1_simulator` | Fast, approximate non-rendered 1v1 simulator for candidate initialization only. |
| 3 (current) | `sf1v1_training` | Exact local-Dota 1v1 environment, on-policy rollout collection, and PPO. |
| 4 | `sf1v1_evaluation` | Frozen-opponent local evaluation and supervised, consent-based human evaluation. |

`librarys` is retained as shared read-only research material, not a stage.

## Stage 1 — `sf1v1_replays`

**Purpose:** derive an offline Shadow Fiend 1v1 bootstrap from `.dem` files.

- Select replays with Shadow Fiend mid-lane examples; record match, patch, hero,
  player, and filter provenance.
- Parse observable state, movement, attack/ability outcomes, camera-independent
  geometry, creep waves, and enemy-hero interactions.
- Infer labels conservatively from state changes. Normal `.dem` files do not
  reliably contain physical key presses or a complete player command stream.
- Keep replay data as behavior-cloning or analysis data only, never local-Dota
  PPO data.

**Promotion rule:** a versioned SF 1v1 dataset and a reproducible behavior
bootstrap checkpoint exist with its source manifest.

## Stage 2 — `sf1v1_simulator`

**Purpose:** provide fast, deliberately approximate 1v1 pretraining.

- Preserve the terrain/GridNav and static NPC data already extracted.
- Add only validated 1v1 state: both SF positions/facing/health/mana, creep
  waves, attack timing, Shadowraze ranges/cooldowns, and a configurable frozen
  opponent policy.
- Calibrate directly measurable constants from local-Dota telemetry. Keep
  unmeasured values explicit; do not call the approximation a Dota server.
- Save candidates with `source: sf1v1_simulator` and
  `real_dota_verified: false`.

**Promotion rule:** deterministic tests cover the 1v1 contract and a candidate
checkpoint can be exported for Stage 3 evaluation. Synthetic transitions do
not enter real-Dota PPO.

## Stage 3 — `sf1v1_training` (current)

**Purpose:** build the exact local-Dota Shadow Fiend 1v1 environment and train
only from its instrumented, on-policy rollouts.

### Curriculum

1. **Lane foundation:** the existing fixed-progression last-hit drill remains
   a regression test and behavior bootstrap; it is not 1v1 success evidence.
2. **Passive enemy SF (implementation complete; local runtime validation
   pending):** spawn a fixed level-1 opposing Shadow Fiend in a reproducible
   lane state. Observe visible opponent geometry, quantized health/mana bars,
   and facing only; do not grant hidden health, cooldown, or intent oracles.
   The current contract is `sf1v1_passive_v1` with 32 observations. Fresh
   data is separated into `data/training` (PPO eligible when validated) and
   `data/evaluations` (held out).
3. **Frozen scripted opponent:** enable bounded movement, attack, and one
   verified ability at a time. Add mirrored action/observation telemetry,
   projectiles, damage attribution, death/respawn, gold/XP, and items only
   when each is tested locally.
4. **PPO self-play candidates:** use frozen checkpoints as opponents. A new
   policy is evaluated before it can become an opponent; do not train both
   sides from the same rollout batch.

### Evidence contract

- Each rollout records observation/action/reward/terminal/action mask, old
  log probability/value, game time, reward version, observation version, and
  checkpoint identity.
- Local episode configuration records both checkpoint identities, side/team,
  map, opponent mode, Dota build, and seed where available.
- Rewards progress from lane control and last-hits to hero damage, survival,
  tower/objective state, and only then match outcome. Reward changes create a
  new version; old archives are not comparable.
- The lab runner may start Dota Tools, launch
  `dota_launch_custom_game rl_ppo_local template_map`, join the local Radiant
  slot, start/stop the bridge it owns, and run bounded local batches. All
  console commands remain loopback-only and audit-logged.

**Promotion rule:** at least three independent local batches against each
frozen opponent show stable improvement over the declared baseline, with
matching reward/observation versions, valid action masks, no reset/cadence
faults, and reproducible reports. Only then is a local-Dota PPO update
allowed.

## Detailed execution plan

This is the implementation order for the desired outcome: a locally trained,
evaluated Shadow Fiend that can play a controlled 1v1 lane. Each milestone is
small enough to test before more mechanics are added. A new reward,
observation, action, or terminal definition creates a new version; archives
from an older version are never silently reused for PPO.

### M0 — establish a trustworthy passive-opponent baseline

**Goal:** prove that the current `sf1v1_passive_v1` contract runs end-to-end
in a fresh local lobby.

1. Start the 32-observation bootstrap checkpoint and local bridge.
2. Launch `rl_ppo_local` on `template_map`; verify `rl_ppo_opponent` reports
   exactly one passive Shadow Fiend on the opposite team.
3. Collect one natural terminal batch, inspect it, and check its metadata:
   `source=local_instrumented_lobby`, `collection_role=training`, 32-wide
   observations, monotonic game time, valid masks, terminal final step, and
   environment metadata naming the passive opponent.
4. Run one held-out batch under the same checkpoint in `data/evaluations`.

**Exit evidence:** the source and installed addon hashes match; both archives
are readable; no Lua errors, missing transition, pending-decision, reset, or
cadence fault occurs. This is a smoke test only, not a PPO update and not 1v1
success.

### M1 — make the combat state observable and controllable

**Goal:** remove the current non-causal setup where the agent sees an enemy
hero but cannot meaningfully interact with it.

1. Define an explicit, versioned control layout. Keep `idle`, eight-direction
   movement, `stop`, and `hold`; replace implicit “attack nearest creep” with
   target-aware actions such as `attack_nearest_enemy_creep` and
   `attack_enemy_hero`. Do not expose entity indices as policy actions.
2. Make all safe movement directions available; do not mask retreating merely
   because a creep is nearby. Masks should describe engine-invalid actions,
   not strategy advice.
3. Add visible self state: quantized own health/mana, facing, attack recovery,
   movement state, and visible cooldown readiness. Add visible opponent state:
   relative position, facing, quantized health/mana, and distance. Preserve
   the rule that hidden cooldowns, exact opponent intent, and invisible units
   are never included.
4. Add a short fixed observation history (for example the last 4–8 decision
   states) or a recurrent policy. Last-hit timing requires knowing recent
   health-bar motion, attacks, and movement; one static frame is insufficient.
5. Log the executed order, selected semantic target class, whether an attack
   began, whether it landed, and its visible outcome. The logger—not inference
   from reward—must establish damage attribution.

**Tests:** action-mask width matches the new action contract; every allowed
action executes or produces a logged engine rejection; controlled and enemy
target actions are distinguishable; no hidden-value field appears in the
observation encoder.

**Exit evidence:** a scripted manual test can make the controlled SF attack a
creep and the enemy SF, while the rollout reports the correct semantic target
and observation/action version.

### M2 — frozen opponent and deterministic episode lifecycle

**Goal:** turn the lane drill into a controlled one-sided 1v1 environment.

1. Implement a frozen opponent policy outside the learning policy. Begin with
   a reproducible schedule: hold lane position, then bounded movement, then
   creep last-hit attacks, then one verified Shadowraze. Only one new behavior
   is added per version.
2. Give every episode a configuration record: map, controlled side, opponent
   mode/checkpoint identity, opponent side, Dota build when available, seed or
   deterministic scenario identifier, initial positions, level, gold, and
   item set.
3. Replace “enemy wave cleared” as the sole terminal condition. Use a fixed
   game-time horizon plus hero death/respawn boundaries. A wave clearing by
   allied creeps must be reported as an event, not awarded as a policy win.
4. Reset both heroes, creeps, modifiers, gold, XP, cooldowns, Necromastery
   stacks, and projectiles that belong to the scenario. Verify the opponent is
   recreated or returned to the same state on every reset.
5. Record hero damage dealt/taken, death, last hit, deny, net-worth change,
   and time-in-lane as separate counters. Do not infer them from position.

**Tests:** repeated seeded resets have the same observable start state; idle
and active policies cannot receive the same lane-success outcome simply
because allied creeps clear a wave; death/respawn settles the bridge before
the next episode begins.

**Exit evidence:** three repeated fixed-opponent episodes have identical
initial state and valid terminal settlement, with distinct event counters for
creeps and heroes.

### M3 — causal reward design and calibration

**Goal:** make reward answer “did this action improve the lane?” rather than
“did game time advance?”

1. Start from sparse, attributable events: hero last hit, deny, hero damage
   dealt, hero damage received, death, and survival at horizon. Keep each
   reward component in rollout metadata or a sidecar so reports can separate
   them.
2. Normalize damage rewards by the relevant maximum health and cap each
   component to prevent one long combat from dominating the batch.
3. Penalize hero death strongly enough to dominate speculative damage, but do
   not reward a wave clearing event unless the policy has a measured causal
   contribution.
4. Add lane-resource terms only after their attribution is verified: gold/XP
   delta and tower damage. Match outcome comes last, after a reliable local
   match lifecycle exists.
5. Calibrate attack cooldown, wind-up, projectile travel, and creep pressure
   from local telemetry. Use the measurements in the simulator as
   approximations; the local Dota rollout remains authoritative.

**Tests:** forced last-hit, forced deny, forced hero damage, forced hero death,
and idle wave-clear scenarios each yield only their expected reward components.

**Exit evidence:** a reward audit report explains every nonzero component for
sampled episodes, and an idle policy cannot earn a positive lane-control
score merely from environmental progression.

### M4 — fast simulator alignment, without contaminating PPO

**Goal:** use `sf1v1_simulator` for speed while preserving the exact local-Dota
training boundary.

1. Mirror only the tested M1–M3 observable/action contract in Gymnasium.
2. Implement the same frozen-opponent modes and scenario seeds, with explicit
   flags for all approximated mechanics.
3. Compare simulator traces to local telemetry for movement, attack timing,
   projectile arrival, creep health trend, damage, and reset state. Revise
   approximation constants from measurements rather than assumptions.
4. Pretrain or behavior-clone simulator data into a candidate checkpoint only.
   Mark every such checkpoint `real_dota_verified: false` until it passes M5.

**Exit evidence:** deterministic simulator tests pass and a simulator-trained
candidate can run the local smoke test. Simulator transitions never appear in
`data/training` or a local-Dota PPO update.

### M5 — disciplined local-Dota PPO loop

**Goal:** improve one frozen policy against one declared opponent without
off-policy mixing or evaluation leakage.

1. Freeze a baseline checkpoint and frozen-opponent configuration.
2. Collect at least three fresh `data/training` archives from exactly one
   policy SHA-256 and one environment/reward/observation version. A batch must
   end terminally and pass `dota-ppo inspect-rollouts`.
3. Merge only that cohort, run one PPO update, and save its checkpoint plus
   update manifest. Do not continue collecting with the old bridge after the
   checkpoint changes.
4. Restart the bridge on the new checkpoint and collect a new cohort. Never
   train both sides from the same collected transitions.
5. Keep learning-rate, epochs, batch size, entropy, clipping, advantage
   normalization, action distribution, reward components, and checkpoint SHA
   in the manifest. Do not judge improvement from policy/value loss alone.

**Exit evidence:** no archive is rejected for version, collection role,
checkpoint SHA, invalid mask, non-monotonic time, or terminal state; each PPO
update is traceable to a single cohort and environment version.

### M6 — held-out evaluation and frozen-opponent promotion

**Goal:** decide whether a candidate is genuinely better than the baseline.

1. Run baseline and candidate in `data/evaluations`, never `data/training`.
2. Use at least three independent batches for every opponent mode, starting
   side, and declared scenario seed. Hold opponent settings fixed within each
   comparison.
3. Report: deaths, damage dealt/taken, last hits, denies, net worth/XP,
   survival time, tower/objective metrics when present, action validity,
   reset/cadence faults, and reward components. Compare distribution and
   consistency, not just one aggregate reward.
4. Promote a candidate to a frozen opponent only when it improves the
   predeclared primary metrics across repeated held-out runs without a safety
   or lifecycle regression.

**Exit evidence:** versioned JSON reports, the frozen baseline/candidate
checkpoint hashes, raw held-out archive list, and a decision record explaining
promotion or rejection.

### M7 — controlled self-play, then supervised human evaluation

**Goal:** broaden the opponent distribution only after frozen-opponent results
are reliable.

1. Build an opponent pool of promoted, immutable checkpoints and sample one
   per complete episode. Record both checkpoint hashes in the archive.
2. Train only the controlled side. Opponent policies stay frozen throughout
   a collection batch; periodically add a promoted policy to the pool after
   held-out evaluation.
3. Add abilities, item choices, tower pressure, and match outcomes one
   independently tested version at a time. Re-run M3–M6 for each new
   mechanics contract.
4. Human evaluation, if desired, is consent-based, local, unranked, and
   supervised. It produces evaluation reports, not unverified PPO labels.

**Exit evidence:** the project can reproduce results against a documented
opponent pool and clearly states which Dota mechanics remain unsupported.

### Investigation checklist when a candidate fails to improve

Before changing PPO hyperparameters, inspect these in order:

1. Is the archive from the current environment, reward, observation, and
   checkpoint SHA, with `collection_role=training`?
2. Do event counters show that policy actions cause last hits/damage/deaths,
   rather than the environment progressing by itself?
3. Are the needed actions actually unmasked, executed, and represented in the
   action distribution?
4. Can the policy observe the information needed at the decision time,
   including recent timing history?
5. Is the collected cohort sufficiently large and varied before multiple PPO
   epochs are applied?
6. Does the candidate improve held-out, frozen-opponent metrics—not just PPO
   loss or one rollout reward?

## Stage 4 — `sf1v1_evaluation`

**Purpose:** measure a frozen agent in local 1v1 games without changing it.

- Evaluate against held-out scripted and frozen-checkpoint opponents across
  starting sides, seeds, and lane conditions.
- Report lane metrics, deaths, damage, net worth/XP, objectives, and match
  outcomes separately; no single reward number is sufficient.
- Any human testing is separately scoped, consent-based, supervised, local,
  and unranked. It is never public matchmaking.

**Promotion rule:** publish the exact checkpoints, environment version,
opponent set, reports, limitations, and reproducible local instructions.

## Historical assets

The previous single-hero lane archives, calibration reports, and simulator
checkpoints are retained under `sf1v1_training` and `sf1v1_simulator` as
historical bootstrap evidence. They are not proof that the 1v1 goal is met,
and they must not be relabeled as 1v1 data.

## Prompt for the next implementation session

> Read this file in full. Work on the next incomplete Stage 3 curriculum step:
> add the smallest observable, testable part of a local Shadow Fiend 1v1
> environment. Preserve the separation between replay, simulator, and exact
> local-Dota data. Add tests and report the concrete evidence; do not claim a
> 1v1 promotion without the Stage 3 rule.
