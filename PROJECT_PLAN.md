# Dota 2 PPO Project Plan

## Scope and boundaries

Build a single-hero (Shadow Fiend) research agent in small, measurable stages.
Replay-derived actions, synthetic simulator data, and exact local-Dota PPO
rollouts are different data sources and must never be mixed.

All game interaction is limited to a human-started local `rl_ppo_local` custom
lobby and its loopback bridge at `127.0.0.1`. There is no Dota MCP, UI
automation, binary modification, protocol injection, or public matchmaking in
this project.

## Stage 1 - `replay`

**Purpose:** create an offline behavior-cloning bootstrap from `.dem` files.

- Parse replay state, positions, combat, objectives, and player data.
- Infer movement labels from successive positions for one narrow hero/lane
  curriculum.
- Keep the limitation explicit: ordinary `.dem` files do not reliably contain
  the physical keys or complete command stream that caused every action.

**Exit criteria:** parsed match directories and a replay-derived movement
behavior-cloning dataset exist, with provenance recorded.

## Stage 2 - `replay_training` (current stage)

**Purpose:** collect exact on-policy local-Dota trajectories and demonstrate
repeatable last-hit improvement in one controlled lane drill.

### Current lane curriculum: `lane_wave_clear_v4_fixed_progression`

- Observation contract: `lane-v3`, 25 features and 24 action IDs; ability and
  item actions remain masked. It gives a target-relative, 20-segment health-bar
  value (not exact hit points), recent health-bar loss rate, normalized hero
  attack damage, attack range/recovery, allied creeps attacking the target, and
  nearest allied-creep geometry/health. The policy must infer last-hit timing
  from those visible cues rather than receiving a `last_hit_ready` oracle.
- Reward: `+1` for a hero last hit, `-2` for hero death, approach shaping, and
  a small terminal last-hit bonus.
- Terminal conditions: hero death, 75-second fallback limit, or the four
  tracked enemy creeps being cleared.
- On a wave-clear terminal, the exact final transition is committed first;
  Shadow Fiend and a fresh scripted wave then reset. A settled-state recovery
  path handles a missed terminal callback.
- Shadow Fiend receives no experience, has zero ability points, and clears
  Necromastery level/stacks for every wave, keeping the micro-curriculum
  stationary instead of making later waves easier.
- Every new archive records `reward_version:
  lane_wave_clear_v4_fixed_progression` and `observation_version: lane_v3`.
  Older archives are not comparable with, or training data for, this
  curriculum.

### Verified evidence

- The loopback bridge saves observation, sampled action, reward, terminal flag,
  action mask, old log-probability, old value, game time, reward version, and
  a SHA-256 identity of the policy checkpoint that sampled the transition.
- Local attack targeting and the real entity-killed -> last-hit reward path were
  exercised; a forced integration probe saved `+1.03685`.
- A 600-event local calibration measured approximately 0.2667-second decision
  spacing, 305 move speed, 525 attack range, 550 creep max health, and the
  measured values used by the headless lane approximation.
- `headless_lane_calibrated.pt` was trained on CUDA from the calibration report
  and is explicitly marked `real_dota_verified: false`.
- The first complete timeout-only A/B evaluation was cadence-comparable
  (~3.75 decisions/game-second) but rejected the calibrated candidate: reward
  per step was `0.00119` versus baseline `0.00555`, last-hit signals were 12
  versus 17, and death signals were 5 versus 4. It is not eligible for PPO
  promotion.
- In a fresh local lobby, three consecutive forced enemy-wave clears each
  emitted `lane episode committed; starting fresh wave`; the new-wave reset
  path is now smoke-tested.
- The first complete v4 A/B set passed all collection-quality gates: three
  160-game-second archives per checkpoint, valid action masks, terminal
  episodes, and comparable cadence (baseline 3.705 versus candidate 3.732
  decisions/game-second). The calibrated headless candidate was rejected:
  reward per step was `-0.01067` versus the baseline's `-0.00890`, it produced
  6 versus 9 last-hit signals, and both had 13 death signals. It is not
  eligible for real-Dota PPO promotion.
- The `lane_v3` 25-feature observation contract is implemented in the local
  bridge and the separate headless pretrainer. Fresh `lane_expert_bc_v3.pt`
  and calibration-informed `headless_lane_context_v3.pt` checkpoints exist;
  neither has yet been tested in a local-Dota `lane_v3` archive.
- A CleanRL/TorchRL audit corrected a headless-only GAE trajectory-ordering
  error, added a CUDA-resident vectorized simulator path, and made
  allied-creep kills terminal without awarding a false last hit. The rebuilt
  candidate consumed 131,072 approximate CUDA samples at 3,393 samples/second;
  it remains `real_dota_verified: false`.

### Immediate next action

1. Start a fresh local lobby/bridge using the prepared `lane-v3` checkpoints,
   verify it produces a 25-feature archive without bridge errors, then
   collect three complete normal-speed archives per policy. Compare only equal
   reward and observation versions with `dota-ppo compare-rollouts`.
2. Only after repeatable local improvement, collect fresh on-policy data from
   the chosen policy and run one PPO update. Archive the data, report, reward
   version, observation version, and resulting checkpoint together.

**Exit criteria:** several independent `lane_wave_clear_v4_fixed_progression` batches show stable
last-hit reward improvement over the frozen movement/behavior-cloning baseline,
with no reset, action-mask, or cadence faults.

## Fast offline pretraining (supporting Stage 2)

`replay_training/src/dota_ppo/headless_lane.py` is a vectorized CUDA
approximation for fast policy initialization. It is not Dota and is never PPO
data. Its checkpoints carry `source: headless_lane_simulator` and
`real_dota_verified: false`; normal `dota-ppo ppo` accepts only
`local_instrumented_lobby` archives.

`simulator/dota_lane_env.py` is a separate, single-environment Gymnasium
reference for terrain-aware offline experiments. It reads
`simulator/data/dota_heightmap.npy` with its recorded map bounds and raw-height
range; movement is bounded and slope-gated, and observations contain ground
height/slope context plus a quantized creep health bar. It now also consumes
the original static `dota.gnv` GridNav file (64-unit cells, conservative
walkability bit) for spawn and movement constraints. GridNav is not a full
server-physics model and the environment's `terrain_lane_v1` data is not
interchangeable with `lane_v3` local-Dota rollouts. Its only valid role is to
test environment logic or initialize a candidate before local evaluation.

Use the local calibration report only for directly measured simulator fields.
Attack buffer and attack cooldown remain explicit defaults until a controlled
local measurement exists. See `replay_training/LANE_CALIBRATION.md` and
`replay_training/HEADLESS_LANE_PRETRAINING.md`.

## Stage 3 - `rl_with_bots`

**Purpose:** move from the lane drill to controlled local matches against
scripted bots, without skipping Stage 2 evidence.

- Fixed hero, lane, bot difficulty, reward version, and short episode contract.
- Add abilities, inventory, target selection, ally/enemy context, objectives,
  and match rewards incrementally.
- Evaluate against a frozen local bot baseline before checkpoint promotion.

**Exit criteria:** the policy completes local bot matches without bridge errors
and beats the frozen baseline across a defined evaluation set.

## Stage 4 - supervised human evaluation (future; not active)

This stage is not started by the current project. Any future human testing must
be separately scoped, consent-based, and supervised; it must not use public
matchmaking, ranked queues, or UI automation. It requires Stage 3 evidence
first and a new evaluation and safety plan.

## Local collection boundary

A person starts and observes the local custom match. The bounded supervisor may
validate completed archives, evaluate a batch, and perform one PPO update; it
does not start or control Dota. A `rl_ppo_finish` call is for an intentional
early stop and produces an incomplete batch, so it is not valid for the
150-second A/B gate.

## Research references - `librarys`

Fourteen pinned research checkouts are documented in `librarys/LIBRARIES.md`.
They cover replay parsing, PPO/vector-environment engineering, imitation,
future multi-agent interfaces, Dota constants, and historical Dota AI work.
They are references only, not runtime dependencies. None is a maintained,
faithful, high-speed Dota simulator; the local client remains the authority for
on-policy PPO evidence.

## Promotion rule

Do not skip stages. A checkpoint advances only when the prior stage's exit
criteria are met and its source data, reward version, evaluation report, model
file, and known limitations are recorded together.

## Prompt for the next implementation session

> Work on the next incomplete Stage 2 action. First read this file and inspect
> the current artifacts and tests. Preserve the separation between replay
> behavior cloning, headless simulator pretraining, and exact local-Dota PPO
> rollouts. Implement the smallest testable local curriculum improvement, add
> or update tests, verify it, and provide only the human-operated local test
> command needed next. Do not claim a promotion without the stated evidence.
