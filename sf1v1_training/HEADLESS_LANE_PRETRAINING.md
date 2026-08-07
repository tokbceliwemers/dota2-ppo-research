# Headless lane PPO pretraining

`headless-lane-ppo` is a fast, non-rendered approximation used to initialise
the lane-v3 policy. It has no Dota process, network, VConsole, mouse, keyboard,
or UI dependency.

It is **not** a Dota substitute. The simulator checkpoint is marked
`real_dota_verified: false`; it may initialise a policy bridge, but it must be
evaluated through the local instrumented lobby before any promotion.

```powershell
Set-Location C:\Users\skaya\Desktop\dota2\sf1v1_training
dota-ppo headless-lane-ppo `
  --checkpoint checkpoints\lane_expert_bc_v3.pt `
  --output checkpoints\headless_lane_context_v3.pt `
  --calibration-report data\calibration\lane_001_report.json `
  --updates 16 --environments 1024 --horizon 96 --epochs 4 `
  --minibatch-size 16384 --device cuda
```

The command trains PPO only from `headless_lane_simulator` rollouts. The normal
`dota-ppo ppo` command still rejects every source except exact
`local_instrumented_lobby` archives, so a simulator rollout cannot accidentally
be used as real Dota PPO data.

Before a long simulator run, collect one real local calibration batch as
described in [LANE_CALIBRATION.md](LANE_CALIBRATION.md). The optional
`--calibration-report` loads only measured fields into the CUDA pretrainer and
saves the exact applied values in checkpoint metadata. Fields not measured in
the report, such as attack range buffer and attack cooldown, remain explicit
defaults.

The CUDA pretrainer keeps simulator state and policy sampling on the GPU, then
transfers the completed batch only once for the PPO update. Its default 16,384
sample minibatch reduces small-kernel overhead for this compact policy. It does
not change the separate, exact local-Dota PPO training path.

The collector stores each simulated environment's time series contiguously
before calculating GAE. This matters because a vectorized batch is not one
long trajectory. An allied-creep kill ends an approximation episode with no
last-hit reward, matching the intended drill semantics.
