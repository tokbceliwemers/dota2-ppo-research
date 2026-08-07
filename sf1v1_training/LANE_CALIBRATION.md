# Local lane calibration

Calibration records real local-Dota transition measurements separately from PPO
archives. It does not launch Dota, interact with its UI, or train a PPO model.

## Collect one calibration batch

1. In PowerShell, start the bridge before entering the local `rl_ppo_local`
   custom match:

   ```powershell
   Set-Location C:\Users\skaya\Desktop\dota2\sf1v1_training
   dota-ppo bridge checkpoints\lane_expert_bc_v3.pt `
     --rollouts data\rollouts\calibration_001.npz `
     --calibration data\calibration\lane_001.jsonl `
     --device cuda
   ```

2. Start the local custom match manually, wait for Shadow Fiend to spawn, and
   let the normal 160-game-second batch finish. Do not use 2x/4x speed for the
   first calibration batch.

3. In Dota's local console, run `rl_ppo_finish` only if you need to finish
   early. Wait for `Saved PPO rollout` before stopping the Python bridge.

4. Analyse the event stream:

   ```powershell
   dota-ppo analyze-calibration data\calibration\lane_001.jsonl `
     --output data\calibration\lane_001_report.json
   ```

The report measures tick spacing, move speed, attack range, attack-damage
fraction, and last-hit health fraction. It leaves attack buffer, attack
cooldown, passive creep damage, and nearby-creep damage explicitly unmeasured.
Those need controlled local probes, not guessed values.

Fresh bridge events also record the live `GetSecondsPerAttack(false)` value as
`hero_attack_cooldown`. `analyze-calibration` reports its median when present;
older JSONL files do not contain it and remain valid, but cannot calibrate that
field. The attack-order buffer is still unmeasured and must not be guessed from
the report.

The terrain Gymnasium trainer can load the report with
`--calibration-report data\calibration\lane_002_report.json`. Neither the
JSONL file nor its accompanying rollout may be used as a substitute for a
fresh exact PPO rollout.
