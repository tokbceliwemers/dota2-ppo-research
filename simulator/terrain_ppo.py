"""PPO pretraining on :class:`DotaTerrainLaneEnv`, never real-Dota PPO.

The saved checkpoint intentionally uses the same ActorCritic format as the
local bridge.  It is an *initialization candidate* only: do not feed its
synthetic transitions to ``dota-ppo ppo`` and do not promote it without local
rollout evaluation.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dota_ppo.data import Rollouts
from dota_ppo.model import ActorCritic
from dota_ppo.observations import OBSERVATION_DIM, OBSERVATION_VERSION
from dota_ppo.train import PPOConfig, _checkpoint, _ppo_update_for_source, load_model, select_device

from dota_lane_env import ACTION_DIM, DotaTerrainLaneEnv


SOURCE = "terrain_headless_gymnasium"


def collect_rollout(model: ActorCritic, environments: int, horizon: int, data_directory: Path,
                    device: torch.device, seed: int) -> Rollouts:
    """Collect environment-major trajectories so GAE never mixes environments."""
    if environments < 1 or horizon < 2:
        raise ValueError("environments must be positive and horizon must be at least two")
    envs = [DotaTerrainLaneEnv(data_directory) for _ in range(environments)]
    observations, infos = zip(*(env.reset(seed=seed + index) for index, env in enumerate(envs)))
    observations = list(observations); infos = list(infos)
    steps: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    model.eval()
    for _ in range(horizon):
        batch_observations = np.asarray(observations, dtype=np.float32)
        batch_masks = np.asarray([info["action_mask"] for info in infos], dtype=bool)
        with torch.no_grad():
            action, log_prob, value = model.act(torch.as_tensor(batch_observations, device=device),
                                                torch.as_tensor(batch_masks, device=device))
        actions = action.cpu().numpy()
        rewards = np.empty(environments, dtype=np.float32)
        dones = np.zeros(environments, dtype=bool)
        next_observations: list[np.ndarray] = []; next_infos: list[dict[str, object]] = []
        for index, env in enumerate(envs):
            observation, reward, terminated, truncated, info = env.step(int(actions[index]))
            done = terminated or truncated
            rewards[index], dones[index] = reward, done
            if done:
                observation, info = env.reset()
            next_observations.append(observation); next_infos.append(info)
        steps.append((batch_observations, actions, rewards, dones, log_prob.cpu().numpy(), value.cpu().numpy(), batch_masks))
        observations, infos = next_observations, next_infos

    def environment_major(index: int) -> np.ndarray:
        values = np.stack([step[index] for step in steps], axis=0)
        return np.swapaxes(values, 0, 1).reshape((-1, *values.shape[2:]))

    dones = environment_major(3).astype(bool)
    # A collection cutoff is an explicit bootstrap boundary for every lane.
    dones.reshape(environments, horizon)[:, -1] = True
    rollout = Rollouts(environment_major(0), environment_major(1).astype(np.int64), environment_major(2), dones,
                       SOURCE, environment_major(4), environment_major(5), environment_major(6).astype(bool), None)
    rollout.validate(ACTION_DIM)
    return rollout


def train_terrain_ppo(checkpoint_in: Path, checkpoint_out: Path, data_directory: Path, *, updates: int,
                      environments: int, horizon: int, epochs: int, minibatch_size: int,
                      device: torch.device, seed: int) -> dict[str, object]:
    model = load_model(checkpoint_in, device)
    if model.observation_dim != OBSERVATION_DIM or model.action_dim != ACTION_DIM:
        raise ValueError("terrain PPO requires the current lane-v3 observation and action dimensions")
    config = PPOConfig(epochs=epochs, minibatch_size=minibatch_size)
    started = time.perf_counter(); metrics: list[dict[str, float]] = []
    for update in range(updates):
        rollout = collect_rollout(model, environments, horizon, data_directory, device, seed + update * environments)
        metrics.append(_ppo_update_for_source(rollout, model, device, config, SOURCE))
    elapsed = time.perf_counter() - started
    _checkpoint(checkpoint_out, model, None, {
        "algorithm": "terrain_headless_gymnasium_ppo",
        "source": SOURCE,
        "real_dota_verified": False,
        "observation_version": OBSERVATION_VERSION,
        "updates": updates, "environments": environments, "horizon": horizon,
        "config": asdict(config), "metrics": metrics[-1],
        "data_directory": str(data_directory),
        "limitations": "Approximate terrain/GridNav/NPC environment; local-Dota evaluation required.",
    })
    samples = updates * environments * horizon
    return {"source": SOURCE, "samples": samples, "seconds": elapsed,
            "samples_per_second": samples / max(elapsed, 1e-9), "metrics": metrics[-1],
            "output": str(checkpoint_out), "real_dota_verified": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--updates", type=int, default=8); parser.add_argument("--environments", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=96); parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024); parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.updates < 1:
        raise SystemExit("--updates must be positive")
    print(train_terrain_ppo(args.checkpoint, args.output, args.data_directory, updates=args.updates,
                            environments=args.environments, horizon=args.horizon, epochs=args.epochs,
                            minibatch_size=args.minibatch_size, device=select_device(args.device), seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
