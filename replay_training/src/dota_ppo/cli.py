"""Command line workflow: parse .dem -> approximate replay bootstrap -> PPO -> export."""

from __future__ import annotations

import argparse
from pathlib import Path

from .actions import ACTION_DIM
from .bridge import serve_bridge
from .calibration import analyze_events
from .comparison import compare_rollout_sets
from .curriculum import build_synthetic_lane_expert
from .data import (load_rollouts, load_trajectories, merge_rollouts,
                   require_current_local_observation_version, require_current_local_reward_version,
                   require_policy_checkpoint)
from .evaluation import evaluate_rollouts
from .export import export_torchscript, serve
from .input_logs import canonicalize_jsonl, join_orders_to_states
from .headless_lane import train_headless_lane
from .replays import build_replay_dataset, parse_dem_files
from .supervisor import SupervisorConfig, run_supervisor
from .train import PPOConfig, behavior_clone, select_device, train_ppo


def _paths(value: str) -> list[Path]:
    path = Path(value)
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.dem")) if path.is_dir() else sorted(Path().glob(value))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dota-ppo")
    commands = p.add_subparsers(dest="command", required=True)
    parse = commands.add_parser("parse"); parse.add_argument("dems"); parse.add_argument("--output", type=Path, required=True)
    dataset = commands.add_parser("dataset"); dataset.add_argument("parsed", type=Path); dataset.add_argument("--hero", default="nevermore"); dataset.add_argument("--output", type=Path, required=True); dataset.add_argument("--stride", type=int, default=1)
    synthetic = commands.add_parser("synthetic-lane-data", help="generate fast scripted lane demonstrations for BC warm start")
    synthetic.add_argument("--output", type=Path, required=True); synthetic.add_argument("--steps", type=int, default=100_000); synthetic.add_argument("--seed", type=int, default=7)
    controls = commands.add_parser("canonicalize-inputs", help="map controlled-lobby semantic orders to fixed deployment keys")
    controls.add_argument("input_log", type=Path); controls.add_argument("--output", type=Path, required=True); controls.add_argument("--layout", type=Path)
    join = commands.add_parser("join-human-orders", help="align canonical local human orders with parsed player state for behavior cloning")
    join.add_argument("parsed_match", type=Path); join.add_argument("labels", type=Path); join.add_argument("--output", type=Path, required=True); join.add_argument("--max-tick-gap", type=int, default=30)
    merge = commands.add_parser("merge-rollouts", help="combine exact local-lobby rollout archives before PPO")
    merge.add_argument("rollouts", type=Path, nargs="+"); merge.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate-rollouts", help="summarize comparable local-lobby rollout metrics")
    evaluate.add_argument("rollouts", type=Path, nargs="+"); evaluate.add_argument("--output", type=Path)
    compare = commands.add_parser("compare-rollouts", help="compare frozen baseline and candidate exact local-lobby evaluation sets")
    compare.add_argument("baseline", type=Path, nargs="+"); compare.add_argument("--candidate", type=Path, nargs="+", required=True)
    compare.add_argument("--minimum-archives", type=int, default=3); compare.add_argument("--minimum-game-seconds", type=float, default=150.0); compare.add_argument("--output", type=Path, required=True)
    bc = commands.add_parser("bc"); bc.add_argument("dataset", type=Path); bc.add_argument("--output", type=Path, required=True); bc.add_argument("--device", default="cuda"); bc.add_argument("--epochs", type=int, default=30)
    ppo = commands.add_parser("ppo"); ppo.add_argument("rollouts", type=Path); ppo.add_argument("--checkpoint", type=Path, required=True); ppo.add_argument("--output", type=Path, required=True); ppo.add_argument("--device", default="cuda"); ppo.add_argument("--epochs", type=int, default=10)
    headless = commands.add_parser("headless-lane-ppo", help="fast approximate lane PPO pretraining; requires real-Dota validation before promotion")
    headless.add_argument("--checkpoint", type=Path, required=True); headless.add_argument("--output", type=Path, required=True); headless.add_argument("--device", default="cuda")
    headless.add_argument("--updates", type=int, default=8); headless.add_argument("--environments", type=int, default=1024); headless.add_argument("--horizon", type=int, default=96); headless.add_argument("--epochs", type=int, default=4); headless.add_argument("--minibatch-size", type=int, default=16_384); headless.add_argument("--seed", type=int, default=7); headless.add_argument("--calibration-report", type=Path)
    export = commands.add_parser("export"); export.add_argument("checkpoint", type=Path); export.add_argument("--output", type=Path, required=True); export.add_argument("--device", default="cpu")
    server = commands.add_parser("serve"); server.add_argument("model", type=Path); server.add_argument("--device", default="cuda")
    bridge = commands.add_parser("bridge", help="run the localhost policy and exact-rollout collector for a local custom lobby")
    bridge.add_argument("checkpoint", type=Path); bridge.add_argument("--rollouts", type=Path, required=True); bridge.add_argument("--port", type=int, default=8765); bridge.add_argument("--device", default="cuda"); bridge.add_argument("--human-orders", type=Path); bridge.add_argument("--calibration", type=Path)
    calibrate = commands.add_parser("analyze-calibration", help="estimate approximate lane-simulator parameters from local calibration JSONL")
    calibrate.add_argument("events", type=Path, nargs="+"); calibrate.add_argument("--output", type=Path, required=True)
    supervise = commands.add_parser("supervise", help="watch newly collected exact local rollouts, evaluate them, and run bounded PPO updates")
    supervise.add_argument("--rollout-dir", type=Path, default=Path("data/rollouts")); supervise.add_argument("--checkpoint", type=Path, required=True); supervise.add_argument("--run-dir", type=Path, required=True)
    supervise.add_argument("--batch-archives", type=int, default=3); supervise.add_argument("--max-batches", type=int, default=1); supervise.add_argument("--max-hours", type=float, default=8.0); supervise.add_argument("--poll-seconds", type=float, default=10.0)
    supervise.add_argument("--min-archive-age-seconds", type=float, default=2.0); supervise.add_argument("--min-steps", type=int, default=100); supervise.add_argument("--min-decisions-per-game-second", type=float, default=3.0); supervise.add_argument("--min-game-seconds", type=float, default=150.0); supervise.add_argument("--max-failures", type=int, default=3); supervise.add_argument("--epochs", type=int, default=10); supervise.add_argument("--device", default="cuda"); supervise.add_argument("--include-existing", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.command == "parse":
        paths = _paths(args.dems)
        if not paths: raise SystemExit(f"No .dem files found for {args.dems}")
        print("\n".join(map(str, parse_dem_files(paths, args.output))))
    elif args.command == "dataset":
        result = build_replay_dataset([p for p in args.parsed.iterdir() if p.is_dir()], args.output, args.hero, args.stride)
        print(result)
    elif args.command == "synthetic-lane-data":
        print(build_synthetic_lane_expert(str(args.output), args.steps, args.seed))
    elif args.command == "canonicalize-inputs":
        print(canonicalize_jsonl(args.input_log, args.output, args.layout))
    elif args.command == "join-human-orders":
        print(join_orders_to_states(args.parsed_match, args.labels, args.output, max_tick_gap=args.max_tick_gap))
    elif args.command == "merge-rollouts":
        print(merge_rollouts(args.rollouts, args.output))
    elif args.command == "evaluate-rollouts":
        print(evaluate_rollouts(args.rollouts, args.output))
    elif args.command == "compare-rollouts":
        print(compare_rollout_sets(args.baseline, args.candidate, minimum_archives=args.minimum_archives,
                                   minimum_game_seconds=args.minimum_game_seconds, output=args.output))
    elif args.command == "bc":
        print(behavior_clone(load_trajectories(args.dataset), ACTION_DIM, select_device(args.device), args.output, epochs=args.epochs))
    elif args.command == "ppo":
        config = PPOConfig(epochs=args.epochs)
        require_current_local_reward_version(args.rollouts)
        require_current_local_observation_version(args.rollouts)
        require_policy_checkpoint(args.rollouts, args.checkpoint)
        print(train_ppo(load_rollouts(args.rollouts), args.checkpoint, args.output, select_device(args.device), config))
    elif args.command == "headless-lane-ppo":
        print(train_headless_lane(args.checkpoint, args.output, select_device(args.device), updates=args.updates,
                                  environments=args.environments, horizon=args.horizon, epochs=args.epochs, seed=args.seed,
                                  calibration_report=args.calibration_report, minibatch_size=args.minibatch_size))
    elif args.command == "export":
        print(export_torchscript(args.checkpoint, args.output, args.device))
    elif args.command == "bridge":
        serve_bridge(args.checkpoint, args.rollouts, port=args.port, device_name=args.device, human_orders=args.human_orders,
                     calibration_output=args.calibration)
    elif args.command == "analyze-calibration":
        print(analyze_events(args.events, args.output))
    elif args.command == "supervise":
        config = SupervisorConfig(args.rollout_dir, args.checkpoint, args.run_dir, args.batch_archives, args.max_batches,
                                  args.max_hours, args.poll_seconds, args.min_archive_age_seconds, args.min_steps,
                                  args.min_decisions_per_game_second, args.max_failures, args.epochs, args.device,
                                  args.include_existing, min_game_seconds=args.min_game_seconds)
        print(run_supervisor(config))
    else:
        serve(args.model, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
