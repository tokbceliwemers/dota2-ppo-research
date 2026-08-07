#!/usr/bin/env python3
"""Detached, localhost-only collector for the Dota PPO lane drill.

It starts no public game and uses only the approved local addon.  Its state is
written to ``sf1v1_training/logs/local_lab_run.json`` so an MCP client can
inspect a long run without holding an interactive connection open.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from vconsole2_mcp import (NetConsoleTarget, RconError, bridge_health, execute_local_console,
                            launch_local_addon, sf1v1_training_root, safe_run_name, start_bridge,
                            start_dota_tools)


def write_state(path: Path, **values: object) -> None:
    path.parent.mkdir(exist_ok=True)
    values["updated_unix"] = time.time()
    values["runner_pid"] = os.getpid()
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def wait_for(predicate, timeout_seconds: float, description: str) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(1.0)
    raise RconError(f"Timed out waiting for {description} after {timeout_seconds:.0f} seconds.")


def stable_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    first_size = path.stat().st_size
    time.sleep(1.0)
    return path.is_file() and path.stat().st_size == first_size


def controlled_hero_ready() -> bool:
    """Join the local Radiant slot and confirm that the addon owns a hero.

    ``dota_launch_custom_game`` alone creates a map session but no Player 0.
    The documented follow-up is ``jointeam good``; retrying it while the map
    loads is harmless and avoids starting a bridge-only, hero-less drill.
    """
    execute_local_console("jointeam good", timeout=5.0, tool="local_lab_runner", require_opt_in=False)
    output = execute_local_console("rl_ppo_progress", timeout=5.0, tool="local_lab_runner", require_opt_in=False)
    return "RL PPO progression: level=" in output


def passive_opponent_ready() -> bool:
    """Reject a collection if the required Stage 3 opponent did not spawn."""
    output = execute_local_console("rl_ppo_opponent", timeout=5.0, tool="local_lab_runner", require_opt_in=False)
    return "RL PPO opponent: mode=passive identity=passive_nevermore_v1" in output


def stop_owned_bridge(process_id: int) -> None:
    """Stop only the process tree this runner itself created.

    The console addon remains open for the next run, but the Python listener
    must not leak across completed batches or it would block the next policy.
    """
    if process_id <= 0:
        return
    subprocess.run(["taskkill", "/PID", str(process_id), "/T", "/F"], capture_output=True,
                   text=True, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def run(args: argparse.Namespace) -> dict[str, object]:
    root = sf1v1_training_root()
    args.run_name = safe_run_name(args.run_name)
    state_path = root / "logs" / "local_lab_run.json"
    collection_dir = root / "data" / args.collection_role
    template = collection_dir / f"{args.run_name}_{{batch:03d}}.npz"
    calibration = f"{args.run_name}.jsonl" if args.calibration else None
    target = NetConsoleTarget.from_environment()
    target.require_loopback()
    write_state(state_path, status="starting", run_name=args.run_name, batches=args.batches,
                checkpoint=args.checkpoint_name, collection_role=args.collection_role,
                rollout_template=str(template), calibration=calibration)
    planned = [collection_dir / f"{args.run_name}_{batch:03d}.npz"
               for batch in range(1, args.batches + 1)]
    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise RconError("Refusing to overwrite existing local archive(s): " + ", ".join(existing))
    bridge_process_id: int | None = None
    try:
        if not bridge_health(target.port):
            start_dota_tools(target.port)
            wait_for(lambda: bridge_health(target.port), args.startup_timeout, "Dota netconsole")
        if bridge_health(args.bridge_port):
            raise RconError(f"A PPO bridge already listens on port {args.bridge_port}; refusing to replace it.")
        _, bridge_process_id = start_bridge(args.checkpoint_name, template.name, args.device, args.bridge_port,
                                            calibration, args.collection_role, return_pid=True)
        wait_for(lambda: bridge_health(args.bridge_port), 30.0, "PPO bridge")
        launch_local_addon(args.addon, args.map_name, timeout=10.0)
        wait_for(controlled_hero_ready, args.startup_timeout, "controlled Shadow Fiend on the local team")
        wait_for(passive_opponent_ready, args.startup_timeout, "passive enemy Shadow Fiend")

        archives: list[Path] = []
        for batch in range(1, args.batches + 1):
            archive = collection_dir / f"{args.run_name}_{batch:03d}.npz"
            write_state(state_path, status="collecting", run_name=args.run_name, batch=batch,
                        batches=args.batches, archive=str(archive), archives=[str(item) for item in archives])
            wait_for(lambda: stable_file(archive), args.batch_timeout, f"archive {archive.name}")
            archives.append(archive)
            if batch < args.batches:
                execute_local_console("rl_ppo_restart", timeout=10.0, tool="local_lab_runner", require_opt_in=False)

        report = root / "reports" / f"{args.run_name}_metrics.json"
        command = ["dota-ppo", "evaluate-rollouts", *(str(path) for path in archives), "--output", str(report)]
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60, check=False)
        if completed.returncode != 0:
            raise RconError(f"evaluate-rollouts failed: {completed.stderr.strip() or completed.stdout.strip()}")
        result = {"status": "complete", "run_name": args.run_name, "archives": [str(item) for item in archives],
                  "report": str(report), "evaluation": completed.stdout.strip()}
        write_state(state_path, **result)
        return result
    finally:
        if bridge_process_id is not None:
            stop_owned_bridge(bridge_process_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--addon", default="rl_ppo_local")
    parser.add_argument("--map-name", default="template_map")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bridge-port", type=int, default=8765)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--batch-timeout", type=float, default=600.0)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--collection-role", choices=("training", "evaluations"), default="evaluations",
                        help="training archives may enter PPO; evaluations remain held out.")
    args = parser.parse_args()
    if not 1 <= args.batches <= 6:
        raise SystemExit("--batches must be between 1 and 6")
    try:
        result = run(args)
    except (RconError, OSError, subprocess.SubprocessError) as error:
        root = sf1v1_training_root()
        write_state(root / "logs" / "local_lab_run.json", status="failed", run_name=args.run_name, error=str(error))
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
