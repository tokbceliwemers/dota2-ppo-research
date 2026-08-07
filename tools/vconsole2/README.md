# Bounded local VConsole2 MCP

This directory is the versioned project copy. Replace every `C:\\Path\\To`
example below with paths on your own machine.

This dependency-free Python MCP server runs the **local-only** Dota PPO lab
through its localhost netconsole. It does not automate the VConsole2 window,
send input, alter binaries, connect to a remote/public server, or use
matchmaking.

## What it exposes

- `netconsole_status` checks `127.0.0.1:2121`.
- `netconsole_logs` returns at most 200 lines / 64 KiB and sends no command.
- `netconsole_execute_local` sends a single allowlisted diagnostic command.
- `netconsole_execute_any_local` sends one arbitrary console line only after
  the explicit local opt-in described below; every command is audited.
- `dota_tools_start` launches a local Dota Tools process with `-tools`,
  `-console`, and `-netconport` when no Dota process is already running.
- `local_addon_launch` starts `rl_ppo_local` locally through
  `dota_launch_custom_game rl_ppo_local template_map`.
  The default allowlist is only `status,rl_ppo_progress`.
- `local_bridge_status` and `local_bridge_start` manage only the local Python
  PPO HTTP bridge. They never launch or control Dota.
- `local_lab_runner_start` / `local_lab_runner_status` run and observe one
  bounded, detached local collection on `template_map`. It refuses duplicate
  archive names, joins the local Radiant slot (`jointeam good`), verifies both
  the controlled Shadow Fiend and the passive enemy Shadow Fiend, and records progress in
  `sf1v1_training/logs/local_lab_run.json`. When it finishes or fails, it
  stops only the loopback bridge process it started, so the next run is not
  blocked by a stale listener.

`-netconport` creates a remotely accessible console, so localhost binding is
essential; do not expose it to a LAN or the Internet. The arbitrary command
tool is intentionally opt-in and only accepts loopback targets. Its audit log
is `sf1v1_training/logs/local_console_audit.jsonl`.

## Setup

Start Dota yourself with this launch option before opening the local custom
lobby:

```text
-netconport 2121
```

Then add this MCP entry to `C:\Users\skaya\.codex\config.toml` and restart
Codex so it loads the new stdio server:

```toml
[mcp_servers.vconsole2_local]
command = 'C:\\Path\\To\\python.exe'
args = ['C:\\Path\\To\\dota2\\tools\\vconsole2\\vconsole2_mcp.py']
startup_timeout_sec = 15

[mcp_servers.vconsole2_local.env]
VCONSOLE_NETCONSOLE_HOST = '127.0.0.1'
VCONSOLE_NETCONSOLE_PORT = '2121'
VCONSOLE_LOCAL_COMMANDS = 'status,rl_ppo_progress'
VCONSOLE_ALLOW_ARBITRARY_LOCAL_COMMANDS = 'true'
DOTA_TOOLS_EXE = 'C:\\Path\\To\\dota 2 beta\\game\\bin\\win64\\dota2.exe'
SF1V1_TRAINING_ROOT = 'C:\\Path\\To\\dota2\\sf1v1_training'
```

The allowlisted diagnostic tool remains the default. The separate arbitrary
tool becomes available only when `VCONSOLE_ALLOW_ARBITRARY_LOCAL_COMMANDS` is
`true`; it is still restricted to `127.0.0.1` and creates an audit record.

`local_bridge_start` accepts only an existing `.pt` filename from
`sf1v1_training/checkpoints` and a non-overwriting `.npz` filename containing
the literal `{batch...}` placeholder. By default it writes PPO-eligible data
to `data/training`; select the held-out `evaluations` role only for measurement.
Its optional `calibration_name` accepts
only a `.jsonl` filename under `data/calibration`. It writes process output to
`sf1v1_training/logs/local_bridge.log`.

## Test

```powershell
Set-Location C:\Path\To\dota2\tools\vconsole2
python -m unittest discover -s tests -v
```
