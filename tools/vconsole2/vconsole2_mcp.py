#!/usr/bin/env python3
"""Local-only Source 2 lab MCP for the Dota PPO custom-game drill.

It uses no window handles, keyboard injection, binary modification, or public
servers. Every engine command is sent only to a loopback ``-netconport``
listener and written to a local audit log.
"""

from __future__ import annotations

import json
import os
import re
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any


SERVERDATA_RESPONSE_VALUE = 0
SERVERDATA_EXECCOMMAND = 2
SERVERDATA_AUTH_RESPONSE = 2
SERVERDATA_AUTH = 3
MAX_RCON_PACKET_SIZE = 4 * 1024 * 1024
DEFAULT_LOCAL_COMMANDS = frozenset({"status", "rl_ppo_progress"})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
DEFAULT_DOTA_TOOLS_EXE = Path(r"C:\Program Files (x86)\Steam\steamapps\common\dota 2 beta\game\bin\win64\dota2.exe")
DEFAULT_LAB_ADDON = "rl_ppo_local"
DEFAULT_LAB_MAP = "template_map"


class RconError(RuntimeError):
    """Raised for an RCON connection, authentication, or protocol error."""


@dataclass(frozen=True)
class RconTarget:
    host: str
    port: int
    password: str

    @classmethod
    def from_environment(cls) -> "RconTarget":
        host = os.environ.get("VCONSOLE_RCON_HOST", "127.0.0.1")
        password = os.environ.get("VCONSOLE_RCON_PASSWORD", "")
        raw_port = os.environ.get("VCONSOLE_RCON_PORT", "27015")
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise RconError("VCONSOLE_RCON_PORT must be an integer.") from exc
        if not host:
            raise RconError("VCONSOLE_RCON_HOST cannot be empty.")
        if not 1 <= port <= 65535:
            raise RconError("VCONSOLE_RCON_PORT must be between 1 and 65535.")
        if not password:
            raise RconError(
                "VCONSOLE_RCON_PASSWORD is not configured. Set it in the MCP server environment."
            )
        return cls(host, port, password)

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class NetConsoleTarget:
    host: str
    port: int

    @classmethod
    def from_environment(cls) -> "NetConsoleTarget":
        host = os.environ.get("VCONSOLE_NETCONSOLE_HOST", "127.0.0.1")
        raw_port = os.environ.get("VCONSOLE_NETCONSOLE_PORT", "2121")
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise RconError("VCONSOLE_NETCONSOLE_PORT must be an integer.") from exc
        if not host:
            raise RconError("VCONSOLE_NETCONSOLE_HOST cannot be empty.")
        if not 1 <= port <= 65535:
            raise RconError("VCONSOLE_NETCONSOLE_PORT must be between 1 and 65535.")
        return cls(host, port)

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def require_loopback(self) -> None:
        if self.host.lower() not in LOOPBACK_HOSTS:
            raise RconError("The local training MCP permits only a loopback netconsole host.")


class RconClient:
    def __init__(self, target: RconTarget, timeout: float = 5.0) -> None:
        self.target = target
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self._request_ids = count(1)

    def __enter__(self) -> "RconClient":
        try:
            self.sock = socket.create_connection((self.target.host, self.target.port), self.timeout)
            self.sock.settimeout(self.timeout)
            self.authenticate()
            return self
        except OSError as exc:
            self.close()
            raise RconError(f"Could not connect to RCON server at {self.target.address}: {exc}") from exc

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def _send_packet(self, request_id: int, packet_type: int, body: str) -> None:
        if self.sock is None:
            raise RconError("RCON socket is not connected.")
        encoded = body.encode("utf-8")
        payload = struct.pack("<ii", request_id, packet_type) + encoded + b"\x00\x00"
        self.sock.sendall(struct.pack("<i", len(payload)) + payload)

    def _read_exactly(self, size: int) -> bytes:
        if self.sock is None:
            raise RconError("RCON socket is not connected.")
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise RconError("RCON server closed the connection.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_packet(self) -> tuple[int, int, str]:
        size = struct.unpack("<i", self._read_exactly(4))[0]
        if size < 10 or size > MAX_RCON_PACKET_SIZE:
            raise RconError(f"Invalid RCON packet size: {size}.")
        payload = self._read_exactly(size)
        request_id, packet_type = struct.unpack("<ii", payload[:8])
        if payload[-2:] != b"\x00\x00":
            raise RconError("Malformed RCON packet terminator.")
        return request_id, packet_type, payload[8:-2].decode("utf-8", errors="replace")

    def authenticate(self) -> None:
        request_id = next(self._request_ids)
        self._send_packet(request_id, SERVERDATA_AUTH, self.target.password)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            response_id, response_type, _ = self._read_packet()
            if response_type == SERVERDATA_AUTH_RESPONSE:
                if response_id == -1:
                    raise RconError("RCON authentication was rejected.")
                if response_id == request_id:
                    return
        raise RconError("Timed out waiting for RCON authentication response.")

    def execute(self, command: str, timeout: float | None = None) -> str:
        if not command or not command.strip():
            raise RconError("Command cannot be empty.")
        if "\x00" in command:
            raise RconError("Command cannot contain NUL characters.")
        request_id = next(self._request_ids)
        self._send_packet(request_id, SERVERDATA_EXECCOMMAND, command)

        # Source RCON may split a response into packets but has no reliable
        # terminator packet. Read all immediately available response packets,
        # using a short idle timeout after the first response.
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        output: list[str] = []
        received = False
        assert self.sock is not None
        original_timeout = self.sock.gettimeout()
        try:
            while time.monotonic() < deadline:
                remaining = max(0.01, deadline - time.monotonic())
                self.sock.settimeout(min(0.15 if received else remaining, remaining))
                try:
                    response_id, response_type, body = self._read_packet()
                except socket.timeout:
                    if received:
                        break
                    continue
                except RconError as exc:
                    # Some RCON implementations close immediately after a
                    # completed response instead of leaving the session idle.
                    if received and str(exc) == "RCON server closed the connection.":
                        break
                    raise
                if response_id == request_id and response_type == SERVERDATA_RESPONSE_VALUE:
                    output.append(body)
                    received = True
        finally:
            self.sock.settimeout(original_timeout)
        return "".join(output)


class NetConsoleClient:
    """A direct client for Source 2's optional -netconport TCP listener."""

    def __init__(self, target: NetConsoleTarget, timeout: float = 5.0) -> None:
        self.target = target
        self.timeout = timeout
        self.sock: socket.socket | None = None

    def __enter__(self) -> "NetConsoleClient":
        try:
            self.sock = socket.create_connection((self.target.host, self.target.port), self.timeout)
            self.sock.settimeout(self.timeout)
            return self
        except OSError as exc:
            self.close()
            raise RconError(
                f"Could not connect to netconsole at {self.target.address}: {exc}. "
                "Start the game with -netconport <port> and use a localhost-only port."
            ) from exc

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def execute(self, command: str, timeout: float | None = None) -> str:
        if not command or not command.strip():
            raise RconError("Command cannot be empty.")
        if "\x00" in command or "\n" in command or "\r" in command:
            raise RconError("Netconsole commands cannot contain NUL or newline characters.")
        if self.sock is None:
            raise RconError("Netconsole socket is not connected.")
        self.sock.sendall(command.encode("utf-8") + b"\n")
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        output: list[bytes] = []
        received = False
        original_timeout = self.sock.gettimeout()
        try:
            while time.monotonic() < deadline:
                remaining = max(0.01, deadline - time.monotonic())
                self.sock.settimeout(min(0.15 if received else remaining, remaining))
                try:
                    chunk = self.sock.recv(4096)
                except socket.timeout:
                    if received:
                        break
                    continue
                if not chunk:
                    break
                output.append(chunk)
                received = True
        finally:
            self.sock.settimeout(original_timeout)
        return b"".join(output).decode("utf-8", errors="replace")

    def read_logs(self, *, max_lines: int, max_bytes: int, timeout: float) -> str:
        """Read a bounded amount of text from a localhost netconsole session."""
        if self.sock is None:
            raise RconError("Netconsole socket is not connected.")
        deadline = time.monotonic() + timeout
        output: list[bytes] = []
        size = 0
        original_timeout = self.sock.gettimeout()
        try:
            while time.monotonic() < deadline and size < max_bytes:
                remaining = max(0.01, deadline - time.monotonic())
                self.sock.settimeout(min(0.2, remaining))
                try:
                    chunk = self.sock.recv(min(4096, max_bytes - size))
                except socket.timeout:
                    break
                if not chunk:
                    break
                output.append(chunk)
                size += len(chunk)
                if b"".join(output).count(b"\n") >= max_lines:
                    break
        finally:
            self.sock.settimeout(original_timeout)
        return limit_log_text(b"".join(output).decode("utf-8", errors="replace"), max_lines, max_bytes)


def limit_log_text(text: str, max_lines: int, max_bytes: int) -> str:
    """Make returned console output safe to display in an MCP transcript."""
    encoded = text.encode("utf-8", errors="replace")[:max_bytes]
    bounded = encoded.decode("utf-8", errors="replace")
    lines = bounded.splitlines()
    result = "\n".join(lines[:max_lines])
    if len(lines) > max_lines or len(text.encode("utf-8", errors="replace")) > max_bytes:
        result += "\n[truncated by local MCP limit]"
    return result


def local_command_set() -> frozenset[str]:
    raw = os.environ.get("VCONSOLE_LOCAL_COMMANDS", "")
    if not raw.strip():
        return DEFAULT_LOCAL_COMMANDS
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def validate_local_command(command: str) -> None:
    if not command or not command.strip() or "\x00" in command or "\n" in command or "\r" in command:
        raise RconError("Command must be one non-empty console line.")
    verb = command.strip().split(maxsplit=1)[0].lower()
    if verb not in local_command_set():
        allowed = ", ".join(sorted(local_command_set()))
        raise RconError(f"Command '{verb}' is not allowed by VCONSOLE_LOCAL_COMMANDS. Allowed: {allowed}.")


def validate_console_line(command: str) -> str:
    command = command.strip()
    if not command or "\x00" in command or "\n" in command or "\r" in command:
        raise RconError("Command must be one non-empty console line.")
    return command


def arbitrary_local_commands_enabled() -> bool:
    return os.environ.get("VCONSOLE_ALLOW_ARBITRARY_LOCAL_COMMANDS", "").strip().lower() in {"1", "true", "yes"}


def audit_console_command(tool: str, target: NetConsoleTarget, command: str, *, success: bool,
                          output: str = "", error: str | None = None) -> None:
    """Append command metadata without polluting MCP stdout."""
    root = sf1v1_training_root()
    path = root / "logs" / "local_console_audit.jsonl"
    path.parent.mkdir(exist_ok=True)
    payload = {
        "timestamp_unix": time.time(), "tool": tool, "host": target.host, "port": target.port,
        "command": command, "success": success, "output_bytes": len(output.encode("utf-8", errors="replace")),
    }
    if error is not None:
        payload["error"] = error
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def execute_local_console(command: str, *, timeout: float, tool: str, require_opt_in: bool) -> str:
    command = validate_console_line(command)
    if require_opt_in and not arbitrary_local_commands_enabled():
        raise RconError("Arbitrary local commands are disabled. Set VCONSOLE_ALLOW_ARBITRARY_LOCAL_COMMANDS=true in the MCP environment.")
    target = NetConsoleTarget.from_environment()
    target.require_loopback()
    try:
        with NetConsoleClient(target, timeout) as client:
            output = client.execute(command, timeout)
    except (RconError, OSError) as error:
        audit_console_command(tool, target, command, success=False, error=str(error))
        raise
    audit_console_command(tool, target, command, success=True, output=output)
    return output


def dota_tools_executable() -> Path:
    configured = os.environ.get("DOTA_TOOLS_EXE")
    executable = Path(configured) if configured else DEFAULT_DOTA_TOOLS_EXE
    executable = executable.resolve()
    if executable.name.lower() != "dota2.exe" or not executable.is_file():
        raise RconError(f"DOTA_TOOLS_EXE must point to the installed dota2.exe: {executable}")
    return executable


def dota_process_running() -> bool:
    """Use tasklist rather than UI/process injection to avoid duplicate clients."""
    completed = subprocess.run(["tasklist", "/FI", "IMAGENAME eq dota2.exe", "/NH"], capture_output=True,
                               text=True, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return "dota2.exe" in completed.stdout.lower()


def start_dota_tools(port: int) -> str:
    if not 1 <= port <= 65535:
        raise RconError("port must be between 1 and 65535.")
    target = NetConsoleTarget("127.0.0.1", port)
    if bridge_health(port):
        return f"Dota netconsole is already listening on {target.address}."
    if dota_process_running():
        raise RconError("A Dota process is already running without this netconsole listener; refusing to start a second client.")
    executable = dota_tools_executable()
    root = sf1v1_training_root()
    log_path = root / "logs" / "dota_tools.log"
    log_path.parent.mkdir(exist_ok=True)
    command = [str(executable), "-tools", "-console", "-netconport", str(port)]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(command, cwd=executable.parents[2], stdin=subprocess.DEVNULL,
                                   stdout=log_file, stderr=subprocess.STDOUT, creationflags=creationflags,
                                   close_fds=True)
    return f"Started local Dota Tools process {process.pid}; waiting for netconsole at {target.address}. Log: {log_path}"


def validate_addon_name(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_/-]+", value):
        raise RconError(f"{label} contains unsupported characters.")
    return value


def launch_local_addon(addon: str, map_name: str, timeout: float) -> str:
    addon = validate_addon_name(addon, "addon")
    map_name = validate_addon_name(map_name, "map_name")
    command = f"dota_launch_custom_game {addon} {map_name}"
    return execute_local_console(command, timeout=timeout, tool="local_addon_launch", require_opt_in=False)


def sf1v1_training_root() -> Path:
    configured = os.environ.get("SF1V1_TRAINING_ROOT") or os.environ.get("REPLAY_TRAINING_ROOT")
    root = Path(configured) if configured else Path(__file__).resolve().parents[1] / "dota2" / "sf1v1_training"
    root = root.resolve()
    if not (root / "src" / "dota_ppo" / "bridge.py").is_file():
        raise RconError(f"SF1V1_TRAINING_ROOT is not an sf1v1_training checkout: {root}")
    return root


def safe_filename(value: str, *, suffix: str, label: str) -> str:
    path = Path(value)
    if not value or path.name != value or path.suffix.lower() != suffix:
        raise RconError(f"{label} must be a single filename ending in {suffix}.")
    return value


def safe_run_name(value: str) -> str:
    """Allow a human-readable evaluation stem, never a path or shell fragment."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", value):
        raise RconError("run_name must start with a letter and contain only letters, digits, _ or -.")
    return value


def read_local_lab_state() -> dict[str, Any]:
    path = sf1v1_training_root() / "logs" / "local_lab_run.json"
    if not path.is_file():
        return {"status": "idle", "state_file": str(path)}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RconError(f"Local lab state file is malformed: {error}") from error
    if not isinstance(state, dict):
        raise RconError("Local lab state file must contain a JSON object.")
    state["state_file"] = str(path)
    return state


def start_local_lab_runner(checkpoint_name: str, run_name: str, batches: int, device: str,
                           calibration: bool, collection_role: str = "evaluations") -> str:
    """Detach one bounded local-only rollout collector with visible state/log files."""
    if device not in {"cpu", "cuda"}:
        raise RconError("device must be 'cpu' or 'cuda'.")
    if not 1 <= batches <= 6:
        raise RconError("batches must be between 1 and 6.")
    if collection_role not in {"training", "evaluations"}:
        raise RconError("collection_role must be 'training' or 'evaluations'.")
    root = sf1v1_training_root()
    safe_filename(checkpoint_name, suffix=".pt", label="checkpoint_name")
    run_name = safe_run_name(run_name)
    if not (root / "checkpoints" / checkpoint_name).is_file():
        raise RconError(f"Checkpoint does not exist: {root / 'checkpoints' / checkpoint_name}")
    state = read_local_lab_state()
    if state.get("status") in {"starting", "collecting"}:
        raise RconError("A local lab collection is already marked active; inspect local_lab_runner_status first.")
    planned = [root / "data" / collection_role / f"{run_name}_{batch:03d}.npz" for batch in range(1, batches + 1)]
    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise RconError("Refusing to overwrite existing local archive(s): " + ", ".join(existing))
    log_path = root / "logs" / "local_lab_runner.log"
    command = [sys.executable, str(Path(__file__).with_name("local_lab_runner.py")),
               "--checkpoint-name", checkpoint_name, "--run-name", run_name,
               "--batches", str(batches), "--addon", DEFAULT_LAB_ADDON,
               "--map-name", DEFAULT_LAB_MAP, "--device", device,
               "--collection-role", collection_role]
    if calibration:
        command.append("--calibration")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(command, cwd=Path(__file__).parent, stdin=subprocess.DEVNULL,
                                   stdout=log_file, stderr=subprocess.STDOUT,
                                   creationflags=creationflags, close_fds=True)
    return (f"Started local {collection_role} runner process {process.pid} for {batches} {DEFAULT_LAB_MAP} archive(s). "
            f"State: {root / 'logs' / 'local_lab_run.json'} Log: {log_path}")


def bridge_health(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def start_bridge(checkpoint_name: str, rollout_name: str, device: str, port: int,
                 calibration_name: str | None = None, collection_role: str = "training", *,
                 return_pid: bool = False) -> str | tuple[str, int]:
    """Start only the loopback policy bridge; it never starts or controls Dota."""
    if device not in {"cpu", "cuda"}:
        raise RconError("device must be 'cpu' or 'cuda'.")
    if not 1 <= port <= 65535:
        raise RconError("port must be between 1 and 65535.")
    if collection_role not in {"training", "evaluations"}:
        raise RconError("collection_role must be 'training' or 'evaluations'.")
    if bridge_health(port):
        raise RconError(f"A listener is already active on 127.0.0.1:{port}; refusing to replace it.")
    root = sf1v1_training_root()
    checkpoint = root / "checkpoints" / safe_filename(checkpoint_name, suffix=".pt", label="checkpoint_name")
    if not checkpoint.is_file():
        raise RconError(f"Checkpoint does not exist: {checkpoint}")
    rollout = root / "data" / collection_role / safe_filename(rollout_name, suffix=".npz", label="rollout_name")
    if "{batch" not in rollout_name:
        raise RconError("rollout_name must include a literal {batch...} placeholder for non-overwriting archives.")
    calibration: Path | None = None
    if calibration_name:
        calibration = root / "data" / "calibration" / safe_filename(
            calibration_name, suffix=".jsonl", label="calibration_name")
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "local_bridge.log"
    bridge_role = "training" if collection_role == "training" else "evaluation"
    command = ["dota-ppo", "bridge", str(checkpoint), "--rollouts", str(rollout), "--device", device,
               "--port", str(port), "--collection-role", bridge_role]
    if calibration is not None:
        command.extend(("--calibration", str(calibration)))
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
                                   creationflags=creationflags, close_fds=True)
    calibration_note = f" Calibration: {calibration}" if calibration is not None else ""
    message = (f"Started {collection_role} loopback PPO bridge process {process.pid} on 127.0.0.1:{port}. "
               f"Dota was not started or controlled.{calibration_note} Log: {log_path}")
    return (message, process.pid) if return_pid else message


def text_result(message: str, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": message}]}
    if is_error:
        result["isError"] = True
    return result


TOOLS = [
    {
        "name": "netconsole_status",
        "description": "Validate a Source 2 netconsole endpoint, started by launching the game with -netconport <port>. This is a direct TCP command interface, not VConsole2 UI automation.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "netconsole_logs",
        "description": "Read a bounded number of log lines from a human-started, localhost-only Source 2 -netconport listener. It sends no console command.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_lines": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Maximum returned lines; default 80."},
                "max_bytes": {"type": "integer", "minimum": 256, "maximum": 65536, "description": "Maximum returned UTF-8 bytes; default 16384."},
                "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 10, "description": "How long to wait for output; default 1 second."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "netconsole_execute_local",
        "description": "Execute one allowlisted diagnostic command through the localhost netconsole. The allowed command verbs come only from VCONSOLE_LOCAL_COMMANDS; default: status and rl_ppo_progress.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "One permitted local diagnostic command."},
                "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 10, "description": "Optional response timeout; default 3 seconds."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "name": "netconsole_execute_any_local",
        "description": "Execute one arbitrary console line only against the configured loopback netconsole. Requires the explicit VCONSOLE_ALLOW_ARBITRARY_LOCAL_COMMANDS=true opt-in and appends an audit record locally.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "One localhost-only Source 2 console line."},
                "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 30, "description": "Optional response timeout; default 5 seconds."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "name": "dota_tools_status",
        "description": "Report whether a local Dota process and the configured loopback netconsole are present.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "dota_tools_start",
        "description": "Launch the installed Dota 2 Tools client locally with -tools, -console, and the loopback netconsole port. It refuses to start a second Dota process.",
        "inputSchema": {"type": "object", "properties": {"port": {"type": "integer", "minimum": 1, "maximum": 65535}}, "additionalProperties": False},
    },
    {
        "name": "local_addon_launch",
        "description": "Launch a local Workshop addon through the loopback console using dota_launch_custom_game. This never creates or joins a public lobby.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "addon": {"type": "string", "description": "Local addon name; default rl_ppo_local."},
                "map_name": {"type": "string", "description": "Map name; default template_map."},
                "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 30, "description": "Optional response timeout; default 5 seconds."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "local_bridge_status",
        "description": "Report whether the localhost PPO bridge is listening. It does not inspect or control Dota.",
        "inputSchema": {"type": "object", "properties": {"port": {"type": "integer", "minimum": 1, "maximum": 65535}}, "additionalProperties": False},
    },
    {
        "name": "local_bridge_start",
        "description": "Start the sf1v1_training loopback PPO bridge from a checkpoint under its checkpoints directory. It never starts Dota or automates the game UI.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "checkpoint_name": {"type": "string", "description": "Existing .pt filename in sf1v1_training/checkpoints."},
                "rollout_name": {"type": "string", "description": "Output .npz filename in data/training containing literal {batch...}, e.g. train_{batch:03d}.npz."},
                "calibration_name": {"type": "string", "description": "Optional .jsonl filename in data/calibration for fresh local telemetry."},
                "collection_role": {"type": "string", "enum": ["training", "evaluations"], "description": "Default training; evaluations are held out from PPO."},
                "device": {"type": "string", "enum": ["cuda", "cpu"], "description": "Default cuda."},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535, "description": "Default 8765."},
            },
            "required": ["checkpoint_name", "rollout_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "local_lab_runner_start",
        "description": "Start one bounded, detached local-only lane collection. It starts Dota Tools if needed, starts the loopback bridge, launches rl_ppo_local on template_map, collects 1-6 non-overwriting archives, and writes a local evaluation report. It never joins a public game.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "checkpoint_name": {"type": "string", "description": "Existing .pt filename in sf1v1_training/checkpoints."},
                "run_name": {"type": "string", "description": "New archive/report prefix using letters, digits, _ or -."},
                "batches": {"type": "integer", "minimum": 1, "maximum": 6, "description": "Completed archives to collect; default 3."},
                "device": {"type": "string", "enum": ["cuda", "cpu"], "description": "Default cuda."},
                "calibration": {"type": "boolean", "description": "Also record fresh local calibration telemetry; default false."},
                "collection_role": {"type": "string", "enum": ["training", "evaluations"], "description": "Default evaluations; choose training only for an on-policy PPO batch."},
            },
            "required": ["checkpoint_name", "run_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "local_lab_runner_status",
        "description": "Read the bounded local-lab runner state file. It does not send a Dota command or control a game.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def handle_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        timeout = float(arguments.get("timeout_seconds", 5.0))
        if not 0.1 <= timeout <= 30:
            raise RconError("timeout_seconds must be between 0.1 and 30.")
        if name == "netconsole_status":
            target = NetConsoleTarget.from_environment()
            target.require_loopback()
            with NetConsoleClient(target):
                pass
            return text_result(f"Connected to netconsole at {target.address}.")
        if name == "netconsole_logs":
            target = NetConsoleTarget.from_environment()
            target.require_loopback()
            max_lines = int(arguments.get("max_lines", 80))
            max_bytes = int(arguments.get("max_bytes", 16384))
            log_timeout = float(arguments.get("timeout_seconds", 1.0))
            if not 1 <= max_lines <= 200 or not 256 <= max_bytes <= 65536 or not 0.1 <= log_timeout <= 10:
                raise RconError("Invalid log limit. Use 1-200 lines, 256-65536 bytes, and 0.1-10 seconds.")
            with NetConsoleClient(target, log_timeout) as client:
                output = client.read_logs(max_lines=max_lines, max_bytes=max_bytes, timeout=log_timeout)
            return text_result(output or "No netconsole text arrived within the requested limit.")
        if name == "netconsole_execute_local":
            command = str(arguments.get("command", ""))
            validate_local_command(command)
            local_timeout = float(arguments.get("timeout_seconds", 3.0))
            if not 0.1 <= local_timeout <= 10:
                raise RconError("timeout_seconds must be between 0.1 and 10.")
            output = execute_local_console(command, timeout=local_timeout, tool="netconsole_execute_local",
                                           require_opt_in=False)
            return text_result(limit_log_text(output, 200, 65536) or "Command sent successfully (the listener returned no text output).")
        if name == "netconsole_execute_any_local":
            output = execute_local_console(str(arguments.get("command", "")), timeout=timeout,
                                           tool="netconsole_execute_any_local", require_opt_in=True)
            return text_result(limit_log_text(output, 200, 65536) or "Command sent successfully (the listener returned no text output).")
        if name == "dota_tools_status":
            target = NetConsoleTarget.from_environment()
            target.require_loopback()
            return text_result(json.dumps({"dota_process_running": dota_process_running(),
                                           "netconsole_listening": bridge_health(target.port),
                                           "netconsole": target.address}, indent=2))
        if name == "dota_tools_start":
            port = int(arguments.get("port", NetConsoleTarget.from_environment().port))
            return text_result(start_dota_tools(port))
        if name == "local_addon_launch":
            output = launch_local_addon(str(arguments.get("addon", DEFAULT_LAB_ADDON)),
                                        str(arguments.get("map_name", DEFAULT_LAB_MAP)), timeout)
            return text_result(limit_log_text(output, 200, 65536) or "Local addon launch command sent successfully.")
        if name == "local_bridge_status":
            port = int(arguments.get("port", 8765))
            if not 1 <= port <= 65535:
                raise RconError("port must be between 1 and 65535.")
            return text_result(f"Local PPO bridge {'is listening' if bridge_health(port) else 'is not listening'} on 127.0.0.1:{port}.")
        if name == "local_bridge_start":
            calibration_name = arguments.get("calibration_name")
            return text_result(start_bridge(str(arguments.get("checkpoint_name", "")), str(arguments.get("rollout_name", "")),
                                            str(arguments.get("device", "cuda")), int(arguments.get("port", 8765)),
                                            None if calibration_name is None else str(calibration_name),
                                            str(arguments.get("collection_role", "training"))))
        if name == "local_lab_runner_start":
            return text_result(start_local_lab_runner(str(arguments.get("checkpoint_name", "")),
                                                       str(arguments.get("run_name", "")),
                                                       int(arguments.get("batches", 3)),
                                                       str(arguments.get("device", "cuda")),
                                                       bool(arguments.get("calibration", False)),
                                                       str(arguments.get("collection_role", "evaluations"))))
        if name == "local_lab_runner_status":
            return text_result(json.dumps(read_local_lab_state(), indent=2))
        return text_result(f"Unknown tool: {name}", is_error=True)
    except (RconError, OSError, ValueError) as exc:
        return text_result(str(exc), is_error=True)


def send_response(request_id: Any, result: dict[str, Any] | None = None, error: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            method = request.get("method")
            request_id = request.get("id")
            if method == "initialize":
                send_response(request_id, {
                    "protocolVersion": request.get("params", {}).get("protocolVersion", "2024-11-05"),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "vconsole2-local", "version": "1.1.0"},
                })
            elif method == "tools/list":
                send_response(request_id, {"tools": TOOLS})
            elif method == "tools/call":
                params = request.get("params", {})
                send_response(request_id, handle_tool(params.get("name", ""), params.get("arguments", {})))
            elif request_id is not None:
                send_response(request_id, error={"code": -32601, "message": f"Method not found: {method}"})
        except json.JSONDecodeError:
            # A malformed notification has no safe id to return; keep serving.
            print("Ignoring malformed JSON-RPC input.", file=sys.stderr, flush=True)
        except Exception as exc:  # Prevent a malformed client request from killing the stdio server.
            if 'request_id' in locals() and request_id is not None:
                send_response(request_id, error={"code": -32603, "message": str(exc)})
            else:
                print(f"Unhandled MCP server error: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
