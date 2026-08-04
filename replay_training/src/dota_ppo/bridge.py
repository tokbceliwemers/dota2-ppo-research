"""Loopback-only policy service and exact PPO rollout collector.

The service is deliberately small: a local custom-game adapter requests an
action, executes it, then reports its observed reward.  The service keeps the
original observation, sampled action, old log-probability, value, and mask so
the resulting archive is mathematically usable by PPO.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actions import ACTION_NAMES
from .calibration import validate_event
from .controls import canonicalize_input
from .data import Rollouts, checkpoint_sha256, save_rollouts
from .observations import OBSERVATION_DIM, OBSERVATION_VERSION
from .train import load_model, select_device


@dataclass(frozen=True)
class PendingDecision:
    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    old_log_prob: float
    old_value: float
    game_time: float | None
    reward_version: str


class PolicyBridge:
    """Stateful policy sampler; bind it only to trusted localhost clients."""

    def __init__(self, checkpoint: Path, rollout_output: Path, device_name: str = "cuda", human_orders: Path | None = None,
                 calibration_output: Path | None = None) -> None:
        self.device = select_device(device_name)
        self.model = load_model(checkpoint, self.device)
        if self.model.observation_dim != OBSERVATION_DIM:
            raise ValueError(
                f"{checkpoint} uses {self.model.observation_dim} observations; "
                f"the current {OBSERVATION_VERSION} local bridge requires {OBSERVATION_DIM}"
            )
        self.policy_checkpoint_sha256 = checkpoint_sha256(checkpoint)
        self.rollout_output = rollout_output
        self.flush_number = 0
        self.human_orders = human_orders
        self.calibration_output = calibration_output
        self.pending: dict[str, PendingDecision] = {}
        self.rows: list[tuple[PendingDecision, float, bool]] = []
        self.action_requests = 0
        self.transition_requests = 0
        self.calibration_events = 0
        self.lock = threading.Lock()

    def _next_rollout_path(self) -> Path:
        """Render an optional ``{batch}`` filename placeholder safely."""
        template = str(self.rollout_output)
        if "{batch" not in template:
            return self.rollout_output
        try:
            return Path(template.format(batch=self.flush_number + 1))
        except (KeyError, ValueError) as error:
            raise ValueError("rollout template may use only a {batch} placeholder") from error

    def act(self, request: dict[str, Any]) -> dict[str, object]:
        observation = np.asarray(request["observation"], dtype=np.float32)
        if observation.shape != (self.model.observation_dim,):
            raise ValueError(f"observation must have {self.model.observation_dim} values")
        mask = np.asarray(request.get("action_mask", [True] * self.model.action_dim), dtype=bool)
        if mask.shape != (self.model.action_dim,) or not mask.any():
            raise ValueError(f"action_mask must contain {self.model.action_dim} values and permit at least one action")
        game_time = request.get("game_time")
        if game_time is not None:
            game_time = float(game_time)
            if not np.isfinite(game_time):
                raise ValueError("game_time must be finite")
        reward_version = str(request.get("reward_version", "unspecified"))
        if not reward_version or len(reward_version) > 128:
            raise ValueError("reward_version must be a non-empty string of at most 128 characters")
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation[None, :], device=self.device)
            mask_tensor = torch.as_tensor(mask[None, :], device=self.device)
            distribution, values = self.model.distribution(obs_tensor, mask_tensor)
            action_tensor = distribution.probs.argmax(dim=-1) if request.get("deterministic", False) else distribution.sample()
            action = int(action_tensor.item())
            log_prob = float(distribution.log_prob(action_tensor).item())
            value = float(values.item())
        decision_id = uuid.uuid4().hex
        with self.lock:
            self.pending[decision_id] = PendingDecision(observation, mask, action, log_prob, value, game_time, reward_version)
            self.action_requests += 1
        return {"decision_id": decision_id, "action": action, "action_name": ACTION_NAMES[action], "old_log_prob": log_prob, "old_value": value}

    def transition(self, request: dict[str, Any]) -> dict[str, object]:
        decision_id = str(request["decision_id"])
        reward, done = float(request["reward"]), bool(request["done"])
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        with self.lock:
            decision = self.pending.pop(decision_id, None)
            if decision is None:
                raise ValueError("unknown or already committed decision_id")
            self.rows.append((decision, reward, done))
            self.transition_requests += 1
        return {"accepted": True, "steps": len(self.rows)}

    def record_human_order(self, request: dict[str, Any]) -> dict[str, object]:
        """Audit semantic orders from a consenting local participant.

        A VScript order filter cannot see physical keyboard bindings, so
        ``raw_key`` is optional. The semantic inventory/ability slot is enough
        to canonicalize it to the bot layout.
        """
        label = canonicalize_input(request)
        if self.human_orders is None:
            raise ValueError("human-order logging was not enabled for this bridge")
        self.human_orders.parent.mkdir(parents=True, exist_ok=True)
        with self.human_orders.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(request, sort_keys=True) + "\n")
        return {"accepted": True, "action_label": label.action_label, "canonical_key": label.canonical_key, "action_id": label.action_id}

    def record_calibration(self, request: dict[str, Any]) -> dict[str, object]:
        """Append a versioned live-lane measurement without touching PPO rows."""
        if self.calibration_output is None:
            return {"accepted": False, "enabled": False}
        event = validate_event(request)
        self.calibration_output.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            with self.calibration_output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            self.calibration_events += 1
            count = self.calibration_events
        return {"accepted": True, "events": count}

    def flush(self, request: dict[str, Any] | None = None) -> dict[str, object]:
        """Save the current batch, optionally terminally committing one final action.

        The game client's asynchronous callbacks can deliver `/act` just as a
        participant enters the finish console command.  In that case Lua may
        no longer hold the decision id while this collector still does.  The
        adapter supplies the final observed reward, allowing this method to
        retain the exact sampled action as a legitimate terminal transition
        instead of losing the entire batch.
        """
        request = request or {}
        final_reward = request.get("final_reward")
        if final_reward is not None:
            final_reward = float(final_reward)
            if not np.isfinite(final_reward):
                raise ValueError("final_reward must be finite")
        with self.lock:
            if self.pending:
                if final_reward is None:
                    raise ValueError(f"cannot flush with {len(self.pending)} uncommitted decisions")
                if len(self.pending) != 1:
                    raise ValueError(f"cannot safely finalize {len(self.pending)} uncommitted decisions")
                _, decision = self.pending.popitem()
                self.rows.append((decision, final_reward, True))
            if not self.rows:
                raise ValueError("cannot flush an empty rollout")
            rows = list(self.rows)
            # A Source HTTP callback can arrive after a later transition even
            # though each decision carries its authoritative Dota game time.
            # Restore chronological trajectory order before persistence; this
            # keeps terminal flags attached to their exact sampled decisions.
            reordered = False
            if all(row[0].game_time is not None for row in rows):
                original_times = [float(row[0].game_time) for row in rows]
                rows.sort(key=lambda row: float(row[0].game_time))
                reordered = original_times != [float(row[0].game_time) for row in rows]
            decisions, rewards, dones = zip(*rows)
            reward_versions = {row.reward_version for row in decisions}
            if len(reward_versions) != 1:
                raise ValueError("cannot save a rollout containing mixed reward versions")
            data = Rollouts(
                observations=np.stack([row.observation for row in decisions]),
                actions=np.asarray([row.action for row in decisions], dtype=np.int64),
                rewards=np.asarray(rewards, dtype=np.float32), dones=np.asarray(dones, dtype=bool),
                source="local_instrumented_lobby",
                old_log_probs=np.asarray([row.old_log_prob for row in decisions], dtype=np.float32),
                old_values=np.asarray([row.old_value for row in decisions], dtype=np.float32),
                action_masks=np.stack([row.action_mask for row in decisions]),
                game_times=np.asarray([row.game_time for row in decisions], dtype=np.float32)
                if all(row.game_time is not None for row in decisions) else None,
            )
            output = self._next_rollout_path()
            save_rollouts(output, data, {"checkpoint_action_dim": self.model.action_dim,
                                         "checkpoint_observation_dim": self.model.observation_dim,
                                         "observation_version": OBSERVATION_VERSION,
                                         "bridge": "loopback_http",
                                         "reward_version": next(iter(reward_versions)),
                                         "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
                                         "transition_ordering": "game_time_sorted" if reordered else "arrival_order",
                                         "decision_game_time_recorded": all(row.game_time is not None for row in decisions)})
            steps = len(rows)
            self.rows.clear()
            self.flush_number += 1
        return {"saved": str(output), "steps": steps}

    def health(self) -> dict[str, object]:
        """Expose bounded local readiness evidence for scenario recovery."""
        with self.lock:
            return {
                "ok": True, "action_dim": self.model.action_dim,
                "action_requests": self.action_requests,
                "transition_requests": self.transition_requests,
                "calibration_events": self.calibration_events,
                "pending_decisions": len(self.pending),
            }


def _handler(bridge: PolicyBridge) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:  # avoid request spam in training output
            return

        def _reply(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._reply(HTTPStatus.OK, bridge.health())
            else:
                self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65_536:
                    raise ValueError("request body must be 1..65536 bytes")
                request = json.loads(self.rfile.read(size))
                # dkjson serializes an empty Lua table as [] rather than {}.
                # `/flush` has no fields, so accept that harmless representation.
                if self.path == "/flush" and request == []:
                    request = {}
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                if self.path == "/act": result = bridge.act(request)
                elif self.path == "/transition": result = bridge.transition(request)
                elif self.path == "/human-order": result = bridge.record_human_order(request)
                elif self.path == "/calibration": result = bridge.record_calibration(request)
                elif self.path == "/flush": result = bridge.flush(request)
                else:
                    self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"}); return
                self._reply(HTTPStatus.OK, result)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._reply(HTTPStatus.BAD_REQUEST, {"error": str(error)})
    return Handler


def serve_bridge(checkpoint: Path, rollout_output: Path, *, port: int = 8765, device_name: str = "cuda", human_orders: Path | None = None,
                 calibration_output: Path | None = None) -> None:
    bridge = PolicyBridge(checkpoint, rollout_output, device_name, human_orders, calibration_output)
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(bridge))
    print(f"PPO bridge listening on http://127.0.0.1:{port}; Ctrl+C stops it.")
    try:
        server.serve_forever()
    finally:
        server.server_close()
