"""TorchScript deployment export and a newline-delimited JSON policy bridge."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .actions import ACTION_NAMES
from .train import load_model, select_device


class _Policy(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(); self.model = model

    def forward(self, observations: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(observations, action_mask)
        return logits.argmax(dim=-1)


def export_torchscript(checkpoint: Path, output: Path, device_name: str = "cpu") -> dict[str, object]:
    device = select_device(device_name)
    model = load_model(checkpoint, device)
    wrapper = _Policy(model).eval()
    example_obs = torch.zeros(1, model.observation_dim, device=device)
    example_mask = torch.ones(1, model.action_dim, dtype=torch.bool, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.trace(wrapper, (example_obs, example_mask)).save(str(output))
    spec = {"observation_dim": model.observation_dim, "actions": list(ACTION_NAMES), "protocol": "JSONL: {observation:[float], action_mask?:[bool]} -> {action:int, action_name:str}"}
    output.with_suffix(".json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec


def serve(model_path: Path, device_name: str = "cuda") -> None:
    """Run the local bridge; a custom-lobby adapter sends one JSON request per line."""
    device = select_device(device_name)
    model = torch.jit.load(str(model_path), map_location=device).eval()
    for line in sys.stdin:
        request = json.loads(line)
        observation = np.asarray(request["observation"], dtype=np.float32)[None, :]
        mask = np.asarray(request.get("action_mask", [True] * len(ACTION_NAMES)), dtype=bool)[None, :]
        with torch.no_grad():
            action = int(model(torch.as_tensor(observation, device=device), torch.as_tensor(mask, device=device)).item())
        print(json.dumps({"action": action, "action_name": ACTION_NAMES[action]}), flush=True)
