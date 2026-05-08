"""Checkpoint helpers kept for evaluation compatibility."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint: str | Path,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=map_location)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"  [warn] missing parameters: {len(missing)}")
    if unexpected:
        print(f"  [warn] unexpected parameters: {len(unexpected)}")
    return ckpt if isinstance(ckpt, dict) else {"state_dict": state}


def train_from_config(config: Mapping[str, Any]):
    from scripts.train import main

    return main(config)
