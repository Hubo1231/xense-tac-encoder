"""Training entrypoint."""
from __future__ import annotations

import dataclasses
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

# Allow `python scripts/train.py` to run from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model.config as _config
from model.factory import (
    build_dataloader_from_config,
    build_loss_from_config,
    build_model_from_config,
    build_optimizer_from_config,
    resolve_device,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


@dataclass
class TrainResult:
    checkpoint: str | None
    epochs: int
    global_step: int
    last_losses: dict[str, float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed)


def _move_batch(batch, device: torch.device):
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    if isinstance(batch, dict):
        if "image" in batch:
            batch = batch["image"]
        else:
            raise KeyError("Dict batch must contain an 'image' key")
    return batch.to(device, non_blocking=True)


def _loss_items(losses: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: value.detach().item()
        for key, value in losses.items()
        if torch.is_tensor(value) and value.numel() == 1
    }


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    epoch: int,
    global_step: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": dict(config),
            "model": config.get("model", {}),
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


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


def _wandb_config(config: _config.TrainConfig | Mapping[str, Any], config_dict: Mapping[str, Any]) -> dict[str, Any]:
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)
    return dict(config_dict)


def init_wandb(config: _config.TrainConfig | Mapping[str, Any], config_dict: Mapping[str, Any]):
    wandb_cfg = config_dict.get("wandb", {})
    train_cfg = config_dict.get("training", {})
    enabled = bool(
        wandb_cfg.get("enabled", False)
        or train_cfg.get("wandb_enabled", False)
        or config_dict.get("wandb_enabled", False)
    )
    if not enabled:
        return None
    if wandb is None:
        raise ImportError("启用 wandb 需要 `pip install wandb`")

    project = wandb_cfg.get("project") or config_dict.get("project_name", "eval-tactile-encoder")
    name = wandb_cfg.get("name") or config_dict.get("exp_name") or config_dict.get("name")
    mode = wandb_cfg.get("mode")
    tags = wandb_cfg.get("tags")
    run_id = wandb_cfg.get("id")
    resume = wandb_cfg.get("resume")

    return wandb.init(
        project=project,
        name=name,
        config=_wandb_config(config, config_dict),
        mode=mode,
        tags=tags,
        id=run_id,
        resume=resume,
    )


def main(config: _config.TrainConfig) -> TrainResult:
    config_dict = _config.to_dict(config)
    seed = config_dict.get("seed")
    if seed is not None:
        set_seed(int(seed))

    train_cfg = config_dict.get("training", {})
    device = resolve_device(config_dict, "training")
    loader = build_dataloader_from_config(config_dict, "train", device=device)
    model = build_model_from_config(config_dict).to(device)
    loss_fn = build_loss_from_config(config_dict)
    optimizer = build_optimizer_from_config(model, config_dict)
    wandb_run = init_wandb(config, config_dict)

    epochs = int(train_cfg.get("epochs", 20))
    log_every = int(train_cfg.get("log_every", 20))
    output = train_cfg.get("output")
    global_step = 0
    last_losses: dict[str, float] = {}

    try:
        model.train()
        for epoch in range(1, epochs + 1):
            for batch in loader:
                batch = _move_batch(batch, device)
                outputs = model(batch)
                losses = loss_fn(outputs, batch)

                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()

                last_losses = _loss_items(losses)
                if log_every > 0 and global_step % log_every == 0:
                    msg = " ".join(f"{key}={value:.4f}" for key, value in last_losses.items())
                    print(f"epoch {epoch:>3d} step {global_step:>6d} | {msg}")
                    if wandb_run is not None:
                        wandb.log(
                            {f"train/{key}": value for key, value in last_losses.items()}
                            | {"train/epoch": epoch},
                            step=global_step,
                        )
                global_step += 1

        if output:
            save_checkpoint(output, model, optimizer, config_dict, epoch=epochs, global_step=global_step)
            print(f"model saved to {output}")
            if wandb_run is not None:
                wandb.save(str(output), base_path=str(Path(output).parent))

        if wandb_run is not None:
            wandb.summary["epochs"] = epochs
            wandb.summary["global_step"] = global_step
            for key, value in last_losses.items():
                wandb.summary[f"last/{key}"] = value

        return TrainResult(
            checkpoint=str(output) if output else None,
            epochs=epochs,
            global_step=global_step,
            last_losses=last_losses,
        )
    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main(_config.cli())
