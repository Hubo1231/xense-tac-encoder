"""Training entrypoint."""
from __future__ import annotations

import dataclasses
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

# Allow `python scripts/train.py` to run from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.training.config as _config
from src.training.data_loader import create_torch_data_loader
from src.training.factory import resolve_device
from src.training.losses import LossWeights

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def _config_as_dict(config: _config.TrainConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)


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
    config: _config.TrainConfig,
    epoch: int,
    global_step: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = _config_as_dict(config)
    torch.save(
        {
            "config": config_dict,
            "model": dataclasses.asdict(config.model),
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


def _build_optimizer(model: torch.nn.Module, config: _config.TrainConfig) -> torch.optim.Optimizer:
    name = config.optimizer.name.lower()
    params = {
        "lr": config.optimizer.lr,
        "weight_decay": config.optimizer.weight_decay,
    }
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **params)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **params)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **params)
    raise KeyError(f"未知 optimizer: {name}; 可选: adamw, adam, sgd")


def _attach_loss_weights(model: torch.nn.Module, config: _config.TrainConfig) -> None:
    if hasattr(model, "loss_weights"):
        model.loss_weights = LossWeights(
            mse=config.loss.mse,
            grad=config.loss.grad,
            kld=config.loss.kld,
            ssim=config.loss.ssim,
        )


def init_wandb(config: _config.TrainConfig):
    if not config.wandb_enabled:
        return None
    if wandb is None:
        raise ImportError("启用 wandb 需要 `pip install wandb`")

    return wandb.init(
        project=config.project_name,
        name=config.resolved_exp_name,
        config=_config_as_dict(config),
        mode=config.wandb_mode,
    )


def main(config: _config.TrainConfig):

    set_seed(config.seed)

    device = resolve_device({"device": config.device}, "training")
    data_loader = create_torch_data_loader(config, split="train", pin_memory=device.type == "cuda")
    model = config.model.create(device=device)
    _attach_loss_weights(model, config)
    optimizer = _build_optimizer(model, config)
    wandb_run = init_wandb(config)

    epochs = int(config.epochs)
    log_every = int(config.log_interval)
    output = str(config.checkpoint_path)
    global_step = 0
    last_losses: dict[str, float] = {}

    try:
        model.train()
        for epoch in range(1, epochs + 1):
            for batch in data_loader:
                batch = _move_batch(batch, device)
                losses = model.compute_loss(batch)

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
            save_checkpoint(output, model, optimizer, config, epoch=epochs, global_step=global_step)
            print(f"model saved to {output}")
            if wandb_run is not None:
                wandb.save(str(output), base_path=str(Path(output).parent))

        if wandb_run is not None:
            wandb.summary["epochs"] = epochs
            wandb.summary["global_step"] = global_step
            for key, value in last_losses.items():
                wandb.summary[f"last/{key}"] = value

    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main(_config.cli())
