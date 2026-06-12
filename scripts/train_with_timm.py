#!/usr/bin/env python3
"""Train a tactile-image VAE with a timm encoder.

借助 timm 提供的几个工具：
  * timm.create_model            - 创建 backbone
  * timm.data.resolve_data_config / create_transform - 预处理
  * timm.optim.create_optimizer_v2 - 优化器
  * timm.scheduler.create_scheduler_v2 - 学习率调度

每个 backbone 都通过 timm 的官方调用方式得到 (B, num_features) 的图像
embedding：

    model = timm.create_model(name, pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    embedding = model(transforms(img).unsqueeze(0))  # (B, C)

运行：
    python scripts/train_with_timm.py --config configs/vae/vit_base_patch16_dinov3_lvd1689m.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms.functional as tvF
import torchvision.utils
import yaml
from tqdm.auto import tqdm

os.environ.setdefault("WANDB_ERROR_REPORTING", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timm import utils
from timm.data import create_transform, resolve_data_config
from timm.models import create_model, safe_model_name
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from src.models.mae import TactileMAE
from src.models.simmim import TactileSimMIM
from src.models.vae import TactileVAE
from src.training.data_loader import TactileDataset, list_images, split_image_paths
from src.training.losses import LossWeights

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


_logger = logging.getLogger("train_with_timm")

DEFAULT_CONFIG = "configs/vae/vit_base_patch16_dinov3_lvd1689m.yaml"


# Keys required regardless of the training task (`task: vae | mae`).
COMMON_REQUIRED_KEYS: tuple[str, ...] = (
    # Dataset
    "data_dir", "eval_ratio",
    # Model — every backbone goes through the standard
    #   model(transforms(img).unsqueeze(0))  ->  (B, num_features)
    # path with num_classes=0 hardcoded; no features_only switch.
    "model", "pretrained", "pretrained_path",
    "initial_checkpoint", "resume", "no_resume_opt",
    "in_chans",
    # Device
    "device",
    # Optimizer / scheduler
    "opt", "lr", "weight_decay", "clip_grad",
    "sched", "epochs", "warmup_epochs", "warmup_lr", "min_lr", "cooldown_epochs",
    # Loader / logging
    "batch_size", "workers", "seed",
    "log_interval", "val_interval", "output", "experiment",
    "log_wandb",
)

# VAE-only keys (latent projection + reconstruction loss weights).
VAE_REQUIRED_KEYS: tuple[str, ...] = (
    "latent_dim", "decoder_hidden_channels",
    "w_mse", "w_grad", "w_kld", "w_ssim",
    "w_mix", "mix_alpha", "use_ms_ssim",
)

# MAE defaults — every field is optional in YAML; the `mae:` block overrides these.
MAE_DEFAULTS: dict[str, Any] = {
    "mask_ratio": 0.75,
    "decoder_embed_dim": 512,
    "decoder_depth": 4,
    "decoder_num_heads": 16,
    "norm_pix_loss": True,
}

# SimMIM defaults — backbone-agnostic masked image modeling; `simmim:` block overrides.
SIMMIM_DEFAULTS: dict[str, Any] = {
    "mask_patch_size": 32,
    "mask_ratio": 0.6,
}

VALID_TASKS = ("vae", "mae", "simmim")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _validate_config(config: MutableMapping[str, Any]) -> None:
    task = config.setdefault("task", "vae")
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {VALID_TASKS}, got {task!r}.")

    required = list(COMMON_REQUIRED_KEYS)
    if task == "vae":
        required += list(VAE_REQUIRED_KEYS)
    missing = sorted(k for k in required if k not in config)
    if missing:
        raise ValueError(f"Config (task={task}) is missing required keys: {', '.join(missing)}")
    if not 0.0 < config["eval_ratio"] < 1.0:
        raise ValueError("eval_ratio must be in (0, 1)")


def _load_config(config_path: Path, task_override: str | None = None) -> tuple[SimpleNamespace, str]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    config = _load_yaml(config_path)
    config["config"] = str(config_path)
    if task_override is not None:
        # CLI --task wins over the config's own `task`; applied before validation so the
        # task-specific required-key set is checked correctly.
        config["task"] = task_override
    _validate_config(config)
    args_text = yaml.safe_dump(config, default_flow_style=False, sort_keys=False)
    return SimpleNamespace(**config), args_text


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_timm_backbone(args: SimpleNamespace) -> nn.Module:
    """Create a timm backbone exactly as in the official example::

        model = timm.create_model(name, pretrained=True, num_classes=0)
        output = model(transforms(img).unsqueeze(0))  # (B, num_features)

    ``num_classes`` is hardcoded to 0 so timm strips the classifier head and
    every supported backbone returns a flat ``(B, num_features)`` embedding.
    """
    pretrained_path = getattr(args, "pretrained_path", "") or ""

    factory_kwargs: dict[str, Any] = {
        "pretrained": bool(args.pretrained) or bool(pretrained_path),
        "num_classes": 0,
        "in_chans": int(args.in_chans),
    }
    if pretrained_path:
        factory_kwargs["pretrained_cfg_overlay"] = {
            "file": pretrained_path,
            "num_classes": -1,  # don't try to remap classifier weights — we have no head.
        }

    extra = dict(getattr(args, "model_kwargs", {}) or {})
    return create_model(args.model, **factory_kwargs, **extra)


def _resolve_embedding_dim(encoder: nn.Module) -> int:
    """Read the ``model(x) -> (B, C)`` embedding dim from timm metadata.

    ``timm.data.resolve_model_data_config`` only carries preprocessing fields
    (input_size / mean / std / interpolation / crop_pct / crop_mode), so the
    feature dim has to come from the model object itself.

    Important: prefer ``head_hidden_size`` over ``num_features``. They match
    for most backbones (ResNet, ConvNeXt, ViT…), but for backbones whose head
    expands the channel count before pooling — most notably EfficientViT-B3
    (``num_features=512`` vs ``head_hidden_size=2560``) — only
    ``head_hidden_size`` matches what ``model(x)`` actually returns when
    ``num_classes=0``. ``TactileVAE.encode`` re-validates the dim at the
    first real forward, so a wrong reading here surfaces immediately rather
    than silently corrupting training.
    """
    for attr in ("head_hidden_size", "num_features"):
        value = getattr(encoder, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(
        "timm encoder must expose head_hidden_size or num_features to determine embedding dim."
    )


def _build_vae(
    args: SimpleNamespace,
    timm_backbone: nn.Module,
    data_config: Mapping[str, Any],
) -> TactileVAE:
    feature_dim = _resolve_embedding_dim(timm_backbone)
    args.encoder_feature_dim = feature_dim
    img_size = int(data_config["input_size"][-1])
    if img_size <= 0:
        raise ValueError(f"Resolved timm input size must be positive, got {img_size}.")
    if img_size % 32 != 0 and getattr(args, "decoder_hidden_spatial", None) is None:
        raise ValueError("Resolved timm input size must be divisible by 32 unless decoder_hidden_spatial is set.")
    hidden_spatial = getattr(args, "decoder_hidden_spatial", None) or (img_size // 32)
    return TactileVAE(
        encoder_backbone=timm_backbone,
        feature_dim=feature_dim,
        latent_dim=args.latent_dim,
        decoder_hidden_channels=args.decoder_hidden_channels,
        decoder_hidden_spatial=hidden_spatial,
        loss_weights=LossWeights(
            mse=args.w_mse,
            grad=args.w_grad,
            kld=args.w_kld,
            ssim=args.w_ssim,
            mix=args.w_mix,
            mix_alpha=args.mix_alpha,
            use_ms_ssim=bool(args.use_ms_ssim),
        ),
        image_size=img_size,
    )


def _build_mae(
    args: SimpleNamespace,
    timm_backbone: nn.Module,
    data_config: Mapping[str, Any],
) -> TactileMAE:
    """Build a TactileMAE; requires a timm ViT (patch tokens)."""
    is_vit = (
        hasattr(timm_backbone, "patch_embed")
        and hasattr(timm_backbone, "blocks")
        and hasattr(timm_backbone, "_pos_embed")
        and getattr(timm_backbone, "num_prefix_tokens", None) is not None
    )
    if not is_vit:
        raise ValueError(
            f"task=mae only supports ViT backbones (need patch_embed/blocks/_pos_embed); "
            f"model={args.model!r} is not a ViT. Use a vit_*_dinov3 config, or set task=vae."
        )

    img_size = int(data_config["input_size"][-1])
    patch_size = timm_backbone.patch_embed.patch_size
    patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else int(patch_size)

    mae_cfg = {**MAE_DEFAULTS, **dict(getattr(args, "mae", {}) or {})}
    args.mae = mae_cfg  # write back resolved values for logging / checkpoint args
    return TactileMAE(
        encoder_backbone=timm_backbone,
        image_size=img_size,
        patch_size=patch_size,
        in_chans=int(args.in_chans),
        mask_ratio=float(mae_cfg["mask_ratio"]),
        decoder_embed_dim=int(mae_cfg["decoder_embed_dim"]),
        decoder_depth=int(mae_cfg["decoder_depth"]),
        decoder_num_heads=int(mae_cfg["decoder_num_heads"]),
        norm_pix_loss=bool(mae_cfg["norm_pix_loss"]),
    )


def _build_simmim(
    args: SimpleNamespace,
    timm_backbone: nn.Module,
    data_config: Mapping[str, Any],
) -> TactileSimMIM:
    """Build a TactileSimMIM; works with any timm backbone via forward_features."""
    img_size = int(data_config["input_size"][-1])
    cfg = {**SIMMIM_DEFAULTS, **dict(getattr(args, "simmim", {}) or {})}
    args.simmim = cfg  # write back resolved values for logging / checkpoint args
    return TactileSimMIM(
        encoder_backbone=timm_backbone,
        image_size=img_size,
        in_chans=int(args.in_chans),
        mask_patch_size=int(cfg["mask_patch_size"]),
        mask_ratio=float(cfg["mask_ratio"]),
    )


def _build_model(
    args: SimpleNamespace,
    timm_backbone: nn.Module,
    data_config: Mapping[str, Any],
) -> nn.Module:
    """Dispatch on ``args.task`` to build a VAE, an MAE, or a SimMIM model."""
    if args.task == "mae":
        return _build_mae(args, timm_backbone, data_config)
    if args.task == "simmim":
        return _build_simmim(args, timm_backbone, data_config)
    return _build_vae(args, timm_backbone, data_config)


def _build_datasets(
    args: SimpleNamespace,
    data_config: Mapping[str, Any],
) -> tuple[TactileDataset, TactileDataset]:
    paths = list_images(args.data_dir)
    splits = split_image_paths(paths, eval_ratio=args.eval_ratio, seed=args.seed)
    # Match the official timm example exactly:
    #   transforms = timm.data.create_transform(**data_config, is_training=False)
    # For tactile reconstruction we deliberately skip training-time augmentation;
    # color jitter / RandErase would corrupt the very signal the VAE has to
    # reconstruct, and the eval transform already does resize + center crop +
    # ImageNet-style normalization sized to the backbone's pretrained_cfg.
    transform = create_transform(**data_config, is_training=False)
    return (
        TactileDataset(splits["train"], transform=transform),
        TactileDataset(splits["eval"], transform=transform),
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)


def _build_loader(
    dataset: torch.utils.data.Dataset,
    *,
    args: SimpleNamespace,
    is_training: bool,
) -> torch.utils.data.DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=is_training,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=is_training and len(dataset) >= args.batch_size,
        persistent_workers=args.workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def init_wandb_run(args: SimpleNamespace, *, exp_name: str, output_dir: Path) -> Any | None:
    if not args.log_wandb:
        return None
    if not HAS_WANDB:
        _logger.warning("wandb requested but package is not installed.")
        return None
    wandb_root = output_dir / "wandb"
    wandb_root.mkdir(parents=True, exist_ok=True)
    try:
        return wandb.init(
            project=args.wandb_project,
            name=getattr(args, "wandb_name", "") or exp_name,
            config=vars(args),
            dir=str(wandb_root),
        )
    except Exception as exc:
        _logger.warning("wandb init failed; continuing without wandb logging: %s", exc)
        return None


def _move_batch(batch: Any, device: torch.device) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    if isinstance(batch, dict):
        batch = batch["image"]
    return batch.to(device=device, non_blocking=True)


def _loss_items(losses: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: value.detach().item()
        for key, value in losses.items()
        if torch.is_tensor(value) and value.numel() == 1
    }


def _avg_lr(optimizer: torch.optim.Optimizer) -> float:
    return sum(g["lr"] for g in optimizer.param_groups) / len(optimizer.param_groups)


def _flatten_metrics(prefix: str, metrics: Mapping[str, float] | None) -> dict[str, float]:
    if metrics is None:
        return {}
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def _denormalize_batch(batch: torch.Tensor, data_config: Mapping[str, Any]) -> torch.Tensor:
    mean = torch.tensor(data_config["mean"], device=batch.device, dtype=batch.dtype).view(1, -1, 1, 1)
    std = torch.tensor(data_config["std"], device=batch.device, dtype=batch.dtype).view(1, -1, 1, 1)
    return (batch * std + mean).clamp(0.0, 1.0)


@torch.inference_mode()
def get_wandb_reconstructions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    args: SimpleNamespace,
    *,
    device: torch.device,
    data_config: Mapping[str, Any],
    epoch: int,
) -> list[Any]:
    if not (args.log_wandb and args.wandb_log_images and HAS_WANDB):
        return []
    if (epoch + 1) % args.wandb_image_interval != 0:
        return []

    model.eval()
    batch = _move_batch(next(iter(loader)), device)
    count = min(args.wandb_num_images, batch.shape[0])
    inputs = batch[:count]
    recon = model.reconstruct(inputs).detach()
    inputs = _denormalize_batch(inputs.detach(), data_config)
    recon = _denormalize_batch(recon, data_config)

    images = []
    for idx in range(count):
        comparison = torch.stack([inputs[idx], recon[idx]], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=2, padding=2)
        images.append(wandb.Image(tvF.to_pil_image(grid.cpu()), caption=f"sample {idx}: input | recon"))
    return images


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: SimpleNamespace,
    lr_scheduler: Any,
    metrics: Mapping[str, float],
    best_metric: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
        "metrics": dict(metrics),
        "best_metric": best_metric,
    }
    if lr_scheduler is not None and hasattr(lr_scheduler, "state_dict"):
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(payload, path)


def _load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: Any = None,
    model_only: bool = False,
) -> tuple[int, float | None]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        _logger.warning("Missing parameters while loading %s: %d", path, len(missing))
    if unexpected:
        _logger.warning("Unexpected parameters while loading %s: %d", path, len(unexpected))
    if model_only or not isinstance(checkpoint, dict):
        return 0, None
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if lr_scheduler is not None and "lr_scheduler" in checkpoint and hasattr(lr_scheduler, "load_state_dict"):
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return int(checkpoint.get("epoch", -1)) + 1, checkpoint.get("best_metric")


def _update_meters(
    meters: MutableMapping[str, Any], items: Mapping[str, float], batch_size: int
) -> None:
    """Accumulate every scalar loss term except ``total`` into per-key meters.

    Works for both tasks without hardcoding term names: VAE emits mse/grad/kld/(ssim/mix),
    MAE emits recon. New keys get a meter on first sight.
    """
    for key, value in items.items():
        if key == "total":
            continue
        meter = meters.get(key)
        if meter is None:
            meter = meters[key] = utils.AverageMeter()
        meter.update(value, batch_size)


def _meters_summary(loss_avg: float, meters: Mapping[str, Any]) -> OrderedDict[str, float]:
    summary: OrderedDict[str, float] = OrderedDict(loss=loss_avg)
    for key, meter in meters.items():
        summary[key] = meter.avg
    return summary


def _meters_postfix(loss_avg: float, meters: Mapping[str, Any]) -> dict[str, str]:
    postfix = {"loss": f"{loss_avg:.4f}"}
    for key, meter in meters.items():
        postfix[key] = f"{meter.avg:.4f}"
    return postfix


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    args: SimpleNamespace,
    *,
    device: torch.device,
) -> OrderedDict[str, float]:
    model.train()
    losses_m = utils.AverageMeter()
    term_meters: OrderedDict[str, Any] = OrderedDict()
    batch_time_m = utils.AverageMeter()

    last_idx = len(loader) - 1
    end = time.time()
    progress = tqdm(loader, desc=f"Train {epoch}", dynamic_ncols=True, leave=False)
    for batch_idx, batch in enumerate(progress):
        batch = _move_batch(batch, device)
        loss_dict = model.compute_loss(batch)
        loss = loss_dict["total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad is not None:
            utils.dispatch_clip_grad(model.parameters(), args.clip_grad, mode="norm")
        optimizer.step()

        items = _loss_items(loss_dict)
        bs = batch.shape[0]
        losses_m.update(items["total"], bs)
        _update_meters(term_meters, items, bs)
        batch_time_m.update(time.time() - end)
        end = time.time()
        lr = _avg_lr(optimizer)
        progress.set_postfix(lr=f"{lr:.2e}", **_meters_postfix(losses_m.avg, term_meters))

        if batch_idx % args.log_interval == 0 or batch_idx == last_idx:
            terms = " ".join(f"{k.upper()} {m.avg:.4f}" for k, m in term_meters.items())
            _logger.info(
                "Train: %d [%4d/%d] Loss %.4f (%.4f) %s Time %.3fs LR %.3e",
                epoch, batch_idx, len(loader),
                losses_m.val, losses_m.avg, terms, batch_time_m.val, lr,
            )

    return _meters_summary(losses_m.avg, term_meters)


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    args: SimpleNamespace,
    *,
    device: torch.device,
) -> OrderedDict[str, float]:
    model.eval()
    losses_m = utils.AverageMeter()
    term_meters: OrderedDict[str, Any] = OrderedDict()
    batch_time_m = utils.AverageMeter()

    end = time.time()
    last_idx = len(loader) - 1
    progress = tqdm(loader, desc="Eval", dynamic_ncols=True, leave=False)
    for batch_idx, batch in enumerate(progress):
        batch = _move_batch(batch, device)
        loss_dict = model.compute_loss(batch)

        items = _loss_items(loss_dict)
        bs = batch.shape[0]
        losses_m.update(items["total"], bs)
        _update_meters(term_meters, items, bs)
        batch_time_m.update(time.time() - end)
        end = time.time()
        progress.set_postfix(**_meters_postfix(losses_m.avg, term_meters))

        if batch_idx % args.log_interval == 0 or batch_idx == last_idx:
            terms = " ".join(f"{k.upper()} {m.avg:.4f}" for k, m in term_meters.items())
            _logger.info(
                "Eval: [%4d/%d] Loss %.4f (%.4f) %s Time %.3fs",
                batch_idx, len(loader),
                losses_m.val, losses_m.avg, terms, batch_time_m.val,
            )

    return _meters_summary(losses_m.avg, term_meters)


def _parse_cli() -> tuple[Path, str | None]:
    parser = argparse.ArgumentParser(description="Train a tactile-image self-supervised model with a timm encoder.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=(
            "Path to a YAML config under configs/{vae,mae,simmim}/. Either an absolute path, "
            "a path relative to the repo root, or just the filename (e.g. "
            "'resnet50_a1_in1k.yaml')."
        ),
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default=None,
        choices=VALID_TASKS,
        help=(
            "Override the config's `task` (vae|mae|simmim). Lets one config set run different "
            "self-supervised tasks without duplicating YAML, e.g. `--task simmim`."
        ),
    )
    args = parser.parse_args()
    task_override = args.task

    repo_root = Path(__file__).resolve().parents[1]
    raw_paths = [Path(args.config)]
    if raw_paths[0].suffix == "":
        raw_paths.append(Path(f"{args.config}.yaml"))

    for raw in raw_paths:
        if raw.is_absolute() and raw.exists():
            return raw, task_override

        rel = repo_root / raw
        if rel.exists():
            return rel, task_override

        in_configs = repo_root / "configs" / raw
        if in_configs.exists():
            return in_configs, task_override

    task_dirs = [task_override] if task_override is not None else list(VALID_TASKS)
    for raw in raw_paths:
        for task_dir in task_dirs:
            in_task_configs = repo_root / "configs" / task_dir / raw
            if in_task_configs.exists():
                return in_task_configs, task_override
    raise FileNotFoundError(
        f"Could not locate config '{args.config}'. Tried absolute/relative paths, configs/, "
        "and configs/{vae,mae,simmim}/."
    )


def main() -> None:
    utils.setup_default_logging()
    config_path, task_override = _parse_cli()
    _logger.info("Loading config from %s", config_path)
    args, args_text = _load_config(config_path, task_override=task_override)

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    utils.random_seed(args.seed, 0)

    timm_backbone = _build_timm_backbone(args)
    data_config = resolve_data_config({}, model=timm_backbone, verbose=True)
    args.img_size = int(data_config["input_size"][-1])
    args.input_size = tuple(data_config["input_size"])
    model = _build_model(args, timm_backbone, data_config).to(device)

    if args.initial_checkpoint:
        _load_checkpoint(args.initial_checkpoint, model=model, model_only=True)

    param_count = sum(p.numel() for p in model.parameters())
    _logger.info(
        "Model %s %s created: params %.2fM, encoder features %s",
        safe_model_name(args.model),
        args.task.upper(),
        param_count / 1e6,
        getattr(args, "encoder_feature_dim", "unknown"),
    )

    dataset_train, dataset_eval = _build_datasets(args, data_config)
    _logger.info("Dataset: train=%d eval=%d root=%s", len(dataset_train), len(dataset_eval), args.data_dir)

    loader_train = _build_loader(dataset_train, args=args, is_training=True)
    loader_eval = _build_loader(dataset_eval, args=args, is_training=False)

    optimizer = create_optimizer_v2(model, opt=args.opt, lr=args.lr, weight_decay=args.weight_decay)
    _logger.info("Created %s optimizer (lr=%.3e, wd=%.3e)", type(optimizer).__name__, args.lr, args.weight_decay)

    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        sched=args.sched,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        min_lr=args.min_lr,
        cooldown_epochs=args.cooldown_epochs,
    )

    start_epoch = 0
    best_metric: float | None = None
    if args.resume:
        start_epoch, best_metric = _load_checkpoint(
            args.resume,
            model=model,
            optimizer=None if args.no_resume_opt else optimizer,
            lr_scheduler=None if args.no_resume_opt else lr_scheduler,
        )
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    exp_name = args.experiment or "-".join(
        [datetime.now().strftime("%Y%m%d-%H%M%S"), safe_model_name(args.model), f"{args.task}{args.img_size}"]
    )
    output_dir = Path(utils.get_outdir(args.output, exp_name))
    with open(output_dir / "args.yaml", "w", encoding="utf-8") as f:
        f.write(args_text)

    wandb_run = init_wandb_run(args, exp_name=exp_name, output_dir=output_dir)
    if wandb_run is not None and HAS_WANDB:
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("eval/*", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")

    results = []
    try:
        for epoch in range(start_epoch, num_epochs):
            train_metrics = train_one_epoch(epoch, model, loader_train, optimizer, args, device=device)

            eval_metrics = None
            if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == num_epochs:
                eval_metrics = validate(model, loader_eval, args, device=device)

            metric = (eval_metrics or train_metrics)["loss"]
            is_best = best_metric is None or metric < best_metric
            if is_best:
                best_metric = metric
                _save_checkpoint(
                    output_dir / "model_best.pt",
                    epoch=epoch, model=model, optimizer=optimizer, args=args,
                    lr_scheduler=lr_scheduler,
                    metrics=eval_metrics or train_metrics, best_metric=best_metric,
                )
            _save_checkpoint(
                output_dir / "checkpoint_last.pt",
                epoch=epoch, model=model, optimizer=optimizer, args=args,
                lr_scheduler=lr_scheduler,
                metrics=eval_metrics or train_metrics, best_metric=best_metric,
            )

            lr = _avg_lr(optimizer)
            utils.update_summary(
                epoch, train_metrics, eval_metrics,
                filename=str(output_dir / "summary.csv"),
                lr=lr, write_header=epoch == start_epoch, log_wandb=False,
            )

            if wandb_run is not None:
                payload: dict[str, Any] = {"epoch": epoch, "lr": lr, "best/loss": best_metric}
                payload.update(_flatten_metrics("train", train_metrics))
                payload.update(_flatten_metrics("eval", eval_metrics))
                recon_images = get_wandb_reconstructions(
                    model, loader_eval, args, device=device, data_config=data_config, epoch=epoch,
                )
                if recon_images:
                    payload["reconstructions"] = recon_images
                wandb.log(payload, step=epoch + 1)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, metric)

            entry = {"epoch": epoch, "train": dict(train_metrics)}
            if eval_metrics is not None:
                entry["validation"] = dict(eval_metrics)
            results.append(entry)
            _logger.info("Epoch %d done. metric=%.6f best=%.6f", epoch, metric, best_metric)

    except KeyboardInterrupt:
        _logger.warning("Interrupted by user.")
    finally:
        if wandb_run is not None:
            wandb.finish()

    _logger.info("Best loss: %s", best_metric)
    print(f"--result\n{json.dumps(results[-10:], indent=4)}")


if __name__ == "__main__":
    main()
