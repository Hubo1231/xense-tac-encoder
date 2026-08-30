
"""多头监督（multitask）预训练入口。

与 scripts/train_with_timm.py 同级但完全独立：通过 ``import train_with_timm as twt``
复用其构件——

  * twt._build_timm_backbone / twt._resolve_embedding_dim   - timm backbone 与 feature_dim
  * twt._save_checkpoint / twt._load_checkpoint             - checkpoint / resume
  * twt.init_wandb_run / twt._loss_items / twt._avg_lr / twt._flatten_metrics - 日志
  * twt.utils.AverageMeter / update_summary                 - 指标累积与 summary.csv
  * timm create_optimizer_v2 / create_scheduler_v2 / resolve_data_config / create_transform

差异点（本脚本自行实现）：
  * 数据是带标注的 dict batch：``{"image": (B,3,H,W), "targets": {name: tensor}}``；
  * 支持 ``dataset_type: h5`` 直接消费 xensim collection2 的 HDF5 文件，
    训练前对回归目标（depth/flow 等）按 train split 计算并 z-score 归一化；
  * ``model_arch: physical`` 使用 todo.md 描述的
    TactilePhysicalMultiTask（FastViT-T12 + Tiny FPN + dense depth/flow +
    PhysicalPooler bottleneck），否则保留旧 TactileMultiTask；
  * ``_move_batch`` 递归把整个 dict 搬上 GPU，batch size 从 batch["image"] 取。
  * 训练日志除各任务 loss 外，还记录回归头的残差均值（``*_residual``）和
    未裁剪前的全局梯度范数 ``grad_norm``。

运行：
    python scripts/train_multitask.py --config configs/multitask/fastvit_t12_physical_collection2.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from tqdm.auto import tqdm

os.environ.setdefault("WANDB_ERROR_REPORTING", "false")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

import train_with_timm as twt  # noqa: E402
from timm.data import create_transform, resolve_data_config  # noqa: E402
from timm.models import safe_model_name  # noqa: E402
from timm.optim import create_optimizer_v2  # noqa: E402
from timm.scheduler import create_scheduler_v2  # noqa: E402

from src.models.multitask import (  # noqa: E402
    HeadSpec,
    TactileMultiTask,
    TactilePhysicalMultiTask,
    TactileViTPhysicalMultiTask,
    parse_head_specs,
)
from src.datasets.h5_dataset import H5TactileDataset, compute_target_stats  # noqa: E402
from src.datasets.labeled_dataset import LabeledTactileDataset  # noqa: E402

_logger = logging.getLogger("train_multitask")

DEFAULT_CONFIG = "configs/multitask/fastvit_t12_physical_collection2.yaml"

# multitask 必填键 = twt 通用键（已不含 VAE 专属键）+ heads 声明。
# metadata_path 只对 dataset_type=labeled 强制。
MULTITASK_REQUIRED_KEYS: tuple[str, ...] = twt.COMMON_REQUIRED_KEYS + (
    "heads",
)

DEFAULT_PHYSICAL_LOSS_WEIGHTS: dict[str, float] = {
    "dense_depth": 1.0,
    "dense_flow": 1.0,
    "bottleneck_depth": 0.25,
    "bottleneck_flow": 0.25,
}

DEFAULT_VIT_PHYSICAL_LOSS_WEIGHTS: dict[str, float] = {
    "patch_depth": 1.0,
    "patch_flow": 1.0,
    "bottleneck_depth": 0.25,
    "bottleneck_flow": 0.25,
}


def _validate_config(config: dict[str, Any]) -> None:
    """校验 multitask 配置：缺键/非法值直接 ValueError，不做静默默认。"""
    task = config.setdefault("task", "multitask")
    if task != "multitask":
        raise ValueError(f"train_multitask.py 只接受 task=multitask，got {task!r}。")
    missing = sorted(k for k in MULTITASK_REQUIRED_KEYS if k not in config)
    if missing:
        raise ValueError(f"Config (task=multitask) is missing required keys: {', '.join(missing)}")
    if not 0.0 < config["eval_ratio"] < 1.0:
        raise ValueError("eval_ratio must be in (0, 1)")

    dataset_type = config.setdefault("dataset_type", "labeled")
    if dataset_type == "h5":
        if not config.get("h5_paths") and not config.get("h5_dir"):
            raise ValueError("dataset_type=h5 时必须提供 h5_paths 或 h5_dir。")
    elif dataset_type == "labeled":
        if "metadata_path" not in config:
            raise ValueError("dataset_type=labeled 时必须提供 metadata_path。")
    else:
        raise ValueError(f"dataset_type 必须是 h5 或 labeled，got {dataset_type!r}。")

    config.setdefault("normalize_targets", True)
    config.setdefault("image_key", "rgb")
    config.setdefault("model_arch", "linear")
    config.setdefault("log_batches", True)
    if config["model_arch"] == "physical":
        config.setdefault("loss_weights", DEFAULT_PHYSICAL_LOSS_WEIGHTS)
    elif config["model_arch"] == "dinov3_physical":
        config.setdefault("loss_weights", DEFAULT_VIT_PHYSICAL_LOSS_WEIGHTS)
    # heads 块格式在此校验（train: false 的头允许存在，训练时跳过）
    train_specs = parse_head_specs(config["heads"])
    if not train_specs:
        raise ValueError("heads 块里至少需要一个 train: true 的头。")


def load_config(config_path: Path) -> tuple[SimpleNamespace, str]:
    """加载并校验 YAML 配置，返回 (args, args_text)（供评测脚本复用）。"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    config = twt._load_yaml(config_path)
    config["config"] = str(config_path)
    _validate_config(config)
    args_text = yaml.safe_dump(config, default_flow_style=False, sort_keys=False)
    return SimpleNamespace(**config), args_text


def resolve_config_path(config_arg: str) -> Path:
    """解析 --config：绝对路径 / 相对仓库根 / configs/multitask/ 下按文件名找（自动补 .yaml）。"""
    raw_paths = [Path(config_arg)]
    if raw_paths[0].suffix == "":
        raw_paths.append(Path(f"{config_arg}.yaml"))

    for raw in raw_paths:
        if raw.is_absolute() and raw.exists():
            return raw
        rel = _REPO_ROOT / raw
        if rel.exists():
            return rel
        in_configs = _REPO_ROOT / "configs" / "multitask" / raw
        if in_configs.exists():
            return in_configs
    raise FileNotFoundError(
        f"Could not locate config '{config_arg}'. Tried absolute/relative paths and configs/multitask/."
    )


def _parse_cli() -> Path:
    parser = argparse.ArgumentParser(
        description="Train a multi-head supervised tactile model with a timm encoder "
                    "(supports HDF5 collection2 and physical FPN heads)."
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=(
            "Path to a YAML config under configs/multitask/. Either an absolute path, "
            "a path relative to the repo root, or just the filename."
        ),
    )
    return resolve_config_path(parser.parse_args().config)


def _move_batch(batch: Any, device: torch.device) -> Any:
    """递归把 dict batch 里的 tensor 搬上 GPU，保留整个 dict 结构。"""
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: _move_batch(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_batch(value, device) for value in batch)
    return batch


def _compute_grad_norm(model: torch.nn.Module) -> float:
    """计算所有可训练参数的全局 L2 梯度范数（未裁剪前）。"""
    total_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total_norm_sq += param.grad.detach().double().pow(2).sum().item()
    return float(total_norm_sq**0.5)


def _record_metric(
    meters: OrderedDict[str, Any],
    key: str,
    value: float,
    *,
    count: int = 1,
) -> None:
    """把非 loss 指标（如 grad_norm）写入 AverageMeter，供日志/summary 复用。"""
    meter = meters.get(key)
    if meter is None:
        meter = meters[key] = twt.utils.AverageMeter()
    meter.update(value, count)


def _resolve_h5_paths(args: SimpleNamespace) -> list[Path]:
    """从 h5_paths 或 h5_dir 解析 collection2 的 h5 文件列表（排序保证确定性）。"""
    explicit = getattr(args, "h5_paths", None)
    if explicit:
        if isinstance(explicit, (str, Path)):
            paths = [Path(explicit)]
        else:
            paths = [Path(p) for p in explicit]
    else:
        h5_dir = Path(getattr(args, "h5_dir", None) or args.data_dir)
        if not h5_dir.is_dir():
            raise NotADirectoryError(f"h5_dir 不存在或不是目录: {h5_dir}")
        paths = sorted(h5_dir.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"未在 {getattr(args, 'h5_dir', args.data_dir)} 找到任何 .h5 文件。")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"以下 h5 文件不存在: {', '.join(missing)}")
    return paths


def _ensure_target_stats(
    h5_paths: list[Path],
    target_specs: dict[str, HeadSpec],
    *,
    stats_path: Path,
    args: SimpleNamespace,
) -> dict[str, dict[str, float]]:
    """确保回归目标在训练前有 z-score 统计。

    仅计算 train split（与训练/评测划分一致），避免用 eval split 泄漏统计；
    已存在的 stats 项保留，只补缺失的 depth/flow 等目标。
    """
    stats: dict[str, dict[str, float]] = {}
    if stats_path.exists():
        try:
            with stats_path.open("r", encoding="utf-8") as f:
                stats = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"stats.json 不是合法 JSON: {stats_path}") from exc

    regression_names = [
        name for name, spec in target_specs.items() if spec.type == "regression"
    ]
    missing = [name for name in regression_names if name not in stats]
    if not missing:
        return stats

    _logger.info(
        "Computing normalization stats on train split for %s -> %s",
        missing, stats_path,
    )
    new_stats = compute_target_stats(
        h5_paths,
        missing,
        split="train",
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    stats.update(new_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def _build_datasets(
    args: SimpleNamespace,
    data_config: dict[str, Any],
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    # 与 twt 一致：只用 eval transform（resize + center crop + normalize），
    # 不做训练增强——颜色抖动等会破坏触觉信号本身。
    transform = create_transform(**data_config, is_training=False)
    # target_specs 含全部目标（含 train: false 的 eval-only 列），由数据集负责读列；
    # 模型只用其中 train: true 的头。
    target_specs = {spec.name: spec for spec in parse_head_specs(args.heads, include_eval_only=True)}

    if getattr(args, "dataset_type", "labeled") == "h5":
        h5_paths = _resolve_h5_paths(args)
        stats_path = Path(
            getattr(args, "stats_path", None)
            or h5_paths[0].with_name("stats.json")
        )
        if bool(getattr(args, "normalize_targets", True)):
            _ensure_target_stats(
                h5_paths, target_specs, stats_path=stats_path, args=args
            )
        common = dict(
            target_specs=target_specs,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
            transform=transform,
            image_key=getattr(args, "image_key", "rgb"),
            stats_path=stats_path,
        )
        return (
            H5TactileDataset(h5_paths, split="train", **common),
            H5TactileDataset(h5_paths, split="eval", **common),
        )

    common = dict(
        target_specs=target_specs,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        transform=transform,
    )
    return (
        LabeledTactileDataset(args.data_dir, args.metadata_path, split="train", **common),
        LabeledTactileDataset(args.data_dir, args.metadata_path, split="eval", **common),
    )


def train_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    args: SimpleNamespace,
    *,
    device: torch.device,
) -> OrderedDict[str, float]:
    model.train()
    losses_m = twt.utils.AverageMeter()
    term_meters: OrderedDict[str, Any] = OrderedDict()
    batch_time_m = twt.utils.AverageMeter()

    last_idx = len(loader) - 1
    end = time.time()
    progress = tqdm(
        loader,
        desc=f"Train {epoch}",
        dynamic_ncols=True,
        leave=False,
    )
    for batch_idx, batch in enumerate(progress):
        batch = _move_batch(batch, device)
        loss_dict = model.compute_loss(batch)
        loss = loss_dict["total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = _compute_grad_norm(model)
        if args.clip_grad is not None:
            twt.utils.dispatch_clip_grad(model.parameters(), args.clip_grad, mode="norm")
        optimizer.step()

        items = twt._loss_items(loss_dict)
        bs = batch["image"].shape[0]
        losses_m.update(items["total"], bs)
        twt._update_meters(term_meters, items, bs)
        _record_metric(term_meters, "grad_norm", grad_norm, count=1)
        batch_time_m.update(time.time() - end)
        end = time.time()
        lr = twt._avg_lr(optimizer)
        progress.set_postfix(lr=f"{lr:.2e}", **twt._meters_postfix(losses_m.avg, term_meters))

        if args.log_batches and (batch_idx % args.log_interval == 0 or batch_idx == last_idx):
            terms = " ".join(f"{k.upper()} {m.avg:.4f}" for k, m in term_meters.items())
            _logger.info(
                "Train: %d [%4d/%d] Loss %.4f (%.4f) %s Time %.3fs LR %.3e",
                epoch, batch_idx, len(loader),
                losses_m.val, losses_m.avg, terms, batch_time_m.val, lr,
            )

    return twt._meters_summary(losses_m.avg, term_meters)


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    args: SimpleNamespace,
    *,
    device: torch.device,
) -> OrderedDict[str, float]:
    model.eval()
    losses_m = twt.utils.AverageMeter()
    term_meters: OrderedDict[str, Any] = OrderedDict()
    batch_time_m = twt.utils.AverageMeter()

    end = time.time()
    last_idx = len(loader) - 1
    progress = tqdm(
        loader,
        desc="Eval",
        dynamic_ncols=True,
        leave=False,
    )
    for batch_idx, batch in enumerate(progress):
        batch = _move_batch(batch, device)
        loss_dict = model.compute_loss(batch)

        items = twt._loss_items(loss_dict)
        bs = batch["image"].shape[0]
        losses_m.update(items["total"], bs)
        twt._update_meters(term_meters, items, bs)
        batch_time_m.update(time.time() - end)
        end = time.time()
        progress.set_postfix(**twt._meters_postfix(losses_m.avg, term_meters))

        if args.log_batches and (batch_idx % args.log_interval == 0 or batch_idx == last_idx):
            terms = " ".join(f"{k.upper()} {m.avg:.4f}" for k, m in term_meters.items())
            _logger.info(
                "Eval: [%4d/%d] Loss %.4f (%.4f) %s Time %.3fs",
                batch_idx, len(loader),
                losses_m.val, losses_m.avg, terms, batch_time_m.val,
            )

    return twt._meters_summary(losses_m.avg, term_meters)


def main() -> None:
    twt.utils.setup_default_logging()
    config_path = _parse_cli()
    _logger.info("Loading config from %s", config_path)
    args, args_text = load_config(config_path)

    device = twt._resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    twt.utils.random_seed(args.seed, 0)

    timm_backbone = twt._build_timm_backbone(args)
    data_config = resolve_data_config({}, model=timm_backbone, verbose=True)
    args.img_size = int(data_config["input_size"][-1])
    args.input_size = tuple(data_config["input_size"])

    feature_dim = twt._resolve_embedding_dim(timm_backbone)
    args.encoder_feature_dim = feature_dim
    head_specs: list[HeadSpec] = parse_head_specs(args.heads)
    # 写回解析后的 heads，便于 args.yaml / checkpoint 里看到生效配置
    args.heads = {spec.name: {k: v for k, v in vars(spec).items() if v is not None} for spec in head_specs}
    model_arch_name = getattr(args, "model_arch", "linear")
    if model_arch_name == "physical":
        model = TactilePhysicalMultiTask(
            encoder_backbone=timm_backbone,
            feature_dim=feature_dim,
            image_size=args.img_size,
            loss_weights=dict(args.loss_weights),
        ).to(device)
        model_arch = "physical"
    elif model_arch_name == "dinov3_physical":
        model = TactileViTPhysicalMultiTask(
            encoder_backbone=timm_backbone,
            feature_dim=feature_dim,
            image_size=args.img_size,
            loss_weights=dict(args.loss_weights),
        ).to(device)
        model_arch = "dinov3_physical"
    else:
        model = TactileMultiTask(
            encoder_backbone=timm_backbone,
            feature_dim=feature_dim,
            head_specs=head_specs,
            image_size=args.img_size,
        ).to(device)
        model_arch = "linear"

    if args.initial_checkpoint:
        twt._load_checkpoint(args.initial_checkpoint, model=model, model_only=True)

    param_count = sum(p.numel() for p in model.parameters())
    _logger.info(
        "Model %s MULTITASK(%s) created: params %.2fM, encoder features %d, heads %s",
        safe_model_name(args.model), model_arch, param_count / 1e6, feature_dim, list(model.heads.keys()),
    )

    dataset_train, dataset_eval = _build_datasets(args, data_config)
    if getattr(args, "dataset_type", "labeled") == "h5":
        _logger.info(
            "Dataset: train=%d eval=%d type=h5 root=%s",
            len(dataset_train), len(dataset_eval), args.data_dir,
        )
    else:
        _logger.info(
            "Dataset: train=%d eval=%d type=labeled root=%s metadata=%s",
            len(dataset_train), len(dataset_eval), args.data_dir, args.metadata_path,
        )

    loader_train = twt._build_loader(dataset_train, args=args, is_training=True)
    loader_eval = twt._build_loader(dataset_eval, args=args, is_training=False)

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
        start_epoch, best_metric = twt._load_checkpoint(
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
    output_dir = Path(twt.utils.get_outdir(args.output, exp_name))
    with open(output_dir / "args.yaml", "w", encoding="utf-8") as f:
        f.write(args_text)

    wandb_run = twt.init_wandb_run(args, exp_name=exp_name, output_dir=output_dir)
    if wandb_run is not None and twt.HAS_WANDB:
        import wandb

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
                twt._save_checkpoint(
                    output_dir / "model_best.pt",
                    epoch=epoch, model=model, optimizer=optimizer, args=args,
                    lr_scheduler=lr_scheduler,
                    metrics=eval_metrics or train_metrics, best_metric=best_metric,
                )
            twt._save_checkpoint(
                output_dir / "checkpoint_last.pt",
                epoch=epoch, model=model, optimizer=optimizer, args=args,
                lr_scheduler=lr_scheduler,
                metrics=eval_metrics or train_metrics, best_metric=best_metric,
            )

            lr = twt._avg_lr(optimizer)
            twt.utils.update_summary(
                epoch, train_metrics, eval_metrics,
                filename=str(output_dir / "summary.csv"),
                lr=lr, write_header=epoch == start_epoch, log_wandb=False,
            )

            if wandb_run is not None:
                import wandb

                # multitask 无重建图，跳过 twt 的 wandb image 逻辑
                payload: dict[str, Any] = {"epoch": epoch, "lr": lr, "best/loss": best_metric}
                payload.update(twt._flatten_metrics("train", train_metrics))
                payload.update(twt._flatten_metrics("eval", eval_metrics))
                wandb.log(payload, step=epoch + 1)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, metric)

            entry = {"epoch": epoch, "train": dict(train_metrics)}
            if eval_metrics is not None:
                entry["validation"] = dict(eval_metrics)
            results.append(entry)
            train_parts = " ".join(f"{k}={v:.6f}" for k, v in train_metrics.items())
            eval_parts = " ".join(f"{k}={v:.6f}" for k, v in (eval_metrics or {}).items())
            _logger.info(
                "Epoch %d done. train: %s%s | best=%.6f",
                epoch,
                train_parts,
                f" | eval: {eval_parts}" if eval_metrics else "",
                best_metric,
            )

    except KeyboardInterrupt:
        _logger.warning("Interrupted by user.")
    finally:
        if wandb_run is not None:
            import wandb

            wandb.finish()

    _logger.info("Best loss: %s", best_metric)
    print(f"--result\n{json.dumps(results[-10:], indent=4)}")


if __name__ == "__main__":
    main()
