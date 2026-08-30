#!/usr/bin/env python3
"""训练后独立评测：冻结编码器 + 多头指标 + 可选 linear probe。

用法：
    # 只评测已训练头
    python scripts/evaluate_multitask.py --config configs/multitask/<cfg>.yaml \
        --checkpoint outputs/<exp>/model_best.pt

    # 额外对 train: false 的 eval-only 目标拟合 linear probe 并评测
    python scripts/evaluate_multitask.py --config configs/multitask/<cfg>.yaml \
        --checkpoint outputs/<exp>/model_best.pt --probe --probe-epochs 200 \
        --output-json outputs/<exp>/eval_metrics.json

指标：
  * 回归头：MAE / RMSE / R²（用 stats.json 反标准化回物理单位再算）；
  * 分类头：accuracy / macro-F1（纯 torch 实现，不引 sklearn）；
  * probe：回归目标用闭式 ridge 回归（含 bias 列，λ 默认 1e-3），
    分类目标训一层 Linear + softmax（Adam，--probe-epochs 步）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

import train_multitask as tmt  # noqa: E402
import train_with_timm as twt  # noqa: E402
from timm.data import create_transform, resolve_data_config  # noqa: E402

from src.models.multitask import (  # noqa: E402
    HeadSpec,
    TactileMultiTask,
    TactilePhysicalMultiTask,
    TactileViTPhysicalMultiTask,
    parse_head_specs,
)
from src.datasets.h5_dataset import H5TactileDataset  # noqa: E402
from src.datasets.labeled_dataset import LabeledTactileDataset  # noqa: E402

_logger = logging.getLogger("evaluate_multitask")


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen TactileMultiTask encoder: trained-head metrics + optional linear probe."
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=tmt.DEFAULT_CONFIG,
        help="同 train_multitask：绝对路径 / 相对仓库根 / configs/multitask/ 下的文件名。",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="训练产出的 checkpoint（model_best.pt 等）。")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="对 config 中 train: false 的 eval-only 目标拟合 linear probe 并评测。",
    )
    parser.add_argument("--probe-epochs", type=int, default=200, help="分类 probe 的 Adam 训练步数（默认 200）。")
    parser.add_argument("--probe-ridge-lambda", type=float, default=1e-3, help="回归 probe 的 ridge 系数 λ（默认 1e-3）。")
    parser.add_argument("--output-json", type=str, default=None, help="可选：把指标表写成 JSON 文件。")
    args = parser.parse_args()
    args.config = tmt.resolve_config_path(args.config)
    if args.probe_epochs < 1:
        raise ValueError("--probe-epochs 必须 >= 1。")
    if args.probe_ridge_lambda <= 0:
        raise ValueError("--probe-ridge-lambda 必须 > 0。")
    return args


# --------------------------------------------------------------------------- #
# 指标（纯 torch）
# --------------------------------------------------------------------------- #

def _regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """MAE / RMSE / R²，输入需已反标准化到物理单位，形状 (N, dim)。"""
    pred = pred.double()
    target = target.double()
    err = pred - target
    mae = err.abs().mean().item()
    rmse = err.pow(2).mean().sqrt().item()
    ss_res = err.pow(2).sum().item()
    ss_tot = (target - target.mean(dim=0, keepdim=True)).pow(2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _classification_metrics(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict[str, float]:
    """accuracy / macro-F1（纯 torch；某类在 eval 上无样本也无预测时 F1 记 0）。"""
    pred = pred_logits.argmax(dim=-1)
    target = target.long()
    accuracy = (pred == target).double().mean().item()
    f1_sum = 0.0
    for cls in range(num_classes):
        tp = ((pred == cls) & (target == cls)).sum().item()
        fp = ((pred == cls) & (target != cls)).sum().item()
        fn = ((pred != cls) & (target == cls)).sum().item()
        denom = 2 * tp + fp + fn
        f1_sum += (2.0 * tp / denom) if denom > 0 else 0.0
    return {"accuracy": accuracy, "macro_f1": f1_sum / num_classes}


# --------------------------------------------------------------------------- #
# 前向收集
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def _collect_trained_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    spec_by_name: dict[str, HeadSpec] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """eval split 上前向各训练头，收集 {name: {"pred", "target"}}（CPU tensor）。"""
    model.eval()
    collected: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"pred": [], "target": []} for name in model.heads.keys()
    }
    for batch in loader:
        batch = tmt._move_batch(batch, device)
        predictions = model(batch["image"])
        for name, pred in predictions.items():
            target = batch["targets"][name]
            if spec_by_name is not None and spec_by_name[name].spatial:
                target = _prepare_physical_target(name, pred, batch["targets"])
                pred, target = _flatten_spatial(pred, target)
            collected[name]["pred"].append(pred.cpu())
            collected[name]["target"].append(target.cpu())
    return {
        name: {"pred": torch.cat(parts["pred"]), "target": torch.cat(parts["target"])}
        for name, parts in collected.items()
    }


def _prepare_physical_target(
    name: str,
    pred: torch.Tensor,
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    """把 H5 目标转成与物理模型预测一致的 (B, C, H, W)。"""
    if name in ("flow", "flow_z"):
        target = targets["flow"]
        if target.ndim == 4 and target.shape[-1] == 2:
            target = target.permute(0, 3, 1, 2)
        elif target.ndim == 3:
            target = target.unsqueeze(1)
    elif name in ("depth", "depth_z"):
        target = targets["depth"]
        if target.ndim == 3:
            target = target.unsqueeze(1)
    else:
        raise ValueError(f"未知物理目标名 {name!r}。")
    target = F.interpolate(
        target, size=pred.shape[-2:], mode="bilinear", align_corners=False
    )
    return target.to(dtype=pred.dtype, device=pred.device)


def _flatten_spatial(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把 (B, C, H, W) 展平为 (N, C)，供回归指标计算。"""
    pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
    target = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
    return pred, target


@torch.inference_mode()
def _collect_physical_predictions(
    model: TactilePhysicalMultiTask,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """收集 physical 模型的 dense/bottleneck 预测与对齐后的目标。"""
    model.eval()
    collected: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"pred": [], "target": []} for name in model.heads.keys()
    }
    for batch in loader:
        batch = tmt._move_batch(batch, device)
        predictions = model(batch["image"])
        for name, pred in predictions.items():
            target = _prepare_physical_target(name, pred, batch["targets"])
            pred, target = _flatten_spatial(pred, target)
            collected[name]["pred"].append(pred.cpu())
            collected[name]["target"].append(target.cpu())
    return {
        name: {"pred": torch.cat(parts["pred"]), "target": torch.cat(parts["target"])}
        for name, parts in collected.items()
    }


@torch.inference_mode()
def _collect_embeddings(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    target_names: list[str],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """冻结 encoder 抽整个 split 的 embedding 与目标列（CPU tensor）。"""
    model.eval()
    feats: list[torch.Tensor] = []
    targets: dict[str, list[torch.Tensor]] = {name: [] for name in target_names}
    for batch in loader:
        batch = tmt._move_batch(batch, device)
        feats.append(model.encode(batch["image"]).cpu())
        for name in target_names:
            targets[name].append(batch["targets"][name].cpu())
    return torch.cat(feats), {name: torch.cat(parts) for name, parts in targets.items()}


# --------------------------------------------------------------------------- #
# 反标准化与 probe
# --------------------------------------------------------------------------- #

def _denormalize(name: str, values: torch.Tensor, stats: dict[str, dict[str, float]]) -> torch.Tensor:
    """回归目标按 stats.json 反标准化回物理单位；无 stats 时原样返回。"""
    entry = stats.get(name)
    if entry is None:
        return values
    std = float(entry["std"])
    if std <= 0.0:  # 与 LabeledTactileDataset._normalize 的常数列退化逻辑对齐
        std = 1.0
    return values * std + float(entry["mean"])


def _fit_ridge(feats: torch.Tensor, target: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    """闭式 ridge 回归（含 bias 列）：W = (X^T X + λI)^{-1} X^T y。

    返回 (C+1, dim) 的权重；预测为 [X, 1] @ W。float64 求解保证数值稳定。
    """
    x = torch.cat([feats, torch.ones(feats.shape[0], 1)], dim=1).double()
    y = target.double()
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    gram = x.T @ x
    gram.diagonal().add_(ridge_lambda)  # 原位加 λI；与目标的 (B,1)/(B,dim) 对齐由调用方保证
    return torch.linalg.solve(gram, x.T @ y)


def _ridge_predict(feats: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = torch.cat([feats, torch.ones(feats.shape[0], 1)], dim=1).double()
    return (x @ weight).float()


def _fit_softmax_probe(
    feats: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    epochs: int,
    device: torch.device,
) -> nn.Linear:
    """一层 Linear + cross_entropy 分类探针（全 batch Adam）。"""
    probe = nn.Linear(feats.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    x = feats.to(device)
    y = target.long().to(device)
    probe.train()
    for step in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(x), y)
        loss.backward()
        optimizer.step()
        if (step + 1) % max(1, epochs // 5) == 0:
            _logger.info("probe step %d/%d loss=%.4f", step + 1, epochs, loss.item())
    probe.eval()
    return probe


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #

_METRIC_COLUMNS: tuple[str, ...] = ("mae", "rmse", "r2", "accuracy", "macro_f1")


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = ["target", "type", "source", *_METRIC_COLUMNS]
    table = []
    for row in rows:
        line = [row["target"], row["type"], row["source"]]
        for col in _METRIC_COLUMNS:
            value = row["metrics"].get(col)
            line.append("-" if value is None else f"{value:.4f}")
        table.append(line)

    widths = [max(len(header[i]), *(len(line[i]) for line in table)) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*["-" * w for w in widths]))
    for line in table:
        print(fmt.format(*line))


def _evaluate_predictions(
    name: str,
    spec: HeadSpec,
    pred: torch.Tensor,
    target: torch.Tensor,
    stats: dict[str, dict[str, float]],
) -> dict[str, float]:
    if spec.type == "regression":
        pred = _denormalize(name, pred, stats)
        target = _denormalize(name, target, stats)
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        return _regression_metrics(pred, target)
    return _classification_metrics(pred, target, int(spec.num_classes))


def main() -> None:
    twt.utils.setup_default_logging()
    cli = _parse_cli()
    _logger.info("Loading config from %s", cli.config)
    args, _ = tmt.load_config(cli.config)

    device = twt._resolve_device(args.device)

    # 重建 backbone + 模型（只含训练头），加载 checkpoint 后全部冻结
    timm_backbone = twt._build_timm_backbone(args)
    data_config = resolve_data_config({}, model=timm_backbone, verbose=False)
    feature_dim = twt._resolve_embedding_dim(timm_backbone)
    all_specs = parse_head_specs(args.heads, include_eval_only=True)
    train_specs = [spec for spec in all_specs if spec.train]
    eval_only_specs = [spec for spec in all_specs if not spec.train]
    model_arch = getattr(args, "model_arch", "linear")
    if model_arch == "physical":
        model = TactilePhysicalMultiTask(
            encoder_backbone=timm_backbone,
            feature_dim=feature_dim,
            image_size=int(data_config["input_size"][-1]),
            loss_weights=dict(args.loss_weights),
        ).to(device)
    elif model_arch == "dinov3_physical":
        model = TactileViTPhysicalMultiTask(
            encoder_backbone=timm_backbone,
            feature_dim=feature_dim,
            image_size=int(data_config["input_size"][-1]),
            loss_weights=dict(args.loss_weights),
        ).to(device)
    else:
        model = TactileMultiTask(
            encoder_backbone=timm_backbone,
            feature_dim=feature_dim,
            head_specs=train_specs,
            image_size=int(data_config["input_size"][-1]),
        ).to(device)
    twt._load_checkpoint(cli.checkpoint, model=model, model_only=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    _logger.info(
        "Loaded checkpoint %s; arch=%s heads=%s",
        cli.checkpoint, model_arch, list(model.heads.keys()),
    )

    # 数据集（eval split 含全部目标列；probe 另需 train split）
    transform = create_transform(**data_config, is_training=False)
    target_specs = {spec.name: spec for spec in all_specs}
    if getattr(args, "dataset_type", "labeled") == "h5":
        h5_paths = tmt._resolve_h5_paths(args)
        stats_path = Path(
            getattr(args, "stats_path", None)
            or h5_paths[0].with_name("stats.json")
        )
        if bool(getattr(args, "normalize_targets", True)):
            tmt._ensure_target_stats(
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
        dataset_eval = H5TactileDataset(h5_paths, split="eval", **common)
        dataset_train_for_probe = None
    else:
        common = dict(
            target_specs=target_specs,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
            transform=transform,
        )
        dataset_eval = LabeledTactileDataset(
            args.data_dir, args.metadata_path, split="eval", **common
        )
        dataset_train_for_probe = None
    loader_eval = twt._build_loader(dataset_eval, args=args, is_training=False)
    stats = dataset_eval.target_stats

    rows: list[dict[str, Any]] = []
    spec_by_name = {spec.name: spec for spec in all_specs}

    # 1) 已训练头
    if model_arch in ("physical", "dinov3_physical"):
        collected = _collect_physical_predictions(model, loader_eval, device=device)
        for name, parts in collected.items():
            base_name = name.removesuffix("_z")
            pred = _denormalize(base_name, parts["pred"], stats)
            target = _denormalize(base_name, parts["target"], stats)
            metrics = _regression_metrics(pred, target)
            source = "trained_bottleneck" if name.endswith("_z") else "trained_dense"
            rows.append(
                {"target": name, "type": "regression", "source": source, "metrics": metrics}
            )
    else:
        collected = _collect_trained_predictions(
            model, loader_eval, device=device, spec_by_name=spec_by_name
        )
        for name, parts in collected.items():
            metrics = _evaluate_predictions(name, spec_by_name[name], parts["pred"], parts["target"], stats)
            rows.append({"target": name, "type": spec_by_name[name].type, "source": "trained", "metrics": metrics})

    # 2) eval-only 目标的 linear probe
    if cli.probe:
        if not eval_only_specs:
            _logger.warning("--probe 指定了但 config 中没有 train: false 的目标，跳过。")
        else:
            if dataset_train_for_probe is None:
                if getattr(args, "dataset_type", "labeled") == "h5":
                    dataset_train = H5TactileDataset(h5_paths, split="train", **common)
                else:
                    dataset_train = LabeledTactileDataset(
                        args.data_dir, args.metadata_path, split="train", **common
                    )
            else:
                dataset_train = dataset_train_for_probe
            loader_train = twt._build_loader(dataset_train, args=args, is_training=False)
            probe_names = [spec.name for spec in eval_only_specs]
            feats_train, targets_train = _collect_embeddings(model, loader_train, probe_names, device=device)
            feats_eval, targets_eval = _collect_embeddings(model, loader_eval, probe_names, device=device)

            for spec in eval_only_specs:
                if spec.type == "regression":
                    weight = _fit_ridge(feats_train, targets_train[spec.name], cli.probe_ridge_lambda)
                    pred = _ridge_predict(feats_eval, weight)
                else:
                    probe = _fit_softmax_probe(
                        feats_train, targets_train[spec.name], int(spec.num_classes),
                        epochs=cli.probe_epochs, device=device,
                    )
                    with torch.inference_mode():
                        pred = probe(feats_eval.to(device)).cpu()
                metrics = _evaluate_predictions(spec.name, spec, pred, targets_eval[spec.name], stats)
                rows.append({"target": spec.name, "type": spec.type, "source": "probe", "metrics": metrics})

    _print_table(rows)

    if cli.output_json:
        output_path = Path(cli.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        _logger.info("Wrote metrics to %s", output_path)


if __name__ == "__main__":
    main()
