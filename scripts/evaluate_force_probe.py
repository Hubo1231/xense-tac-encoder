#!/usr/bin/env python3
"""z → force_grid (35×20×3) probe 评测（docs/force_probe_eval_design.md）。

流程：加载冻结 encoder → 抽 train/eval 两个 split 全部有效样本的 z 与
force_grid / force_xyz / contact_mask → 拟合 linear(ridge) 与 MLP 两个 probe
→ 在 eval split 上一次性报告指标（全体 / 非零 masked / 分桶 / 接触检测 /
合力一致性），并附常数预测基线对照。

划分协议（设计文档 §3，严格执行）：
  * train/eval 划分原样复现 encoder 训练时的 split——在训练 h5_dir 全部文件
    拼接的全局样本序号上跑 ``split_row_indices(eval_ratio=0.2, seed=42)``，
    再取 force h5 对应的文件片段，剔除无 force_grid 的样本；
  * probe 只在 train split 上拟合；train split 再切 10% 作 probe-val
    用于选超参（ridge λ / MLP lr）与早停；eval split 只用于最终报告。

用法：
    .venv/bin/python scripts/evaluate_force_probe.py \
        --config configs/multitask/fastvit_t12_physical_collection2.yaml \
        --checkpoint outputs/<run>/model_best.pt \
        --force-config configs/multitask/force_probe.yaml

    # 初始权重对照组（不加载 multitask checkpoint，pooler/头随机初始化）：
    .venv/bin/python scripts/evaluate_force_probe.py \
        --config configs/multitask/vit_base_patch16_dinov3_lvd1689m.yaml \
        --checkpoint none --pretrained
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

import train_multitask as tmt  # noqa: E402
import train_with_timm as twt  # noqa: E402
import evaluate_multitask as emt  # noqa: E402
from timm.data import create_transform, resolve_data_config  # noqa: E402

from src.models.multitask import (  # noqa: E402
    TactileMultiTask,
    TactilePhysicalMultiTask,
    TactileViTPhysicalMultiTask,
    parse_head_specs,
)
from src.datasets.h5_dataset import H5TactileDataset, list_h5_samples  # noqa: E402
from src.datasets.labeled_dataset import split_row_indices  # noqa: E402

_logger = logging.getLogger("evaluate_force_probe")

GRID_SHAPE = (35, 20, 3)
N_NODES = GRID_SHAPE[0] * GRID_SHAPE[1]
OUT_DIM = N_NODES * GRID_SHAPE[2]


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen-encoder probe: z -> force_grid (35x20x3), linear(ridge) + MLP heads."
    )
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="backbone 训练配置（决定模型结构与 z 维度）。")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练产出的 checkpoint（model_best.pt 等）；"
                             "传 'none' 表示只用 backbone 预训练权重（初始权重对照组），"
                             "multitask 头/pooler 保持随机初始化。")
    parser.add_argument("--pretrained", action="store_true",
                        help="覆盖 config 的 pretrained 为 True（如 DINOv3 训练配置是 "
                             "pretrained: false，初始权重对照组需要下载官方预训练权重）。")
    parser.add_argument("--force-config", type=str,
                        default="configs/multitask/force_probe.yaml",
                        help="force 数据与 probe 协议配置。")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="结果目录，默认 <checkpoint 所在目录>/force_probe_eval/；"
                             "--checkpoint none 时默认 outputs/force_probe_eval/<model>_init/。")
    parser.add_argument("--batch-size", type=int, default=64, help="抽 embedding 的 batch 大小。")
    args = parser.parse_args()
    args.config = tmt.resolve_config_path(args.config)
    args.force_config = tmt.resolve_config_path(args.force_config)
    return args


# --------------------------------------------------------------------------- #
# 划分复现（与 encoder 训练 split 严格一致）
# --------------------------------------------------------------------------- #

def _reconstruct_split(
    force_h5: Path,
    train_h5_dir: Path,
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """复现训练时的全局 train/eval 划分，返回 force 文件内的 (train_idx, eval_idx)。

    训练脚本 ``_resolve_h5_paths`` 把 train_h5_dir 下 *.h5 按文件名排序拼接为
    全局样本序号，``H5TactileDataset`` 在该序号上跑 ``split_row_indices``。
    这里原样重复这一过程，再取 force 文件（按文件名定位）对应的片段，
    最后剔除无 force_grid 的样本（划分之后过滤，不改变成员关系）。
    """
    train_paths = sorted(train_h5_dir.glob("*.h5"))
    if not train_paths:
        raise FileNotFoundError(f"{train_h5_dir} 下没有 .h5 文件。")
    names_per_file = [list_h5_samples(p) for p in train_paths]
    match = [i for i, p in enumerate(train_paths) if p.name == force_h5.name]
    if len(match) != 1:
        raise ValueError(
            f"无法在 {train_h5_dir} 中唯一定位 {force_h5.name}（命中 {len(match)} 个）。"
        )
    file_idx = match[0]
    offset = sum(len(n) for n in names_per_file[:file_idx])
    file_names = names_per_file[file_idx]

    force_names = list_h5_samples(force_h5)
    if force_names != file_names:
        raise ValueError(
            f"{force_h5.name} 与训练目录同名文件的样本列表不一致，无法对齐 split。"
        )

    n_total = sum(len(n) for n in names_per_file)
    split = split_row_indices(n_total, eval_ratio=eval_ratio, seed=seed)
    lo, hi = offset, offset + len(file_names)

    import h5py

    with h5py.File(force_h5, "r") as f:
        valid = np.array(["force_grid" in f[name] for name in force_names], dtype=bool)

    def _to_local(rows: list[int]) -> list[int]:
        local = [r - lo for r in rows if lo <= r < hi]
        return [r for r in local if valid[r]]

    train_idx, eval_idx = _to_local(split["train"]), _to_local(split["eval"])
    _logger.info(
        "split 复现: 全局 %d 样本（%d 个训练文件），本文件 [%d, %d) 共 %d 样本；"
        "有效 train=%d eval=%d（剔除无 force_grid 样本）。",
        n_total, len(train_paths), lo, hi, len(file_names), len(train_idx), len(eval_idx),
    )
    if not train_idx or not eval_idx:
        raise ValueError("复现的 split 与本文件交集为空，请检查 train_h5_dir 配置。")
    return train_idx, eval_idx


# --------------------------------------------------------------------------- #
# 指标（全部在物理单位 N 上计算）
# --------------------------------------------------------------------------- #

def _masked_regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """1-D 展开后的 MAE/RMSE/R²（调用方保证已按 mask 选取元素）。"""
    return emt._regression_metrics(pred.reshape(-1, 1), target.reshape(-1, 1))


def compute_force_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    force_xyz: torch.Tensor,
    *,
    eps: float,
    eps_sensitivity: tuple[float, ...],
) -> dict[str, Any]:
    """force_grid 全套指标：全体 / 非零 masked(ε 敏感性) / 分桶 / 接触检测 / 合力一致性。

    pred/target: (N, 35, 20, 3)，force_xyz: (N, 3)，单位 N。
    """
    n = pred.shape[0]
    p = pred.reshape(n, N_NODES, 3).double()
    t = target.reshape(n, N_NODES, 3).double()
    result: dict[str, Any] = {
        "overall": emt._regression_metrics(pred.reshape(n, -1), target.reshape(n, -1))
    }

    t_mag = t.norm(dim=-1)  # (N, 700) 真值节点模长
    p_mag = p.norm(dim=-1)

    # 非零 masked 指标（ε 主值 + 敏感性）
    masked: dict[str, Any] = {}
    for e in (eps, *eps_sensitivity):
        node_mask = t_mag > e
        m3 = node_mask.unsqueeze(-1).expand_as(t)
        entry: dict[str, Any] = {"n_nodes": int(node_mask.sum())}
        if node_mask.any():
            entry.update(_masked_regression_metrics(p[m3], t[m3]))
        masked[f"{e:g}"] = entry
    result["masked"] = masked

    # 分桶 MAE：[0,ε), [ε,0.1), [0.1,1), [1,∞)
    buckets: dict[str, Any] = {}
    for lo, hi in ((0.0, eps), (eps, 0.1), (0.1, 1.0), (1.0, math.inf)):
        node_mask = (t_mag >= lo) & (t_mag < hi)
        m3 = node_mask.unsqueeze(-1).expand_as(t)
        label = f"[{lo:g},{'inf' if math.isinf(hi) else format(hi, 'g')})"
        entry = {"n_nodes": int(node_mask.sum())}
        if node_mask.any():
            entry["mae"] = (p[m3] - t[m3]).abs().mean().item()
        buckets[label] = entry
    result["buckets"] = buckets

    # 接触检测（节点模长 > ε 的二分类，micro 平均）
    t_contact = t_mag > eps
    p_contact = p_mag > eps
    tp = int((t_contact & p_contact).sum())
    fp = int((~t_contact & p_contact).sum())
    fn = int((t_contact & ~p_contact).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    result["contact"] = {
        "precision": precision, "recall": recall, "f1": f1, "iou": iou,
        "tp": tp, "fp": fp, "fn": fn,
    }

    # 合力一致性：预测力场求和 vs 已存 force_xyz
    p_sum = p.sum(dim=1)  # (N, 3)
    fx = force_xyz.double()
    diff = p_sum - fx
    l2 = diff.norm(dim=-1)  # 逐样本合力 L2 误差（N）
    rel = l2 / fx.norm(dim=-1).clamp_min(1e-12)
    result["force_sum"] = {
        "l2_err_mean": l2.mean().item(),      # 合力绝对误差主指标（N）
        "l2_err_median": l2.median().item(),
        "mae": diff.abs().mean().item(),      # 逐分量 MAE（N）
        "rel_err_mean": rel.mean().item(),
        "rel_err_median": rel.median().item(),
    }
    return result


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, eps: float) -> float:
    """probe-val 选超参/早停用的 masked MAE（真值节点模长 > ε 的元素）。"""
    n = pred.shape[0]
    p = pred.reshape(n, N_NODES, 3)
    t = target.reshape(n, N_NODES, 3)
    m3 = (t.norm(dim=-1) > eps).unsqueeze(-1).expand_as(t)
    if not m3.any():
        return (p - t).abs().mean().item()
    return (p[m3] - t[m3]).abs().mean().item()


# --------------------------------------------------------------------------- #
# 头 A：闭式 ridge（复用 evaluate_multitask._fit_ridge）
# --------------------------------------------------------------------------- #

def _select_ridge(
    z_ptr: torch.Tensor,
    y_ptr: torch.Tensor,
    z_pval: torch.Tensor,
    y_pval: torch.Tensor,
    lambdas: list[float],
    eps: float,
) -> tuple[float, dict[str, float]]:
    """在 probe-val masked MAE 上扫 λ，返回 (最优 λ, 各 λ 的 val masked MAE)。"""
    scores: dict[str, float] = {}
    best_lambda, best_score = lambdas[0], math.inf
    y_flat = y_ptr.reshape(y_ptr.shape[0], -1)
    for lam in lambdas:
        weight = emt._fit_ridge(z_ptr, y_flat, lam)
        pred = emt._ridge_predict(z_pval, weight)
        score = _masked_mae(pred, y_pval, eps)
        scores[f"{lam:g}"] = score
        _logger.info("ridge λ=%g probe-val masked MAE=%.6f", lam, score)
        if score < best_score:
            best_lambda, best_score = lam, score
    return best_lambda, scores


# --------------------------------------------------------------------------- #
# 头 B：MLP probe
# --------------------------------------------------------------------------- #

class _MlpProbe(nn.Module):
    """Linear(d, h) → GELU → Dropout(p) → Linear(h, 2100)。"""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, OUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_mlp(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    lr: float,
    mlp_cfg: SimpleNamespace,
    device: torch.device,
    seed: int,
    max_epochs: int,
    z_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> tuple[_MlpProbe, int, float]:
    """训练 MLP probe（总 MSE loss，AdamW）。

    提供 val 时按 probe-val masked MAE 早停并回滚到最优权重；
    返回 (probe, 最优 epoch 数, 最优 val masked MAE)。
    """
    torch.manual_seed(seed)
    probe = _MlpProbe(z_train.shape[1], int(mlp_cfg.hidden_dim), float(mlp_cfg.dropout)).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=lr, weight_decay=float(mlp_cfg.weight_decay)
    )
    x = z_train.to(device)
    y = y_train.reshape(y_train.shape[0], -1).to(device)
    x_val = z_val.to(device) if z_val is not None else None
    y_val_dev = y_val.to(device) if y_val is not None else None
    batch = int(mlp_cfg.batch_size)
    patience = int(mlp_cfg.patience)

    best_state, best_epoch, best_val = None, 0, math.inf
    bad_epochs = 0
    for epoch in range(1, max_epochs + 1):
        probe.train()
        perm = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], batch):
            idx = perm[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(probe(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
        if x_val is None:
            continue
        probe.eval()
        with torch.inference_mode():
            val_mae = _masked_mae(probe(x_val).cpu(), y_val_dev.cpu(), eps)
        if val_mae < best_val:
            best_val, best_epoch = val_mae, epoch
            best_state = copy.deepcopy({k: v.cpu() for k, v in probe.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                _logger.info("MLP lr=%g 早停于 epoch %d（最优 epoch %d）", lr, epoch, best_epoch)
                break
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    return probe, max(best_epoch, 1), best_val


def _select_mlp(
    z_ptr: torch.Tensor,
    y_ptr: torch.Tensor,
    z_pval: torch.Tensor,
    y_pval: torch.Tensor,
    *,
    mlp_cfg: SimpleNamespace,
    device: torch.device,
    seed: int,
    eps: float,
) -> tuple[float, int, dict[str, Any]]:
    """按 probe-val masked MAE 扫 lr，返回 (最优 lr, 最优 epoch, 各 lr 详情)。"""
    details: dict[str, Any] = {}
    best_lr, best_epochs, best_score = float(mlp_cfg.lrs[0]), 1, math.inf
    for lr in mlp_cfg.lrs:
        probe, epochs, val_mae = _train_mlp(
            z_ptr, y_ptr, lr=float(lr), mlp_cfg=mlp_cfg, device=device, seed=seed,
            max_epochs=int(mlp_cfg.max_epochs), z_val=z_pval, y_val=y_pval, eps=eps,
        )
        details[f"{float(lr):g}"] = {"val_masked_mae": val_mae, "best_epoch": epochs}
        _logger.info("MLP lr=%g probe-val masked MAE=%.6f (best epoch %d)", float(lr), val_mae, epochs)
        del probe
        if val_mae < best_score:
            best_lr, best_epochs, best_score = float(lr), epochs, val_mae
    return best_lr, best_epochs, details


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #

def _fmt(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _print_metrics(name: str, metrics: dict[str, Any], eps: float,
                   eps_sensitivity: tuple[float, ...]) -> None:
    o = metrics["overall"]
    print(f"\n[{name}] 全体(2100 维): MAE={o['mae']:.5f} RMSE={o['rmse']:.5f} R²={o['r2']:.4f}")
    for e in (eps, *eps_sensitivity):
        m = metrics["masked"][f"{e:g}"]
        print(
            f"  非零 masked (ε={e:g}, n={m['n_nodes']}): "
            f"MAE={_fmt(m.get('mae'), 5)} RMSE={_fmt(m.get('rmse'), 5)} R²={_fmt(m.get('r2'))}"
        )
    bucket_str = "  ".join(
        f"{k}: MAE={_fmt(v.get('mae'), 5)}(n={v['n_nodes']})" for k, v in metrics["buckets"].items()
    )
    print(f"  分桶: {bucket_str}")
    c = metrics["contact"]
    print(
        f"  接触检测(ε={eps:g}): P={c['precision']:.4f} R={c['recall']:.4f} "
        f"F1={c['f1']:.4f} IoU={c['iou']:.4f}"
    )
    fs = metrics["force_sum"]
    print(
        f"  合力一致性: L2={fs['l2_err_mean']:.5f} MAE={fs['mae']:.5f} "
        f"rel_err(mean)={fs['rel_err_mean']:.4f} rel_err(median)={fs['rel_err_median']:.4f}"
    )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> None:
    twt.utils.setup_default_logging()
    cli = _parse_cli()
    args, _ = tmt.load_config(cli.config)
    force_cfg = SimpleNamespace(**twt._load_yaml(cli.force_config))
    force_cfg.mlp = SimpleNamespace(**force_cfg.mlp)

    device = twt._resolve_device(getattr(args, "device", "auto"))

    backbone_only = cli.checkpoint.strip().lower() == "none"
    if cli.pretrained:
        args.pretrained = True  # 覆盖训练配置（如 DINOv3 训练时 pretrained: false）
    if backbone_only:
        # 初始权重对照组：固定 pooler/头的随机初始化，保证可复现。
        # 注意此时 z 经过一层随机投影（PhysicalPooler / AttentionPool），
        # probe 仍合法（在 train split 上拟合），但报告中需注明。
        torch.manual_seed(int(getattr(args, "seed", 42)))

    # 重建 backbone + 模型，加载 checkpoint 后全参数冻结（同 evaluate_multitask）
    timm_backbone = twt._build_timm_backbone(args)
    data_config = resolve_data_config({}, model=timm_backbone, verbose=False)
    feature_dim = twt._resolve_embedding_dim(timm_backbone)
    model_arch = getattr(args, "model_arch", "linear")
    image_size = int(data_config["input_size"][-1])
    if model_arch == "physical":
        model = TactilePhysicalMultiTask(
            encoder_backbone=timm_backbone, feature_dim=feature_dim,
            image_size=image_size, loss_weights=dict(args.loss_weights),
        ).to(device)
    elif model_arch == "dinov3_physical":
        model = TactileViTPhysicalMultiTask(
            encoder_backbone=timm_backbone, feature_dim=feature_dim,
            image_size=image_size, loss_weights=dict(args.loss_weights),
        ).to(device)
    else:
        model = TactileMultiTask(
            encoder_backbone=timm_backbone, feature_dim=feature_dim,
            head_specs=[s for s in parse_head_specs(args.heads, include_eval_only=True) if s.train],
            image_size=image_size,
        ).to(device)
    if backbone_only:
        _logger.info(
            "--checkpoint none: 仅用 backbone 预训练权重（pretrained=%s, pretrained_path=%s），"
            "multitask 头/pooler 为随机初始化（seed=%d）。",
            bool(args.pretrained), getattr(args, "pretrained_path", "") or "",
            int(getattr(args, "seed", 42)),
        )
    else:
        twt._load_checkpoint(cli.checkpoint, model=model, model_only=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    _logger.info(
        "Loaded checkpoint %s; arch=%s z_dim=%d", cli.checkpoint, model_arch, feature_dim
    )

    # force 数据集（全部样本；划分由 _reconstruct_split 复现后用 Subset 索引）
    force_h5 = Path(force_cfg.h5_path)
    transform = create_transform(**data_config, is_training=False)
    target_specs = {
        spec.name: spec
        for spec in parse_head_specs(force_cfg.heads, include_eval_only=True)
    }
    dataset = H5TactileDataset(
        force_h5,
        target_specs,
        split=None,
        transform=transform,
        image_key=getattr(force_cfg, "image_key", "rgb"),
    )
    train_idx, eval_idx = _reconstruct_split(
        force_h5,
        Path(force_cfg.train_h5_dir),
        eval_ratio=float(getattr(args, "eval_ratio", force_cfg.eval_ratio)),
        seed=int(getattr(args, "seed", force_cfg.seed)),
    )

    target_names = ["force_grid", "force_xyz", "contact_mask"]

    def _make_loader(indices: list[int]) -> torch.utils.data.DataLoader:
        subset = torch.utils.data.Subset(dataset, indices)
        return torch.utils.data.DataLoader(
            subset, batch_size=cli.batch_size, shuffle=False,
            num_workers=int(getattr(args, "workers", 4)), pin_memory=True,
        )

    z_train_all, t_train = emt._collect_embeddings(
        model, _make_loader(train_idx), target_names, device=device
    )
    z_eval, t_eval = emt._collect_embeddings(
        model, _make_loader(eval_idx), target_names, device=device
    )
    _logger.info("embeddings: train=%s eval=%s", tuple(z_train_all.shape), tuple(z_eval.shape))

    # train split 再切 10% 作 probe-val（只在 train 内部，eval 全程不可见）
    eps = float(force_cfg.eps)
    eps_sensitivity = tuple(float(e) for e in force_cfg.eps_sensitivity)
    val_split = split_row_indices(
        len(train_idx), eval_ratio=float(force_cfg.probe_val_ratio), seed=int(force_cfg.seed)
    )
    ptr, pval = val_split["train"], val_split["eval"]
    z_ptr, z_pval = z_train_all[ptr], z_train_all[pval]
    y_ptr = t_train["force_grid"][ptr]
    y_pval = t_train["force_grid"][pval]
    y_train = t_train["force_grid"]
    y_eval = t_eval["force_grid"]
    _logger.info("probe-train=%d probe-val=%d eval=%d", len(ptr), len(pval), len(eval_idx))

    lambdas = [float(v) for v in force_cfg.ridge_lambdas]

    # 头 A：ridge —— probe-train 上拟合、probe-val 选 λ，再用全 train split 重拟合
    best_lambda, ridge_scores = _select_ridge(z_ptr, y_ptr, z_pval, y_pval, lambdas, eps)
    weight = emt._fit_ridge(z_train_all, y_train.reshape(y_train.shape[0], -1), best_lambda)
    pred_linear = emt._ridge_predict(z_eval, weight).reshape(-1, *GRID_SHAPE)
    _logger.info("ridge 选定 λ=%g，已在全 train split 上重拟合。", best_lambda)

    # 头 B：MLP —— probe-train 训练 + probe-val 早停选 lr，再按最优 epoch 在全 train 上重训
    best_lr, best_epochs, mlp_details = _select_mlp(
        z_ptr, y_ptr, z_pval, y_pval,
        mlp_cfg=force_cfg.mlp, device=device, seed=int(force_cfg.seed), eps=eps,
    )
    mlp, _, _ = _train_mlp(
        z_train_all, y_train, lr=best_lr, mlp_cfg=force_cfg.mlp, device=device,
        seed=int(force_cfg.seed), max_epochs=best_epochs,
    )
    with torch.inference_mode():
        pred_mlp = mlp(z_eval.to(device)).cpu().reshape(-1, *GRID_SHAPE)
    _logger.info("MLP 选定 lr=%g epochs=%d，已在全 train split 上重训。", best_lr, best_epochs)

    # 基线：常数预测（train split 逐维均值，几乎全零）
    pred_const = y_train.mean(dim=0, keepdim=True).expand_as(y_eval).clone()

    results: dict[str, Any] = {
        "checkpoint": str(cli.checkpoint),
        "weights_source": (
            f"backbone_only(pretrained={bool(args.pretrained)}, "
            f"pretrained_path={getattr(args, 'pretrained_path', '') or ''}; "
            "pooler/heads 随机初始化, z 含一层随机投影)"
            if backbone_only
            else f"multitask checkpoint: {cli.checkpoint}"
        ),
        "config": str(cli.config),
        "force_config": str(cli.force_config),
        "model_arch": model_arch,
        "backbone": getattr(args, "model", ""),
        "pretrained": bool(getattr(args, "pretrained", False)),
        "z_dim": feature_dim,
        "eps": eps,
        "eps_sensitivity": list(eps_sensitivity),
        "split": {
            "n_samples_h5": len(dataset),
            "n_train": len(train_idx),
            "n_probe_train": len(ptr),
            "n_probe_val": len(pval),
            "n_eval": len(eval_idx),
        },
        "hyperparams": {
            "ridge_lambda": best_lambda,
            "ridge_lambda_scores": ridge_scores,
            "mlp_lr": best_lr,
            "mlp_epochs": best_epochs,
            "mlp_lr_scores": mlp_details,
        },
        "heads": {},
        "baseline_const": None,
    }
    for name, pred in (
        ("linear", pred_linear),
        ("mlp", pred_mlp),
        ("baseline_const", pred_const),
    ):
        metrics = compute_force_metrics(
            pred, y_eval, t_eval["force_xyz"], eps=eps, eps_sensitivity=eps_sensitivity
        )
        if name == "baseline_const":
            results["baseline_const"] = metrics
        else:
            results["heads"][name] = metrics
        _print_metrics(name, metrics, eps, eps_sensitivity)

    if cli.output_dir:
        output_dir = Path(cli.output_dir)
    elif backbone_only:
        safe_name = str(getattr(args, "model", "backbone")).replace(".", "_")
        output_dir = _REPO_ROOT / "outputs" / "force_probe_eval" / f"{safe_name}_init"
    else:
        output_dir = Path(cli.checkpoint).resolve().parent / "force_probe_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "force_probe_metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    _logger.info("Wrote metrics to %s", output_path)
    dataset.close()


if __name__ == "__main__":
    main()
