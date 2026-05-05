"""端到端评估流程：把 latency / 重建质量 / 潜空间统计串起来。"""
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from .losses import LossWeights, TactileVAELoss
from .metrics import count_parameters, latent_variance, measure_latency


@dataclass
class EvalResult:
    backbone: str
    latency_ms: float
    params_million: float
    mse: float
    grad_loss: float
    latent_variance: float
    n_samples: int

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _extract_latent(outputs) -> Optional[torch.Tensor]:
    if isinstance(outputs, dict):
        for key in ("mu", "z", "latent", "embedding"):
            value = outputs.get(key)
            if torch.is_tensor(value):
                return value
    if isinstance(outputs, (tuple, list)) and len(outputs) >= 2 and torch.is_tensor(outputs[1]):
        return outputs[1]
    return None


@torch.no_grad()
def evaluate_encoder(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    backbone_name: str = "unknown",
    loss_weights: Optional[LossWeights] = None,
    loss_fn=None,
    measure_latency_only_encoder: bool = True,
    latency_input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> EvalResult:
    """对单个重建模型完成一次完整评测。

    Args:
        model: 已构建（可能未训练）的重建模型。
        dataloader: 验证/测试集，返回归一化后的 (B, 3, H, W) 张量。
        device: 推理设备。
        backbone_name: 仅用于结果标记。
        loss_weights: 用于评测时计算 MSE / Grad；不影响 latency。
        measure_latency_only_encoder: System-0 路径只用 encoder，建议 True。
    """
    model = model.to(device).eval()
    loss_fn = loss_fn or TactileVAELoss(loss_weights)

    target_for_latency = getattr(model, "encoder", model) if measure_latency_only_encoder else model
    latency_ms = measure_latency(target_for_latency, device, input_shape=latency_input_shape)

    total_mse, total_grad = 0.0, 0.0
    latent_chunks = []
    n_samples = 0

    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        if isinstance(batch, dict):
            batch = batch["image"]
        batch = batch.to(device, non_blocking=True)
        outputs = model(batch)
        losses = loss_fn(outputs, batch)
        bs = batch.size(0)
        total_mse += losses["mse"].item() * bs
        total_grad += losses["grad"].item() * bs
        latent = _extract_latent(outputs)
        if latent is not None:
            latent_chunks.append(latent.detach().cpu())
        n_samples += bs

    avg_mse = total_mse / max(n_samples, 1)
    avg_grad = total_grad / max(n_samples, 1)
    var = latent_variance(torch.cat(latent_chunks, dim=0)) if latent_chunks else float("nan")

    return EvalResult(
        backbone=backbone_name,
        latency_ms=latency_ms,
        params_million=count_parameters(getattr(model, "encoder", model)) / 1e6,
        mse=avg_mse,
        grad_loss=avg_grad,
        latent_variance=var,
        n_samples=n_samples,
    )


def format_results_table(results) -> str:
    header = (
        f"{'backbone':<22}{'latency(ms)':>13}{'params(M)':>11}"
        f"{'mse':>10}{'grad':>10}{'latent_var':>13}{'n':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.backbone:<22}{r.latency_ms:>13.2f}{r.params_million:>11.2f}"
            f"{r.mse:>10.4f}{r.grad_loss:>10.4f}{r.latent_variance:>13.4f}{r.n_samples:>8d}"
        )
    return "\n".join(lines)
