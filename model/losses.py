"""无 marker 触觉图像的定制损失。

核心思想：
  - MSE 监督整体光强；
  - 图像梯度 L1 强迫网络保留接触区域的阴影边界（无 marker 时主要靠光影变化）；
  - KL 散度约束潜在空间，防止崩塌；
  - 可选 SSIM 提升结构相似度。
"""
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

try:  # SSIM 为可选依赖
    from pytorch_msssim import ssim as _ssim
    _HAS_SSIM = True
except ImportError:  # pragma: no cover
    _HAS_SSIM = False


def image_gradient(image: torch.Tensor):
    """返回 (dx, dy) — 一阶差分，等价于 Sobel 的简化版。"""
    dy = image[..., 1:, :] - image[..., :-1, :]
    dx = image[..., :, 1:] - image[..., :, :-1]
    return dx, dy


@dataclass
class LossWeights:
    mse: float = 1.0
    grad: float = 2.0     # 无 marker 场景下应显著加权梯度项
    kld: float = 0.01
    ssim: float = 0.0     # >0 时启用 SSIM


def unpack_reconstruction(outputs) -> torch.Tensor:
    """Extract reconstructed image tensor from common model output formats."""
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, dict):
        for key in ("recon", "reconstruction", "x_hat", "pred"):
            value = outputs.get(key)
            if torch.is_tensor(value):
                return value
    if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
        return outputs[0]
    raise TypeError("Cannot extract reconstruction tensor from model outputs")


def unpack_vae_stats(outputs):
    if isinstance(outputs, dict):
        mu = outputs.get("mu")
        logvar = outputs.get("logvar")
        if torch.is_tensor(mu) and torch.is_tensor(logvar):
            return mu, logvar
    if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
        mu, logvar = outputs[1], outputs[2]
        if torch.is_tensor(mu) and torch.is_tensor(logvar):
            return mu, logvar
    raise TypeError("VAE loss requires outputs containing mu and logvar")


def _reconstruction_terms(recon: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(recon, target)

    dx_r, dy_r = image_gradient(recon)
    dx_t, dy_t = image_gradient(target)
    grad = F.l1_loss(dx_r, dx_t) + F.l1_loss(dy_r, dy_t)
    return mse, grad


class TactileVAELoss:
    """组合 VAE 损失，调用接口与 nn.Module 类似但保持纯函数语义。"""

    def __init__(self, weights: Optional[LossWeights] = None) -> None:
        self.w = weights or LossWeights()
        if self.w.ssim > 0 and not _HAS_SSIM:
            raise ImportError("启用 SSIM 需要 `pip install pytorch-msssim`")

    def __call__(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if mu is None or logvar is None:
            outputs = recon
            recon = unpack_reconstruction(outputs)
            mu, logvar = unpack_vae_stats(outputs)

        mse, grad = _reconstruction_terms(recon, target)

        # KL(N(mu, sigma^2) || N(0, I))，对 batch 与 latent 维度同时取均值
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total = self.w.mse * mse + self.w.grad * grad + self.w.kld * kld
        out = {"total": total, "mse": mse, "grad": grad, "kld": kld}

        if self.w.ssim > 0:
            # SSIM 需要 [0,1] 数据范围；这里假设输入已 ImageNet 归一化，
            # 以方差范围近似覆盖，使用 data_range=2.0 简单兜底
            ssim_val = _ssim(recon, target, data_range=2.0, size_average=True)
            ssim_loss = 1.0 - ssim_val
            total = total + self.w.ssim * ssim_loss
            out["ssim"] = ssim_loss
            out["total"] = total
        return out


class ReconstructionLoss:
    """Image reconstruction loss without latent regularization."""

    def __init__(self, weights: Optional[LossWeights] = None) -> None:
        self.w = weights or LossWeights(kld=0.0)
        if self.w.ssim > 0 and not _HAS_SSIM:
            raise ImportError("启用 SSIM 需要 `pip install pytorch-msssim`")

    def __call__(self, outputs, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon = unpack_reconstruction(outputs)
        mse, grad = _reconstruction_terms(recon, target)
        total = self.w.mse * mse + self.w.grad * grad
        out = {"total": total, "mse": mse, "grad": grad}

        if self.w.ssim > 0:
            ssim_val = _ssim(recon, target, data_range=2.0, size_average=True)
            ssim_loss = 1.0 - ssim_val
            out["ssim"] = ssim_loss
            out["total"] = total + self.w.ssim * ssim_loss
        return out


def _coerce_weights(weights=None, **kwargs) -> LossWeights:
    values = {}
    if isinstance(weights, LossWeights):
        values.update(weights.__dict__)
    elif isinstance(weights, dict):
        values.update(weights)
    elif weights is not None:
        raise TypeError(f"Unsupported loss weights type: {type(weights)!r}")

    for key in ("mse", "grad", "kld", "ssim"):
        if key in kwargs:
            values[key] = kwargs.pop(key)
    if kwargs:
        raise TypeError(f"Unsupported loss params: {sorted(kwargs)}")
    return LossWeights(**values)


def available_losses() -> list:
    return ["mse_grad", "reconstruction", "tactile_vae", "vae"]


def build_loss(name: str = "tactile_vae", **kwargs):
    weights = _coerce_weights(**kwargs)
    if name in ("tactile_vae", "vae"):
        return TactileVAELoss(weights)
    if name in ("reconstruction", "mse_grad"):
        return ReconstructionLoss(weights)
    raise KeyError(f"未知 loss: {name}; 可选: {available_losses()}")
