"""Shared loss utilities for tactile reconstruction models.

核心思想：
  - MSE 监督整体光强；
  - 图像梯度 L1 强迫网络保留接触区域的阴影边界（无 marker 时主要靠光影变化）；
  - KL 散度约束潜在空间，防止崩塌；
  - 可选 SSIM 提升结构相似度；
  - 可选 SSIM+L1 混合损失（mix_loss）：α(1-SSIM) + (1-α)L1，常用 α=0.84。
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:  # SSIM / MS-SSIM 为可选依赖
    from pytorch_msssim import ms_ssim as _ms_ssim
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
    mix: float = 0.0      # >0 时启用 SSIM+L1 混合损失
    mix_alpha: float = 0.84
    use_ms_ssim: bool = False


def reconstruction_terms(recon: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(recon, target)

    dx_r, dy_r = image_gradient(recon)
    dx_t, dy_t = image_gradient(target)
    grad = F.l1_loss(dx_r, dx_t) + F.l1_loss(dy_r, dy_t)
    return mse, grad


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, I)), averaged over batch and latent dims."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def ssim_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not _HAS_SSIM:
        raise ImportError("启用 SSIM 需要 `pip install pytorch-msssim`")
    # SSIM 需要 [0,1] 数据范围；这里假设输入已 ImageNet 归一化，
    # 以方差范围近似覆盖，使用 data_range=2.0 简单兜底。
    return 1.0 - _ssim(recon, target, data_range=2.0, size_average=True)


def mix_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.84,
    use_ms_ssim: bool = False,
    data_range: float = 2.0,
) -> torch.Tensor:
    """SSIM + L1 混合损失：``α(1-SSIM) + (1-α)L1``。

    与参考实现一致：α 越大越偏向结构相似度；α=0.84 是常用经验值。
    输入仍为 ImageNet 归一化空间，data_range 默认 2.0 与 ssim_loss 一致。
    use_ms_ssim=True 时切到 MS-SSIM，要求图像分辨率 ≳160。
    """
    if not _HAS_SSIM:
        raise ImportError("启用 mix loss 需要 `pip install pytorch-msssim`")
    l1 = F.l1_loss(recon, target)
    fn = _ms_ssim if use_ms_ssim else _ssim
    ssim_val = fn(recon, target, data_range=data_range, size_average=True)
    return alpha * (1.0 - ssim_val) + (1.0 - alpha) * l1
