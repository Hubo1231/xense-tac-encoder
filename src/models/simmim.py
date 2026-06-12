"""触觉 RGB SimMIM 掩码图像建模模型。

与 ``TactileMAE`` 的关系：

- ``TactileMAE`` 在 patch token 级别丢弃 token，**只适用于 ViT**（需要 ``patch_embed``/``blocks``）。
- ``TactileSimMIM`` 在**像素/输入层**打掩码，把（被遮挡的）整图喂进 backbone 的 ``forward_features``
  得到特征图，再用一个轻量「1×1 卷积 + PixelShuffle」解码器重建整图，只在被 mask 的像素上算 L1 损失。
  它把 backbone 当**黑盒**（只调 ``forward_features``），因此**对任意 backbone 通用**：卷积网络
  （ResNet / ConvNeXt / EfficientViT / MobileNetV4 / FastViT）返回 (B,C,h,w) 特征图直接用；ViT 返回
  token 序列则丢掉前缀 token 后 reshape 成特征图。ConvNeXt-V2 的 FCMAE 正是这一类卷积掩码建模。

参考 SimMIM（Xie et al., 2022）：掩码块大小默认 32，掩码比例 0.6，解码头为单层线性/卷积，L1 重建损失。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseModel


class TactileSimMIM(BaseModel):
    """对任意 timm backbone 的 SimMIM 式掩码图像建模。

    Args:
        encoder_backbone: 任意 timm backbone（``num_classes=0`` 创建），需暴露 ``forward_features``。
        image_size: 输入边长（由 timm ``data_config`` 解析）。
        in_chans: 通道数（触觉 RGB = 3）。
        mask_patch_size: 掩码块边长（在输入分辨率上），默认 32。
        mask_ratio: 被 mask 的块比例，默认 0.6。
    """

    def __init__(
        self,
        encoder_backbone: nn.Module,
        *,
        image_size: int,
        in_chans: int = 3,
        mask_patch_size: int = 32,
        mask_ratio: float = 0.6,
    ) -> None:
        super().__init__(image_size=image_size, input_channels=in_chans, latent_dim=0)
        if not hasattr(encoder_backbone, "forward_features"):
            raise ValueError("TactileSimMIM 需要 backbone 暴露 forward_features()。")
        if image_size % mask_patch_size != 0:
            raise ValueError(f"image_size={image_size} 必须能被 mask_patch_size={mask_patch_size} 整除。")

        self.encoder = encoder_backbone
        self.in_chans = int(in_chans)
        self.mask_patch_size = int(mask_patch_size)
        self.mask_ratio = float(mask_ratio)
        self.num_prefix_tokens = int(getattr(encoder_backbone, "num_prefix_tokens", 0) or 0)

        # ---- 干跑一次，确定特征图维度与下采样步长 ----
        was_training = encoder_backbone.training
        encoder_backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_chans, image_size, image_size)
            feat = self._to_feature_map(encoder_backbone.forward_features(dummy))
        if was_training:
            encoder_backbone.train()

        self.feat_dim = int(feat.shape[1])
        self.feat_hw = int(feat.shape[-1])
        if feat.shape[-1] != feat.shape[-2]:
            raise ValueError(f"SimMIM 仅支持方形特征图，得到 {tuple(feat.shape)}。")
        if image_size % self.feat_hw != 0:
            raise ValueError(
                f"输入 {image_size} 不能被特征图边长 {self.feat_hw} 整除，无法用 PixelShuffle 还原整图。"
            )
        self.encoder_stride = image_size // self.feat_hw

        # ---- 轻量解码器：1×1 卷积 + PixelShuffle 还原到输入分辨率 ----
        self.decoder = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.encoder_stride**2 * self.in_chans, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )
        # 像素空间的可学习掩码值（被遮挡区域用它替换）。
        self.mask_token = nn.Parameter(torch.zeros(1, self.in_chans, 1, 1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    # ------------------------------------------------------------------
    def _to_feature_map(self, out: torch.Tensor) -> torch.Tensor:
        """把 ``forward_features`` 输出统一成 (B, C, h, w) 特征图。

        - 卷积 backbone：已是 (B, C, h, w)，直接返回。
        - ViT：(B, P+N, C)，丢掉前缀 token 后 reshape 成方形特征图。
        """
        if out.ndim == 4:
            return out
        if out.ndim == 3:
            tokens = out[:, self.num_prefix_tokens:, :]      # (B, N, C)
            b, n, c = tokens.shape
            hw = int(n**0.5)
            if hw * hw != n:
                raise ValueError(f"ViT patch 数 {n} 不是完全平方数，无法 reshape 成特征图。")
            return tokens.transpose(1, 2).reshape(b, c, hw, hw)
        raise ValueError(f"无法识别的 forward_features 输出维度: {out.ndim}（期待 3 或 4）。")

    def _random_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """生成 (B, 1, H, W) 掩码，1=被 mask。在 mask_patch_size 粒度上随机选块。"""
        g = self.image_size // self.mask_patch_size          # 每边块数
        n = g * g
        len_mask = max(1, int(round(n * self.mask_ratio)))
        noise = torch.rand(batch_size, n, device=device)
        ids = torch.argsort(noise, dim=1)
        mask = torch.zeros(batch_size, n, device=device)
        mask.scatter_(1, ids[:, :len_mask], 1.0)
        mask = mask.reshape(batch_size, 1, g, g)
        mask = mask.repeat_interleave(self.mask_patch_size, dim=2).repeat_interleave(
            self.mask_patch_size, dim=3
        )
        return mask                                          # (B, 1, H, W)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._to_feature_map(self.encoder.forward_features(x))

    # ------------------------------------------------------------------
    # BaseModel 接口
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """无 mask 地编码，返回 (B, C, h, w) 特征图，用于下游特征提取。"""
        return self._encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mask = self._random_mask(x.shape[0], x.device)
        x_masked = x * (1.0 - mask) + self.mask_token * mask
        feat = self._encode(x_masked)
        pred = self.decoder(feat)                            # (B, C, H, W)
        return pred, mask

    def compute_loss(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        pred, mask = self.forward(batch)
        # 只在被 mask 的像素上算 L1，按掩码像素数 × 通道数归一化（SimMIM 标准做法）。
        loss = F.l1_loss(pred, batch, reduction="none")
        loss = (loss * mask).sum() / (mask.sum() * self.in_chans + 1e-5)
        return {"total": loss, "recon": loss}

    @torch.inference_mode()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """可视化：被 mask 区域用预测、可见区域用原图，拼成完整图。"""
        pred, mask = self.forward(x)
        return x * (1.0 - mask) + pred * mask
