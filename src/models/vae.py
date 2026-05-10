"""触觉 RGB VAE 模型。"""
from typing import Tuple

import torch
import torch.nn as nn

from src.models.base_model import BaseModel
from src.training.losses import LossWeights, kl_divergence, mix_loss, reconstruction_terms, ssim_loss


class TactileDecoder(nn.Module):
    """简单反卷积解码器：从 (B, latent_dim) 重建到 (B, 3, 224, 224)。"""

    def __init__(self, latent_dim: int, hidden_channels: int = 512, hidden_spatial: int = 7) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_spatial = hidden_spatial
        self.fc = nn.Linear(latent_dim, hidden_channels * hidden_spatial * hidden_spatial)

        # 5 次 stride=2 反卷积：7 -> 14 -> 28 -> 56 -> 112 -> 224
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, 256, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            # 不加 Tanh，直接输出与归一化后输入对齐的连续值，配合 MSE/Grad loss
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.hidden_channels, self.hidden_spatial, self.hidden_spatial)
        return self.deconv(h)


class TactileAutoencoder(BaseModel):
    """Deterministic encoder-decoder baseline for reconstruction tasks."""

    def __init__(
        self,
        encoder_backbone: nn.Module,
        feature_dim: int,
        latent_dim: int = 256,
        decoder_hidden_channels: int = 512,
        decoder_hidden_spatial: int = 7,
        loss_weights: LossWeights | None = None,
        image_size: int = 224,
    ) -> None:
        super().__init__(image_size=image_size, input_channels=3, latent_dim=latent_dim)
        self.encoder = encoder_backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_latent = nn.Linear(feature_dim, latent_dim)
        self.decoder = TactileDecoder(latent_dim, decoder_hidden_channels, decoder_hidden_spatial)
        self.loss_weights = loss_weights or LossWeights(kld=0.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = self.pool(feat).flatten(1)
        return self.fc_latent(feat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    def compute_loss(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        recon, _ = self.forward(batch)
        mse, grad = reconstruction_terms(recon, batch)
        total = self.loss_weights.mse * mse + self.loss_weights.grad * grad
        losses = {"total": total, "mse": mse, "grad": grad}
        if self.loss_weights.ssim > 0:
            ssim = ssim_loss(recon, batch)
            losses["ssim"] = ssim
            total = total + self.loss_weights.ssim * ssim
        if self.loss_weights.mix > 0:
            mix = mix_loss(
                recon, batch,
                alpha=self.loss_weights.mix_alpha,
                use_ms_ssim=self.loss_weights.use_ms_ssim,
            )
            losses["mix"] = mix
            total = total + self.loss_weights.mix * mix
        losses["total"] = total
        return losses


class TactileVAE(BaseModel):
    """编码器 + 重参数化 + 解码器。

    encoder_backbone 通过 timm 风格的接口
    ``output = encoder(transforms(img).unsqueeze(0))`` 返回 (B, feature_dim)
    的图像 embedding；TactileVAE 再用 fc_mu / fc_logvar 投影到 latent，
    走标准 VAE 重参数化与反卷积解码。
    """

    def __init__(
        self,
        encoder_backbone: nn.Module,
        feature_dim: int,
        latent_dim: int = 256,
        decoder_hidden_channels: int = 512,
        decoder_hidden_spatial: int = 7,
        loss_weights: LossWeights | None = None,
        image_size: int = 224,
    ) -> None:
        super().__init__(image_size=image_size, input_channels=3, latent_dim=latent_dim)
        self.encoder = encoder_backbone
        self.feature_dim = feature_dim
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)
        self.decoder = TactileDecoder(latent_dim, decoder_hidden_channels, decoder_hidden_spatial)
        self.loss_weights = loss_weights or LossWeights()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        if feat.ndim != 2:
            raise ValueError(
                f"TactileVAE encoder must return (B, C) embeddings, got shape {tuple(feat.shape)}."
            )
        if feat.shape[1] != self.feature_dim:
            raise ValueError(
                f"Encoder embedding dim {feat.shape[1]} does not match configured feature_dim {self.feature_dim}."
            )
        return self.fc_mu(feat), self.fc_logvar(feat)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def compute_loss(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        recon, mu, logvar = self.forward(batch)
        mse, grad = reconstruction_terms(recon, batch)
        kld = kl_divergence(mu, logvar)
        total = self.loss_weights.mse * mse + self.loss_weights.grad * grad + self.loss_weights.kld * kld
        losses = {"total": total, "mse": mse, "grad": grad, "kld": kld}
        if self.loss_weights.ssim > 0:
            ssim = ssim_loss(recon, batch)
            losses["ssim"] = ssim
            total = total + self.loss_weights.ssim * ssim
        if self.loss_weights.mix > 0:
            mix = mix_loss(
                recon, batch,
                alpha=self.loss_weights.mix_alpha,
                use_ms_ssim=self.loss_weights.use_ms_ssim,
            )
            losses["mix"] = mix
            total = total + self.loss_weights.mix * mix
        losses["total"] = total
        return losses
