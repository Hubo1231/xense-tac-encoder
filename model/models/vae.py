"""触觉 RGB VAE 模型。"""
from typing import Tuple

import torch
import torch.nn as nn

from ..base_model import BaseModel


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
    ) -> None:
        super().__init__(image_size=224, input_channels=3, latent_dim=latent_dim)
        self.encoder = encoder_backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_latent = nn.Linear(feature_dim, latent_dim)
        self.decoder = TactileDecoder(latent_dim, decoder_hidden_channels, decoder_hidden_spatial)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = self.pool(feat).flatten(1)
        return self.fc_latent(feat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z


class TactileVAE(BaseModel):
    """编码器 + 重参数化 + 解码器。

    encoder_backbone 输出 (B, C, h, w)；通过 AdaptiveAvgPool 到 1x1 后投影到 latent。
    将空间信息聚合后再投影，避免 backbone 输出维度差异导致 fc 层巨大。
    """

    def __init__(
        self,
        encoder_backbone: nn.Module,
        feature_dim: int,
        latent_dim: int = 256,
        decoder_hidden_channels: int = 512,
        decoder_hidden_spatial: int = 7,
    ) -> None:
        super().__init__(image_size=224, input_channels=3, latent_dim=latent_dim)
        self.encoder = encoder_backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)
        self.decoder = TactileDecoder(latent_dim, decoder_hidden_channels, decoder_hidden_spatial)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        feat = self.pool(feat).flatten(1)
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
