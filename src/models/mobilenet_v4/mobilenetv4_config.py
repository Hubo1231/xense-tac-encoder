"""MobileNetV4 VAE task configuration."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from src.models.base_model import BaseModel, BaseModelConfig, ModelType
from src.models.mobilenet_v4.mobilenetv4_conv_aa_large import (
    ARCHITECTURE,
    MODEL_NAME,
    mobilenetv4_conv_aa_large,
)
from src.models.vae import TactileAutoencoder, TactileVAE


ModelVariant = Literal["vae", "autoencoder"]


class _MobileNetV4Features(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)


@dataclasses.dataclass(frozen=True)
class Mobilenetv4Config(BaseModelConfig):
    """Config for training MobileNetV4 as the encoder in a reconstruction task."""

    model_variant: ModelVariant = "vae"
    model_name: str = MODEL_NAME
    architecture: str = ARCHITECTURE

    image_size: int = 224
    input_channels: int = 3
    latent_dim: int = 256
    pretrained: bool = False
    checkpoint_path: str | Path | None = None
    strict: bool = True

    decoder_hidden_channels: int = 512
    decoder_hidden_spatial: int | None = 7

    channel_multiplier: float = 1.0
    group_size: int | None = None
    aa_layer: Any = "avg"
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if self.channel_multiplier <= 0:
            raise ValueError("channel_multiplier must be greater than zero.")

    @property
    def model_type(self) -> ModelType:
        if self.model_variant == "vae":
            return ModelType.TACTILE_VAE
        if self.model_variant == "autoencoder":
            return ModelType.TACTILE_AUTOENCODER
        return ModelType.CUSTOM

    def _create_backbone(self, *, device: str | torch.device | None = None) -> tuple[nn.Module, int, int]:
        model = mobilenetv4_conv_aa_large(
            pretrained=self.pretrained,
            checkpoint_path=self.checkpoint_path,
            strict=self.strict,
            num_classes=0,
            in_chans=self.input_channels,
            channel_multiplier=self.channel_multiplier,
            group_size=self.group_size,
            aa_layer=self.aa_layer,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            device=torch.device(device) if device is not None else None,
        )
        return _MobileNetV4Features(model), model.num_features, self.decoder_hidden_spatial or 7

    def create(self, *, device: str | torch.device | None = None) -> BaseModel:
        backbone, feature_dim, spatial = self._create_backbone(device=device)
        model_cls = TactileVAE if self.model_variant == "vae" else TactileAutoencoder
        model = model_cls(
            encoder_backbone=backbone,
            feature_dim=feature_dim,
            latent_dim=self.latent_dim,
            decoder_hidden_channels=self.decoder_hidden_channels,
            decoder_hidden_spatial=spatial,
        )
        if device is not None:
            model = model.to(device)
        return model

MobileNetV4Config = Mobilenetv4Config


__all__ = ["Mobilenetv4Config", "MobileNetV4Config"]
