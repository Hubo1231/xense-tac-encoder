"""模型构建入口。"""
from typing import Callable, Dict

import torch.nn as nn

from .backbones import available_backbones, build_backbone
from .vae import TactileAutoencoder, TactileVAE
from src.training.losses import LossWeights


ModelFactory = Callable[..., nn.Module]
_MODEL_REGISTRY: Dict[str, ModelFactory] = {}


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    def _wrap(fn: ModelFactory) -> ModelFactory:
        _MODEL_REGISTRY[name] = fn
        return fn
    return _wrap


def available_models() -> list:
    return sorted(_MODEL_REGISTRY.keys())


@register_model("tactile_vae")
def build_tactile_vae(
    backbone_name: str,
    latent_dim: int = 256,
    pretrained: bool = False,
    decoder_hidden_channels: int = 512,
    decoder_hidden_spatial: int | None = None,
    loss_weights: LossWeights | dict | None = None,
) -> TactileVAE:
    """根据 backbone 名称组装一个完整的 TactileVAE。"""
    backbone, feat_dim, spatial = build_backbone(backbone_name, pretrained=pretrained)
    if isinstance(loss_weights, dict):
        loss_weights = LossWeights(**loss_weights)
    return TactileVAE(
        encoder_backbone=backbone,
        feature_dim=feat_dim,
        latent_dim=latent_dim,
        decoder_hidden_channels=decoder_hidden_channels,
        decoder_hidden_spatial=decoder_hidden_spatial or spatial,
        loss_weights=loss_weights,
    )


@register_model("tactile_autoencoder")
def build_tactile_autoencoder(
    backbone_name: str,
    latent_dim: int = 256,
    pretrained: bool = False,
    decoder_hidden_channels: int = 512,
    decoder_hidden_spatial: int | None = None,
    loss_weights: LossWeights | dict | None = None,
) -> TactileAutoencoder:
    """根据 backbone 名称组装一个确定性重建 Autoencoder。"""
    backbone, feat_dim, spatial = build_backbone(backbone_name, pretrained=pretrained)
    if isinstance(loss_weights, dict):
        loss_weights = LossWeights(**loss_weights)
    return TactileAutoencoder(
        encoder_backbone=backbone,
        feature_dim=feat_dim,
        latent_dim=latent_dim,
        decoder_hidden_channels=decoder_hidden_channels,
        decoder_hidden_spatial=decoder_hidden_spatial or spatial,
        loss_weights=loss_weights,
    )


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"未知 model: {name}; 可选: {available_models()}")
    return _MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "TactileAutoencoder",
    "TactileVAE",
    "available_backbones",
    "available_models",
    "build_backbone",
    "build_model",
    "build_tactile_autoencoder",
    "build_tactile_vae",
]
