"""候选编码器 backbone 注册表。

每个 build_* 函数返回 (backbone_module, feature_dim, spatial_size)：
  - backbone_module: 输入 (B,3,H,W)，输出 (B,C,h,w) 特征图
  - feature_dim:    C
  - spatial_size:   h == w，用于 VAE 解码器入口尺寸推断

约定：尽量去掉分类头，保留卷积/下采样部分；同时支持 224 输入下的固定空间尺寸。
"""
from typing import Callable, Dict, Tuple

import torch.nn as nn
from torchvision import models


BackboneFactory = Callable[[bool], Tuple[nn.Module, int, int]]
_REGISTRY: Dict[str, BackboneFactory] = {}


def register(name: str) -> Callable[[BackboneFactory], BackboneFactory]:
    def _wrap(fn: BackboneFactory) -> BackboneFactory:
        _REGISTRY[name] = fn
        return fn
    return _wrap


def available_backbones() -> list:
    return sorted(_REGISTRY.keys())


def build_backbone(name: str, pretrained: bool = False) -> Tuple[nn.Module, int, int]:
    if name not in _REGISTRY:
        raise KeyError(f"未知 backbone: {name}; 可选: {available_backbones()}")
    return _REGISTRY[name](pretrained)


# ---------- ResNet 系列 ----------
def _strip_resnet(net: nn.Module) -> nn.Module:
    # 去掉 avgpool 与 fc，��留到最后一个 stage 的特征图（224 输入下为 7x7）
    return nn.Sequential(*list(net.children())[:-2])


@register("resnet18")
def _resnet18(pretrained: bool):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    return _strip_resnet(models.resnet18(weights=weights)), 512, 7


@register("resnet34")
def _resnet34(pretrained: bool):
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    return _strip_resnet(models.resnet34(weights=weights)), 512, 7


@register("resnet50")
def _resnet50(pretrained: bool):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    return _strip_resnet(models.resnet50(weights=weights)), 2048, 7


# ---------- 轻量级移动端 backbone ----------
@register("mobilenet_v3_small")
def _mbv3s(pretrained: bool):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    net = models.mobilenet_v3_small(weights=weights)
    return net.features, 576, 7


@register("mobilenet_v3_large")
def _mbv3l(pretrained: bool):
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    net = models.mobilenet_v3_large(weights=weights)
    return net.features, 960, 7


class _MobileNetV4Features(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_features(x)


@register("mobilenetv4_conv_aa_large")
def _mbv4_conv_aa_large(pretrained: bool):
    from src.models.mobilenet_v4.mobilenetv4_conv_aa_large import mobilenetv4_conv_aa_large

    net = mobilenetv4_conv_aa_large(pretrained=pretrained, num_classes=0)
    return _MobileNetV4Features(net), net.num_features, 7


@register("efficientnet_b0")
def _effb0(pretrained: bool):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    return net.features, 1280, 7


@register("shufflenet_v2_x1_0")
def _shuffle(pretrained: bool):
    weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
    net = models.shufflenet_v2_x1_0(weights=weights)
    # ShuffleNet 没有干净的 features 属性，手动取到 conv5
    backbone = nn.Sequential(
        net.conv1, net.maxpool, net.stage2, net.stage3, net.stage4, net.conv5,
    )
    return backbone, 1024, 7


# ---------- 小型 ConvNeXt（适合 System-0 部署） ----------
@register("convnext_tiny")
def _convnext_tiny(pretrained: bool):
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    net = models.convnext_tiny(weights=weights)
    return net.features, 768, 7
