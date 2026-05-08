"""MobileNetV4 Conv AA Large model definition extracted from timm.

This file contains the architecture used by:

    timm/mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k

It keeps the model definition local while reusing timm's low-level mobile
blocks and EfficientNet builder utilities.
"""
from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Linear, SelectAdaptivePool2d, create_conv2d, get_norm_act_layer
from timm.models._efficientnet_blocks import SqueezeExcite
from timm.models._efficientnet_builder import (
    BlockArgs,
    EfficientNetBuilder,
    decode_arch_def,
    efficientnet_init_weights,
    resolve_act_layer,
    resolve_bn_args,
    round_channels,
)
from timm.models._helpers import load_state_dict as timm_load_state_dict
from timm.models._manipulate import checkpoint_seq


MODEL_NAME = "mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k"
ARCHITECTURE = "mobilenetv4_conv_aa_large"
SOURCE_FILE = "timm/models/mobilenetv3.py"

DEFAULT_CFG: Dict[str, Any] = {
    "url": "",
    "hf_hub_id": "timm/mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k",
    "architecture": ARCHITECTURE,
    "tag": "e230_r448_in12k_ft_in1k",
    "custom_load": False,
    "input_size": (3, 448, 448),
    "test_input_size": (3, 544, 544),
    "fixed_input_size": False,
    "interpolation": "bicubic",
    "crop_pct": 0.95,
    "test_crop_pct": 1.0,
    "crop_mode": "center",
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "num_classes": 1000,
    "pool_size": (14, 14),
    "first_conv": "conv_stem",
    "classifier": "classifier",
    "license": "apache-2.0",
}


class MobileNetV3(nn.Module):
    """MobileNetV3/MobileNetV4 classification backbone used by timm."""

    def __init__(
        self,
        block_args: BlockArgs,
        num_classes: int = 1000,
        in_chans: int = 3,
        stem_size: int = 16,
        fix_stem: bool = False,
        num_features: int = 1280,
        head_bias: bool = True,
        head_norm: bool = False,
        pad_type: str = "",
        act_layer: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        aa_layer: Optional[Any] = None,
        se_layer: Optional[Callable[..., nn.Module]] = None,
        se_from_exp: bool = True,
        round_chs_fn: Callable[..., int] = round_channels,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: Optional[float] = None,
        global_pool: str = "avg",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite

        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type, **dd)
        self.bn1 = norm_act_layer(stem_size, inplace=True, **dd)

        builder = EfficientNetBuilder(
            output_stride=32,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            se_from_exp=se_from_exp,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            **dd,
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        self.stage_ends = [f["stage"] for f in self.feature_info]
        self.num_features = builder.in_chs
        self.head_hidden_size = num_features

        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        num_pooled_chs = self.num_features * self.global_pool.feat_mult()
        if head_norm:
            self.conv_head = create_conv2d(
                num_pooled_chs,
                self.head_hidden_size,
                1,
                padding=pad_type,
                bias=False,
                **dd,
            )
            self.norm_head = norm_act_layer(self.head_hidden_size, **dd)
            self.act2 = nn.Identity()
        else:
            self.conv_head = create_conv2d(
                num_pooled_chs,
                self.head_hidden_size,
                1,
                padding=pad_type,
                bias=head_bias,
                **dd,
            )
            self.norm_head = nn.Identity()
            self.act2 = act_layer(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.classifier = Linear(self.head_hidden_size, num_classes, **dd) if num_classes > 0 else nn.Identity()

        efficientnet_init_weights(self)

    def as_sequential(self) -> nn.Sequential:
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.norm_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Any]:
        return {
            "stem": r"^conv_stem|bn1",
            "blocks": r"^blocks\.(\d+)" if coarse else r"^blocks\.(\d+)\.(\d+)",
        }

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.classifier

    def reset_classifier(self, num_classes: int, global_pool: str = "avg") -> None:
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.classifier = Linear(self.head_hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.norm_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        if pre_logits:
            return x
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _arch_def() -> list[list[str]]:
    """Architecture string list for mobilenetv4_conv_aa_large."""
    return [
        [
            "er_r1_k3_s2_e4_c48",
        ],
        [
            "uir_r1_a3_k5_s2_e4_c96",
            "uir_r1_a3_k3_s1_e4_c96",
        ],
        [
            "uir_r1_a3_k5_s2_e4_c192",
            "uir_r3_a3_k3_s1_e4_c192",
            "uir_r1_a3_k5_s1_e4_c192",
            "uir_r5_a5_k3_s1_e4_c192",
            "uir_r1_a3_k0_s1_e4_c192",
        ],
        [
            "uir_r4_a5_k5_s2_e4_c512",
            "uir_r1_a5_k0_s1_e4_c512",
            "uir_r1_a5_k3_s1_e4_c512",
            "uir_r2_a5_k0_s1_e4_c512",
            "uir_r1_a5_k3_s1_e4_c512",
            "uir_r1_a5_k5_s1_e4_c512",
            "uir_r3_a5_k0_s1_e4_c512",
        ],
        [
            "cn_r1_k1_s1_c960",
        ],
    ]


def mobilenetv4_conv_aa_large(
    pretrained: bool = False,
    checkpoint_path: str | Path | None = None,
    strict: bool = True,
    **kwargs: Any,
) -> MobileNetV3:
    """Build MobileNetV4 Conv AA Large.

    Args:
        pretrained: Load local weights when True.
        checkpoint_path: Optional file or directory containing local weights.
        strict: Strictness passed to ``load_state_dict`` when loading weights.
        **kwargs: Model overrides such as ``num_classes``, ``in_chans`` or ``drop_rate``.
    """
    if kwargs.pop("features_only", False):
        raise NotImplementedError("This extracted file only defines the classification model.")

    model_kwargs = _model_kwargs(**kwargs)
    model = MobileNetV3(**model_kwargs)
    cfg = deepcopy(DEFAULT_CFG)
    cfg["num_classes"] = model.num_classes
    model.default_cfg = cfg
    model.pretrained_cfg = cfg

    if pretrained:
        load_local_pretrained(model, checkpoint_path=checkpoint_path, strict=strict)
    return model


def _model_kwargs(
    channel_multiplier: float = 1.0,
    group_size: int | None = None,
    aa_layer: Any = "avg",
    **kwargs: Any,
) -> Dict[str, Any]:
    act_layer = resolve_act_layer(kwargs, "relu")
    return {
        "block_args": decode_arch_def(_arch_def(), group_size=group_size),
        "head_bias": False,
        "head_norm": True,
        "num_features": 1280,
        "stem_size": 24,
        "fix_stem": channel_multiplier < 1.0,
        "round_chs_fn": partial(round_channels, multiplier=channel_multiplier),
        "norm_layer": partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        "act_layer": act_layer,
        "aa_layer": aa_layer,
        "layer_scale_init_value": None,
        **kwargs,
    }


def _resolve_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if path.is_file():
            return path
        if path.is_dir():
            found = _find_checkpoint_in_dir(path)
            if found is not None:
                return found
        raise FileNotFoundError(f"No checkpoint file found at {path}")

    here = Path(__file__).resolve().parent
    for directory in (here, here.parent):
        found = _find_checkpoint_in_dir(directory)
        if found is not None:
            return found
    raise FileNotFoundError(
        "No local checkpoint found. Expected model.safetensors or pytorch_model.bin "
        f"in {here} or {here.parent}."
    )


def _find_checkpoint_in_dir(directory: Path) -> Path | None:
    for name in ("model.safetensors", "pytorch_model.bin", "pytorch_model.pth", "model.pth"):
        path = directory / name
        if path.is_file():
            return path
    return None


def load_local_pretrained(
    model: nn.Module,
    checkpoint_path: str | Path | None = None,
    strict: bool = True,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load local checkpoint weights into ``model``.

    Returns a small report containing the resolved checkpoint and load results.
    """
    path = _resolve_checkpoint_path(checkpoint_path)
    state_dict = timm_load_state_dict(str(path), device=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    return {
        "checkpoint": str(path),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def create_model(
    pretrained: bool = False,
    checkpoint_path: str | Path | None = None,
    strict: bool = True,
    **kwargs: Any,
) -> MobileNetV3:
    """Compatibility alias for constructing the extracted model."""
    return mobilenetv4_conv_aa_large(
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        strict=strict,
        **kwargs,
    )


__all__ = [
    "ARCHITECTURE",
    "DEFAULT_CFG",
    "MODEL_NAME",
    "MobileNetV3",
    "create_model",
    "load_local_pretrained",
    "mobilenetv4_conv_aa_large",
]
