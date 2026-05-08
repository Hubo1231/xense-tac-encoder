"""Factories that build runtime objects from dict-style config payloads.

These helpers are kept for the legacy YAML / dict path used by
``src.utils.evaluation.evaluate_from_config``. The training entrypoint uses
the typed ``TrainConfig`` API in ``src/training/data_loader.py`` directly.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.models import build_model
from src.models.base_model import BaseModelConfig

from .config import _require
from .data_loader import (
    TactileDataset,
    build_timm_transform,
    list_images,
    split_image_paths,
)
from .config import TransformConfig


def resolve_device(config: Mapping[str, Any], section: str = "training") -> torch.device:
    section_cfg = config.get(section, {})
    requested = section_cfg.get("device", config.get("device", "auto"))
    if requested in (None, "auto"):
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def build_model_from_config(config: Mapping[str, Any]) -> torch.nn.Module:
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, BaseModelConfig):
        return model_cfg.create()
    name = model_cfg.get("name", "tactile_vae")
    params = dict(model_cfg.get("params", {}))
    for key, value in model_cfg.items():
        if key not in {"name", "params"}:
            params.setdefault(key, value)
    return build_model(name, **params)


def _build_transform(data_cfg: Mapping[str, Any], *, is_training: bool):
    image_size = int(_require(data_cfg.get("image_size"), "data.image_size"))
    transform_cfg = TransformConfig(
        input_size=(3, image_size, image_size),
        interpolation=data_cfg.get("interpolation", "bicubic"),
        mean=tuple(data_cfg.get("mean", (0.485, 0.456, 0.406))),
        std=tuple(data_cfg.get("std", (0.229, 0.224, 0.225))),
        crop_pct=float(data_cfg.get("crop_pct", 0.95)),
        crop_mode=data_cfg.get("crop_mode", "center"),
        scale=tuple(data_cfg.get("scale", (0.08, 1.0))),
        ratio=tuple(data_cfg.get("ratio", (3.0 / 4.0, 4.0 / 3.0))),
        hflip=float(data_cfg.get("hflip", 0.5)),
        vflip=float(data_cfg.get("vflip", 0.0)),
        color_jitter=float(data_cfg.get("color_jitter", 0.4)),
        auto_augment=data_cfg.get("auto_augment"),
        re_prob=float(data_cfg.get("re_prob", 0.0)),
        re_mode=data_cfg.get("re_mode", "const"),
        re_count=int(data_cfg.get("re_count", 1)),
    )
    return build_timm_transform(transform_cfg, is_training=is_training)


def build_dataset_from_config(config: Mapping[str, Any], split: str):
    """Build a ``TactileDataset`` for ``split`` using the dict config schema.

    Required keys under ``data``: ``root``, ``image_size``, ``seed``.
    ``eval_ratio`` defaults to 0.2.
    """
    if split not in {"train", "eval"}:
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}.")
    data_cfg = config.get("data", {})
    root = _require(data_cfg.get("root"), "data.root")
    seed = int(_require(data_cfg.get("seed"), "data.seed"))
    eval_ratio = float(data_cfg.get("eval_ratio", 0.2))

    paths = list_images(root)
    splits = split_image_paths(paths, eval_ratio=eval_ratio, seed=seed)
    transform = _build_transform(data_cfg, is_training=split == "train")
    return TactileDataset(splits[split], transform=transform)


def build_dataloader_from_config(
    config: Mapping[str, Any],
    split: str,
    device: torch.device | None = None,
) -> DataLoader:
    dataset = build_dataset_from_config(config, split)
    loader_cfg = config.get("dataloader", {}).get(split, {})
    if device is None:
        device = resolve_device(config, "training" if split == "train" else "evaluation")

    batch_size = int(_require(loader_cfg.get("batch_size"), f"dataloader.{split}.batch_size"))
    num_workers = int(_require(loader_cfg.get("num_workers"), f"dataloader.{split}.num_workers"))
    shuffle = bool(loader_cfg.get("shuffle", split == "train"))
    drop_last = bool(loader_cfg.get("drop_last", split == "train"))
    pin_memory = bool(loader_cfg.get("pin_memory", device.type == "cuda"))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def build_optimizer_from_config(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    optim_cfg = config.get("optimizer", {})
    name = optim_cfg.get("name", "adamw").lower()
    params = dict(optim_cfg.get("params", {}))
    for key, value in optim_cfg.items():
        if key not in {"name", "params"}:
            params.setdefault(key, value)

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **params)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **params)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **params)
    raise KeyError(f"未知 optimizer: {name}; 可选: adamw, adam, sgd")
