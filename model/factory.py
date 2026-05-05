"""Factories that build runtime objects from config dictionaries."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader

from .base_model import BaseModelConfig
from .dataset import RandomTactileDataset, TactileDataset
from .losses import build_loss
from .models import build_model


def _without_split_keys(config: Mapping[str, Any]) -> dict[str, Any]:
    split_keys = {"train", "eval", "val", "test"}
    return {key: value for key, value in config.items() if key not in split_keys}


def _merged_split_config(config: Mapping[str, Any], section: str, split: str) -> dict[str, Any]:
    base = _without_split_keys(config.get(section, {}))
    split_cfg = config.get(section, {}).get(split, {})
    if isinstance(split_cfg, str):
        split_cfg = {"root": split_cfg}
    if split_cfg is None:
        split_cfg = {}
    if not isinstance(split_cfg, Mapping):
        raise TypeError(f"{section}.{split} must be a mapping, string path, or null")
    base.update(split_cfg)
    return base


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


def build_loss_from_config(config: Mapping[str, Any]):
    loss_cfg = config.get("loss", {})
    name = loss_cfg.get("name", "tactile_vae")
    params = dict(loss_cfg.get("params", {}))
    for key, value in loss_cfg.items():
        if key not in {"name", "params"}:
            params.setdefault(key, value)
    return build_loss(name, **params)


def build_dataset_from_config(config: Mapping[str, Any], split: str):
    data_cfg = _merged_split_config(config, "data", split)
    name = data_cfg.get("name", "tactile_images")
    image_size = int(data_cfg.get("image_size", 224))
    root = data_cfg.get("root", data_cfg.get("source"))
    use_random = bool(data_cfg.get("random", False)) or root in (None, "")

    if name == "random" or use_random:
        default_length = 256 if split == "train" else 128
        length = int(data_cfg.get("length", data_cfg.get("random_length", default_length)))
        return RandomTactileDataset(length=length, image_size=image_size)
    if name != "tactile_images":
        raise KeyError(f"未知 dataset: {name}; 可选: tactile_images, random")
    return TactileDataset(root, image_size=image_size)


def build_dataloader_from_config(
    config: Mapping[str, Any],
    split: str,
    device: torch.device | None = None,
) -> DataLoader:
    dataset = build_dataset_from_config(config, split)
    loader_cfg = _merged_split_config(config, "dataloader", split)
    if device is None:
        device = resolve_device(config, "training" if split == "train" else "evaluation")

    batch_size = int(loader_cfg.get("batch_size", 32 if split == "train" else 16))
    shuffle = bool(loader_cfg.get("shuffle", split == "train"))
    num_workers = int(loader_cfg.get("num_workers", 4))
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
