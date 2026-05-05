"""Config loading and override helpers."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
import dataclasses
import difflib
from pathlib import Path
from typing import Any
import tyro

import torch
import yaml

from .base_model import BaseModel, BaseModelConfig, ModelType
from .models import build_model


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def deep_update(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def set_by_path(config: MutableMapping[str, Any], dotted_path: str, value: Any) -> None:
    keys = [key for key in dotted_path.split(".") if key]
    if not keys:
        raise ValueError("Override path cannot be empty")

    node: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, MutableMapping):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def get_by_path(config: Mapping[str, Any], dotted_path: str, default: Any = None) -> Any:
    node: Any = config
    for key in dotted_path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(config: MutableMapping[str, Any], overrides: Sequence[str] | None) -> MutableMapping[str, Any]:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        key, raw_value = item.split("=", 1)
        set_by_path(config, key.strip(), parse_override_value(raw_value.strip()))
    return config


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    config = load_yaml(path)

    base_path = config.pop("base", None)
    if base_path:
        parent = load_config(path.parent / base_path)
        config = dict(deep_update(parent, config))

    return dict(apply_overrides(config, overrides))


@dataclasses.dataclass(frozen=True)
class ModelConfig(BaseModelConfig):
    # Reconstruction model registered in model.models.
    name: str = "tactile_vae"
    # Visual backbone registered in model.models.backbones.
    backbone_name: str = "resnet18"
    latent_dim: int = 256
    image_size: int = 224
    input_channels: int = 3
    pretrained: bool = False
    decoder_hidden_channels: int = 512
    decoder_hidden_spatial: int | None = None

    @property
    def model_type(self) -> ModelType:
        if self.name in ("tactile_vae", "vae"):
            return ModelType.TACTILE_VAE
        if self.name in ("tactile_autoencoder", "autoencoder", "ae"):
            return ModelType.TACTILE_AUTOENCODER
        return ModelType.CUSTOM

    def create(self, *, device: str | torch.device | None = None) -> BaseModel:
        model = build_model(
            self.name,
            backbone_name=self.backbone_name,
            latent_dim=self.latent_dim,
            pretrained=self.pretrained,
            decoder_hidden_channels=self.decoder_hidden_channels,
            decoder_hidden_spatial=self.decoder_hidden_spatial,
        )
        if device is not None:
            model = model.to(device)
        return model

    def as_dict(self) -> dict[str, Any]:
        params = {
            "backbone_name": self.backbone_name,
            "latent_dim": self.latent_dim,
            "pretrained": self.pretrained,
            "decoder_hidden_channels": self.decoder_hidden_channels,
        }
        if self.decoder_hidden_spatial is not None:
            params["decoder_hidden_spatial"] = self.decoder_hidden_spatial
        return {"name": self.name, "params": params}


@dataclasses.dataclass(frozen=True)
class LossConfig:
    name: str = "tactile_vae"
    mse: float = 1.0
    grad: float = 2.0
    kld: float = 0.01
    ssim: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": {
                "mse": self.mse,
                "grad": self.grad,
                "kld": self.kld,
                "ssim": self.ssim,
            },
        }


@dataclasses.dataclass(frozen=True)
class DataSplitConfig:
    root: str | None = None
    length: int | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"root": self.root}
        if self.length is not None:
            out["length"] = self.length
        return out


@dataclasses.dataclass(frozen=True)
class DataConfig:
    name: str = "tactile_images"
    image_size: int = 224
    train: DataSplitConfig = dataclasses.field(default_factory=lambda: DataSplitConfig(length=256))
    eval: DataSplitConfig = dataclasses.field(default_factory=lambda: DataSplitConfig(length=128))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "image_size": self.image_size,
            "train": self.train.as_dict(),
            "eval": self.eval.as_dict(),
        }


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "lr": self.lr, "weight_decay": self.weight_decay}


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: str
    # Project name used by wandb and metadata.
    project_name: str = "eval-tactile-encoder"
    # Experiment name. Defaults to name.
    exp_name: str | None = None

    # Core experiment definition.
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)

    # Base directory for checkpoints and eval outputs.
    checkpoint_base_dir: str = "runs"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global train batch size.
    batch_size: int = 32
    # Eval batch size used by evaluation configs.
    eval_batch_size: int = 16
    # Number of workers used by train/eval data loaders.
    num_workers: int = 4
    # Number of epochs to run.
    epochs: int = 20

    # Runtime options.
    device: str = "auto"
    log_interval: int = 20
    overwrite: bool = False
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = False
    wandb_mode: str | None = None

    @property
    def checkpoint_path(self) -> Path:
        exp_name = self.exp_name or self.name
        return (Path(self.checkpoint_base_dir) / f"{exp_name}.pt").resolve()

    @property
    def eval_output_path(self) -> Path:
        exp_name = self.exp_name or self.name
        return (Path(self.checkpoint_base_dir) / f"{exp_name}_eval.json").resolve()

    def as_dict(self) -> dict[str, Any]:
        checkpoint_path = str(self.checkpoint_path)
        return {
            "name": self.name,
            "project_name": self.project_name,
            "exp_name": self.exp_name,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_workers": self.num_workers,
            "epochs": self.epochs,
            "log_interval": self.log_interval,
            "overwrite": self.overwrite,
            "resume": self.resume,
            "wandb_enabled": self.wandb_enabled,
            "wandb": {
                "enabled": self.wandb_enabled,
                "project": self.project_name,
                "name": self.exp_name or self.name,
                "mode": self.wandb_mode,
            },
            "model": self.model.as_dict(),
            "loss": self.loss.as_dict(),
            "data": self.data.as_dict(),
            "dataloader": {
                "train": {
                    "batch_size": self.batch_size,
                    "shuffle": True,
                    "num_workers": self.num_workers,
                    "drop_last": True,
                },
                "eval": {
                    "batch_size": self.eval_batch_size,
                    "shuffle": False,
                    "num_workers": self.num_workers,
                    "drop_last": False,
                },
            },
            "optimizer": self.optimizer.as_dict(),
            "training": {
                "device": self.device,
                "epochs": self.epochs,
                "log_every": self.log_interval,
                "output": checkpoint_path,
                "wandb_enabled": self.wandb_enabled,
            },
            "evaluation": {
                "device": self.device,
                "checkpoint": checkpoint_path,
                "max_batches": 0,
                "measure_latency_only_encoder": True,
                "output_json": str(self.eval_output_path),
            },
        }

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


def _vae_model(backbone_name: str, *, pretrained: bool = False, latent_dim: int = 256) -> ModelConfig:
    return ModelConfig(
        name="tactile_vae",
        backbone_name=backbone_name,
        latent_dim=latent_dim,
        pretrained=pretrained,
    )


def _ae_model(backbone_name: str, *, pretrained: bool = False, latent_dim: int = 256) -> ModelConfig:
    return ModelConfig(
        name="tactile_autoencoder",
        backbone_name=backbone_name,
        latent_dim=latent_dim,
        pretrained=pretrained,
    )


_TACTILE_DATA = DataConfig()


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    # VAE reconstruction configs. These are the primary configs for comparing visual backbones.
    TrainConfig(
        name="vae_mobilenet_v2_large",
        model=_vae_model("mobilenet_v2_large"),
        data=_TACTILE_DATA,
        batch_size=48,
        num_workers=4,
        wandb_enabled=False,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}

ConfigLike = TrainConfig | Mapping[str, Any]


def available_configs() -> list[str]:
    return sorted(_CONFIGS_DICT)


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]


def to_dict(config: ConfigLike) -> dict[str, Any]:
    if isinstance(config, TrainConfig):
        return config.as_dict()
    return normalize_train_config(dict(config))


def _override_paths(overrides: Sequence[str] | None) -> set[str]:
    paths = set()
    for item in overrides or []:
        if "=" in item:
            paths.add(item.split("=", 1)[0].strip())
    return paths


def _sync_scalar(
    config: dict[str, Any],
    top_level_path: str,
    nested_paths: Sequence[str],
    explicit_paths: set[str],
) -> None:
    explicit_nested_paths = [path for path in nested_paths if path in explicit_paths]
    if explicit_nested_paths and top_level_path not in explicit_paths:
        value = get_by_path(config, explicit_nested_paths[0])
        set_by_path(config, top_level_path, value)
        for nested_path in nested_paths:
            if nested_path not in explicit_paths:
                set_by_path(config, nested_path, value)
        return

    value = get_by_path(config, top_level_path)
    if value is not None:
        for nested_path in nested_paths:
            if nested_path not in explicit_paths:
                set_by_path(config, nested_path, value)


def normalize_train_config(config: dict[str, Any], explicit_paths: set[str] | None = None) -> dict[str, Any]:
    """Synchronize openpi-style top-level train fields with runtime config sections."""
    explicit_paths = explicit_paths or set()
    _sync_scalar(config, "batch_size", ("dataloader.train.batch_size",), explicit_paths)
    _sync_scalar(config, "eval_batch_size", ("dataloader.eval.batch_size",), explicit_paths)
    _sync_scalar(config, "num_workers", ("dataloader.train.num_workers", "dataloader.eval.num_workers"), explicit_paths)
    _sync_scalar(config, "epochs", ("training.epochs",), explicit_paths)
    _sync_scalar(config, "log_interval", ("training.log_every",), explicit_paths)
    _sync_scalar(config, "device", ("training.device", "evaluation.device"), explicit_paths)
    _sync_scalar(config, "wandb_enabled", ("wandb.enabled", "training.wandb_enabled"), explicit_paths)
    return config


def load_train_config(config: str | Path, overrides: Sequence[str] | None = None) -> dict[str, Any]:
    config_str = str(config)
    explicit_paths = _override_paths(overrides)
    if config_str in _CONFIGS_DICT:
        return normalize_train_config(dict(apply_overrides(get_config(config_str).as_dict(), overrides)), explicit_paths)
    if not Path(config).exists():
        return get_config(config_str).as_dict()
    return normalize_train_config(load_config(config, overrides), explicit_paths)


def apply_legacy_train_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "backbone": "model.params.backbone_name",
        "data": "data.train.root",
        "image_size": "data.image_size",
        "latent_dim": "model.params.latent_dim",
        "lr": "optimizer.lr",
        "w_mse": "loss.params.mse",
        "w_grad": "loss.params.grad",
        "w_kld": "loss.params.kld",
        "w_ssim": "loss.params.ssim",
        "output": "training.output",
        "wandb_project": "wandb.project",
        "wandb_name": "wandb.name",
        "wandb_mode": "wandb.mode",
    }
    for arg_name, config_path in mapping.items():
        value = getattr(args, arg_name)
        if value is not None:
            set_by_path(config, config_path, value)

    if args.pretrained:
        set_by_path(config, "model.params.pretrained", True)
    if args.epochs is not None:
        set_by_path(config, "epochs", args.epochs)
        set_by_path(config, "training.epochs", args.epochs)
    if args.batch_size is not None:
        set_by_path(config, "batch_size", args.batch_size)
        set_by_path(config, "dataloader.train.batch_size", args.batch_size)
    if args.num_workers is not None:
        set_by_path(config, "num_workers", args.num_workers)
        set_by_path(config, "dataloader.train.num_workers", args.num_workers)
        set_by_path(config, "dataloader.eval.num_workers", args.num_workers)
    if args.device is not None:
        set_by_path(config, "device", args.device)
        set_by_path(config, "training.device", args.device)
        set_by_path(config, "evaluation.device", args.device)
    if args.log_every is not None:
        set_by_path(config, "log_interval", args.log_every)
        set_by_path(config, "training.log_every", args.log_every)
    if args.wandb is not None:
        set_by_path(config, "wandb_enabled", args.wandb)
        set_by_path(config, "wandb.enabled", args.wandb)
        set_by_path(config, "training.wandb_enabled", args.wandb)
    return normalize_train_config(config)


if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}

def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})
