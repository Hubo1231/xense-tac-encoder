"""Typed training configs selected by tyro."""
from __future__ import annotations

import dataclasses
import difflib
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import tyro
import yaml

from src.models.base_model import BaseModelConfig
from src.models.mobilenet_v4.mobilenetv4_config import Mobilenetv4Config


OptimizerVariant = Literal["adamw", "adam", "sgd"]

# The dataset class is hard-coded; ``name`` is no longer a config knob.
DATASET_NAME: str = "tactile_images"


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


def apply_overrides(config: MutableMapping[str, Any], overrides: Sequence[str] | None) -> MutableMapping[str, Any]:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        key, raw_value = item.split("=", 1)
        set_by_path(config, key.strip(), yaml.safe_load(raw_value.strip()))
    return config


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    config = load_yaml(path)
    base_path = config.pop("base", None)
    if base_path:
        parent = load_config(path.parent / base_path)
        config = dict(deep_update(parent, config))
    return dict(apply_overrides(config, overrides))


def _require(value: Any, name: str) -> Any:
    """Raise ValueError if ``value`` is None or an empty string."""
    if value is None:
        raise ValueError(f"{name} is required and must not be None.")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} is required and must not be empty.")
    return value


@dataclasses.dataclass(frozen=True)
class TransformConfig:
    """Image preprocessing config consumed by ``transform_dataset``.

    Mirrors the keyword arguments of ``timm.data.create_transform``. Defaults
    follow the pretrained_cfg of
    ``mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k`` (bicubic, ImageNet
    mean/std, ``crop_pct=0.95``, center crop).
    """

    input_size: tuple[int, int, int] = (3, 224, 224)
    interpolation: str = "bicubic"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    crop_pct: float = 0.95
    crop_mode: Literal["center", "squash", "border"] = "center"
    scale: tuple[float, float] = (0.08, 1.0)
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)
    hflip: float = 0.5
    vflip: float = 0.0
    color_jitter: float = 0.4
    auto_augment: str | None = None
    re_prob: float = 0.0
    re_mode: str = "const"
    re_count: int = 1


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Resolved dataset config consumed by ``create_torch_dataset``.

    Single-root layout: all RGB images live under ``root`` and are split
    deterministically into train/eval using ``eval_ratio`` and ``seed``.
    """

    root: str | None = None
    image_size: int | None = None
    seed: int | None = None
    eval_ratio: float = 0.2
    transform: TransformConfig = dataclasses.field(default_factory=TransformConfig)

    def __post_init__(self) -> None:
        _require(self.root, "DataConfig.root")
        _require(self.image_size, "DataConfig.image_size")
        if self.image_size is not None and int(self.image_size) <= 0:
            raise ValueError(f"DataConfig.image_size must be positive, got {self.image_size}.")
        _require(self.seed, "DataConfig.seed")
        if not 0.0 < float(self.eval_ratio) < 1.0:
            raise ValueError(
                f"DataConfig.eval_ratio must be in (0, 1), got {self.eval_ratio}."
            )


@dataclasses.dataclass(frozen=True)
class Mobilenetv4DataConfig:
    """User-facing dataset factory for tactile RGB images.

    All images are read from a single ``root`` directory (recursively). The
    train/eval split is deterministic given ``seed``. Preprocessing defaults
    follow the pretrained_cfg of
    ``mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k``.
    """

    root: str | None = None
    eval_ratio: float = 0.2
    image_size: int | None = None
    seed: int | None = None

    interpolation: str = "bicubic"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    crop_pct: float = 0.95
    crop_mode: Literal["center", "squash", "border"] = "center"
    scale: tuple[float, float] = (0.08, 1.0)
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)
    hflip: float = 0.5
    vflip: float = 0.0
    color_jitter: float = 0.4
    auto_augment: str | None = None
    re_prob: float = 0.0
    re_mode: str = "const"
    re_count: int = 1

    def create(self, model_config: BaseModelConfig) -> DataConfig:
        root = _require(self.root, "Mobilenetv4DataConfig.root")
        image_size = self.image_size if self.image_size is not None else getattr(model_config, "image_size", None)
        _require(image_size, "Mobilenetv4DataConfig.image_size or model_config.image_size")
        if int(image_size) <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}.")
        seed = _require(self.seed, "Mobilenetv4DataConfig.seed")
        in_chans = int(getattr(model_config, "input_channels", 3))

        transform_cfg = TransformConfig(
            input_size=(in_chans, int(image_size), int(image_size)),
            interpolation=self.interpolation,
            mean=tuple(self.mean),
            std=tuple(self.std),
            crop_pct=float(self.crop_pct),
            crop_mode=self.crop_mode,
            scale=tuple(self.scale),
            ratio=tuple(self.ratio),
            hflip=float(self.hflip),
            vflip=float(self.vflip),
            color_jitter=float(self.color_jitter),
            auto_augment=self.auto_augment,
            re_prob=float(self.re_prob),
            re_mode=self.re_mode,
            re_count=int(self.re_count),
        )
        return DataConfig(
            root=str(root),
            image_size=int(image_size),
            seed=int(seed),
            eval_ratio=float(self.eval_ratio),
            transform=transform_cfg,
        )


@dataclasses.dataclass(frozen=True)
class LossConfig:
    mse: float = 1.0
    grad: float = 2.0
    kld: float = 0.01
    ssim: float = 0.0


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    name: OptimizerVariant = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    checkpoint: str | None = None
    max_batches: int = 0
    measure_latency_only_encoder: bool = True
    output_json: str | None = None


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Top-level experiment config.

    Required user-facing fields (no defaults; missing raises at construction):
    ``name``, ``model``, ``data``, ``batch_size``, ``eval_batch_size``,
    ``num_workers``, ``seed``, ``epochs``.
    """

    name: str
    model: Mobilenetv4Config
    data: Mobilenetv4DataConfig

    batch_size: int
    eval_batch_size: int
    num_workers: int
    seed: int
    epochs: int

    project_name: str = "xense-tac-encoder"
    exp_name: str | None = None

    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)

    checkpoint_base_dir: str = "runs"

    device: str = "auto"
    log_interval: int = 20
    overwrite: bool = False
    resume: bool = False

    wandb_enabled: bool = False
    wandb_mode: str | None = None
    evaluation: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)

    @property
    def resolved_exp_name(self) -> str:
        return self.exp_name or self.name

    @property
    def checkpoint_dir(self) -> Path:
        return (Path(self.checkpoint_base_dir) / self.name / self.resolved_exp_name).resolve()

    @property
    def checkpoint_path(self) -> Path:
        return (Path(self.checkpoint_base_dir) / f"{self.resolved_exp_name}.pt").resolve()

    @property
    def eval_output_path(self) -> Path:
        return (Path(self.checkpoint_base_dir) / f"{self.resolved_exp_name}_eval.json").resolve()

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if int(self.batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if int(self.eval_batch_size) <= 0:
            raise ValueError(f"eval_batch_size must be positive, got {self.eval_batch_size}.")
        if int(self.num_workers) < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}.")
        if int(self.epochs) <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}.")


# Default data root is the directory the user said they will populate by hand.
DEFAULT_DATA_ROOT: str = "/home/li/hubo/xense-tac-encoder/data"


_CONFIGS = [
    TrainConfig(
        name="mobilenetv4_conv_aa_large_vae",
        model=Mobilenetv4Config(
            model_variant="vae",
            image_size=224,
            latent_dim=256,
            pretrained=False,
            decoder_hidden_channels=512,
            decoder_hidden_spatial=7,
        ),
        data=Mobilenetv4DataConfig(
            root=DEFAULT_DATA_ROOT,
            eval_ratio=0.2,
            image_size=224,
            seed=42,
        ),
        loss=LossConfig(mse=1.0, grad=2.0, kld=0.01, ssim=0.0),
        optimizer=OptimizerConfig(name="adamw", lr=1e-4, weight_decay=0.01),
        batch_size=4,
        eval_batch_size=4,
        num_workers=4,
        seed=42,
        epochs=10,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


TactileEncoderConfig = Mobilenetv4Config
ModelConfig = Mobilenetv4Config
TactileImagesDataConfig = Mobilenetv4DataConfig
