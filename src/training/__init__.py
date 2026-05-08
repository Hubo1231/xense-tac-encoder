"""Xense tactile RGB encoder package."""

__all__ = [
    "BaseModel",
    "BaseModelConfig",
    "ModelType",
    "TactileEncoderConfig",
    "TrainConfig",
    "available_configs",
    "get_config",
]


def __getattr__(name: str):
    if name in {"BaseModel", "BaseModelConfig", "ModelType"}:
        from src.models.base_model import BaseModel, BaseModelConfig, ModelType

        return {
            "BaseModel": BaseModel,
            "BaseModelConfig": BaseModelConfig,
            "ModelType": ModelType,
        }[name]
    if name in {"TactileEncoderConfig", "TrainConfig", "available_configs", "get_config"}:
        from .config import TactileEncoderConfig, TrainConfig, available_configs, get_config

        return {
            "TactileEncoderConfig": TactileEncoderConfig,
            "TrainConfig": TrainConfig,
            "available_configs": available_configs,
            "get_config": get_config,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
