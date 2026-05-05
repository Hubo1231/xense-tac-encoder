"""Base model interfaces for tactile image reconstruction models."""
from __future__ import annotations

import abc
import dataclasses
import enum
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ImageBatch = torch.Tensor
ModelOutput = torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]


class ModelType(enum.Enum):
    """Supported model families."""

    TACTILE_VAE = "tactile_vae"
    TACTILE_AUTOENCODER = "tactile_autoencoder"
    CUSTOM = "custom"


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all reconstruction models.

    Concrete model configs should inherit from this class and implement
    `model_type`, `create`, and `as_dict`. This mirrors openpi's model config
    contract while using PyTorch tensors and modules for this repository.
    """

    # Input image shape expected by the model, in CHW format.
    image_size: int = 224
    input_channels: int = 3
    # Size of the bottleneck representation used by reconstruction models.
    latent_dim: int = 256

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model family."""

    @abc.abstractmethod
    def create(self, *, device: str | torch.device | None = None) -> "BaseModel":
        """Create a new model instance."""

    @abc.abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Serialize this config into the runtime config dictionary format."""

    def load(
        self,
        checkpoint: str | Path | dict[str, Any],
        *,
        strict: bool = False,
        map_location: str | torch.device = "cpu",
        remove_extra_params: bool = True,
        device: str | torch.device | None = None,
    ) -> "BaseModel":
        """Create a model and load weights from a checkpoint or state dict."""

        model = self.create(device=device)
        if isinstance(checkpoint, (str, Path)):
            ckpt = torch.load(checkpoint, map_location=map_location)
        else:
            ckpt = checkpoint

        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

        if remove_extra_params:
            expected = model.state_dict()
            state = {key: value for key, value in state.items() if key in expected}

        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing:
            logger.warning("Missing parameters when loading %s: %d", self.model_type.value, len(missing))
        if unexpected:
            logger.warning("Unexpected parameters when loading %s: %d", self.model_type.value, len(unexpected))
        return model

    def inputs_spec(
        self,
        *,
        batch_size: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ImageBatch:
        """Return a tensor with the expected input shape.

        PyTorch does not have a native ShapeDtypeStruct equivalent, so this
        returns an empty tensor carrying the same shape, dtype, and device that
        real batches should have.
        """

        return torch.empty(
            batch_size,
            self.input_channels,
            self.image_size,
            self.image_size,
            device=device,
            dtype=dtype,
        )

    def fake_obs(
        self,
        *,
        batch_size: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ImageBatch:
        return torch.ones_like(self.inputs_spec(batch_size=batch_size, device=device, dtype=dtype))

    def fake_target(
        self,
        *,
        batch_size: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ImageBatch:
        return self.fake_obs(batch_size=batch_size, device=device, dtype=dtype)


class BaseModel(nn.Module, abc.ABC):
    """Base class for PyTorch tactile reconstruction models."""

    def __init__(self, *, image_size: int = 224, input_channels: int = 3, latent_dim: int = 256) -> None:
        super().__init__()
        self.image_size = image_size
        self.input_channels = input_channels
        self.latent_dim = latent_dim

    @abc.abstractmethod
    def encode(self, x: ImageBatch) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Encode an image batch into latent features."""

    @abc.abstractmethod
    def decode(self, z: torch.Tensor) -> ImageBatch:
        """Decode latent features into reconstructed images."""

    @abc.abstractmethod
    def forward(self, x: ImageBatch) -> ModelOutput:
        """Run the model forward pass."""

    def reconstruct(self, x: ImageBatch) -> ImageBatch:
        """Return only the reconstructed image from common output formats."""

        output = self.forward(x)
        if torch.is_tensor(output):
            return output
        if isinstance(output, dict):
            for key in ("recon", "reconstruction", "x_hat", "pred"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return output[0]
        raise TypeError(f"Cannot extract reconstruction from output type: {type(output)!r}")

    def compute_loss(self, batch: ImageBatch, loss_fn) -> dict[str, torch.Tensor]:
        """Compute reconstruction loss for an image batch using a provided loss function."""

        outputs = self.forward(batch)
        return loss_fn(outputs, batch)
