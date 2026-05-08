"""PyTorch dataset / dataloader for tactile RGB image reconstruction.

Single-root layout: all images live recursively under ``data_config.root``.
Train/eval is a deterministic 8:2 random split keyed by ``data_config.seed``
(``eval_ratio`` controls the eval fraction; defaults to 0.2).

Required config fields are enforced with explicit ``ValueError`` instead of
silent defaults — see ``DataConfig`` / ``Mobilenetv4DataConfig`` /
``TrainConfig`` in ``src/training/config.py``.
"""
from __future__ import annotations

import logging
import multiprocessing
import random
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Generic, Protocol, SupportsIndex, TypeVar

import torch
from PIL import Image
from timm.data import create_transform
from torch.utils.data import DataLoader as TorchDataLoaderBase
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

from src.training import config as _config


T_co = TypeVar("T_co", covariant=True)
DataTransform = Callable[[Any], Any]
IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SPLITS: tuple[str, ...] = ("train", "eval")


def list_images(root: str | Path) -> list[Path]:
    """Recursively list image files under ``root``, sorted for determinism."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Image root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Image root is not a directory: {root_path}")
    paths = sorted(p for p in root_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise ValueError(f"No images with extensions {IMAGE_EXTS} found under {root_path}")
    return paths


def split_image_paths(
    paths: Sequence[Path],
    *,
    eval_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    """Deterministic 8:2 (or any ``eval_ratio``) random split.

    ``paths`` should be a sorted, stable list — the function shuffles a copy
    using ``random.Random(seed)`` so the split is reproducible across runs.
    """
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f"eval_ratio must be in (0, 1), got {eval_ratio}.")
    n = len(paths)
    if n < 2:
        raise ValueError(f"Need at least 2 images to split, got {n}.")

    indices = list(range(n))
    random.Random(int(seed)).shuffle(indices)
    n_eval = max(1, int(round(n * float(eval_ratio))))
    n_eval = min(n_eval, n - 1)  # guarantee at least one train sample
    eval_idx = sorted(indices[:n_eval])
    train_idx = sorted(indices[n_eval:])
    return {
        "train": [paths[i] for i in train_idx],
        "eval": [paths[i] for i in eval_idx],
    }


class Dataset(Protocol[T_co]):
    """Dataset protocol with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class DataLoader(Protocol[T_co]):
    """Minimal iterable data loader protocol."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError


class TactileDataset(TorchDataset):
    """Load tactile RGB images from a folder or an explicit path sequence.

    With ``transform=None`` the dataset emits raw PIL images so callers can
    wrap it via ``transform_dataset`` (the timm-style pipeline).
    """

    def __init__(
        self,
        source: str | Path | Sequence[str | Path],
        *,
        transform: Callable | None = None,
    ) -> None:
        if source is None:
            raise ValueError("TactileDataset.source is required.")
        if isinstance(source, (str, Path)):
            self.image_paths: list[Path] = list_images(source)
        else:
            paths = [Path(p) for p in source]
            if not paths:
                raise ValueError("TactileDataset received an empty path sequence.")
            self.image_paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Any:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image) if self.transform is not None else image


class RandomTactileDataset(TorchDataset):
    """Random normalized tensors. Kept for unit tests and smoke checks."""

    def __init__(self, length: int, image_size: int) -> None:
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}.")
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}.")
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        del index
        return torch.randn(3, self.image_size, self.image_size)


class TransformedDataset(TorchDataset):
    """Apply one transform or a sequence of transforms to another dataset."""

    def __init__(self, dataset: TorchDataset, transforms: DataTransform | Sequence[DataTransform]) -> None:
        self._dataset = dataset
        if isinstance(transforms, Sequence):
            self._transforms = list(transforms)
        else:
            self._transforms = [transforms]

    def __getitem__(self, index: SupportsIndex) -> Any:
        item = self._dataset[index]
        for transform in self._transforms:
            item = transform(item)
        return item

    def __len__(self) -> int:
        return len(self._dataset)


class TorchDataLoader(Generic[T_co]):
    """Thin wrapper around ``torch.utils.data.DataLoader``.

    When ``num_batches`` is set, iteration restarts from the dataset until
    that many batches have been yielded. Without ``num_batches`` it behaves
    like a normal finite PyTorch data loader.
    """

    def __init__(
        self,
        dataset: TorchDataset,
        batch_size: int,
        *,
        shuffle: bool = False,
        sampler: Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        drop_last: bool = True,
        pin_memory: bool = False,
        collate_fn: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if num_batches is not None and num_batches <= 0:
            raise ValueError(f"num_batches must be positive, got {num_batches}.")
        if drop_last and hasattr(dataset, "__len__") and len(dataset) < batch_size:
            raise ValueError(
                f"batch_size ({batch_size}) is larger than dataset size ({len(dataset)})."
            )

        mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
        generator = torch.Generator()
        generator.manual_seed(int(seed))

        self._num_batches = num_batches
        self._data_loader = TorchDataLoaderBase(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            prefetch_factor=16 if num_workers > 0 else None,
            collate_fn=collate_fn,
            worker_init_fn=_seed_worker,
            drop_last=drop_last,
            pin_memory=pin_memory,
            generator=generator,
        )

    @property
    def torch_loader(self) -> TorchDataLoaderBase:
        return self._data_loader

    def __iter__(self) -> Iterator[Any]:
        yielded = 0
        while True:
            start = time.monotonic()
            data_iter = iter(self._data_loader)
            logging.debug("[TorchDataLoader] iter() returned in %.1fs", time.monotonic() - start)

            for batch in data_iter:
                if self._num_batches is not None and yielded >= self._num_batches:
                    return
                yielded += 1
                yield batch

            if self._num_batches is None or yielded >= self._num_batches:
                return


def build_timm_transform(transform_config: _config.TransformConfig, *, is_training: bool) -> Callable:
    """Build a timm-style transform pipeline from ``TransformConfig``.

    Augmentations (RRC scale/ratio, flips, color jitter, AutoAugment, random
    erasing) only apply when ``is_training=True``; eval uses
    resize → center crop → normalize sized via ``crop_pct`` and ``crop_mode``.
    """
    return create_transform(
        input_size=transform_config.input_size,
        is_training=is_training,
        interpolation=transform_config.interpolation,
        mean=transform_config.mean,
        std=transform_config.std,
        crop_pct=transform_config.crop_pct,
        crop_mode=transform_config.crop_mode,
        scale=transform_config.scale,
        ratio=transform_config.ratio,
        hflip=transform_config.hflip,
        vflip=transform_config.vflip,
        color_jitter=transform_config.color_jitter,
        auto_augment=transform_config.auto_augment,
        re_prob=transform_config.re_prob,
        re_mode=transform_config.re_mode,
        re_count=transform_config.re_count,
    )


def transform_dataset(
    dataset: TorchDataset,
    data_config: _config.DataConfig,
    *,
    split: str,
) -> TorchDataset:
    """Wrap ``dataset`` with the timm-style transform implied by ``data_config``."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}.")
    transform = build_timm_transform(data_config.transform, is_training=split == "train")
    return TransformedDataset(dataset, transform)


def _resolve_data_config(config: Any) -> _config.DataConfig:
    """Accept either a resolved ``DataConfig`` or a ``TrainConfig``."""
    if isinstance(config, _config.DataConfig):
        return config
    if hasattr(config, "data") and hasattr(config.data, "create"):
        return config.data.create(config.model)
    raise TypeError(
        "Expected DataConfig or TrainConfig with .data.create(model); "
        f"got {type(config).__name__}."
    )


def create_torch_dataset(
    config: Any,
    *,
    split: str,
) -> TorchDataset:
    """Build the ``TactileDataset`` for ``split`` from a (Train|Data)Config.

    The dataset resolves its split by deterministically partitioning the
    images under ``data_config.root`` using ``data_config.seed`` and
    ``data_config.eval_ratio``.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}.")
    data_config = _resolve_data_config(config)

    paths = list_images(_config._require(data_config.root, "data_config.root"))
    splits = split_image_paths(
        paths,
        eval_ratio=float(data_config.eval_ratio),
        seed=int(_config._require(data_config.seed, "data_config.seed")),
    )
    return TactileDataset(splits[split])


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def _distributed_sampler(dataset: TorchDataset, shuffle: bool) -> DistributedSampler | None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None
    return DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=shuffle,
        drop_last=True,
    )


def create_torch_data_loader(
    config: Any,
    *,
    split: str,
    num_batches: int | None = None,
    pin_memory: bool = False,
) -> TorchDataLoader:
    """Build a ``TorchDataLoader`` for ``split`` from a ``TrainConfig``.

    Required fields on ``config``: ``batch_size``, ``eval_batch_size``,
    ``num_workers``, ``seed``. Missing values raise instead of silently
    defaulting.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}.")
    if not (hasattr(config, "data") and hasattr(config.data, "create")):
        raise TypeError(
            "create_torch_data_loader expects a TrainConfig-like object with "
            "a .data.create(model) factory."
        )

    data_config = config.data.create(config.model)
    dataset = create_torch_dataset(data_config, split=split)
    dataset = transform_dataset(dataset, data_config, split=split)

    if split == "train":
        batch_size = int(_config._require(getattr(config, "batch_size", None), "TrainConfig.batch_size"))
        shuffle = True
        drop_last = True
    else:
        batch_size = int(_config._require(getattr(config, "eval_batch_size", None), "TrainConfig.eval_batch_size"))
        shuffle = False
        drop_last = False

    num_workers = int(_config._require(getattr(config, "num_workers", None), "TrainConfig.num_workers"))
    seed = int(_config._require(getattr(config, "seed", None), "TrainConfig.seed"))

    sampler = _distributed_sampler(dataset, shuffle)
    local_batch_size = batch_size
    if sampler is not None:
        world_size = torch.distributed.get_world_size()
        if batch_size % world_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by world_size ({world_size})."
            )
        local_batch_size = batch_size // world_size

    return TorchDataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


__all__ = [
    "DataLoader",
    "Dataset",
    "IMAGE_EXTS",
    "RandomTactileDataset",
    "SPLITS",
    "TactileDataset",
    "TorchDataLoader",
    "TransformedDataset",
    "build_timm_transform",
    "create_torch_data_loader",
    "create_torch_dataset",
    "list_images",
    "split_image_paths",
    "transform_dataset",
]
