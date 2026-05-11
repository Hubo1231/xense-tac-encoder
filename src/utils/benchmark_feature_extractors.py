"""Benchmark image-embedding latency for timm encoder backbones on GPU.

Each model is built via ``timm.create_model(name, num_classes=0, ...)`` so the
classification head is replaced by an identity / global pool. The timed call is
``model(x)``, which therefore returns the pooled image embedding of shape
``(B, C)``. Image loading and preprocessing are kept outside the timed section.

Typical commands:

    # Benchmark a single timm encoder using a project config.
    python src/utils/benchmark_feature_extractors.py \
        --config configs/fastvit_ma36_apple_in1k.yaml \
        --warmup 50 \
        --iters 500

    # Benchmark without loading local checkpoint weights.
    python src/utils/benchmark_feature_extractors.py \
        --config configs/fastvit_ma36_apple_in1k.yaml \
        --ignore-config-pretrained-path \
        --warmup 50 \
        --iters 500

    # Benchmark multiple timm encoders at 448x448.
    python src/utils/benchmark_feature_extractors.py \
        --models mobilenetv4_conv_aa_large resnet18 efficientnet_b0 \
        --input-size 3 448 448 \
        --warmup 50 \
        --iters 500

    # Save results as CSV.
    python src/utils/benchmark_feature_extractors.py \
        --models fastvit_ma36 convnextv2_tiny efficientvit_b3 \
        --output-csv outputs/embedding_latency.csv \
        --warmup 50 \
        --iters 500
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from timm.data import create_transform, resolve_data_config
from timm.models import create_model

try:
    from PIL import Image
except ImportError:  # pragma: no cover - only needed for --image
    Image = None


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL = "mobilenetv4_conv_aa_large"


@dataclass
class BenchmarkResult:
    model: str
    device: str
    dtype: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    warmup: int
    iters: int
    batch_size: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    mean_ms_per_image: float
    params_million: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure image-to-embedding latency for one or more timm encoder backbones."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_MODEL],
        help="timm model names to benchmark.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config. model / pretrained_path / in_chans / model_kwargs are read from it.",
    )
    parser.add_argument("--device", default="cuda", help="GPU device, e.g. cuda or cuda:0. Default: cuda.")
    parser.add_argument(
        "--input-size",
        nargs=3,
        type=int,
        metavar=("C", "H", "W"),
        default=None,
        help="Input tensor shape without batch. Default: timm pretrained config or 3 224 224.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. Default: 1.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations before timing. Default: 20.")
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations. Default: 200.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Tensor/model dtype. Default: float32.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained weights when supported. Disabled by default because weights do not affect speed much.",
    )
    parser.add_argument(
        "--pretrained-path",
        type=Path,
        default=None,
        help="Local timm pretrained checkpoint file. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--ignore-config-pretrained-path",
        action="store_true",
        help="Ignore pretrained_path from --config and create the model without local weights.",
    )
    parser.add_argument(
        "--reparameterize",
        action="store_true",
        help=(
            "Fuse reparameterizable train-time branches after checkpoint loading. "
            "This can also be enabled by setting reparameterize: true in the config."
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional real image used to build the input tensor. Preprocessing is not timed.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save benchmark results as CSV.",
    )
    return parser.parse_args()


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config root must be a mapping: {path}")
    return data


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        if device.type == "cpu":
            raise SystemExit("float16 benchmarking is only supported on CUDA-like devices.")
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def resolve_project_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else REPO_ROOT / path


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def reparameterize_model(module: nn.Module) -> int:
    """Fuse modules that expose a timm-style ``reparameterize()`` method."""
    fused = 0
    for child in reversed(list(module.modules())):
        reparameterize = getattr(child, "reparameterize", None)
        if callable(reparameterize):
            reparameterize()
            fused += 1
    return fused


def build_timm_encoder(
    name: str,
    *,
    config: dict[str, Any],
    pretrained: bool,
    pretrained_path: Path | None,
    ignore_config_pretrained_path: bool,
    reparameterize: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    """Create a timm model with ``num_classes=0`` so ``model(x) -> (B, C)``."""
    cfg_pretrained_path = None if ignore_config_pretrained_path else config.get("pretrained_path")
    resolved_pretrained_path = pretrained_path or (Path(cfg_pretrained_path) if cfg_pretrained_path else None)
    resolved_pretrained_path = resolve_project_path(resolved_pretrained_path)

    factory_kwargs: dict[str, Any] = {}
    if resolved_pretrained_path is not None:
        factory_kwargs["pretrained_cfg_overlay"] = {
            "file": str(resolved_pretrained_path),
            "num_classes": -1,
        }

    model_kwargs = dict(config.get("model_kwargs", {}) or {})
    model = create_model(
        name,
        pretrained=pretrained or resolved_pretrained_path is not None,
        in_chans=int(config.get("in_chans", 3)),
        num_classes=0,
        **factory_kwargs,
        **model_kwargs,
    )
    if reparameterize:
        fused = reparameterize_model(model)
        if fused == 0:
            print(f"warning: --reparameterize requested, but {name} exposes no reparameterizable modules.")

    data_config = resolve_data_config({}, model=model)
    return model, data_config


def build_input_tensor(
    *,
    image_path: Path | None,
    data_config: dict[str, Any],
    input_size: list[int] | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if input_size is not None:
        shape = tuple(input_size)
    else:
        shape = tuple(int(v) for v in data_config.get("input_size", (3, 224, 224)))

    if len(shape) != 3:
        raise SystemExit(f"Input size must be C H W, got: {shape}")
    if batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    if image_path is None:
        return torch.randn((batch_size, *shape), device=device, dtype=dtype)

    if Image is None:
        raise SystemExit("Pillow is required for --image. Install pillow or omit --image.")
    if not image_path.is_file():
        raise SystemExit(f"Image file not found: {image_path}")

    transform_config = dict(data_config)
    transform_config["input_size"] = shape
    transform = create_transform(**transform_config, is_training=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


@torch.inference_mode()
def benchmark_once(
    model: nn.Module,
    input_tensor: torch.Tensor,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[list[float], tuple[int, ...]]:
    if warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if iters < 1:
        raise SystemExit("--iters must be >= 1")

    model.eval()
    output = None
    for _ in range(warmup):
        output = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings_ms: list[float] = []
    if device.type == "cuda":
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(input_tensor)
            end.record()
            torch.cuda.synchronize(device)
            timings_ms.append(start.elapsed_time(end))
    else:
        for _ in range(iters):
            start = time.perf_counter()
            output = model(input_tensor)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    if not torch.is_tensor(output):
        raise SystemExit(
            f"Expected timm model(x) to return a tensor embedding, got: {type(output)!r}"
        )
    return timings_ms, tuple(output.shape)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def summarize_result(
    *,
    model_name: str,
    model: nn.Module,
    input_tensor: torch.Tensor,
    output_shape: tuple[int, ...],
    timings_ms: list[float],
    warmup: int,
    device: torch.device,
    dtype_name: str,
) -> BenchmarkResult:
    mean_ms = statistics.fmean(timings_ms)
    return BenchmarkResult(
        model=model_name,
        device=str(device),
        dtype=dtype_name,
        input_shape=tuple(input_tensor.shape),
        output_shape=output_shape,
        warmup=warmup,
        iters=len(timings_ms),
        batch_size=input_tensor.shape[0],
        mean_ms=mean_ms,
        median_ms=statistics.median(timings_ms),
        p90_ms=percentile(timings_ms, 0.90),
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
        std_ms=statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        mean_ms_per_image=mean_ms / input_tensor.shape[0],
        params_million=count_parameters(model) / 1e6,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'model':<32} {'device':<8} {'input':<18} {'output':<14}"
        f" {'mean(ms)':>10} {'median':>10} {'p90':>10} {'per_img':>10} {'params(M)':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item.model:<32} {item.device:<8} "
            f"{str(item.input_shape):<18} {str(item.output_shape):<14} "
            f"{item.mean_ms:>10.3f} {item.median_ms:>10.3f} {item.p90_ms:>10.3f} "
            f"{item.mean_ms_per_image:>10.3f} {item.params_million:>10.2f}"
        )


def save_csv(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BenchmarkResult.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = item.__dict__.copy()
            row["input_shape"] = json.dumps(row["input_shape"])
            row["output_shape"] = json.dumps(row["output_shape"])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    model_names = list(args.models)
    if args.config and args.models == [DEFAULT_MODEL] and config.get("model"):
        model_names = [config["model"]]

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    results: list[BenchmarkResult] = []
    for model_name in model_names:
        model, data_config = build_timm_encoder(
            model_name,
            config=config,
            pretrained=args.pretrained or bool(config.get("pretrained", False)),
            pretrained_path=args.pretrained_path,
            ignore_config_pretrained_path=args.ignore_config_pretrained_path,
            reparameterize=args.reparameterize or bool(config.get("reparameterize", False)),
        )

        model = model.to(device=device, dtype=dtype)
        input_tensor = build_input_tensor(
            image_path=args.image,
            data_config=data_config,
            input_size=args.input_size,
            batch_size=args.batch_size,
            device=device,
            dtype=dtype,
        )
        timings_ms, output_shape = benchmark_once(
            model,
            input_tensor,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        results.append(
            summarize_result(
                model_name=model_name,
                model=model,
                input_tensor=input_tensor,
                output_shape=output_shape,
                timings_ms=timings_ms,
                warmup=args.warmup,
                device=device,
                dtype_name=args.dtype,
            )
        )

    print_results(results)
    if args.output_csv is not None:
        save_csv(args.output_csv, results)
        print(f"\nsaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
