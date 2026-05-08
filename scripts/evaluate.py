"""Config-driven evaluation entrypoint."""
import argparse
import dataclasses
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import available_backbones
from src.training.config import get_config, load_config, set_by_path
from src.utils.evaluation import evaluate_from_config
from src.utils.evaluator import format_results_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="vae_resnet18", help="Named preset or YAML config path")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. --set evaluation.checkpoint=runs/resnet18.pt",
    )

    # Legacy shortcuts kept for old comparison commands.
    p.add_argument("--backbones", nargs="+", default=None, help=f"可选: {available_backbones()}")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--w-grad", type=float, default=None)
    p.add_argument("--w-kld", type=float, default=None)
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def apply_legacy_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "checkpoint": "evaluation.checkpoint",
        "data": "data.root",
        "image_size": "data.image_size",
        "latent_dim": "model.params.latent_dim",
        "batch_size": "dataloader.eval.batch_size",
        "num_workers": "dataloader.eval.num_workers",
        "max_batches": "evaluation.max_batches",
        "device": "evaluation.device",
        "w_grad": "loss.grad",
        "w_kld": "loss.kld",
        "output_json": "evaluation.output_json",
    }
    for arg_name, config_path in mapping.items():
        value = getattr(args, arg_name)
        if value is not None:
            set_by_path(config, config_path, value)

    if args.pretrained:
        set_by_path(config, "model.params.pretrained", True)
    return config


def _config_to_eval_dict(config) -> dict[str, Any]:
    data = config.data.create(config.model)
    model_name = "tactile_vae" if config.model.model_variant == "vae" else "tactile_autoencoder"
    return {
        "name": config.name,
        "project_name": config.project_name,
        "seed": config.seed,
        "device": config.device,
        "model": {
            "name": model_name,
            "params": {
                "backbone_name": "mobilenetv4_conv_aa_large",
                "latent_dim": config.model.latent_dim,
                "pretrained": config.model.pretrained,
                "decoder_hidden_channels": config.model.decoder_hidden_channels,
                "decoder_hidden_spatial": config.model.decoder_hidden_spatial,
            },
        },
        "loss": dataclasses.asdict(config.loss),
        "data": {
            "root": data.root,
            "image_size": data.image_size,
            "eval_ratio": data.eval_ratio,
            "seed": data.seed,
        },
        "dataloader": {
            "train": {
                "batch_size": config.batch_size,
                "shuffle": True,
                "num_workers": config.num_workers,
                "drop_last": True,
            },
            "eval": {
                "batch_size": config.eval_batch_size,
                "shuffle": False,
                "num_workers": config.num_workers,
                "drop_last": False,
            },
        },
        "optimizer": dataclasses.asdict(config.optimizer),
        "evaluation": {
            "device": config.device,
            "checkpoint": config.evaluation.checkpoint or str(config.checkpoint_path),
            "max_batches": config.evaluation.max_batches,
            "measure_latency_only_encoder": config.evaluation.measure_latency_only_encoder,
            "output_json": config.evaluation.output_json or str(config.eval_output_path),
        },
    }


def _load_eval_config(config: str, overrides: list[str]) -> dict[str, Any]:
    path = Path(config)
    if config in {name for name in get_config.__globals__["_CONFIGS_DICT"]}:
        loaded = _config_to_eval_dict(get_config(config))
    elif path.exists():
        loaded = load_config(path)
        loaded.setdefault("name", path.stem)
    else:
        loaded = _config_to_eval_dict(get_config(config))
    for item in overrides:
        key, raw_value = item.split("=", 1)
        set_by_path(loaded, key, json.loads(raw_value) if raw_value[:1] in "[{\"" else raw_value)
    return loaded


def main() -> None:
    args = parse_args()
    base_config = apply_legacy_args(_load_eval_config(args.config, args.overrides), args)
    backbones = args.backbones or [base_config.get("model", {}).get("params", {}).get("backbone_name", "resnet18")]
    if len(backbones) > 1 and args.checkpoint:
        raise SystemExit("--checkpoint 模式下只允许指定一个 backbone")

    results = []
    for backbone in backbones:
        config = deepcopy(base_config)
        set_by_path(config, "model.params.backbone_name", backbone)
        set_by_path(config, "evaluation.name", backbone)
        if len(backbones) > 1:
            set_by_path(config, "evaluation.output_json", None)
            set_by_path(config, "evaluation.checkpoint", None)
        print(f"\n==> evaluating backbone: {backbone}")
        results.append(evaluate_from_config(config))

    if len(results) > 1:
        print("\n==== summary ====")
        print(format_results_table(results))

    output_json = base_config.get("evaluation", {}).get("output_json")
    if output_json and len(results) > 1:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([r.as_dict() for r in results], indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nmetrics saved to {out}")


if __name__ == "__main__":
    main()
