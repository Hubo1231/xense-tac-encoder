"""Config-driven evaluation entrypoint."""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tactile_encoder.config import load_config, set_by_path
from tactile_encoder.evaluation import evaluate_from_config
from tactile_encoder.evaluator import format_results_table
from tactile_encoder.models import available_backbones


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/vae_resnet18.yaml", help="YAML config path")
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
        "data": "data.eval.root",
        "image_size": "data.image_size",
        "latent_dim": "model.params.latent_dim",
        "batch_size": "dataloader.eval.batch_size",
        "num_workers": "dataloader.eval.num_workers",
        "max_batches": "evaluation.max_batches",
        "device": "evaluation.device",
        "w_grad": "loss.params.grad",
        "w_kld": "loss.params.kld",
        "output_json": "evaluation.output_json",
    }
    for arg_name, config_path in mapping.items():
        value = getattr(args, arg_name)
        if value is not None:
            set_by_path(config, config_path, value)

    if args.pretrained:
        set_by_path(config, "model.params.pretrained", True)
    return config


def main() -> None:
    args = parse_args()
    base_config = apply_legacy_args(load_config(args.config, args.overrides), args)
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
