"""Config-driven evaluation utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .evaluator import evaluate_encoder, format_results_table
from .factory import (
    build_dataloader_from_config,
    build_loss_from_config,
    build_model_from_config,
    resolve_device,
)
from .training import load_checkpoint


def _limited_batches(loader, max_batches: int):
    if max_batches <= 0:
        return loader
    batches = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batches.append(batch)

    class _Wrapped:
        def __iter__(self):
            return iter(batches)

        def __len__(self):
            return len(batches)

    return _Wrapped()


def evaluate_from_config(config: Mapping[str, Any]):
    eval_cfg = config.get("evaluation", {})
    device = resolve_device(config, "evaluation")
    loader = build_dataloader_from_config(config, "eval", device=device)
    loader = _limited_batches(loader, int(eval_cfg.get("max_batches", 0)))

    model = build_model_from_config(config)
    checkpoint = eval_cfg.get("checkpoint") or config.get("checkpoint")
    if checkpoint:
        load_checkpoint(model, checkpoint, strict=bool(eval_cfg.get("strict", False)))

    loss_fn = build_loss_from_config(config)
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "model")
    params = model_cfg.get("params", {})
    run_name = eval_cfg.get("name") or params.get("backbone_name") or model_name
    data_cfg = config.get("data", {})
    eval_data_cfg = data_cfg.get("eval", {})
    image_size = int(eval_data_cfg.get("image_size", data_cfg.get("image_size", 224)))

    result = evaluate_encoder(
        model=model,
        dataloader=loader,
        device=device,
        backbone_name=run_name,
        loss_fn=loss_fn,
        measure_latency_only_encoder=bool(eval_cfg.get("measure_latency_only_encoder", True)),
        latency_input_shape=(1, 3, image_size, image_size),
    )

    print(format_results_table([result]))

    output_json = eval_cfg.get("output_json")
    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"metrics saved to {out}")

    return result
