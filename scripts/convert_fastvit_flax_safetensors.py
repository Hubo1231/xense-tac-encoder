#!/usr/bin/env python3
"""把 Flax 风格的 FastViT safetensors 权重转换为 timm PyTorch state_dict 格式。

背景：checkpoint/ 下的 FastViT 权重（如 fastvit_t12_apple_dist_in1k/params.safetensors、
SimMIM 训练产出的 trained_params.safetensors）是 Flax 命名格式的 safetensors：

    stages_0/blocks_0/token_mixer/mixer/conv_kxk_0/conv/kernel  (HWIO)
    stages_0/blocks_0/token_mixer/mixer/conv_kxk_0/bn/{scale,bias,mean,var}
    stages_0/blocks_0/token_mixer/norm_identity/{scale,bias,mean,var}
    stages_0/blocks_0/layer_scale/gamma  (C,)

timm 的 checkpoint_filter_fn 只处理 Apple 官方 torch 格式（network.N.*），
无法识别该格式（加载时全部报 unexpected keys，权重实际未生效）。本脚本按以下
规则转换：

    路径段 `xxx_N` -> `xxx.N`（stem/stages/blocks/conv_kxk/proj 等带下标模块）
    `norm_identity` -> `norm.identity`（RepMixer 中 norm 分支的 BatchNorm）
    `conv/fc1/fc2/se 的 kernel` (HWIO) -> `.weight` (OIHW，permute 3,2,0,1)
    `bn.scale/bias/mean/var` -> `bn.weight/bias/running_mean/running_var`
    `identity.scale/...`（裸 BatchNorm 分支）同理
    `layer_scale.gamma` (C,) -> (C, 1, 1)

用法：
    python scripts/convert_fastvit_flax_safetensors.py \
        checkpoint/trained_params.safetensors checkpoint/fastvit_t12_simmim_timm.safetensors

可选 --verify-against-timm 用官方 apple_dist_in1k 权重做前向数值校验
（转换官方 params.safetensors 时用于验证转换规则本身的正确性）。
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_logger = logging.getLogger("convert_fastvit_flax_safetensors")

# 路径段末尾带 _N 下标的模块名（_N 是模块列表/Sequential 索引）
_INDEXED_SEGMENTS = ("stem", "stages", "blocks", "conv_kxk", "proj")


def convert_key(key: str) -> str:
    parts = key.split("/")
    out: list[str] = []
    for part in parts:
        match = re.match(r"^(.*)_(\d+)$", part)
        if match and match.group(1) in _INDEXED_SEGMENTS:
            out.append(f"{match.group(1)}.{match.group(2)}")
        else:
            out.append(part)
    key = ".".join(out)
    # RepMixer 的 norm 分支是一个 MobileOneBlock，其恒等分支 BatchNorm 名为 identity
    key = key.replace("norm_identity.", "norm.identity.")
    # BatchNorm：scale/mean/var -> weight/running_mean/running_var
    key = key.replace("bn.scale", "bn.weight")
    key = key.replace("bn.mean", "bn.running_mean")
    key = key.replace("bn.var", "bn.running_var")
    key = re.sub(r"(identity)\.scale$", r"\1.weight", key)
    key = re.sub(r"(identity)\.mean$", r"\1.running_mean", key)
    key = re.sub(r"(identity)\.var$", r"\1.running_var", key)
    # 卷积核（conv / fc1 / fc2 在 SE 与 ConvMlp 里都是 Conv2d）
    if key.endswith(".kernel"):
        key = key[: -len(".kernel")] + ".weight"
    return key


def convert_tensor(key: str, value: torch.Tensor) -> torch.Tensor:
    if key.endswith(".weight") and value.ndim == 4:
        # Flax HWIO -> PyTorch OIHW
        return value.permute(3, 2, 0, 1).contiguous()
    if key.endswith("layer_scale.gamma") and value.ndim == 1:
        return value.reshape(-1, 1, 1).contiguous()
    return value


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {convert_key(k): convert_tensor(convert_key(k), v) for k, v in state_dict.items()}


def _verify_against_timm(converted: dict[str, torch.Tensor]) -> None:
    """加载转换结果与 timm 官方权重，比较同一输入的前向输出。"""
    import timm

    model_converted = timm.create_model("fastvit_t12.apple_dist_in1k", num_classes=0, pretrained=False)
    missing, unexpected = model_converted.load_state_dict(converted, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert not missing, f"missing keys: {missing[:5]}"
    model_converted.eval()

    model_official = timm.create_model("fastvit_t12.apple_dist_in1k", num_classes=0, pretrained=True)
    model_official.eval()

    torch.manual_seed(0)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out_converted = model_converted(x)
        out_official = model_official(x)
    max_diff = (out_converted - out_official).abs().max().item()
    _logger.info("verify: 输出形状 %s，与官方权重前向最大绝对误差 %.3e", tuple(out_converted.shape), max_diff)
    assert max_diff < 1e-4, f"转换结果与官方权重前向不一致（max diff {max_diff:.3e}）"
    _logger.info("verify: 转换规则校验通过。")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=str, help="Flax 格式 safetensors 输入路径。")
    parser.add_argument("dst", type=str, help="timm 格式 safetensors 输出路径。")
    parser.add_argument(
        "--verify-against-timm",
        action="store_true",
        help="转换后与 timm 官方 fastvit_t12.apple_dist_in1k 权重做前向数值校验"
        "（仅适用于官方权重的转换，需联网下载官方权重）。",
    )
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    state_dict = load_file(str(src), device="cpu")
    converted = convert_state_dict(state_dict)
    _logger.info("converted %d tensors: %s -> %s", len(converted), src, dst)

    if args.verify_against_timm:
        _verify_against_timm(converted)

    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(dst))
    _logger.info("wrote %s", dst)


if __name__ == "__main__":
    main()
