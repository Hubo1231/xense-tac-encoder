"""通用度量：推理延迟、参数量、潜空间统计。"""
import time
from typing import Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def measure_latency(
    encoder: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    warmup: int = 10,
    iters: int = 100,
) -> float:
    """返回单次前向的平均毫秒数（仅 encoder，对应 System-0 真实路径）。"""
    encoder.eval()
    dummy = torch.randn(*input_shape, device=device)
    for _ in range(warmup):
        _ = encoder(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = encoder(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000.0


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def latent_variance(mus: torch.Tensor) -> float:
    """所有样本编码 mean 在每一维的方差，再对维度取平均。

    过低 → 特征崩塌（外壳主导，触点信息被忽略）；过高也未必好，需要结合重建误差判断。
    """
    return torch.var(mus, dim=0).mean().item()
