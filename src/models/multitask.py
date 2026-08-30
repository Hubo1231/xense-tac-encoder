"""多头监督模型：timm encoder 的 embedding 上挂多个回归/分类头。

编码器沿用 timm 黑盒约定 ``model(x) -> (B, feature_dim)``（num_classes=0），
各监督头由 YAML 的 ``heads:`` 块声明式配置（见 ``parse_head_specs``）：

    heads:
      force_z:  {type: regression, dim: 1, weight: 1.0, loss: mse}
      slip:     {type: classification, num_classes: 2, weight: 0.5}
      material: {type: classification, num_classes: 8, train: false}  # eval-only

``train: false`` 的头训练时不构建，留给 scripts/evaluate_multitask.py 用冻结
编码器做 linear probe。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseModel

HEAD_TYPES: tuple[str, ...] = ("regression", "classification")
REGRESSION_LOSSES: tuple[str, ...] = ("mse", "huber")


@dataclass(frozen=True)
class HeadSpec:
    """单个监督头的声明式配置。

    - ``hidden_dim=None`` 时头为单个 Linear；否则为 MLP（Linear → GELU → Linear）。
    - ``dim`` 仅回归头使用（输出维度）；``num_classes`` 仅分类头使用。
    - ``loss`` 仅回归头使用：``mse``（默认）或 ``huber``（smooth_l1）。
    - ``train=False`` 表示 eval-only 目标，训练时不建头，由评测脚本 probe。
    - ``normalize=False`` 表示回归目标跳过 stats.json 的 z-score（如 force_grid
      直接在物理单位 N 上训练/评测）；默认 True 保持旧行为。
    """

    name: str
    type: str  # "regression" | "classification"
    weight: float = 1.0
    dim: int = 1
    num_classes: int | None = None
    loss: str = "mse"
    hidden_dim: int | None = None
    train: bool = True
    spatial: bool = False
    shape: tuple[int, ...] | None = None
    normalize: bool = True


def parse_head_specs(heads_cfg: dict[str, Any], *, include_eval_only: bool = False) -> list[HeadSpec]:
    """解析 YAML 的 ``heads:`` 块为 ``HeadSpec`` 列表。

    默认只返回 ``train: true`` 的头；``include_eval_only=True`` 时返回全部
    （评测脚本需要 eval-only 目标的列名/类型来加载数据和拟合 probe）。
    未知 type、缺 num_classes 等非法配置直接 ValueError，不做静默默认。
    """
    if not isinstance(heads_cfg, dict) or not heads_cfg:
        raise ValueError("config 的 `heads` 块必须是非空 mapping。")

    specs: list[HeadSpec] = []
    for name, raw in heads_cfg.items():
        if not isinstance(raw, dict):
            raise ValueError(f"heads.{name} 必须是 mapping，got {type(raw).__name__}。")

        head_type = raw.get("type")
        if head_type not in HEAD_TYPES:
            raise ValueError(f"heads.{name}.type 必须是 {HEAD_TYPES} 之一，got {head_type!r}。")

        train = bool(raw.get("train", True))
        weight = float(raw.get("weight", 1.0))
        if weight <= 0.0:
            raise ValueError(f"heads.{name}.weight 必须 > 0，got {weight}。")

        hidden_dim = raw.get("hidden_dim")
        if hidden_dim is not None:
            hidden_dim = int(hidden_dim)
            if hidden_dim < 1:
                raise ValueError(f"heads.{name}.hidden_dim 必须 >= 1，got {hidden_dim}。")

        spatial = bool(raw.get("spatial", False))
        normalize = bool(raw.get("normalize", True))
        shape = raw.get("shape")
        if shape is not None:
            if not isinstance(shape, (list, tuple)) or not shape:
                raise ValueError(f"heads.{name}.shape 必须是非空 list/tuple，got {shape!r}。")
            shape = tuple(int(s) for s in shape)
            if any(s < 1 for s in shape):
                raise ValueError(f"heads.{name}.shape 元素必须 >= 1，got {shape!r}。")
        if spatial and shape is None:
            raise ValueError(f"heads.{name}.spatial=true 时必须显式提供 shape（如 [175, 100]）。")

        dim = int(raw.get("dim", 1))
        loss = str(raw.get("loss", "mse"))
        num_classes = raw.get("num_classes")
        if head_type == "regression":
            if dim < 1:
                raise ValueError(f"heads.{name}.dim 必须 >= 1，got {dim}。")
            if loss not in REGRESSION_LOSSES:
                raise ValueError(f"heads.{name}.loss 必须是 {REGRESSION_LOSSES} 之一，got {loss!r}。")
        else:  # classification
            if num_classes is None:
                raise ValueError(f"heads.{name} 是 classification，必须显式给出 num_classes。")
            num_classes = int(num_classes)
            if num_classes < 2:
                raise ValueError(f"heads.{name}.num_classes 必须 >= 2，got {num_classes}。")

        if not train and not include_eval_only:
            continue
        specs.append(
            HeadSpec(
                name=str(name),
                type=head_type,
                weight=weight,
                dim=dim,
                num_classes=num_classes,
                loss=loss,
                hidden_dim=hidden_dim,
                train=train,
                spatial=spatial,
                shape=shape,
                normalize=normalize,
            )
        )
    return specs


def _build_head(feature_dim: int, spec: HeadSpec) -> nn.Module:
    """按 HeadSpec 构建头：单 Linear，或 MLP（Linear → GELU → Linear）。"""
    out_dim = spec.dim if spec.type == "regression" else int(spec.num_classes)
    if spec.hidden_dim is None:
        return nn.Linear(feature_dim, out_dim)
    return nn.Sequential(
        nn.Linear(feature_dim, spec.hidden_dim),
        nn.GELU(),
        nn.Linear(spec.hidden_dim, out_dim),
    )


class TactileMultiTask(BaseModel):
    """timm encoder + 多监督头（回归 / 分类 / 空间回归）。

    训练 batch 为 dict：``{"image": (B, 3, H, W), "targets": {name: tensor}}``；
    ``compute_loss`` 逐头前向并加权求和，返回 ``{"total", "<name>_loss", ...}``。
    普通头作用于 ``model(x) -> (B, feature_dim)``；``spatial: true`` 的回归头
    作用于 ``model.forward_features(x)`` 得到的空间特征图（ViT 会丢弃前缀
    token 后 reshape），再插值到 ``shape`` 指定的输出分辨率。
    """

    def __init__(
        self,
        encoder_backbone: nn.Module,
        feature_dim: int,
        head_specs: list[HeadSpec],
        image_size: int = 224,
    ) -> None:
        super().__init__(image_size=image_size, input_channels=3, latent_dim=feature_dim)
        if not head_specs:
            raise ValueError("TactileMultiTask 至少需要一个 train: true 的 head。")
        names = [spec.name for spec in head_specs]
        if len(set(names)) != len(names):
            raise ValueError(f"head 名称重复: {names}。")
        for spec in head_specs:
            if not spec.train:
                raise ValueError(
                    f"heads.{spec.name} 是 eval-only（train: false），不应传给 TactileMultiTask；"
                    "请用 parse_head_specs 的默认行为过滤。"
                )

        self.encoder = encoder_backbone
        self.feature_dim = feature_dim
        self.head_specs = {spec.name: spec for spec in head_specs}
        self.num_prefix_tokens = int(getattr(encoder_backbone, "num_prefix_tokens", 0) or 0)
        self.heads = nn.ModuleDict()
        for spec in head_specs:
            if spec.spatial:
                if spec.type != "regression":
                    raise ValueError(f"spatial 头 {spec.name!r} 只支持 regression。")
                self.heads[spec.name] = self._build_spatial_head(feature_dim, spec)
            else:
                self.heads[spec.name] = _build_head(feature_dim, spec)

    @staticmethod
    def _build_spatial_head(feature_dim: int, spec: HeadSpec) -> nn.Module:
        """ViT/卷积 backbone 的空间特征图 -> 目标网格。"""
        if spec.shape is None:
            raise ValueError(f"spatial 头 {spec.name!r} 必须提供 shape。")
        output_size = tuple(spec.shape[:2])
        return SpatialHead(feature_dim, spec.dim, output_size)

    def _extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """把 ``forward_features`` 输出统一为 (B, C, h, w)，供 spatial 头使用。"""
        if not hasattr(self.encoder, "forward_features"):
            raise ValueError("spatial 头需要 backbone 暴露 forward_features()。")
        out = self.encoder.forward_features(x)
        if out.ndim == 4:
            return out
        if out.ndim == 3:
            tokens = out[:, self.num_prefix_tokens:, :]
            b, n, c = tokens.shape
            hw = int(n**0.5)
            if hw * hw != n:
                raise ValueError(f"ViT patch 数 {n} 不是完全平方数，无法 reshape 成特征图。")
            return tokens.transpose(1, 2).reshape(b, c, hw, hw)
        raise ValueError(
            f"无法识别的 forward_features 输出维度: {out.ndim}（期待 3 或 4）。"
        )

    @staticmethod
    def _as_channel_first(target: torch.Tensor, name: str) -> torch.Tensor:
        """把 H5 空间目标对齐到模型预测的 (B, C, H, W)。"""
        if name == "flow":
            if target.ndim == 4 and target.shape[-1] == 2:
                return target.permute(0, 3, 1, 2)
            if target.ndim == 3:
                return target.unsqueeze(1)
        if target.ndim == 3:
            return target.unsqueeze(1)
        return target

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码图像 batch 为 (B, feature_dim) embedding。"""
        feat = self.encoder(x)
        if feat.ndim != 2:
            raise ValueError(
                f"TactileMultiTask encoder 必须返回 (B, C) embedding，got shape {tuple(feat.shape)}。"
            )
        if feat.shape[1] != self.feature_dim:
            raise ValueError(
                f"Encoder embedding dim {feat.shape[1]} 与配置的 feature_dim {self.feature_dim} 不一致。"
            )
        return feat

    def decode(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover - 仅满足抽象接口
        raise NotImplementedError("TactileMultiTask 是判别式多头模型，没有解码器。")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """前向并返回各头预测；spatial 头返回 (B, C, H, W)，普通头返回 (B, dim|classes)。"""
        predictions: dict[str, torch.Tensor] = {}
        spatial_names = [name for name, spec in self.head_specs.items() if spec.spatial]
        linear_names = [name for name, spec in self.head_specs.items() if not spec.spatial]

        if spatial_names:
            feat_map = self._extract_feature_map(x)
            for name in spatial_names:
                predictions[name] = self.heads[name](feat_map)
        if linear_names:
            feat = self.encode(x)
            for name in linear_names:
                predictions[name] = self.heads[name](feat)
        return predictions

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if not isinstance(batch, dict) or "image" not in batch or "targets" not in batch:
            raise ValueError(
                "TactileMultiTask.compute_loss 需要 dict batch：{'image': tensor, 'targets': {name: tensor}}。"
            )
        targets = batch["targets"]
        if not isinstance(targets, dict):
            raise ValueError(f"batch['targets'] 必须是 dict，got {type(targets).__name__}。")

        predictions = self.forward(batch["image"])
        losses: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for name, pred in predictions.items():
            spec = self.head_specs[name]
            if name not in targets:
                raise ValueError(f"batch['targets'] 缺少头 {name!r} 对应的目标列。")
            target = targets[name]

            if spec.spatial:
                target = self._as_channel_first(target, name).to(dtype=pred.dtype)
                if spec.loss == "huber":
                    loss = F.smooth_l1_loss(pred, target)
                else:
                    loss = F.mse_loss(pred, target)
                losses[f"{name}_residual"] = (pred - target).abs().mean()
            elif spec.type == "regression":
                target = target.to(dtype=pred.dtype)
                if target.ndim == 1:
                    target = target.unsqueeze(-1)  # (B,) -> (B, 1)，与 (B, dim=1) 对齐
                if spec.loss == "huber":
                    loss = F.smooth_l1_loss(pred, target)
                else:
                    loss = F.mse_loss(pred, target)
                losses[f"{name}_residual"] = (pred - target).abs().mean()
            else:  # classification
                if target.dtype != torch.int64:
                    raise ValueError(
                        f"分类头 {name!r} 的目标必须是 int64，got {target.dtype}。"
                    )
                loss = F.cross_entropy(pred, target)

            losses[f"{name}_loss"] = loss
            weighted = spec.weight * loss
            total = weighted if total is None else total + weighted

        assert total is not None  # head_specs 非空，__init__ 已校验
        losses["total"] = total
        return losses


@dataclass(frozen=True)
class PhysicalLossWeights:
    """todo.md 中定义的物理多头监督损失权重。"""

    dense_depth: float = 1.0
    dense_flow: float = 1.0
    bottleneck_depth: float = 0.25
    bottleneck_flow: float = 0.25

    def __post_init__(self) -> None:
        for field in (
            "dense_depth",
            "dense_flow",
            "bottleneck_depth",
            "bottleneck_flow",
        ):
            value = float(getattr(self, field))
            if value <= 0.0:
                raise ValueError(f"loss_weights.{field} 必须 > 0，got {value}。")
            object.__setattr__(self, field, value)


def _conv1x1(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)


def _conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


class TinyFPN(nn.Module):
    """极简 FPN，把 FastViT 的四层空间特征融合成 dense 特征与 F4。

    输入约定与 FastViT-T12 在 256×256 输入下的 forward 结构一致：
      c2: (B, 64, 64, 64)
      c3: (B, 128, 32, 32)
      c4: (B, 256, 16, 16)
      c5: (B, 1024, 8, 8)   # stages[3] 再过 final_conv
    """

    def __init__(self, fpn_dim: int = 256) -> None:
        super().__init__()
        self.lateral2 = _conv1x1(64, fpn_dim)
        self.lateral3 = _conv1x1(128, fpn_dim)
        self.lateral4 = _conv1x1(256, fpn_dim)
        self.lateral5 = _conv1x1(1024, fpn_dim)
        self.smooth2 = _conv3x3(fpn_dim, fpn_dim)
        self.smooth4 = _conv3x3(fpn_dim, fpn_dim)

    def forward(
        self,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        c5: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f5 = self.lateral5(c5)
        f4 = self.lateral4(c4) + F.interpolate(
            f5, size=c4.shape[-2:], mode="bilinear", align_corners=False
        )
        f3 = self.lateral3(c3) + F.interpolate(
            f4, size=c3.shape[-2:], mode="bilinear", align_corners=False
        )
        f2 = self.lateral2(c2) + F.interpolate(
            f3, size=c2.shape[-2:], mode="bilinear", align_corners=False
        )

        dense = self.smooth2(f2)
        dense = F.interpolate(dense, scale_factor=2.0, mode="bilinear", align_corners=False)
        return dense, self.smooth4(f4)


class PhysicalPooler(nn.Module):
    """把 F4（16×16 空间物理特征）压成 z（feature_dim 维 embedding）。"""

    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.reduce = _conv1x1(in_channels, latent_dim)

    def forward(self, f4: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.reduce(f4))
        x = F.adaptive_avg_pool2d(x, 1)
        return x.flatten(1)


class SpatialHead(nn.Module):
    """1×1 卷积后直接插值到触觉物理网格（depth/flow 的 dense 监督）。"""

    def __init__(self, in_channels: int, out_channels: int, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.conv = _conv1x1(in_channels, out_channels)
        self.output_size = tuple(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if tuple(x.shape[-2:]) != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        return x


class LinearSpatialHead(nn.Module):
    """从 z 映射到低分辨率空间标签（bottleneck 监督）。"""

    def __init__(self, latent_dim: int, out_channels: int, spatial_size: tuple[int, int]) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.spatial_size = tuple(spatial_size)
        self.linear = nn.Linear(latent_dim, out_channels * spatial_size[0] * spatial_size[1])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.linear(z)
        return out.view(z.shape[0], self.out_channels, *self.spatial_size)


class TactilePhysicalMultiTask(BaseModel):
    """todo.md 定义的物理感知预训练模型。

    RGB 256×256 -> FastViT-T12 空间特征 -> Tiny FPN：
      - dense 分支预测 depth(175×100) / flow(175×100×2)；
      - F4 经 PhysicalPooler 得到 z(1024)，再线性预测 16×16 的
        depth/flow bottleneck。
    """

    def __init__(
        self,
        encoder_backbone: nn.Module,
        feature_dim: int,
        image_size: int = 256,
        *,
        dense_output_size: tuple[int, int] = (175, 100),
        bottleneck_size: tuple[int, int] = (16, 16),
        fpn_dim: int = 256,
        loss_weights: PhysicalLossWeights | dict[str, float] | None = None,
    ) -> None:
        super().__init__(image_size=image_size, input_channels=3, latent_dim=feature_dim)
        if not hasattr(encoder_backbone, "stem") or not hasattr(encoder_backbone, "stages"):
            raise ValueError(
                "TactilePhysicalMultiTask 需要 FastViT 类 backbone（具有 stem / stages / final_conv）。"
            )
        if not hasattr(encoder_backbone, "final_conv"):
            raise ValueError("TactilePhysicalMultiTask 需要 backbone 具有 final_conv。")

        self.encoder = encoder_backbone
        self.feature_dim = feature_dim
        self.dense_output_size = tuple(dense_output_size)
        self.bottleneck_size = tuple(bottleneck_size)
        self.fpn = TinyFPN(fpn_dim=fpn_dim)
        self.pooler = PhysicalPooler(fpn_dim, feature_dim)
        self.dense_depth_head = SpatialHead(fpn_dim, 1, self.dense_output_size)
        self.dense_flow_head = SpatialHead(fpn_dim, 2, self.dense_output_size)
        self.bottleneck_depth_head = LinearSpatialHead(feature_dim, 1, self.bottleneck_size)
        self.bottleneck_flow_head = LinearSpatialHead(feature_dim, 2, self.bottleneck_size)
        self.heads = nn.ModuleDict(
            {
                "depth": self.dense_depth_head,
                "flow": self.dense_flow_head,
                "depth_z": self.bottleneck_depth_head,
                "flow_z": self.bottleneck_flow_head,
            }
        )
        if isinstance(loss_weights, dict):
            loss_weights = PhysicalLossWeights(**loss_weights)
        self.loss_weights = loss_weights or PhysicalLossWeights()

    def _extract_pyramid(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"TactilePhysicalMultiTask 需要 (B,3,H,W) 图像，got {tuple(x.shape)}。")
        stem = self.encoder.stem(x)
        feat = stem
        stage_features: list[torch.Tensor] = []
        for stage in self.encoder.stages:
            feat = stage(feat)
            stage_features.append(feat)
        if len(stage_features) < 4:
            raise ValueError(
                f"FastViT stages 数量不足 4，got {len(stage_features)}。"
            )
        c2, c3, c4, c5_stage = stage_features[-4:]
        c5 = self.encoder.final_conv(c5_stage)
        return self.fpn(c2, c3, c4, c5)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, f4 = self._extract_pyramid(x)
        return self.pooler(f4)

    def decode(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover - 仅满足抽象接口
        raise NotImplementedError("TactilePhysicalMultiTask 是判别式模型，没有解码器。")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        dense_feat, f4 = self._extract_pyramid(x)
        z = self.pooler(f4)
        return {
            "depth": self.dense_depth_head(dense_feat),
            "flow": self.dense_flow_head(dense_feat),
            "depth_z": self.bottleneck_depth_head(z),
            "flow_z": self.bottleneck_flow_head(z),
        }

    @staticmethod
    def _as_channel_first(target: torch.Tensor, name: str) -> torch.Tensor:
        """把 H5 目标张量对齐到模型预测的 (B, C, H, W)。"""
        if name == "flow":
            if target.ndim == 4 and target.shape[-1] == 2:
                return target.permute(0, 3, 1, 2)
            if target.ndim == 3:
                return target.unsqueeze(1)
        if target.ndim == 3:
            return target.unsqueeze(1)
        return target

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if not isinstance(batch, dict) or "image" not in batch or "targets" not in batch:
            raise ValueError(
                "TactilePhysicalMultiTask.compute_loss 需要 dict batch："
                "{'image': tensor, 'targets': {depth, flow}}。"
            )
        targets = batch["targets"]
        missing = [name for name in ("depth", "flow") if name not in targets]
        if missing:
            raise ValueError(f"batch['targets'] 缺少物理目标列 {missing}。")

        predictions = self.forward(batch["image"])
        dense_targets = {
            "depth": self._as_channel_first(targets["depth"], "depth").to(
                dtype=predictions["depth"].dtype
            ),
            "flow": self._as_channel_first(targets["flow"], "flow").to(
                dtype=predictions["flow"].dtype
            ),
        }
        bottleneck_targets = {
            "depth": F.interpolate(
                dense_targets["depth"], size=self.bottleneck_size,
                mode="bilinear", align_corners=False,
            ),
            "flow": F.interpolate(
                dense_targets["flow"], size=self.bottleneck_size,
                mode="bilinear", align_corners=False,
            ),
        }

        losses: dict[str, torch.Tensor] = {
            "depth_loss": F.mse_loss(predictions["depth"], dense_targets["depth"]),
            "flow_loss": F.mse_loss(predictions["flow"], dense_targets["flow"]),
            "depth_z_loss": F.mse_loss(predictions["depth_z"], bottleneck_targets["depth"]),
            "flow_z_loss": F.mse_loss(predictions["flow_z"], bottleneck_targets["flow"]),
        }
        losses.update(
            {
                "depth_residual": (predictions["depth"] - dense_targets["depth"]).abs().mean(),
                "flow_residual": (predictions["flow"] - dense_targets["flow"]).abs().mean(),
                "depth_z_residual": (predictions["depth_z"] - bottleneck_targets["depth"]).abs().mean(),
                "flow_z_residual": (predictions["flow_z"] - bottleneck_targets["flow"]).abs().mean(),
            }
        )
        losses["total"] = (
            self.loss_weights.dense_depth * losses["depth_loss"]
            + self.loss_weights.dense_flow * losses["flow_loss"]
            + self.loss_weights.bottleneck_depth * losses["depth_z_loss"]
            + self.loss_weights.bottleneck_flow * losses["flow_z_loss"]
        )
        return losses


@dataclass(frozen=True)
class ViTPhysicalLossWeights:
    """todo.md 中 DINOv3 ViT-B/16 物理多监督损失权重。"""

    patch_depth: float = 1.0
    patch_flow: float = 1.0
    bottleneck_depth: float = 0.25
    bottleneck_flow: float = 0.25

    def __post_init__(self) -> None:
        for field in (
            "patch_depth",
            "patch_flow",
            "bottleneck_depth",
            "bottleneck_flow",
        ):
            value = float(getattr(self, field))
            if value <= 0.0:
                raise ValueError(f"loss_weights.{field} 必须 > 0，got {value}。")
            object.__setattr__(self, field, value)


class AttentionPool(nn.Module):
    """把 patch token 序列池化成单个 embedding（todo.md 的 Attention Pool）。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = float(dim) ** -0.5
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, C)
        attn = torch.matmul(self.query, tokens.transpose(1, 2)).mul_(self.scale)
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, tokens).squeeze(1)  # (B, C)


class ViTPhysicalDecoder(nn.Module):
    """从 ViT patch token 生成物理 patch 特征（todo.md 的 Multi-layer physical）。"""

    def __init__(self, embed_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.proj(tokens)  # (B, N, hidden_dim)
        b, n, c = x.shape
        hw = int(n**0.5)
        if hw * hw != n:
            raise ValueError(f"ViT patch 数 {n} 不是完全平方数，无法 reshape 成特征图。")
        return x.transpose(1, 2).reshape(b, c, hw, hw)


class TactileViTPhysicalMultiTask(BaseModel):
    """todo.md 定义的 DINOv3 ViT 物理多监督模型。

    与 FastViT ``TactilePhysicalMultiTask`` 对应，但空间分支来自 ViT patch token：
      - patch token -> ViTPhysicalDecoder -> depth/flow patch 预测 (175×100)；
      - patch token -> AttentionPool -> LayerNorm -> z768 -> Linear depth/flow (16×16)。
    """

    def __init__(
        self,
        encoder_backbone: nn.Module,
        feature_dim: int,
        image_size: int = 256,
        *,
        decoder_hidden_dim: int = 128,
        patch_output_size: tuple[int, int] = (175, 100),
        bottleneck_size: tuple[int, int] = (16, 16),
        loss_weights: ViTPhysicalLossWeights | dict[str, float] | None = None,
    ) -> None:
        super().__init__(image_size=image_size, input_channels=3, latent_dim=feature_dim)
        if not hasattr(encoder_backbone, "forward_features"):
            raise ValueError(
                "TactileViTPhysicalMultiTask 需要 ViT backbone（暴露 forward_features）。"
            )

        patch_embed = getattr(encoder_backbone, "patch_embed", None)
        if patch_embed is None or not hasattr(patch_embed, "patch_size"):
            raise ValueError("TactileViTPhysicalMultiTask 需要 ViT backbone 暴露 patch_embed.patch_size。")
        patch_size = int(tuple(patch_embed.patch_size)[0])
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} 必须能被 patch_size={patch_size} 整除。")

        self.encoder = encoder_backbone
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.patch_hw = image_size // patch_size
        self.patch_output_size = tuple(patch_output_size)
        self.bottleneck_size = tuple(bottleneck_size)
        self.num_prefix_tokens = int(getattr(encoder_backbone, "num_prefix_tokens", 0) or 0)

        self.physical_decoder = ViTPhysicalDecoder(feature_dim, decoder_hidden_dim)
        self.attention_pool = AttentionPool(feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)

        # 与 todo.md 的 Depth/Flow decoder 一致：3×3 128→64、GELU、1×1 64→1/2。
        self.depth_patch_head = nn.Sequential(
            nn.Conv2d(decoder_hidden_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self.flow_patch_head = nn.Sequential(
            nn.Conv2d(decoder_hidden_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2, kernel_size=1),
        )
        self.depth_z_head = LinearSpatialHead(feature_dim, 1, self.bottleneck_size)
        self.flow_z_head = LinearSpatialHead(feature_dim, 2, self.bottleneck_size)
        self.heads = nn.ModuleDict(
            {
                "depth": self.depth_patch_head,
                "flow": self.flow_patch_head,
                "depth_z": self.depth_z_head,
                "flow_z": self.flow_z_head,
            }
        )

        if isinstance(loss_weights, dict):
            loss_weights = ViTPhysicalLossWeights(**loss_weights)
        self.loss_weights = loss_weights or ViTPhysicalLossWeights()

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder.forward_features(x)
        if out.ndim != 3:
            raise ValueError(
                f"ViT forward_features 应返回 (B, N+prefix, C)，got shape {tuple(out.shape)}。"
            )
        tokens = out[:, self.num_prefix_tokens:, :]
        if tokens.shape[1] != self.patch_hw * self.patch_hw:
            raise ValueError(
                f"patch token 数 {tokens.shape[1]} 与预期 {self.patch_hw * self.patch_hw} 不一致。"
            )
        return tokens

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._extract_patch_tokens(x)
        return self.attn_norm(self.attention_pool(tokens))

    def decode(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover - 仅满足抽象接口
        raise NotImplementedError("TactileViTPhysicalMultiTask 是判别式模型，没有解码器。")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self._extract_patch_tokens(x)
        phys = self.physical_decoder(tokens)
        z = self.attn_norm(self.attention_pool(tokens))
        return {
            "depth": F.interpolate(
                self.depth_patch_head(phys),
                size=self.patch_output_size,
                mode="bilinear",
                align_corners=False,
            ),
            "flow": F.interpolate(
                self.flow_patch_head(phys),
                size=self.patch_output_size,
                mode="bilinear",
                align_corners=False,
            ),
            "depth_z": self.depth_z_head(z),
            "flow_z": self.flow_z_head(z),
        }

    @staticmethod
    def _as_channel_first(target: torch.Tensor, name: str) -> torch.Tensor:
        if name == "flow":
            if target.ndim == 4 and target.shape[-1] == 2:
                return target.permute(0, 3, 1, 2)
            if target.ndim == 3:
                return target.unsqueeze(1)
        if target.ndim == 3:
            return target.unsqueeze(1)
        return target

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if not isinstance(batch, dict) or "image" not in batch or "targets" not in batch:
            raise ValueError(
                "TactileViTPhysicalMultiTask.compute_loss 需要 dict batch："
                "{'image': tensor, 'targets': {depth, flow}}。"
            )
        targets = batch["targets"]
        missing = [name for name in ("depth", "flow") if name not in targets]
        if missing:
            raise ValueError(f"batch['targets'] 缺少物理目标列 {missing}。")

        predictions = self.forward(batch["image"])
        dense_targets = {
            "depth": self._as_channel_first(targets["depth"], "depth").to(
                dtype=predictions["depth"].dtype
            ),
            "flow": self._as_channel_first(targets["flow"], "flow").to(
                dtype=predictions["flow"].dtype
            ),
        }
        bottleneck_targets = {
            "depth": F.interpolate(
                dense_targets["depth"], size=self.bottleneck_size,
                mode="bilinear", align_corners=False,
            ),
            "flow": F.interpolate(
                dense_targets["flow"], size=self.bottleneck_size,
                mode="bilinear", align_corners=False,
            ),
        }

        losses: dict[str, torch.Tensor] = {
            "depth_loss": F.mse_loss(predictions["depth"], dense_targets["depth"]),
            "flow_loss": F.mse_loss(predictions["flow"], dense_targets["flow"]),
            "depth_z_loss": F.mse_loss(predictions["depth_z"], bottleneck_targets["depth"]),
            "flow_z_loss": F.mse_loss(predictions["flow_z"], bottleneck_targets["flow"]),
        }
        losses.update(
            {
                "depth_residual": (predictions["depth"] - dense_targets["depth"]).abs().mean(),
                "flow_residual": (predictions["flow"] - dense_targets["flow"]).abs().mean(),
                "depth_z_residual": (predictions["depth_z"] - bottleneck_targets["depth"]).abs().mean(),
                "flow_z_residual": (predictions["flow_z"] - bottleneck_targets["flow"]).abs().mean(),
            }
        )
        losses["total"] = (
            self.loss_weights.patch_depth * losses["depth_loss"]
            + self.loss_weights.patch_flow * losses["flow_loss"]
            + self.loss_weights.bottleneck_depth * losses["depth_z_loss"]
            + self.loss_weights.bottleneck_flow * losses["flow_z_loss"]
        )
        return losses
