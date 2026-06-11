"""触觉 RGB Masked Autoencoder (MAE) 模型。

与 ``TactileVAE`` 的关系：

- ``TactileVAE`` 把 timm backbone 当**黑盒**用，只调 ``model(x) -> (B, C)`` 的池化 embedding，
  再走 fc_mu/fc_logvar + 反卷积解码，对任意 backbone 通用。
- ``TactileMAE`` 需要 patch token 级别的访问，且要让 encoder **只看到可见 patch**，因此无法复用
  ``model(x)`` / ``forward_features`` 这种「全部 token 都过 blocks」的黑盒接口。它改为**按 MAE 顺序
  重新编排 timm ViT 自己的标准子模块**：``patch_embed -> _pos_embed -> 随机丢 token -> blocks -> norm``，
  并自建一个轻量 transformer decoder 重建被 mask 的 patch。decoder 仅预训练用，结束后丢弃，只留 encoder。

实现要点：
- 复用 timm ViT 的 ``_pos_embed``，它已处理好前缀 token（cls/register）、``no_embed_class`` 与位置编码
  插值，比手动切分 ``pos_embed`` 更稳。标准 ViT ``num_prefix_tokens=1``，DINOv3 为 5（1 cls + 4 reg）。
- 损失为被 mask patch 上的（可选 per-patch 归一化）MSE，对齐 facebookresearch/MAE。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from src.models.base_model import BaseModel

try:  # timm 旋转位置编码（RoPE）下，按 kept-token 索引 gather rope 的工具（Eva/DINOv3 用）
    from timm.layers.pos_embed_sincos import apply_keep_indices_nlc
    _HAS_KEEP_IDX = True
except ImportError:  # pragma: no cover
    _HAS_KEEP_IDX = False


def _as_2tuple(value) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


class TactileMAE(BaseModel):
    """基于 timm ViT 的触觉图像 MAE。

    Args:
        encoder_backbone: 一个 timm ``VisionTransformer``（``num_classes=0`` 创建即可）。必须暴露
            ``patch_embed`` / ``pos_embed`` / ``blocks`` / ``norm`` / ``_pos_embed`` / ``num_prefix_tokens``。
        image_size: 输入边长（由 timm ``data_config`` 解析得到）。
        patch_size: patch 边长（从 ``patch_embed.patch_size`` 读）。
        in_chans: 重建通道数（触觉 RGB = 3）。
        mask_ratio: 被 mask 的 patch 比例（MAE 论文默认 0.75）。
        decoder_embed_dim / decoder_depth / decoder_num_heads: 轻量 decoder 的宽度 / 深度 / 头数。
        norm_pix_loss: 是否对每个 patch 做 (x-mean)/std 归一化后再算 MSE。
    """

    def __init__(
        self,
        encoder_backbone: nn.Module,
        *,
        image_size: int,
        patch_size: int,
        in_chans: int = 3,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 16,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__(image_size=image_size, input_channels=in_chans, latent_dim=0)

        # ---- 校验 backbone 是 ViT，并缓存其标准组件 ----
        # 注意：不要求 pos_embed 张量存在——DINOv3 用纯 RoPE，pos_embed 为 None。
        required = ("patch_embed", "blocks", "norm", "_pos_embed", "num_prefix_tokens")
        missing = [a for a in required if not hasattr(encoder_backbone, a)]
        if missing:
            raise ValueError(
                f"TactileMAE 需要 ViT 类 backbone（缺少: {missing}）。"
                "请使用 vit_*（如 vit_*_patch16_dinov3）等 timm ViT。"
            )

        # DINOv3 在 timm 里是 Eva 类，使用旋转位置编码（RoPE）：`_pos_embed` 返回 (x, rope)，
        # 且 blocks 需要 `blk(x, rope=...)`。标准 ViT 无 rope，`_pos_embed` 返回张量、`blk(x)`。
        self._uses_rope = getattr(encoder_backbone, "rope", None) is not None
        self._rope_mixed = bool(getattr(encoder_backbone, "rope_mixed", False))
        if self._uses_rope and self._rope_mixed:
            raise NotImplementedError(
                "暂不支持 rope_mixed=True 的 ViT（每层不同 rope）的 MAE masking；"
                "DINOv3 lvd1689m 系列为 rope_mixed=False，可正常使用。"
            )
        if self._uses_rope and not _HAS_KEEP_IDX:
            raise ImportError("RoPE backbone 需要 timm.layers.pos_embed_sincos.apply_keep_indices_nlc。")

        self.encoder = encoder_backbone
        self.embed_dim = int(encoder_backbone.embed_dim)
        self.num_prefix_tokens = int(encoder_backbone.num_prefix_tokens)
        self.num_patches = int(encoder_backbone.patch_embed.num_patches)
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        self.mask_ratio = float(mask_ratio)
        self.norm_pix_loss = bool(norm_pix_loss)

        ph, pw = _as_2tuple(encoder_backbone.patch_embed.patch_size)
        if ph != pw:
            raise ValueError(f"TactileMAE 仅支持方形 patch，得到 {(ph, pw)}。")

        # ---- 自建 decoder（与 backbone 无关）----
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # decoder 位置编码覆盖「前缀 + 全部 patch」；可学习，trunc_normal 初始化。
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_prefix_tokens + self.num_patches, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size**2 * self.in_chans, bias=True)

        self._init_decoder_weights()

    def _init_decoder_weights(self) -> None:
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for module in (self.decoder_embed, self.decoder_pred):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # patchify / unpatchify
    # ------------------------------------------------------------------
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, L, patch_size**2 * C)。"""
        p = self.patch_size
        b, c, h, w = imgs.shape
        if h % p != 0 or w % p != 0:
            raise ValueError(f"图像尺寸 {(h, w)} 必须能被 patch_size={p} 整除。")
        hp, wp = h // p, w // p
        x = imgs.reshape(b, c, hp, p, wp, p)
        x = torch.einsum("bchpwq->bhwpqc", x)
        return x.reshape(b, hp * wp, p * p * c)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, patch_size**2 * C) -> (B, C, H, W)。"""
        p = self.patch_size
        c = self.in_chans
        b, l, _ = x.shape
        hp = wp = int(l**0.5)
        if hp * wp != l:
            raise ValueError(f"patch 数 {l} 不是完全平方数，无法 unpatchify。")
        x = x.reshape(b, hp, wp, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(b, c, hp * p, wp * p)

    # ------------------------------------------------------------------
    # 随机 mask
    # ------------------------------------------------------------------
    @staticmethod
    def random_masking(
        x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """对 patch 维度按 per-sample 噪声排序，保留前 (1-mask_ratio) 个。

        x: (B, N, D)（仅 patch token）。返回 (x_kept, mask, ids_restore, ids_keep)，mask 中 1 表示被 mask。
        ``ids_keep`` 用于在 RoPE backbone 上同步 gather 被保留 patch 的 rope 频率。
        """
        b, n, d = x.shape
        len_keep = max(1, int(round(n * (1.0 - mask_ratio))))
        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))

        mask = torch.ones(b, n, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_kept, mask, ids_restore, ids_keep

    # ------------------------------------------------------------------
    # encoder / decoder 前向
    # ------------------------------------------------------------------
    def forward_encoder(
        self, imgs: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """patch_embed -> _pos_embed -> 丢 token -> blocks -> norm。

        复用 timm 的 ``_pos_embed``：它对完整 patch 序列加位置编码并 prepend 前缀 token（cls/reg），
        已处理 ``no_embed_class`` 与位置插值。我们在其后、blocks 之前按 mask 丢弃 patch token。
        - 标准 ViT：``_pos_embed`` 返回张量，``blk(x)``，无 rope。
        - DINOv3（Eva）：``_pos_embed`` 返回 (x, rope)，``blk(x, rope=rope)``；mask 后按 ids_keep 同步
          gather rope（attention 仅对 patch token 施加 rope，前缀 token 跳过）。
        mask_ratio=0 时不丢任何 token（用于下游特征提取 / 评估）。
        """
        x = self.encoder.patch_embed(imgs)          # 标准 ViT (B,N,D)；Eva (B,H,W,D)
        pos_out = self.encoder._pos_embed(x)        # Eva 返回 (x, rope)，标准 ViT 返回张量
        if isinstance(pos_out, tuple):
            x, rope = pos_out
        else:
            x, rope = pos_out, None

        p = self.num_prefix_tokens
        prefix, patches = x[:, :p, :], x[:, p:, :]

        mask = ids_restore = None
        if mask_ratio > 0:
            patches, mask, ids_restore, ids_keep = self.random_masking(patches, mask_ratio)
            if rope is not None:
                # rope: (N, dim) 共享 -> 按 kept patch gather -> (B, len_keep, dim) -> (B,1,len_keep,dim)
                rope = apply_keep_indices_nlc(patches, rope, ids_keep).unsqueeze(1)

        x = torch.cat([prefix, patches], dim=1)
        x = self.encoder.norm_pre(x)                # 多数 ViT 为 Identity；pre-norm ViT 有实体
        for blk in self.encoder.blocks:
            x = blk(x, rope=rope) if self._uses_rope else blk(x)
        x = self.encoder.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """补 mask token -> 反排列还原顺序 -> decoder blocks -> 预测像素 (B, N, p*p*C)。"""
        x = self.decoder_embed(x)                    # (B, P+len_keep, Dd)
        p = self.num_prefix_tokens
        prefix, tokens = x[:, :p, :], x[:, p:, :]

        n = ids_restore.shape[1]
        d = tokens.shape[-1]
        n_mask = n - tokens.shape[1]
        mask_tokens = self.mask_token.expand(tokens.shape[0], n_mask, -1)
        tokens = torch.cat([tokens, mask_tokens], dim=1)               # (B, N, Dd)
        tokens = torch.gather(tokens, 1, ids_restore.unsqueeze(-1).expand(-1, -1, d))  # 还原原图顺序

        x = torch.cat([prefix, tokens], dim=1) + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)                     # (B, P+N, p*p*C)
        return x[:, p:, :]                           # 丢前缀，仅留 patch 预测

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                      # 每个 patch 的 MSE
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    # ------------------------------------------------------------------
    # BaseModel 接口
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """无 mask 地编码，返回 (B, P+N, D) token 序列，用于下游特征提取。"""
        latent, _, _ = self.forward_encoder(x, mask_ratio=0.0)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover - 仅满足抽象接口
        raise NotImplementedError("TactileMAE.decode 需要 ids_restore，请用 forward_decoder。")

    def forward(self, x: torch.Tensor):
        latent, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask, ids_restore

    def compute_loss(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(batch, self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(batch, pred, mask)
        return {"total": loss, "recon": loss}

    @torch.inference_mode()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """重建整图用于可视化：预测被 mask patch，可见区域用原图 patch 贴回。"""
        latent, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)            # (B, N, p*p*C)，归一化空间

        target = self.patchify(x)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            pred = pred * (var + 1e-6) ** 0.5 + mean               # 反归一化回像素空间

        # 可见 patch 用原图，mask patch 用预测，拼成完整图。
        mask_e = mask.unsqueeze(-1)                                 # (B, N, 1)
        combined = target * (1 - mask_e) + pred * mask_e
        return self.unpatchify(combined)
