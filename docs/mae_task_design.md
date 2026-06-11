# 触觉 Encoder 预训练：新增 MAE 任务的实现方案

## 0. 如何执行（TL;DR）

任务由 config 顶层的 `task: vae | mae` 决定（缺省 `vae`，旧 config 不受影响）。

```bash
# —— MAE 预训练（ViT，如 dinov3）——
./scripts/run_all_timm.sh vit_base_patch16_dinov3_lvd1689m_mae
# 等价地直接调脚本：
python scripts/train_with_timm.py --config configs/vit_base_patch16_dinov3_lvd1689m_mae.yaml

# —— VAE 预训练（原有任务，任意 backbone）——
./scripts/run_all_timm.sh vit_base_patch16_dinov3_lvd1689m
python scripts/train_with_timm.py --config configs/vit_base_patch16_dinov3_lvd1689m.yaml

# —— 跑全部 configs / 指定多个 ——
./scripts/run_all_timm.sh
./scripts/run_all_timm.sh resnet50_a1_in1k vit_base_patch16_dinov3_lvd1689m_mae
```

要点：
- `task: mae` **只支持 ViT backbone**（需要 patch token），对卷积 backbone 会直接报错；`task: vae` 通用。
- MAE 超参写在 config 的 `mae:` 子块（`mask_ratio` / `decoder_embed_dim` / `decoder_depth` /
  `decoder_num_heads` / `norm_pix_loss`），缺省值见 `scripts/train_with_timm.py` 的 `MAE_DEFAULTS`。
- 输出（checkpoint / summary.csv / args.yaml / wandb 重建图）落在 `outputs/<实验名>/`，实验名含 `mae<img_size>`。
- 日志/可视化复用现有流程：MAE 打印 `loss`+`recon`，wandb 重建图为「可见 patch 贴原图 + mask patch 用预测」的整图。

---

## 1. 背景与现状

当前预训练入口为 `scripts/train_with_timm.py`，批量驱动脚本为 `scripts/run_all_timm.sh`，
逐个读取 `configs/*.yaml` 跑训练。现有训练任务**只有 VAE**：

- `_build_timm_backbone()` 用 `timm.create_model(name, num_classes=0, in_chans=3)` 造 backbone，
  约定 `model(x) -> (B, feature_dim)` 的**扁平图像 embedding**（对任意 backbone 通用：ResNet / ConvNeXt /
  FastViT / EfficientViT / MobileNetV4 / ViT 都能返回 `(B, C)`）。
- `TactileVAE`（`src/models/vae.py`）把该 embedding 经 `fc_mu` / `fc_logvar` 投影到 latent，
  重参数化后用反卷积 `TactileDecoder` 重建整图，损失为 `MSE + grad + KLD + (可选) SSIM/mix`。
- 模型统一对外暴露两个接口：`compute_loss(batch) -> {"total": ..., ...}` 与 `reconstruct(x) -> 图像`，
  训练循环 `train_one_epoch` / `validate` 依赖这两个接口。

参考实现 `/home/li/hubo/AnyTouch2/model/tactile_mae.py` 是基于 HF CLIP ViT 的 **video MAE**：
在 patch token 级别做随机 mask，只把可见 token 送进 transformer，再由 decoder 重建被 mask 的 patch。
本方案借鉴其**「patch 级 mask + 仅编码可见 token + 轻量 decoder 重建 masked patch + 像素归一化 MSE」**思想，
但落地到本仓库的 **timm ViT** 接口上（而非 HF CLIP / 视频），并保持与现有训练循环的兼容。

> 关键约束：**MAE 只对 ViT 类（patch token）backbone 有意义**。卷积 backbone（ResNet / ConvNeXt /
> FastViT / EfficientViT / MobileNetV4）没有 patch token 序列与 `blocks`，无法做标准 MAE。本方案让
> `task: mae` 仅对 ViT 生效，对非 ViT backbone 给出清晰报错。用户当前的 ViT 配置为
> `vit_small/base/large_patch16_dinov3_lvd1689m`，MAE 主要服务于这几个。

## 2. 目标

1. 在 YAML config 中新增开关 `task: vae | mae`，决定本次预训练用哪种自监督任务。
2. 新增 `TactileMAE` 模型，复用 timm ViT 的 `patch_embed` / `pos_embed` / 前缀 token / `blocks` / `norm`，
   外加一个轻量 transformer decoder，实现标准 MAE 预训练。
3. 复用现有训练循环、checkpoint、wandb 可视化等基础设施，改动最小、向后兼容（不写 `task` 时默认 `vae`）。

## 3. Config schema 变更

### 3.1 新增任务开关

所有 config 顶层新增一个字段（缺省回退 `vae`，保证旧 config 不破）：

```yaml
# 训练任务：vae（默认） | mae
task: vae
```

### 3.2 MAE 专属超参（仅 `task: mae` 需要）

新增一个 `mae:` 子块，集中放 MAE 超参，避免污染顶层 VAE 字段：

```yaml
mae:
  mask_ratio: 0.75          # 被 mask 的 patch 比例（MAE 论文默认 0.75）
  decoder_embed_dim: 512    # decoder 宽度（通常远小于 encoder）
  decoder_depth: 4          # decoder transformer 层数（轻量）
  decoder_num_heads: 16
  norm_pix_loss: true       # 对每个 patch 做 (x-mean)/std 归一化后再算 MSE（MAE 论文推荐）
```

### 3.3 配置校验（`train_with_timm.py` 的 `_validate_config`）

把校验改成**任务感知**：

- `task in {"vae", "mae"}`，否则报错。
- `task == "vae"`：保持现有 `REQUIRED_KEYS`（loss 权重等仍然必填）。
- `task == "mae"`：放宽 VAE 专属键（`w_mse/w_grad/w_kld/w_ssim/w_mix/mix_alpha/use_ms_ssim/latent_dim/
  decoder_hidden_channels` 等）为可选；`mae` 子块各字段给默认值，缺失则取默认而非报错。

实现上把 `REQUIRED_KEYS` 拆成 `COMMON_REQUIRED_KEYS`（dataset / 优化器 / 调度 / loader / logging）
+ `VAE_REQUIRED_KEYS`，按 `task` 取并集。

## 4. 模型设计：`src/models/mae.py` 中的 `TactileMAE`

继承 `BaseModel`（`src/models/base_model.py`），对外暴露与 VAE 一致的
`compute_loss(batch) -> dict` 与 `reconstruct(x) -> 图像`，从而无缝接入训练循环。

### 4.0 先看 TactileVAE 怎么用 timm，再看 MAE 为什么必须更深一层

`TactileVAE.encode` 把 backbone 当**黑盒**用，只调一个接口：

```python
feat = self.encoder(x)        # timm: model(x) -> (B, feature_dim) 池化后的图像 embedding
```

这就是 timm 的「池化前向」（`num_classes=0` 时 = `forward_features` 后接全局池化 `forward_head`）。
它对任意 backbone 都返回扁平 `(B, C)`，所以 VAE 能通用。**已实测**（dinov3-base, 256²输入）：

| timm 接口 | 输出 | 含义 | 能否做 MAE |
|---|---|---|---|
| `model(x)` | `(B, 768)` | 池化后的单向量（**VAE 用这个**） | ❌ 丢了 per-patch 信息 |
| `model.forward_features(x)` | `(B, 261, 768)` | 全 token 序列（5 前缀 + 256 patch） | ❌ 全部 patch 都过了 encoder |

**结论：MAE 无法复用 VAE 那条 `model(x)` 路径，也无法只靠 `forward_features`。** 原因是 MAE 的核心是
「encoder 只能看到可见的少数 patch」，mask 必须发生在 `patch_embed` 之后、`blocks` 之前；而 `model(x)` /
`forward_features` 都已经把**全部** token 跑完了 blocks（且 `forward_features` 的 `attn_mask` 参数是注意力
掩码，不能丢 token、省不了算力）。

因此 `TactileMAE` 的做法是：**不重写 ViT，而是按 MAE 顺序重新编排 timm ViT 自己的标准组件**——
`patch_embed → pos_embed → (插入随机 mask) → cls/reg 前缀 token → blocks → norm`。
这些组件正是 timm 内部 `forward_features` 所组合的同一批模块（见下表），我们只是在中间插入「丢 token」这一步，
从而让 encoder **真的只处理可见 patch**（标准 MAE 的算力优势）。

| timm ViT 组件 | 在 MAE encoder 中的角色 |
|---|---|
| `patch_embed(x)` | 图像 → `(B, N, D)` patch token |
| `pos_embed` | 拆成「前缀位置」「patch 位置」分别加 |
| `cls_token` / `reg_token` | mask 之后再 prepend 的前缀 token |
| `blocks` / `norm` | 只对「前缀 + 可见 patch」做 transformer 编码 |

> 一句话：VAE 用 timm 的**黑盒池化接口**；MAE 复用 timm ViT 的**同一批白盒子模块**，只在 `patch_embed`
> 与 `blocks` 之间插入随机丢弃。decoder 则完全自建（见 §4.2 / §4.6），预训练后丢弃，只保留 encoder。

### 4.1 构造参数

```python
class TactileMAE(BaseModel):
    def __init__(
        self,
        encoder_backbone: nn.Module,   # timm VisionTransformer（num_classes=0）
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
```

从 `encoder_backbone` 读取并缓存以下属性（已在 timm 1.0.x ViT 上验证存在）：

- `embed_dim`、`patch_embed`（含 `patch_size` / `num_patches`）、`pos_embed`、`blocks`、`norm`；
- `num_prefix_tokens`（标准 ViT=1 只有 cls；**DINOv3=5**：1 cls + 4 register token）、
  `cls_token`、`reg_token`（DINOv3 有，标准 ViT 为 None）、`no_embed_class`。

> 已实测：`vit_base_patch16_224` → `num_prefix_tokens=1, pos_embed=(1,197,768)`；
> `vit_base_patch16_dinov3.lvd1689m` → `num_prefix_tokens=5, cls+reg token`。
> 因此 mask / pos_embed 逻辑必须**按 `num_prefix_tokens` 泛化**，不能写死 1 个 cls。

### 4.2 Decoder 子模块（完全自建，不依赖 backbone）

decoder 是 MAE 与 VAE 最不一样的地方，单独讲清楚。它的定位与 VAE 的反卷积 `TactileDecoder` 不同：

- **VAE decoder**：从一个 `(B, latent_dim)` 向量反卷积出**整张图**；
- **MAE decoder**：是一个**浅而窄的 transformer**，输入是「可见 patch 的 encoder 特征 + 占位 mask token」的
  完整 token 序列，逐 token 回归出**每个 patch 的像素**，且只对被 mask 的 patch 计损失。

设计原则（沿用 facebookresearch/MAE）：decoder 故意做得**比 encoder 小很多**（如 encoder 12 层 768 宽，
decoder 只 4 层 512 宽），因为它只在预训练时辅助重建，**预训练结束后整个 decoder 丢弃，只保留 encoder**
作为触觉特征提取器。decoder 越轻，越能逼 encoder 把语义都压进特征里。

子模块（构造于 `TactileMAE.__init__`）：

| 模块 | 形状 / 定义 | 作用 |
|---|---|---|
| `decoder_embed` | `nn.Linear(embed_dim, decoder_embed_dim)` | encoder 输出（768）投影到 decoder 宽度（512） |
| `mask_token` | `nn.Parameter(zeros(1,1,decoder_embed_dim))` | 被 mask 位置的可学习占位向量（共享） |
| `decoder_pos_embed` | `Parameter(1, num_prefix+num_patches, decoder_embed_dim)` | decoder 侧的位置编码（sincos 初始化，patch 部分；前缀单独处理） |
| `decoder_blocks` | `nn.ModuleList([timm...Block(...)] * decoder_depth)` | **直接复用 timm 的 `Block`**，与 encoder 同源实现，不重复造轮子 |
| `decoder_norm` | `nn.LayerNorm(decoder_embed_dim)` | 输出前归一化 |
| `decoder_pred` | `nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)` | 每个 token → 一个 patch 的像素（如 16×16×3=768 维） |

> 复用 `timm.models.vision_transformer.Block` 而不是手写注意力，是为了与 encoder 的实现/数值行为保持一致，
> 也少一份维护成本。

### 4.3 patchify / unpatchify

标准 MAE 工具函数，`patch_size` 来自 `patch_embed.patch_size`：

- `patchify(imgs) -> (B, L, patch_size**2 * C)`，`L = (H/P)*(W/P)`；
- `unpatchify(x) -> (B, C, H, W)`，用于可视化与拼回整图。

### 4.4 随机 mask（`random_masking`）

与 MAE 论文一致，对 patch 维度按 per-sample 噪声排序保留前 `(1-mask_ratio)` 个：

```python
def random_masking(self, x, mask_ratio):
    # x: (B, N, D) —— 仅 patch token，不含前缀
    N = x.shape[1]
    len_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, 1, ids_keep[..., None].expand(-1, -1, D))
    mask = torch.ones(B, N); mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)   # 1=被 mask
    return x_kept, mask, ids_restore
```

> 注意：`torch.rand` 在分布式 / 复现实验里要走全局 seed（训练脚本已 `utils.random_seed(seed)`）。

### 4.5 encoder 前向（带 mask，泛化前缀 token）

实现上**直接复用 timm 的 `_pos_embed`**（而非手动切分 `pos_embed`）：它对完整 patch 序列加位置编码并
prepend 前缀 token（cls/reg），已替我们处理好 `no_embed_class`、前缀数量与位置编码插值这三种差异。
我们只在它之后、`blocks` 之前，把 patch 部分按 mask 丢弃；前缀 token 始终保留：

```python
def forward_encoder(self, imgs, mask_ratio):
    x = self.encoder.patch_embed(imgs)               # (B, N, D)
    x = self.encoder._pos_embed(x)                   # (B, P+N, D)：已含前缀 + 位置编码
    P = self.num_prefix_tokens
    prefix, patches = x[:, :P, :], x[:, P:, :]
    if mask_ratio > 0:
        patches, mask, ids_restore = self.random_masking(patches, mask_ratio)
    x = torch.cat([prefix, patches], dim=1)
    x = self.encoder.norm_pre(x)                     # 多数 ViT 为 Identity；pre-norm ViT 有实体
    for blk in self.encoder.blocks:
        x = blk(x)
    x = self.encoder.norm(x)
    return x, mask, ids_restore                      # mask_ratio=0 时 mask/ids_restore 为 None
```

复用 `_pos_embed` 的好处：**直接吃 dinov3 预训练的 pos_embed 与前缀 token 权重**，比重算 sincos 更贴合
「继续预训练已有 encoder」。`mask_ratio=0` 路径（`encode()` 用）不丢 token，可作下游特征提取。

> ✅ **RoPE 兼容性（已实测通过）**：DINOv3 在 timm 里是 **Eva** 类，使用纯 RoPE（`pos_embed=None`，
> 仅旋转位置编码），其 `_pos_embed` 返回 `(x, rope)`、`blocks` 形如 `blk(x, rope=...)`，且注意力**只对
> patch token 施加 rope、跳过前缀 token**。实现据此分两路：标准 ViT 走 `blk(x)`；Eva 走 `blk(x, rope=...)`，
> 并在 mask 后用 timm 的 `apply_keep_indices_nlc` 按 `ids_keep` 同步 gather 被保留 patch 的 rope 频率
> （`(N,dim) -> (B,1,len_keep,dim)`）。已对 `vit_base_patch16_224`（prefix=1, 无 rope）与
> `vit_base_patch16_dinov3.lvd1689m`（prefix=5, RoPE）跑通 compute_loss / backward / reconstruct / encode。
> 仅 `rope_mixed=True`（每层不同 rope）暂未支持并会显式报错；dinov3 lvd1689m 系列为 `rope_mixed=False`。

### 4.6 decoder 前向（重点）

encoder 出来的序列只有 `P 个前缀 + len_keep 个可见 patch`（被 mask 的 patch 根本没进 encoder）。
decoder 要先把序列**补回完整长度并还原到原始 patch 顺序**，再做 transformer 重建。分四步：

```python
def forward_decoder(self, x, ids_restore):
    # x: (B, P + len_keep, D) —— encoder 输出（前缀 + 可见 patch）
    # 步骤 1：投影到 decoder 宽度
    x = self.decoder_embed(x)                        # (B, P+len_keep, Dd)
    P = self.num_prefix_tokens
    prefix, tokens = x[:, :P, :], x[:, P:, :]        # 前缀单独拎出，只对 patch 部分补 mask

    # 步骤 2：用共享 mask_token 把序列补回 N 个 patch
    N = ids_restore.shape[1]
    mask_tokens = self.mask_token.expand(B, N - tokens.shape[1], -1)
    tokens = torch.cat([tokens, mask_tokens], dim=1) # (B, N, Dd)：可见特征 + mask 占位

    # 步骤 3：用 ids_restore 反排列，把 patch 放回它们在原图中的位置
    tokens = torch.gather(tokens, 1, ids_restore[..., None].expand(-1, -1, Dd))
    x = torch.cat([prefix, tokens], dim=1) + self.decoder_pos_embed  # 加 decoder 位置编码

    # 步骤 4：浅 transformer + 线性头回归像素
    for blk in self.decoder_blocks:
        x = blk(x)
    x = self.decoder_norm(x)
    x = self.decoder_pred(x)                         # (B, P+N, p*p*C)
    return x[:, P:, :]                               # 丢掉前缀，仅留 patch 预测 (B, N, p*p*C)
```

四步的要点：
1. **投影**：encoder 与 decoder 宽度不同，先 `decoder_embed` 对齐到 `Dd`。
2. **补 mask token**：被丢弃的 `N - len_keep` 个位置全部填同一个可学习 `mask_token`——decoder 要靠
   **位置编码 + 周围可见 patch 的注意力**去推断这些位置的内容，这正是 MAE 学到表征的关键。
3. **反排列 `ids_restore`**：`random_masking` 当初把 patch 打乱后取前 `len_keep` 个，`ids_restore` 是
   「打乱的逆排列」。这里 `gather` 把 `[可见..., mask...]` 还原成原图 patch 顺序，使 `decoder_pos_embed`
   对得上每个 patch 的真实空间位置。**顺序错则位置编码错位，重建必然崩**——实现时要重点测这一步。
4. **回归像素**：`decoder_pred` 把每个 token 映射到 `p*p*C` 维（一个 patch 的展平像素）。前缀 token 的预测
   不参与损失，直接切掉。

> decoder 只在预训练用；导出 encoder 做下游触觉特征时，`decoder_*` 全部不加载。

### 4.7 损失（仅在被 mask 的 patch 上）

```python
def forward_loss(self, imgs, pred, mask):
    target = self.patchify(imgs)                     # (B, N, p*p*C)
    if self.norm_pix_loss:
        mean = target.mean(-1, keepdim=True)
        var = target.var(-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5
    loss = (pred - target) ** 2
    loss = loss.mean(-1)                             # 每个 patch 的 MSE
    loss = (loss * mask).sum() / mask.sum()          # 只在被 mask patch 上平均
    return loss
```

### 4.8 对外接口（与 VAE 对齐）

```python
def compute_loss(self, batch):
    latent, mask, ids_restore = self.forward_encoder(batch, self.mask_ratio)
    pred = self.forward_decoder(latent, ids_restore)
    loss = self.forward_loss(batch, pred, mask)
    return {"total": loss, "recon": loss, "mask_ratio": torch.tensor(self.mask_ratio)}

@torch.inference_mode()
def reconstruct(self, x):
    # 用于 wandb 可视化：unpatchify(pred)，并用 mask 把可见区域贴回原图
    latent, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)
    pred = self.forward_decoder(latent, ids_restore)
    # 若 norm_pix_loss，可视化时反归一化（用 target 的 mean/std）后再 unpatchify
    return self.unpatchify(pred)

def forward(self, x):
    latent, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)
    pred = self.forward_decoder(latent, ids_restore)
    return pred, mask, ids_restore
```

> `BaseModel.reconstruct` 默认从 `forward` 取第一个返回值，但 MAE 的 `forward` 返回的是 patch 预测而非图像，
> 故 `TactileMAE` **重写 `reconstruct`** 自己 unpatchify。`encode/decode` 抽象方法用 `forward_encoder` /
> `forward_decoder` 适配实现以满足 `BaseModel` 抽象约束。

## 5. 训练脚本 `scripts/train_with_timm.py` 的改动

尽量小改、复用现有循环：

1. **`task` 读取 + 校验**：`_load_config` 后读 `args.task`（默认 `"vae"`），按 §3.3 校验。
2. **backbone 构造**：`_build_timm_backbone` 不变（`num_classes=0` 不影响 `patch_embed/blocks` 的存在）。
3. **模型分发**：新增 `_build_model(args, backbone, data_config)`：
   - `task == "vae"` → 现有 `_build_vae(...)`；
   - `task == "mae"` → 先校验 backbone 是 ViT（见 §6），再造 `TactileMAE`，
     `patch_size` 从 `backbone.patch_embed.patch_size` 读，`image_size` 从 `data_config["input_size"]` 读。
   `main()` 把 `model = _build_vae(...)` 改为 `model = _build_model(...)`。
4. **指标日志泛化**：`train_one_epoch` / `validate` 目前硬编码 `mse/grad/kld/mix` 几个 meter。
   改为**遍历 `loss_dict` 的标量键动态建 meter**（保留 `total` → `loss`），这样 VAE 仍打印 mse/grad/kld/mix，
   MAE 打印 `recon`，互不影响。「best 指标」继续用 `loss`（=total）。
   - 最小改法：保留现有 meter，额外加一个通用 `extra_m: dict[str, AverageMeter]`，对 `items` 里非
     `total` 的键动态累计并在 postfix / summary 里输出。
5. **实验名**：`exp_name` 里的 `f"vae{args.img_size}"` 改为 `f"{args.task}{args.img_size}"`。
6. **wandb 可视化**：`get_wandb_reconstructions` 已走 `model.reconstruct(inputs)`，MAE 重写了 `reconstruct`，
   可直接复用。可选增强：MAE 额外画一张「masked 输入图」（把被 mask patch 涂灰），更直观；本期可后置。
7. **`_move_batch` / 数据**：不变，MAE 同样吃 `(B,3,H,W)` 归一化图像。

## 6. Backbone 兼容性约束

`task: mae` 时，`_build_model` 先做能力检查：

```python
is_vit = hasattr(backbone, "patch_embed") and hasattr(backbone, "blocks") \
         and getattr(backbone, "pos_embed", None) is not None
if not is_vit:
    raise ValueError(
        f"task=mae 仅支持 ViT 类 backbone（需 patch_embed/blocks/pos_embed），"
        f"当前 model={args.model} 不满足。请改用 vit_*_dinov3 等配置，或将该 config 设为 task=vae。"
    )
```

支持矩阵：

| backbone 类型 | 示例 config | VAE | MAE |
|---|---|---|---|
| ViT (DINOv3) | `vit_{small,base,large}_patch16_dinov3_lvd1689m` | ✅ | ✅ |
| ConvNeXt-V2 / ResNet / FastViT / EfficientViT / MobileNetV4 | 其余 configs | ✅ | ❌（清晰报错） |

## 7. 数据与预处理

- MAE 同样使用 `create_transform(**data_config, is_training=False)`（resize + center crop +
  ImageNet 归一化），与现有 VAE 路径一致，**不引入颜色/擦除增强**（保持触觉信号不被破坏）。
- `norm_pix_loss` 在归一化的输入图上计算 patch 统计量，可正常工作；可视化时若需还原到 [0,1] 用
  `_denormalize_batch`（脚本已有）。
- patch 数要求图像边长能被 `patch_size` 整除；dinov3 patch16 + 256 输入 → 16×16=256 patch，满足。

## 8. 涉及文件改动清单

| 文件 | 改动 |
|---|---|
| `src/models/mae.py` | **新增**：`TactileMAE` + `patchify/unpatchify/random_masking` + decoder。 |
| `src/models/__init__.py` | 导出 `TactileMAE`；（可选）注册 `tactile_mae` 工厂。 |
| `scripts/train_with_timm.py` | 读取/校验 `task`；新增 `_build_model` 分发；指标日志泛化；exp_name 含 task。 |
| `configs/*.yaml` | 给现有 config 补 `task: vae`（向后兼容，亦可不写）。 |
| `configs/vit_base_patch16_dinov3_lvd1689m_mae.yaml` | **新增** MAE 示例 config（见 §9）。 |
| `docs/mae_task_design.md` | 本文档。 |
| `scripts/run_all_timm.sh` | 无需改动（已支持按文件名跑指定 config）。 |

## 9. MAE 示例 config（草案）

`configs/vit_base_patch16_dinov3_lvd1689m_mae.yaml`：

```yaml
task: mae

# Dataset
data_dir: data
eval_ratio: 0.2

# Model
model: vit_base_patch16_dinov3.lvd1689m
pretrained: true
pretrained_path: "checkpoint/vit_base_patch16_dinov3_lvd1689m/model.safetensors"
initial_checkpoint: ""
resume: ""
no_resume_opt: false
in_chans: 3
model_kwargs: {}

# MAE 超参
mae:
  mask_ratio: 0.75
  decoder_embed_dim: 512
  decoder_depth: 4
  decoder_num_heads: 16
  norm_pix_loss: true

# Device / Optimizer / Scheduler（与 VAE 路径一致）
device: auto
opt: adamw
lr: 0.0001
weight_decay: 0.05
clip_grad:
sched: cosine
epochs: 400
warmup_epochs: 10
warmup_lr: 0.000001
min_lr: 0.000001
cooldown_epochs: 0

# Loader / logging
batch_size: 64
workers: 4
seed: 42
log_interval: 20
val_interval: 1
output: outputs
experiment: ""

# wandb
log_wandb: true
wandb_project: xense-tac-encoder
wandb_name: "vit_base_patch16_dinov3_lvd1689m_mae"
wandb_log_images: true
wandb_image_interval: 1
wandb_num_images: 8
```

> 注：VAE 专属字段（`latent_dim` / `decoder_hidden_*` / `w_*` / `mix_alpha` / `use_ms_ssim`）在 MAE config 中
> 省略，由 §3.3 的任务感知校验放行。

## 10. 验证与测试计划

1. **单测 / smoke**：构造 `vit_base_patch16_dinov3` backbone + `TactileMAE`，
   随机 `(2,3,256,256)` 输入跑一次 `compute_loss`，断言 `loss` 标量、可反传、`reconstruct` 输出 `(2,3,256,256)`。
2. **前缀 token 泛化**：分别用 `vit_base_patch16_224`（num_prefix=1）与 `dinov3`（num_prefix=5）跑通，
   确认 mask/pos 拆分正确、形状对齐。
3. **报错路径**：用 `resnet50` config + `task: mae` 跑，确认抛出 §6 的清晰错误。
4. **端到端**：`./scripts/run_all_timm.sh vit_base_patch16_dinov3_lvd1689m_mae` 跑 1~2 epoch，
   确认 loss 下降、checkpoint / summary.csv / wandb 重建图正常。
5. **回归**：原 VAE config 行为不变（默认 `task: vae`，指标 mse/grad/kld/mix 照常打印）。

## 11. 风险与待确认项（实现后状态）

- ✅ **DINOv3 前缀 token / RoPE / pos_embed**：已解决。实现复用 timm `_pos_embed`（自动处理前缀与
  `no_embed_class`），dinov3 为纯 RoPE（`pos_embed=None`），按 §4.5 的 rope-gather 方案跑通。
- ✅ **`norm_pix_loss` 下的可视化**：`reconstruct()` 已用 target 的 patch mean/std 反归一化后再 unpatchify，
  并把可见 patch 贴回原图，观感正常。
- ✅ **指标 meter 泛化**：已按动态 meter 实现（VAE→loss/mse/grad/kld/mix，MAE→loss/recon），见 §5.4。
- ⬜ **是否冻结/部分冻结预训练 encoder**：当前默认全量微调（继续预训练）。如需 linear-probe，可后续加
  `freeze_encoder` 选项。
- ⬜ **`rope_mixed=True` 的 ViT**：暂不支持（每层不同 rope），构造时显式报错；dinov3 lvd1689m 为
  `rope_mixed=False`，不受影响。

## 12. 实施顺序建议

1. 写 `src/models/mae.py`（`TactileMAE` + 工具函数），本地 smoke 测试（§10.1/10.2）。
2. 改 `train_with_timm.py`：`task` 校验 + `_build_model` 分发 + 指标泛化 + exp_name。
3. 加 MAE 示例 config，跑 1~2 epoch 端到端验证。
4. 给现有 config 补 `task: vae`，跑一次 VAE 回归。
5. 更新 `README.md` 简述两种任务的用法。
</content>
</invoke>
