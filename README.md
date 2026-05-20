# Xense Tac Encoder

触觉 RGB 图像 encoder 选型与重建训练工具。项目用一个轻量 VAE/Autoencoder 重建任务来比较不同图像 backbone 的 **embedding 延迟、重建质量、参数量和潜空间统计**，用于给 TacXense / System-0 这类前端触觉感知路径筛选合适的图像 encoder。

当前主线是：

1. 使用 `timm.create_model(..., num_classes=0)` 构建候选 encoder，使 `encoder(x)` 直接返回 `(B, C)` 图像 embedding。
2. 用 `TactileVAE` 将 embedding 投影到 latent，再反卷积重建触觉 RGB 图像。
3. 通过重建 loss、latency benchmark 和 latent variance 做横向比较。

## 项目结构

```text
.
├── configs/                         # timm backbone 训练配置
├── checkpoint/                      # 本地预训练权重目录，不建议提交大文件
├── data/                            # 触觉 RGB 图像数据目录
├── outputs/                         # scripts/train_with_timm.py 输出目录
├── runs/                            # scripts/train.py / evaluate.py 默认输出
├── scripts/
│   ├── train_with_timm.py            # 当前主要训练入口：timm encoder + TactileVAE
│   ├── train.py                      # typed config 训练入口，当前主要服务本地 MobileNetV4 路径
│   ├── evaluate.py                   # 旧式/兼容评估入口
│   ├── run_all_timm.sh               # 顺序跑 configs/*.yaml
│   └── benchmark_all_configs.sh      # 顺序 benchmark configs/*.yaml
└── src/
    ├── models/
    │   ├── vae.py                    # TactileVAE / TactileAutoencoder
    │   ├── backbones.py              # 本地 backbone registry
    │   ├── fastvit_t12_apple_dist_in1k.py
    │   └── mobilenet_v4/
    ├── training/                     # data loader / loss / config / factory
    └── utils/
        ├── benchmark_feature_extractors.py
        ├── evaluation.py
        ├── evaluator.py
        └── sample_mp4_rgb_frames.py
```

## 环境安装

推荐在独立 Python/Conda 环境中安装：

```bash
pip install -r requirements.txt
```

`requirements.txt` 中包含训练和 benchmark 所需的核心依赖：`torch`、`torchvision`、`timm`、`safetensors`、`wandb`、`tyro` 等。

如果配置里启用了 `w_mix > 0`、`use_ms_ssim: true` 或 `w_ssim > 0`，还需要安装：

```bash
pip install pytorch-msssim
```

## 数据格式

默认数据目录是 `data/`。数据加载器会递归读取以下格式的 RGB 图像：

```text
.png .jpg .jpeg .bmp .tif .tiff
```

不要求固定子目录结构，所有图片会按照 `eval_ratio` 和 `seed` 做确定性 train/eval 划分。

示例：

```text
data/
├── file-000_000000.png
├── file-000_000001.png
└── episode_01/
    └── frame_000123.png
```

也可以从 MP4 中采样 RGB 帧：

```bash
python src/utils/sample_mp4_rgb_frames.py \
    --video /path/to/video.mp4 \
    --output-dir data \
    --prefix episode_01 \
    --target-fps 10 \
    --overwrite
```

这个脚本依赖 `lerobot` 的视频解码路径。

## 预训练权重

多数 YAML 配置使用本地 HuggingFace/timm checkpoint，例如：

```yaml
pretrained_path: "checkpoint/fastvit_t12_apple_dist_in1k/model.safetensors"
```

请将对应模型权重放在 `checkpoint/<model_name>/` 下。项目已经支持 `.safetensors` 和 PyTorch `.bin/.pth/.pt` 权重，训练脚本会通过 `timm` 的 `pretrained_cfg_overlay` 从本地文件加载。

## 训练

当前推荐入口是 `scripts/train_with_timm.py`：

```bash
python scripts/train_with_timm.py \
    --config configs/fastvit_t12_apple_dist_in1k.yaml
```

也可以只传配置文件名：

```bash
python scripts/train_with_timm.py --config fastvit_t12_apple_dist_in1k.yaml
```

配置里关键字段：

```yaml
data_dir: data
eval_ratio: 0.2

model: fastvit_t12.apple_dist_in1k
pretrained: true
pretrained_path: "checkpoint/fastvit_t12_apple_dist_in1k/model.safetensors"
reparameterize: true
in_chans: 3

latent_dim: 256
decoder_hidden_channels: 512
decoder_hidden_spatial:

w_mse: 0.0
w_grad: 0.0
w_kld: 0.01
w_ssim: 0.0
w_mix: 1.0
mix_alpha: 0.84
use_ms_ssim: true

batch_size: 16
epochs: 25
output: outputs
```

训练输出会写入：

```text
outputs/<timestamp>-<model>-vae<size>/
├── args.yaml
├── summary.csv
├── model_best.pt
└── checkpoint_last.pt
```

批量跑所有配置：

```bash
./scripts/run_all_timm.sh
```

只跑指定配置：

```bash
./scripts/run_all_timm.sh fastvit_t12_apple_dist_in1k resnet50_a1_in1k
```

## 已有配置

`configs/` 下当前包含这些候选 encoder：

```text
mobilenetv4_conv_aa_large
resnet50.a1_in1k
efficientvit_b3.r224_in1k
convnextv2_base.fcmae_ft_in22k_in1k
fastvit_t12.apple_dist_in1k
fastvit_sa12.apple_dist_in1k
fastvit_sa24.apple_dist_in1k
fastvit_sa36.apple_dist_in1k
fastvit_ma36.apple_in1k
fastvit_ma36.apple_dist_in1k
vit_small_patch16_dinov3.lvd1689m
vit_base_patch16_dinov3.lvd1689m
vit_large_patch16_dinov3.lvd1689m
```

所有这些配置都走相同的训练约定：

```python
encoder = timm.create_model(model_name, pretrained=..., num_classes=0)
embedding = encoder(x)  # (B, C)
```

## Benchmark

单个配置的 embedding latency benchmark：

```bash
python src/utils/benchmark_feature_extractors.py \
    --config configs/fastvit_t12_apple_dist_in1k.yaml \
    --warmup 50 \
    --iters 500
```

使用真实图片构造输入，但不把预处理计入耗时：

```bash
python src/utils/benchmark_feature_extractors.py \
    --config configs/fastvit_t12_apple_dist_in1k.yaml \
    --image data/file-000_000000.png \
    --warmup 50 \
    --iters 500
```

保存 CSV：

```bash
python src/utils/benchmark_feature_extractors.py \
    --models fastvit_t12.apple_dist_in1k resnet50.a1_in1k \
    --input-size 3 256 256 \
    --output-csv outputs/embedding_latency.csv
```

批量 benchmark 所有配置：

```bash
./scripts/benchmark_all_configs.sh data/file-000_000000.png
```

## 评估

兼容评估入口是：

```bash
python scripts/evaluate.py \
    --config mobilenetv4_conv_aa_large_vae \
    --set evaluation.checkpoint=runs/mobilenetv4_conv_aa_large_vae.pt \
    --set evaluation.max_batches=20
```

也支持旧式多 backbone 对比参数：

```bash
python scripts/evaluate.py \
    --backbones resnet18 mobilenet_v3_small efficientnet_b0 \
    --data data \
    --pretrained \
    --output-json runs/compare.json
```

评估指标包括：

- `latency_ms`：默认只统计 `model.encoder` 前向，对应部署时只用 encoder 的路径。
- `params_million`：encoder 参数量。
- `mse` / `grad_loss`：重建误差。
- `latent_variance`：latent 在样本维度上的平均方差，过低通常表示潜空间坍塌。
- `n_samples`：参与评估的样本数量。

## 单文件 FastViT-T12 Encoder

[src/models/fastvit_t12_apple_dist_in1k.py](src/models/fastvit_t12_apple_dist_in1k.py) 已提取为自包含 FastViT-T12 Apple distilled ImageNet encoder。这个文件可以复制到其它项目中使用，不需要 `timm` 运行时依赖，只依赖 PyTorch；加载 `.safetensors` 时需要 `safetensors`。

示例：

```python
import torch
from src.models.fastvit_t12_apple_dist_in1k import create_encoder

encoder = create_encoder(
    pretrained=True,
    checkpoint_path="checkpoint/fastvit_t12_apple_dist_in1k/model.safetensors",
    reparameterize=True,
).eval()

x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    embedding = encoder(x)

print(embedding.shape)  # torch.Size([1, 1024])
```

如果只需要特征图：

```python
features = encoder.forward_features(x)  # (B, 1024, 8, 8)
```

注意：训练脚本 `scripts/train_with_timm.py` 仍然使用 `timm` 统一管理多 backbone。单文件 FastViT-T12 主要用于迁移到其它项目或脱离 `timm` 的部署路径。

## Loss 说明

`src/training/losses.py` 中定义了训练使用的重建目标：

- `mse`：像素级 MSE。
- `grad`：一阶图像梯度 L1，强化阴影边界和接触区域结构。
- `kld`：VAE KL 散度。
- `ssim`：可选 SSIM loss。
- `mix`：`alpha * (1 - SSIM/MS-SSIM) + (1 - alpha) * L1`。

当前多数配置使用：

```yaml
w_mse: 0.0
w_grad: 0.0
w_kld: 0.01
w_mix: 1.0
mix_alpha: 0.84
use_ms_ssim: true
```

也就是用 MS-SSIM + L1 混合重建项，加一个较小的 KL 约束。

## 新增模型或配置

新增 timm backbone：

1. 在 `configs/` 新建 YAML。
2. 设置 `model` 为 timm 注册名。
3. 如需本地权重，设置 `pretrained_path`。
4. 确认 `decoder_hidden_spatial` 为空或与输入尺寸匹配；默认按 `input_size // 32` 推断。

新增本地 backbone：

1. 在 `src/models/backbones.py` 中用 `@register("name")` 注册。
2. 返回 `(backbone_module, feature_dim, spatial_size)`。
3. 如果给 `TactileVAE` 使用，最终 encoder 输出需要能转成 `(B, C)` embedding。

新增本地模型配置：

1. 参考 `src/models/mobilenet_v4/mobilenetv4_config.py`。
2. 在 `src/training/config.py` 的 `_CONFIGS` 中加入 `TrainConfig` 预设。

## 常见问题

**找不到图片**

确认 `data_dir` 或 `data.root` 指向的是目录，并且目录下至少有两张支持格式的图片。

**启用 mix loss 后报 `pytorch-msssim` 缺失**

安装：

```bash
pip install pytorch-msssim
```

**本地 checkpoint 加载失败**

检查 YAML 中的 `pretrained_path` 是否存在。相对路径以项目根目录为准。

**FastViT-T12 复制到其它项目后找不到 checkpoint**

显式传入权重路径：

```python
create_encoder(pretrained=True, checkpoint_path="/abs/path/to/model.safetensors")
```
