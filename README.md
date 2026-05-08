# 触觉 RGB 编码器选型工具

模块化复刻原始 VAE 选型流水线，便于通过图像重建任务对多个候选 Encoder 快速对比 **推理延迟 / 重建质量 / 潜空间方差**。

当前仓库的核心目标是给 TacXense/System-0 前端筛选触觉 RGB encoder：先用轻量重建任务稳定比较 backbone，再把低延迟、潜空间不坍塌的 encoder 候选迁移到下游触觉-视觉-语言-动作模型。

## 目录结构

```
xense_tac_encoder/
  config.py         # TacXense 风格 TrainConfig / registry / YAML 兼容层
  factory.py        # 根据 config 实例化 model / data / optimizer
  training.py       # checkpoint 加载兼容层
  evaluation.py     # config 驱动评测入口
  dataset.py        # 触觉图像数据集（不裁剪，保留外壳信息）
  losses.py         # 模型内部 loss 共享工具与权重定义
  metrics.py        # latency / 参数量 / latent variance
  evaluator.py      # 端到端评估流程
  architectures/
    backbones.py    # 候选 backbone 注册表
    vae.py          # TactileVAE / TactileAutoencoder
scripts/
  train.py          # 训练入口与训练循环
  evaluate.py       # 评测脚本
```

`xense_tac_encoder.config` 对齐 TacXense 的配置分层：`TactileEncoderConfig` 描述模型与 backbone variant，`DataConfigFactory` 延迟生成数据配置，`LossConfig` 只保存 loss 权重，具体 loss 在 `TactileVAE` / `TactileAutoencoder` 内部实现，`TrainConfig` 作为训练和评测的顶层入口。可复用实验沉淀为 `get_config()` 命名预设；YAML 文件仅作为外部自定义配置的兼容输入，不再在仓库内维护默认 YAML。

## 安装

```bash
pip install -r requirements.txt
```

## 候选 backbone

`resnet18 / resnet34 / resnet50 / mobilenet_v3_small / mobilenet_v3_large / efficientnet_b0 / shufflenet_v2_x1_0 / convnext_tiny`

## 训练

推荐通过 config 固化实验：

```bash
python scripts/train.py --config vae_resnet18 \
    --set data.train.root=/path/to/tactile_imgs \
    --set data.eval.root=/path/to/tactile_imgs \
    --set training.epochs=30
```

也可以用旧参数快速覆盖：

```bash
python scripts/train.py \
    --backbone resnet18 \
    --data /path/to/tactile_imgs \
    --epochs 30 --batch-size 32 \
    --output runs/resnet18.pt
```

config 中 `data.train.root` 为空时会用随机张量跑通流水线，便于烟测。

## 选型评测

评测训练后的某个 config/checkpoint：

```bash
python scripts/evaluate.py --config vae_resnet18 \
    --set evaluation.checkpoint=runs/vae_resnet18.pt \
    --set data.eval.root=/path/to/tactile_imgs
```

多 backbone 横向对比（无监督初始化即可粗筛）：
```bash
python scripts/evaluate.py \
    --backbones resnet18 mobilenet_v3_small efficientnet_b0 \
    --data /path/to/tactile_imgs \
    --pretrained \
    --output-json runs/compare.json
```

评测训练后的某个权重（旧参数方式）：
```bash
python scripts/evaluate.py \
    --backbones resnet18 \
    --checkpoint runs/resnet18.pt \
    --data /path/to/tactile_imgs
```

## 指标解读

- **latency_ms**：仅统计 encoder 前向，对应 System-0 部署路径
- **mse / grad_loss**：越低越好；`grad_loss` 在无 marker 场景下尤其敏感于阴影边界
- **latent_variance**：过低 → 特征坍塌（外壳主导），需要降低 KLD 权重或加大梯度损失

## 新增 model / loss

- 新增 Encoder backbone：在 `xense_tac_encoder/architectures/backbones.py` 中用 `@register("name")` 返回 `(module, feature_dim, spatial)`。
- 新增重建模型：在 `xense_tac_encoder/architectures/__init__.py` 中用 `@register_model("name")` 注册构建函数。
- 新增 loss 形式：在对应模型类中实现或调整 `compute_loss()`，共享的权重和基础项放在 `xense_tac_encoder/losses.py`。
- 新增命名实验：在 `xense_tac_encoder/config.py` 的 `_CONFIGS` 中添加 `TrainConfig` 预设。
