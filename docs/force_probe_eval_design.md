# z → 逐节点三维力（35×20×3）Probe 评测方案

目标：冻结已训练的触觉编码器，从其 pooled latent `z` 出发，用轻量 probe 头预测凝胶表面
35×20 网格上每个节点的三维接触力（共 35×20×3 = 2100 维），量化表征中关于"力的空间分布"
的信息含量。方案兼容 FastViT 与 DINOv3 两条 multitask 分支。

## 0. 现状盘点（已核实）

- **latent z**：三个 multitask 模型类都暴露 `encode()`（`src/models/multitask.py:219/482/706`）：
  - FastViT 分支（`TactilePhysicalMultiTask`）：`PhysicalPooler` 输出 z = (B, 1024)；
  - DINOv3 分支（`TactileViTPhysicalMultiTask`）：`AttentionPool` 输出 z = (B, 768)。
  - probe 只依赖 `model.encode(x)` 这一统一接口，维度差异由头的输入层吸收，天然兼容。
- **现有 probe 设施**：`scripts/evaluate_multitask.py --probe` 已实现冻结 encoder →
  抽 z → ridge 闭式回归 / softmax 分类 probe 的完整链路，但只支持标量/向量目标。
- **力相关字段（collection2 H5，已逐字段检查）**：
  - `force_xyz (3,) float32`：FEM 接触节点力按轴求和的三维合力（凝胶系，N）；
  - `force_resultant (1,) float32`：合力模长；
  - attrs：`fem_contact_nodes`（接触节点数，实测 1~246/700）等。
  - **force_grid 已补齐**（2026-08 离线回填，
    `data/xensim/outputs/collection2_force/tactile_20260825_143603.h5`）：
    `force_grid (35,20,3) float32`（凝胶系，N，非接触节点显式置零）+
    `contact_mask (35,20) bool`；5994 样本中 5863 个有效，131 个采集失败样本
    无此字段，评测时跳过（`H5TactileDataset(require_keys=["force_grid"])`）。
    原始采集目录 collection2 的同名文件没有该字段，split 复现仍以训练目录为准。
  - ~~没有 35×20×3 逐节点力场~~（历史记录）：35×20=700 是 FEM 表层节点网格
    （`docs/collection2_h5_generation.md:34`），逐节点力只在仿真运行时存在于
    `FEMSimulator.contact_force`（形状 (n_contact, 3)，配合 `contact_idx`）。
- **注意**：DINOv3 配置当前 `pretrained: false`（随机初始化 ViT-B/16）。若评测意图是
  对比"预训练表征"，需先解决 DINOv3 权重加载，否则两 backbone 对比不公平。

## 1. 前置步骤：补齐 force_grid 真值

新增数据集字段契约：

```
force_grid  (35, 20, 3) float32   凝胶坐标系，单位 N，非接触节点严格为 0
```

生成方式：`contact_force` (n_contact, 3) 按 `contact_idx` 散布到 700 节点向量 (700, 3)，
再 `reshape(35, 20, 3)`——与现有 `flow` 的 `disp.reshape(35, 20, 2)` 同一网格约定
（`docs/collection2_h5_generation.md:96`），保证 flow 与 force_grid 空间对齐。

两条可行路径（二选一）：

1. **补采**：修改 `data/xensim/examples/collect_tactile_data*.py`，在写 `force_xyz` 的同一
   位置加写 `force_grid`，重跑采集。最干净，但成本高（全部重仿真）。
2. **离线回放**：利用 H5 已保存的 `object_model / object_xyz_m / object_rpy_deg /
   press_depth_mm / sensor_pose` 等状态重放 FEM 求解，把 `force_grid` 追加进现有 H5。
   成本低，但需验证回放与原始采集的数值一致性（用 `force_xyz`、`fem_contact_nodes`
   做回归校验）。

无论哪条路径，入库前做一致性校验：
`force_grid.sum(axis=(0,1)) ≈ force_xyz`（同一物理量的两种聚合）；
`np.count_nonzero(force_grid 节点模长) ≈ fem_contact_nodes`。

**标准化问题**：`H5TactileDataset` 默认按 stats.json 做 z-score，但 force_grid 大量元素
为 0（接触节点占比 0.1%~35%），全量统计会被零主导。建议：

- probe 训练与评测直接在**原始物理量（N）**上进行，不对 force_grid 做 z-score；
- 或将标准化统计限定在非零元素上，仅用于 loss 内部，评测一律换算回 N。

## 2. Probe 头设计

输入：冻结 encoder 的 `z = model.encode(image)`（FastViT: 1024 维 / DINOv3: 768 维）。
输出：2100 维向量，`reshape(35, 20, 3)`。

### 头 A：纯线性头（Linear probe）

`ŷ = W z + b`，W ∈ R^{2100×d}。

- 首选闭式 ridge 回归（复用 `evaluate_multitask.py:240 _fit_ridge`，含 bias 列，
  λ 默认 1e-3，需在 probe-val 上扫 λ ∈ {1e-4 … 1e2}）；
- 闭式解在 2100 输出维下依然廉价（特征维 d ≤ 1024，按特征维求解正规方程）。

### 头 B：简单 MLP 头

`z → Linear(d, h) → GELU → Dropout(0.1) → Linear(h, 2100)`，h = 1024（一层隐藏层，
保持"简单"，避免 probe 过强导致评测失去表征归因意义；可将 h 与层数作为消融项）。

- 优化：AdamW，lr 1e-3（扫 {1e-2, 1e-3, 1e-4}），weight_decay 1e-4，batch 256，
  全量 epoch ≤ 100，早停看 probe-val 的 masked MAE；
- 损失：总 MSE 为主；因零元素占多数，纯 MSE 会奖励"全零预测"，需配合第 4 节的
  非零指标解读。可加非零加权项作为消融：
  `L = MSE_all + w_nz · MSE(mask_nz)`，w_nz 默认 0（消融取 1~5）。

两个头除输入维度 d 外与 backbone 无关，同一份代码按 `feature_dim` 实例化即可。

## 3. 数据划分协议（probe 训练不得使用评测数据）

现有 `split_row_indices`（`src/datasets/labeled_dataset.py:72`）只有 train/eval 两段
（0.8/0.2，seed 42），没有独立 test。本方案的协议：

```
train split (80%)  →  拟合 probe（ridge 闭式解 / MLP 训练）
   └─ 再切 10% 作 probe-val  →  选超参（λ、lr、epoch、w_nz），早停
eval split  (20%)  →  只用于最终一次性报告指标，不参与任何拟合与选型
```

- encoder 全程冻结（沿用已训练 checkpoint），probe 只在 train split 的 z 上拟合；
- eval split 的 z、force_grid 在超参确定前不可见；
- 如需更稳健的结论，可换 seed ∈ {42, 43, 44} 重跑划分，报告均值±标准差；
- 注意 encoder 本身是在同一 train split 上训练的，这是 linear probing 的标准协议
  （probe 测的是表征的线性/浅层可读性，不是泛化到新分布）。

## 4. 评测指标

所有指标在反标准化后的物理单位（N）上计算，按节点维（700）与分量维（3）展开。

### 4.1 全体指标（全部 2100 维）

- MAE、RMSE、R²（宏观，反映整体回归质量，但被大量零元素稀释）。

### 4.2 非零指标（核心，针对零占比高的问题）

- mask 定义：`m_node = ‖f_node‖₂ > ε`，ε 取力值噪声阈值（建议按数据分布定，
  起点 1e-3 N；同时报 ε ∈ {1e-3, 1e-2, 1e-1} 的敏感性）；
- masked MAE / RMSE / R²：只在真值非零节点上计算；
- 分桶指标：按真值节点力模长分桶（如 [0,ε), [ε,0.1), [0.1,1), [1,∞) N），
  分桶报 MAE，观察大力/小力的回归质量差异。

### 4.3 接触检测指标（零/非零分类）

- 把"节点是否接触"当二分类（预测侧用 ‖ŷ_node‖₂ > ε 判定）：
  per-node Precision / Recall / F1 / IoU；
- 该指标直接暴露"全零预测"的作弊行为（Recall=0）。

### 4.4 物理一致性指标

- 合力误差：‖ŷ.sum(axis=(0,1)) − force_xyz‖ 的 MAE/相对误差——检验预测力场
  求和是否与已有的合力标注一致（与下游可用的 force_xyz 直接可比）；
- 接触质心误差：力加权质心位置误差（mm），检验空间分布定位能力。

### 4.5 可选可视化

- 每个样本的真值/预测力场模长热图（35×20），定性抽查。

## 5. 实验矩阵与报告

2 backbone × 2 head，同划分、同协议。

**实测设置**（2026-08-29 跑通，`scripts/evaluate_force_probe.py`）：

- 数据：`data/xensim/outputs/collection2_force/tactile_20260825_143603.h5`
  （5994 样本，5863 个含 force_grid；131 个采集失败样本已剔除）。
- 划分：在训练 h5_dir（collection2，4 文件共 40094 样本）全局序号上复现
  `split_row_indices(eval_ratio=0.2, seed=42)`，取本文件片段 →
  train=4692 / eval=1171；train 再切 10% 得 probe-train=4223 / probe-val=469。
- checkpoint：FastViT 用 `outputs/20260827-111139-fastvit_t12_apple_dist_in1k-multitask256/model_best.pt`；
  DINOv3 用 `outputs/20260828-005331-vit_base_patch16_dinov3_lvd1689m-multitask256/model_best.pt`。
- ε = 1e-3 N（masked/接触检测主阈值，敏感性见各结果 json 的 ε ∈ {1e-2, 1e-1} 项）。
- 所选超参：FastViT linear λ=0.1、MLP lr=1e-3（91 epoch 早停）；
  DINOv3 linear λ=0.01、MLP lr=1e-3（99 epoch）。
- **注意**：两个 multitask 运行实际都是随机初始化起步：DINOv3 配置本身
  `pretrained: false`；FastViT 配置的 `pretrained_path` 指向的 `params.safetensors`
  是 Flax 命名格式，timm 无法识别（加载时全部报 unexpected keys，权重未生效，
  2026-09-01 核实并修复，见 §5.1）。因此表中两行本质是"两种架构随机初始化 +
  multitask 训练"的对比。

非零指标列为 ε=1e-3 的 masked MAE/RMSE/R²；合力相对误差为 mean（median 见 json）。

| backbone | z 维度 | head | 全体 MAE/RMSE/R² | 非零 MAE/RMSE/R² | 接触 F1 | 合力相对误差 |
|---|---|---|---|---|---|---|
| FastViT-T12 (physical) | 1024 | linear | 0.00088 / 0.00255 / 0.647 | 0.00757 / 0.01306 / 0.784 | 0.072 | 0.244 |
| FastViT-T12 (physical) | 1024 | MLP | 0.00144 / 0.00312 / 0.468 | 0.00951 / 0.01627 / 0.665 | 0.045 | 1.277 |
| ViT-B/16 DINOv3 (dinov3_physical, 随机初始化) | 768 | linear | 0.00081 / 0.00231 / 0.710 | 0.00700 / 0.01172 / 0.826 | 0.080 | 0.242 |
| ViT-B/16 DINOv3 (dinov3_physical, 随机初始化) | 768 | MLP | 0.00111 / 0.00257 / 0.641 | 0.00804 / 0.01346 / 0.771 | 0.055 | 1.064 |
| 常数基线（train 逐维均值） | — | — | 0.00070 / 0.00429 / -0.002 | 0.01755 / 0.03089 / -0.208 | 0.042 | 5.145 |

完整结果 json：`outputs/<run>/force_probe_eval/force_probe_metrics.json`。

观察：

- 两个 backbone 的 **linear probe 均显著优于常数基线**（masked R² 0.78/0.83 vs -0.21），
  说明 z 中线性可读的空间力分布信息确实存在；
- MLP 头一致差于 linear 头（~4700 样本 × 2100 输出维下过拟合），本任务无需更强的头；
- 接触检测 Precision 极低（0.02~0.04）：数据节点力普遍很小（绝大多数接触节点
  模长 < 0.1 N，[1,∞) 桶为空），ε=1e-3 下预测噪声即产生大量假阳性，
  该指标需结合力值量级解读；
- 合力一致性：linear 头相对误差 ~0.24，远优于基线 5.1，与 force_xyz 标注可比。

### 5.1 权重来源对比（训练后 vs 初始预训练）

5 组权重 × (linear + MLP)，协议/split/指标/数据与上表完全相同，只改变 encoder
权重来源（`--checkpoint none` 走 `_build_timm_backbone` 的 pretrained 加载路径，
multitask 头/pooler 保持随机初始化，seed=42 固定）。

权重来源：

- **FastViT 训练后**：`outputs/20260827-111139-fastvit_t12_apple_dist_in1k-multitask256/model_best.pt`；
- **FastViT 初始**：本地 `checkpoint/fastvit_t12_apple_dist_in1k/params_timm.safetensors`——
  原始 `params.safetensors` 为 Flax 命名格式（`/` 分隔、HWIO kernel、BN 的
  `scale/mean/var`），timm 无法直接加载，经
  `scripts/convert_fastvit_flax_safetensors.py` 转换为 timm 格式
  （转换结果与 timm 官方 HF 权重前向逐位一致，max diff=0）；
- **FastViT SimMIM**：`checkpoint/trained_params.safetensors`（SimMIM 掩码重建
  预训练得到的 encoder 权重），同为 Flax 格式，经同一脚本转换为
  `checkpoint/fastvit_t12_simmim_timm.safetensors`；评测配置
  `configs/multitask/fastvit_t12_simmim_collection2.yaml`；
- **DINOv3 训练后**：`outputs/20260828-005331-vit_base_patch16_dinov3_lvd1689m-multitask256/model_best.pt`
  （该运行本身是随机初始化起步的）；
- **DINOv3 初始**：timm `vit_base_patch16_dinov3.lvd1689m` 官方预训练权重，
  实际下载自 timm 的 HF 镜像仓库 `timm/vit_base_patch16_dinov3.lvd1689m`
  （官方 `facebook/dinov3-vitb16-pretrain-lvd1689m` 为门控仓库：无 HF_TOKEN 时
  huggingface.co 返回 403、hf-mirror.com 返回 401，均不可用；timm 仓库非门控，
  权重同为 DINOv3 LVD1689M 官方发布版本）。

**注意 1**：初始权重组的 z 经过一层随机投影（physical 架构的 PhysicalPooler /
AttentionPool 是随机初始化的）。这不影响 probe 合法性（probe 只在 train split
拟合），但意味着初始组的数字是"预训练 backbone + 随机投影"的下界，而非纯
预训练表征的直接读出。

**注意 2**：本节 2026-08-29 版本的"FastViT 初始"两行因上述 Flax 格式问题
加载失败，实际是在随机初始化 backbone 上跑出的（masked R²=0.107 / -0.208），
已由 2026-09-01 重跑的本表结果取代。

| 权重 | head | 全体 MAE/RMSE/R² | 非零 masked MAE/RMSE/R² | 接触 P/R/F1 | 合力相对误差 |
|---|---|---|---|---|---|
| FastViT 训练后 | linear (λ=0.1) | 0.00088 / 0.00255 / 0.647 | 0.00757 / 0.01306 / 0.784 | 0.038 / 0.989 / 0.072 | 0.244 |
| FastViT 训练后 | MLP (lr=1e-3) | 0.00144 / 0.00312 / 0.468 | 0.00951 / 0.01627 / 0.665 | 0.023 / 0.992 / 0.045 | 1.277 |
| FastViT 初始 | linear (λ=1e-4) | 0.00129 / 0.00426 / 0.009 | 0.01284 / 0.02248 / 0.360 | 0.034 / 0.968 / 0.066 | 1.520 |
| FastViT 初始 | MLP (lr=1e-3) | 0.00124 / 0.00412 / 0.075 | 0.01556 / 0.02710 / 0.070 | 0.023 / 0.886 / 0.045 | 1.533 |
| FastViT SimMIM | linear (λ=1e-4) | 0.00079 / 0.00364 / 0.280 | 0.01367 / 0.02394 / 0.274 | 0.050 / 0.936 / 0.094 | 0.810 |
| FastViT SimMIM | MLP (lr=1e-3) | 0.00125 / 0.00439 / -0.051 | 0.01762 / 0.03067 / -0.191 | 0.019 / 0.869 / 0.037 | 4.873 |
| DINOv3 训练后 | linear (λ=0.01) | 0.00081 / 0.00231 / 0.710 | 0.00700 / 0.01172 / 0.826 | 0.042 / 0.993 / 0.080 | 0.242 |
| DINOv3 训练后 | MLP (lr=1e-3) | 0.00111 / 0.00257 / 0.641 | 0.00804 / 0.01346 / 0.771 | 0.028 / 0.995 / 0.055 | 1.064 |
| DINOv3 初始 | linear (λ=1e-4) | 0.00153 / 0.00399 / 0.131 | 0.01172 / 0.02078 / 0.453 | 0.025 / 0.989 / 0.048 | 1.818 |
| DINOv3 初始 | MLP (lr=1e-3) | 0.00193 / 0.00424 / 0.019 | 0.01464 / 0.02530 / 0.190 | 0.020 / 0.992 / 0.039 | 2.249 |
| 常数基线 | — | 0.00070 / 0.00429 / -0.002 | 0.01755 / 0.03089 / -0.208 | 0.022 / 0.555 / 0.042 | 5.145 |

结果 json：`outputs/force_probe_eval/fastvit_t12_apple_dist_in1k_init/`、
`outputs/force_probe_eval/fastvit_t12_simmim/` 与
`outputs/force_probe_eval/vit_base_patch16_dinov3_lvd1689m_init/`（初始/SimMIM 组），
训练后组见各运行目录的 `force_probe_eval/`。

解读：

- **multitask 训练对两个 backbone 都带来巨大提升**（linear masked R²：
  FastViT 0.360→0.784，DINOv3 0.453→0.826），没有任何一组训练后变差——
  depth/flow 空间监督确实把"力的空间分布"信息写进了 z；
- **初始权重的可读性差异**：DINOv3 预训练（0.453）> FastViT ImageNet 蒸馏
  （0.360）> FastViT SimMIM（0.274），均显著优于常数基线（-0.208）。
  SimMIM 的域内掩码重建预训练能写入部分线性可读的力分布信息，但略逊于
  ImageNet 蒸馏初始化，更远低于 DINOv3 自监督特征——掩码重建学到的是
  局部纹理/结构先验，对"力的空间分布"这类物理量的可读性提升有限；
- 训练后两条 backbone 收敛到相近水平（0.784 vs 0.826），初始差距被
  multitask 训练抹平；且两个 multitask 运行均为随机初始化起步，
  说明该任务上 backbone 架构/容量差异小于监督信号的贡献；
- **MLP 头一致差于 linear 头**（~4700 样本 × 2100 输出维下过拟合）；
  SimMIM 组的 MLP 甚至塌缩到接近常数预测（masked R²=-0.191，与基线一致），
  说明其特征中力信息的线性可读性本就有限，而非头容量问题；
- 初始组接触检测 Recall 依然很高（0.89~0.99）但 Precision 极低，与训练后
  组一样受力值量级（绝大多数节点 < 0.1 N）主导，F1 的组间差异不宜过度解读。

## 6. 实现落点（确认方案后再动手）

1. 数据：`data/xensim/examples/collect_tactile_data*.py` 加写 `force_grid`（或新写
   离线回放脚本），更新 `stats.json` 生成逻辑与 `docs/collection2_h5_generation.md`
   的 schema 表；
2. 数据集：`H5TactileDataset` 支持 force_grid（spatial 目标保留形状，跳过 z-score）；
3. probe：扩展 `scripts/evaluate_multitask.py --probe` 支持多维 spatial 目标
   （ridge 直接支持；MLP probe 为新增），或新建 `scripts/evaluate_force_probe.py`；
4. 指标：新增 masked/分桶/接触检测/合力一致性指标函数（纯 torch/numpy，沿用
   `evaluate_multitask.py` 的回归指标风格）；
5. 配置：`configs/multitask/*.yaml` 增加 `force_grid: {type: regression, shape: [35,20,3], train: false}` 形式的 eval-only 头声明。
