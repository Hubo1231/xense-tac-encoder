# collection2 数据集生成细节说明

目标：重新采集一份 xensim 触觉仿真数据集，输出到
`data/xensim/outputs/collection2/`，与现有 `collection/` 相比的变更点：

- `rgb` / `diff` 分辨率 224×224 → **256×256**（适配 FastViT 等 256 输入的 backbone）；
- 新增 **`force_xyz` (3,) float32**：凝胶坐标系下接触合力的三分量（N），
  替代（并保留）标量 `force_resultant`；
- 其余字段（`marker_img` / `marker_pos` / `depth` / `flow` / poses / attrs）保持不变。

---

## 1. 环境准备（独立于训练 .venv）

xensim 与训练环境依赖冲突（`numpy<=1.26.4` vs 训练环境的 2.2.6），**单独建采集环境**：

```bash
cd data/xensim
python3.10 -m venv .venv-collect          # readme 要求 Python 3.9/3.10
source .venv-collect/bin/activate
pip install "xensesdk[viz]"               # 提供 ezgl OpenGL 渲染 / Matrix4x4
pip install -e .                          # 安装 xensim 本体
pip install h5py opencv-python scipy      # 采集脚本依赖
# 可选：FEM 求解加速（无则自动退回 scipy CPU 求解，慢数倍）
# pip install cupy-cuda12x
```

运行依赖检查：

- **OpenGL 上下文**：渲染走 xensesdk 的 ezgl，即使不加 `-v`（无窗口）也需要 GL。
  无显示器机器需 EGL headless 或 `xvfb-run`；先跑
  `python examples/example_simple_sensor.py` 冒烟验证渲染链路；
- **资产确认**（仓库内自带，无需下载）：
  - FEM 模型 `xensim/assets/fem/g1-ws.npz`（35×20=700 表层节点，35×20 网格）
  - 标定查表 `xensim/assets/fem/g1-ws_table.npz`（法线角→RGB 梯度纹理）
  - 物体库 `xensim/assets/obj/poly_sharp/*.STL`（30 个锐边多面体，与现有
    collection 数据一致；如需其他物体库替换 `--model-dir`）。

---

## 2. 代码改动（共 2 处，均在 `data/xensim/examples/collect_tactile_data_poly_sharp.py`）

### 2.1 渲染分辨率 224 → 256

`collect_tactile_data_poly_sharp.py:63`：

```python
RENDER_SIZE = (256, 256)    # 原为 (224, 224)
```

影响范围（已确认）：

- **只影响 `rgb` 和 `diff`**：该参数是 `RGBCamera` 渲染视口尺寸
  （`sensor_simulator.py:199-203`），shader 与分辨率无关；
- `marker_img` 仍 560×320（独立的 `MarkerTextureCamera`）、`depth`/`flow`
  仍 175×100（`DEPTH_SIZE`，不动）、`marker_pos` 仍 20×11——均不受影响；
- 凝胶物理区域（17.3×29.14mm）仍完整渲进方形视口，纵横比压缩特性与
  224 版一致，仅像素密度提高。

### 2.2 写入 force_xyz（三维合力）

在 `_step_save` 的 "Data arrays" 段（现 `:565` 附近）追加：

```python
        # 三维接触合力：FEM 接触节点力按轴求和（凝胶坐标系，单位 N）
        force_xyz = s.fem_solver.contact_force.sum(axis=0).astype(np.float32)
        grp.create_dataset("force_xyz", data=force_xyz)                 # (3,)
        grp.create_dataset("force_resultant", data=np.linalg.norm(force_xyz, keepdims=True).astype(np.float32))  # (1,)
```

依据与约定：

- 数据源：`FEMSimulator.contact_force`（`xensim/fem/simulation.py:267-268`），
  即 `load_force[contact_idx]`——库仑摩擦求解器输出的接触节点三维力，
  物理采样上限为表层 700 节点，实际接触节点数每样本 1~246（现有数据实测）；
- **坐标系**：凝胶（传感器）坐标系，mm/N 单位制（力臂换算那行 `*0.001`
  可佐证力为 N）；
- **符号约定必须实测标定**：历史上注释掉的 `compute_wrench`
  （simulation.py:349-357）写的是 `-np.sum(contact_force, axis=0)`（反力方向）。
  采集前用一个纯垂直下压样本验证：期望法向分量占绝对主导（|normal| ≫ |切向|），
  并确认正负号语义（"凝胶受力"还是"物体受力"），在 h5 文件级 attrs 里写明：
  ```python
  self.h5file.attrs["force_xyz_convention"] = "gel_frame, force_on_gel, N"
  ```
- 保留标量 `force_resultant`（= ‖force_xyz‖）是为了与 collection 的
  下游消费代码（`src/datasets/h5_dataset.py`、stats.json）兼容。

### 2.3 （可选）补写 flow

现有 collection 的 h5 含 `flow (175,100,2)`，但目录内脚本均不写它
（当时的修改版不在仓库）。若 collection2 需要该字段，在同一 save 段补：

```python
        # 表层网格 XY 位移场（相对静止态），双线性上采样到 depth 网格 (175,100,2)
        disp = (s.fem_solver.top_vert - s.fem_solver.top_mesh_xyz)[:, :2]   # (700,2) mm
        disp = disp.reshape(35, 20, 2)
        flow = cv2.resize(disp, (100, 175), interpolation=cv2.INTER_LINEAR)
        grp.create_dataset("flow", data=flow.astype(np.float32))
```

（采样网格语义需与旧数据对齐后再启用，否则建议先不写。）

---

## 3. 输出 HDF5 结构（collection2）

```
tactile_YYYYMMDD_HHMMSS.h5
  attrs:
    total_samples           int     样本总数
    skipped_samples         int     重试耗尽跳过的样本数
    force_xyz_convention    str     新增：力的坐标系/符号/单位说明
  sample_00000/
    attrs:
      object_model          str     模型名（如 poly_sharp_001.STL）
      press_depth_mm        float   实际下压深度（mm）
      target_press_depth_mm float   目标下压深度（mm）
      object_xyz_m          [3]f    物体位置（m，世界系）
      object_rpy_deg        [3]f    物体姿态 RPY（deg）
      contact_xyz_m         [3]f    零接触点（m）
      retry_count           int
      raw_max_penetration_mm / fem_max_penetration_mm /
      fem_contact_pixels / fem_contact_nodes      接触统计
    datasets:
      rgb              (256, 256, 3) uint8    ← 变更
      diff             (256, 256, 3) uint8    ← 变更
      marker_img       (560, 320, 3) uint8
      depth            (175, 100)    float32  mm，>0=侵入
      raw_depth        (175, 100)    float32  （保留 poly_sharp 原版行为）
      marker_pos       (20, 11, 3)   float32  mm，凝胶系
      force_xyz        (3,)          float32  ← 新增：凝胶系三方向合力（N）
      force_resultant  (1,)          float32  ← 新增：‖force_xyz‖（N）
      object_pose      (4, 4)        float32  世界系齐次矩阵
      sensor_pose      (4, 4)        float32  世界系齐次矩阵
      (flow            (175, 100, 2) float32  可选，见 2.3)
```

---

## 4. 采集命令

```bash
cd data/xensim
source .venv-collect/bin/activate

# 冒烟：1 个模型 × 2 样本，可视化检查渲染与力符号
python examples/collect_tactile_data_poly_sharp.py \
  --model-dir xensim/assets/obj/poly_sharp -n 2 -v \
  -o outputs/collection2_smoke

# 正式采集：30 模型 × 200 样本/模型 = 6000（与 collection 单文件量级一致）
python examples/collect_tactile_data_poly_sharp.py \
  --model-dir xensim/assets/obj/poly_sharp \
  -n 200 --seed 42 \
  -o outputs/collection2
```

关键参数（脚本顶部常量，按需调整）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `PRESS_MIN_MM` / `PRESS_MAX_MM` | 0.5 / 1.2 | 下压深度范围（mm） |
| `MAX_PRESS_STEP_MM` | 0.2 | 单步下压上限（保证 FEM 收敛；调小更稳但更慢） |
| `MAX_SAMPLE_RETRIES` | 20 | 单样本重试上限 |
| `MIN_CONTACT_PIXELS` / `MIN_CONTACT_NODES` | 8 / 1 | 有效接触下限 |

---

## 5. 验收检查清单

1. **结构**：随机抽 3 个 h5，逐字段核对 shape/dtype 与第 3 节一致
   （可用 `python examples/read_tactile_data.py <file>` 交互查看）；
2. **力符号**：找一个 `object_rpy_deg` 接近纯垂直的样本，`force_xyz`
   法向分量应主导且符号符合 attrs 里写的约定；
3. **力量级**：`force_resultant` 与 `press_depth_mm` 正相关，1mm 按压
   量级应在 ~10N 附近（对照旧数据 stats：mean 2.8N / 全范围按压 0.5–1.2mm）；
4. **图像**：导出若干 `rgb` PNG 目检——方形视口、无黑边、marker 清晰；
   `diff` 无背景无 marker；
5. **下游兼容**：用 `src/datasets/h5_dataset.py` 加载：
   ```python
   from src.datasets import H5TactileDataset, compute_target_stats
   ds = H5TactileDataset("data/xensim/outputs/collection2/tactile_xxx.h5", None)
   print(len(ds), ds[0]["image"].size)          # 期望 (256, 256)
   stats = compute_target_stats(
       "data/xensim/outputs/collection2/tactile_xxx.h5",
       ["force_resultant", "press_depth_mm"],
       output_path="data/xensim/outputs/collection2/stats.json")
   ```
   （`force_xyz` 为 (3,) 向量目标，`compute_target_stats` 已支持多维展平统计。）

---

## 6. 已知事项 / 风险

- **耗时**：FEM CPU 求解 + 重试机制下，单样本秒级到十秒级；6000 样本
  预计数小时～一天，建议后台跑并先看 smoke 输出；
- **许可**：xensim 为 Binary Runtime License（非商用），数据集用途受其约束；
- **与旧数据不可混训划分**：collection2 与 collection 的物体/位姿分布
  相同但样本独立，train/eval 划分请按文件粒度分开，避免同 episode 泄漏；
- **`flow` 字段语义**：旧数据的 flow 定义未经仓库内代码确证（2.3），
  若下游训练依赖 flow，需先与旧数据生产者确认定义再补写。
