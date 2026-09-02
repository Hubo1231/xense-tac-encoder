"""HDF5 触觉采集数据集（xensim collection 格式，multitask 预训练用）。

数据格式约定（data/xensim/outputs/collection{,2}/*.h5）：

    <file>.h5
        sample_00000/                 # 每个样本一个 group，名字零填充排序
            rgb              (256, 256, 3) uint8     # collection2 触觉 RGB 图像
            diff             (256, 256, 3) uint8     # 与参考帧差分图
            marker_img       (560, 320, 3) uint8     # 标记点相机原图
            depth            (175, 100)    float32   # 深度（mm）
            flow             (175, 100, 2) float32   # 标记点位移场
            marker_pos       (20, 11, 3)   float64   # 标记点三维坐标
            force_xyz        (3,)          float32   # 凝胶系三方向接触力（N）
            force_resultant  (1,)          float32   # 合力（N）
            force_grid       (35, 20, 3)   float32   # 逐节点三维接触力（N，可选；采集失败样本缺失）
            contact_mask     (35, 20)      bool      # 节点是否接触（可选，随 force_grid 存在）
            object_pose      (4, 4)        float32   # 物体位姿
            sensor_pose      (4, 4)        float32   # 传感器位姿
            attrs: contact_xyz_m / object_model / object_rpy_deg /
                   object_xyz_m / press_depth_mm     # 样本级元信息
        sample_00001/ ...

与 :class:`~src.datasets.labeled_dataset.LabeledTactileDataset` 对齐：
``__getitem__`` 返回 ``{"image": transform(PIL), "targets": {name: tensor}}``，
train/eval 划分复用 ``split_row_indices``（作用在全局样本序号上，seed 一致时
与其他数据集划分逻辑相同）。

回归目标取自 group 内的 dataset（如 ``force_resultant``）或 group attrs
（如 ``press_depth_mm``），多维值会被展平；stats.json 存在时按列名 z-score
标准化（格式与 labeled_dataset 相同，可用 ``compute_target_stats`` 生成）。
h5 文件句柄按进程懒打开并缓存，DataLoader 多 worker 下安全。
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from .labeled_dataset import SPLITS, split_row_indices

if TYPE_CHECKING:  # 避免运行期强依赖；真实对象由调用方（训练/评测脚本）传入
    from src.models.multitask import HeadSpec

_logger = logging.getLogger(__name__)


def _import_h5py() -> Any:
    """延迟导入 h5py，缺包时给出中文安装提示（保持 --help 不依赖它）。"""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - 取决于当前环境
        raise ImportError(
            "缺少 h5py（读取 .h5 采集数据需要）。请安装：\n"
            "  pip install h5py\n"
            "或运行 pip install -r requirements.txt 安装全部依赖。"
        ) from exc
    return h5py


def list_h5_samples(h5_path: str | Path) -> list[str]:
    """列出 h5 文件内全部样本 group 名（按名字排序，即采集顺序）。"""
    h5py = _import_h5py()
    with h5py.File(h5_path, "r") as f:
        return sorted(k for k, v in f.items() if isinstance(v, h5py.Group))


class H5TactileDataset(TorchDataset):
    """HDF5 触觉采集数据的 dict-batch 数据集（懒加载，支持多文件拼接）。

    Parameters
    ----------
    h5_paths:
        单个 h5 路径，或 h5 路径列表（多文件按给定顺序拼接为一个数据集）。
    target_specs:
        训练/评测目标，键为目标名，需能在样本 group 的 dataset 或 attrs 中
        找到同名字段（如 ``force_resultant`` / ``press_depth_mm``）。
        可为空 mapping，表示只读图像不读目标。
    split:
        ``None`` 使用全部样本；否则为 ``"train"`` / ``"eval"``，按
        ``split_row_indices`` 在全局样本序号上确定性划分。
    image_key:
        作为 ``"image"`` 返回的 dataset 名，默认 ``"rgb"``。
    stats_path:
        回归目标 z-score 统计（stats.json）；缺省时取第一个 h5 旁的
        ``stats.json``，不存在则不标准化。
    require_keys:
        可选的必备字段名列表（如 ``["force_grid"]``）。在 split 划分**之后**
        过滤掉缺少任一字段的样本（不重新划分，保证与训练 split 一致），
        用于跳过采集失败、缺少该字段的样本。
    """

    def __init__(
        self,
        h5_paths: str | Path | Sequence[str | Path],
        target_specs: Mapping[str, "HeadSpec"] | None = None,
        *,
        split: str | None = None,
        eval_ratio: float = 0.1,
        seed: int = 42,
        transform: Callable | None = None,
        image_key: str = "rgb",
        stats_path: str | Path | None = None,
        require_keys: Sequence[str] | None = None,
    ) -> None:
        if isinstance(h5_paths, (str, Path)):
            h5_paths = [h5_paths]
        self.h5_paths = [Path(p) for p in h5_paths]
        if not self.h5_paths:
            raise ValueError("h5_paths 不能为空。")
        for p in self.h5_paths:
            if not p.is_file():
                raise FileNotFoundError(f"h5 文件不存在: {p}")
        if split is not None and split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS} or None, got {split!r}.")

        self.transform = transform
        self.image_key = image_key
        self.target_specs = dict(target_specs or {})
        self._files: dict[int, Any] = {}  # 按进程缓存的 h5py.File 句柄

        # 全局样本索引：(文件序号, 样本 group 名)，样本名列表在构造时一次性读入
        self._samples: list[tuple[int, str]] = [
            (file_idx, name)
            for file_idx, path in enumerate(self.h5_paths)
            for name in list_h5_samples(path)
        ]
        if not self._samples:
            raise ValueError(f"未在 {[str(p) for p in self.h5_paths]} 中找到任何样本 group。")

        if split is None:
            self._rows = list(range(len(self._samples)))
        else:
            self._rows = split_row_indices(
                len(self._samples), eval_ratio=eval_ratio, seed=seed
            )[split]
            if not self._rows:
                raise ValueError(
                    f"split={split!r} 划分后为空（共 {len(self._samples)} 个样本）。"
                )

        # 必备字段过滤（如 force_grid）：在 split 之后按行剔除缺字段的样本，
        # 不影响划分本身，保证与未过滤时的 train/eval 成员关系一致。
        if require_keys:
            keys = tuple(require_keys)
            n_before = len(self._rows)
            self._rows = [
                row
                for row in self._rows
                if all(key in self._get_group(row) for key in keys)
            ]
            if not self._rows:
                raise ValueError(
                    f"require_keys={keys!r} 过滤后无样本（过滤前 {n_before} 行）。"
                )
            _logger.info(
                "require_keys=%s: %d -> %d 样本（剔除 %d 个缺字段样本）。",
                keys, n_before, len(self._rows), n_before - len(self._rows),
            )

        # 回归目标 z-score 统计；stats_path 缺省取第一个 h5 旁的 stats.json
        stats_file = (
            Path(stats_path)
            if stats_path is not None
            else self.h5_paths[0].with_name("stats.json")
        )
        self.target_stats: dict[str, dict[str, float]] = {}
        if stats_file.exists():
            with stats_file.open("r", encoding="utf-8") as f:
                self.target_stats = json.load(f)
            for name, spec in self.target_specs.items():
                if (
                    spec.type == "regression"
                    and getattr(spec, "normalize", True)
                    and name not in self.target_stats
                ):
                    raise ValueError(f"stats.json 缺少回归目标 {name!r} 的 mean/std。")
        elif any(
            s.type == "regression" and getattr(s, "normalize", True)
            for s in self.target_specs.values()
        ):
            _logger.warning("未找到 %s，回归目标不做 z-score 标准化。", stats_file)

    def __len__(self) -> int:
        return len(self._rows)

    def _get_group(self, row: int) -> Any:
        """取第 ``row`` 个全局样本的 h5 group（句柄按文件缓存，随进程存活）。"""
        file_idx, name = self._samples[row]
        if file_idx not in self._files:
            h5py = _import_h5py()
            self._files[file_idx] = h5py.File(self.h5_paths[file_idx], "r")
        return self._files[file_idx][name]

    def __getstate__(self) -> dict[str, Any]:
        # h5py.File 句柄不可 pickle，spawn 方式起 worker 时剔除（worker 内重新懒打开）
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def _read_target(self, group: Any, name: str, row: int, *, flatten: bool = True) -> Any:
        """从样本 group 读目标值：先查 dataset，再查 attrs。

        ``flatten=False`` 时保留 ndarray 形状，供 depth/flow 等空间回归目标
        直接参与卷积损失；默认展平成 list，保持与旧标量/向量目标兼容。
        """
        if name in group:
            value = group[name][()]
        elif name in group.attrs:
            value = group.attrs[name]
        else:
            raise ValueError(
                f"样本 {group.name} 中找不到目标 {name!r}"
                f"（datasets: {sorted(group.keys())}, attrs: {sorted(group.attrs.keys())}）。"
            )
        if isinstance(value, np.ndarray):
            if flatten:
                value = value.reshape(-1).tolist()
            else:
                value = np.asarray(value)
        if isinstance(value, (list, tuple)) and len(value) == 0:
            raise ValueError(f"目标 {name!r} 第 {row} 行为空数组。")
        return value

    def _normalize(self, name: str, value: Any) -> Any:
        """按 stats.json 对回归目标做 z-score；无 stats 时原样返回。"""
        stats = self.target_stats.get(name)
        if stats is None:
            return value
        mean, std = float(stats["mean"]), float(stats["std"])
        if std <= 0.0:  # 常数列（如占位信号）：退化为只减均值，避免除零
            std = 1.0
        if isinstance(value, np.ndarray):
            return (value.astype(np.float32) - mean) / std
        if isinstance(value, (list, tuple)):
            return [(float(v) - mean) / std for v in value]
        return (float(value) - mean) / std

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        group = self._get_group(row)

        if self.image_key not in group:
            raise ValueError(
                f"样本 {group.name} 缺少图像 dataset {self.image_key!r}"
                f"（现有: {sorted(group.keys())}）。"
            )
        image = Image.fromarray(group[self.image_key][()]).convert("RGB")

        targets: dict[str, torch.Tensor] = {}
        for name, spec in self.target_specs.items():
            value = self._read_target(
                group, name, row, flatten=not bool(getattr(spec, "spatial", False))
            )
            if spec.type == "regression":
                if getattr(spec, "normalize", True):
                    value = self._normalize(name, value)
                tensor = torch.as_tensor(value, dtype=torch.float32)
                if tensor.ndim == 0:
                    tensor = tensor.unsqueeze(0)  # 标量 -> (1,)，与头的 (B, dim=1) 对齐
                targets[name] = tensor
            else:  # classification
                targets[name] = torch.tensor(int(value), dtype=torch.int64)

        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "targets": targets}

    def close(self) -> None:
        """显式关闭本进程缓存的 h5 句柄（进程退出时也可由析构兜底）。"""
        for f in self._files.values():
            f.close()
        self._files.clear()

    def __del__(self) -> None:  # pragma: no cover - 兜底清理
        try:
            self.close()
        except Exception:
            pass


def compute_target_stats(
    h5_paths: str | Path | Sequence[str | Path],
    target_names: Sequence[str],
    *,
    output_path: str | Path | None = None,
    split: str | None = None,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """遍历全部样本，计算各回归目标的 mean/std（与 labeled_dataset 的 stats.json 同格式）。

    多维目标（展平后）按全部元素汇总统计。``output_path`` 提供时把结果写成
    JSON 并原样返回该 dict。
    """
    dataset = H5TactileDataset(
        h5_paths,
        None,
        split=split,
        eval_ratio=eval_ratio,
        seed=seed,
    )
    sums: dict[str, float] = {name: 0.0 for name in target_names}
    sq_sums: dict[str, float] = {name: 0.0 for name in target_names}
    counts: dict[str, int] = {name: 0 for name in target_names}
    try:
        rows = dataset._rows if split is not None else list(range(len(dataset._samples)))
        for row in rows:
            group = dataset._get_group(row)
            for name in target_names:
                value = dataset._read_target(group, name, row, flatten=False)
                values = (
                    value
                    if isinstance(value, (np.ndarray, list, tuple))
                    else [value]
                )
                arr = np.asarray(values, dtype=np.float64)
                sums[name] += float(arr.sum())
                sq_sums[name] += float((arr**2).sum())
                counts[name] += int(arr.size)
    finally:
        dataset.close()

    stats: dict[str, dict[str, float]] = {}
    for name in target_names:
        if counts[name] == 0:
            raise ValueError(f"目标 {name!r} 没有任何有效值。")
        mean = sums[name] / counts[name]
        var = sq_sums[name] / counts[name] - mean**2
        stats[name] = {"mean": mean, "std": float(np.sqrt(max(var, 0.0)))}

    if output_path is not None:
        path = Path(output_path)
        path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats
