"""PNG + Parquet 带标注触觉数据集（multitask 预训练用）。

数据格式约定（由 scripts/build_labeled_dataset.py 产出）：

    image_root/                       # 对应配置的 data_dir
        images/*.png                  # 触觉图像帧（懒加载）
        metadata.parquet              # 每帧一行，至少包含：
                                      #   image_path     相对 image_root 的 PNG 路径
                                      #   episode_index  帧所属 episode 序号
                                      #   timestamp      帧时间戳（秒）
                                      #   frame_index    帧在 episode 内的序号
                                      #   <信号列…>      回归目标（float）
                                      #   <标签列…>      分类目标（int）
        stats.json                    # 回归目标的 z-score 统计：
                                      #   {"<列名>": {"mean": ..., "std": ...}, ...}

train/eval 划分复刻 ``split_image_paths`` 的确定性 seed shuffle，但作用在
parquet 行号上（同一行号集合，train 与 eval 脚本划分必然一致）。

回归目标：stats.json 存在时按列名 z-score 标准化为 float32；分类目标输出
int64 标量。``train: false`` 的 eval-only 目标列同样加载（评测脚本 probe 用）。
"""
from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

if TYPE_CHECKING:  # 避免运行期强依赖；真实对象由调用方（训练/评测脚本）传入
    import pyarrow as pa

    from src.models.multitask import HeadSpec

_logger = logging.getLogger(__name__)

SPLITS: tuple[str, ...] = ("train", "eval")
# metadata.parquet 中不参与训练目标、也不计入 stats.json 的固定元数据列
METADATA_COLUMNS: tuple[str, ...] = ("image_path", "episode_index", "timestamp", "frame_index")


def _import_pyarrow() -> Any:
    """延迟导入 pyarrow，缺包时给出中文安装提示（保持 --help 不依赖它）。"""
    try:
        import pyarrow
    except ImportError as exc:  # pragma: no cover - 取决于当前环境
        raise ImportError(
            "缺少 pyarrow（读写 metadata.parquet 需要）。请用 uv 安装：\n"
            "  uv pip install pyarrow\n"
            "或运行 scripts/install.sh 一键配置环境。"
        ) from exc
    return pyarrow


def load_metadata(metadata_path: str | Path) -> "pa.Table":
    """读取 metadata.parquet 为 pyarrow.Table。"""
    pa = _import_pyarrow()
    import pyarrow.parquet as pq

    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"metadata.parquet 不存在: {path}")
    return pq.read_table(path)


def split_row_indices(n: int, *, eval_ratio: float, seed: int) -> dict[str, list[int]]:
    """确定性 train/eval 行号划分，逻辑复刻 ``split_image_paths``。

    用 ``random.Random(seed)`` shuffle 行号副本，保证跨脚本/跨进程一致。
    """
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f"eval_ratio must be in (0, 1), got {eval_ratio}.")
    if n < 2:
        raise ValueError(f"Need at least 2 rows to split, got {n}.")

    indices = list(range(n))
    random.Random(int(seed)).shuffle(indices)
    n_eval = max(1, int(round(n * float(eval_ratio))))
    n_eval = min(n_eval, n - 1)  # 保证 train 至少一行
    return {
        "train": sorted(indices[n_eval:]),
        "eval": sorted(indices[:n_eval]),
    }


class LabeledTactileDataset(TorchDataset):
    """PNG 图像 + parquet 行目标的 dict-batch 数据集。

    ``__getitem__`` 返回 ``{"image": transform(PIL), "targets": {name: tensor}}``；
    torch ``default_collate`` 原生支持嵌套 dict，无需自定义 collate_fn。

    ``target_specs`` 需同时包含 train 与 eval-only 目标（缺列直接 ValueError）。
    """

    def __init__(
        self,
        image_root: str | Path,
        metadata_path: str | Path,
        target_specs: Mapping[str, "HeadSpec"],
        *,
        split: str,
        eval_ratio: float,
        seed: int,
        transform: Callable | None,
        stats_path: str | Path | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}.")
        if not target_specs:
            raise ValueError("target_specs 不能为空。")
        self.image_root = Path(image_root)
        if not self.image_root.is_dir():
            raise NotADirectoryError(f"image_root 不存在或不是目录: {self.image_root}")
        self.transform = transform
        self.target_specs = dict(target_specs)

        table = load_metadata(metadata_path)
        columns = set(table.column_names)
        if "image_path" not in columns:
            raise ValueError(f"metadata.parquet 缺少 image_path 列，现有列: {sorted(columns)}。")
        missing = [name for name in self.target_specs if name not in columns]
        if missing:
            raise ValueError(
                f"metadata.parquet 缺少目标列 {missing}，现有列: {sorted(columns)}。"
            )

        splits = split_row_indices(table.num_rows, eval_ratio=eval_ratio, seed=seed)
        self._rows: list[int] = splits[split]
        if not self._rows:
            raise ValueError(f"split={split!r} 划分后为空（共 {table.num_rows} 行）。")

        # 只把需要的列物化成 python list，图像保持懒加载
        self._image_paths: list[str] = table.column("image_path").to_pylist()
        self._columns: dict[str, list[Any]] = {
            name: table.column(name).to_pylist() for name in self.target_specs
        }

        # 回归目标 z-score 统计；stats_path 缺省取 metadata.parquet 旁的 stats.json
        stats_file = Path(stats_path) if stats_path is not None else Path(metadata_path).with_name("stats.json")
        self.target_stats: dict[str, dict[str, float]] = {}
        if stats_file.exists():
            with stats_file.open("r", encoding="utf-8") as f:
                self.target_stats = json.load(f)
            for name, spec in self.target_specs.items():
                if spec.type == "regression" and name not in self.target_stats:
                    raise ValueError(f"stats.json 缺少回归目标 {name!r} 的 mean/std。")
        else:
            _logger.warning("未找到 %s，回归目标不做 z-score 标准化。", stats_file)

    def __len__(self) -> int:
        return len(self._rows)

    def _normalize(self, name: str, value: Any) -> Any:
        """按 stats.json 对回归目标做 z-score；无 stats 时原样返回。"""
        stats = self.target_stats.get(name)
        if stats is None:
            return value
        mean, std = float(stats["mean"]), float(stats["std"])
        if std <= 0.0:  # 常数列（如占位信号）：退化为只减均值，避免除零
            std = 1.0
        if isinstance(value, (list, tuple)):
            return [(float(v) - mean) / std for v in value]
        return (float(value) - mean) / std

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        image = Image.open(self.image_root / self._image_paths[row]).convert("RGB")

        targets: dict[str, torch.Tensor] = {}
        for name, spec in self.target_specs.items():
            value = self._columns[name][row]
            if value is None:
                raise ValueError(f"目标列 {name!r} 第 {row} 行为空（null）。")
            if spec.type == "regression":
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
