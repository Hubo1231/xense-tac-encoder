"""数据处理：把原始数据清洗、转换、标注为训练可用的格式。

职责：
- 从 :mod:`src.datasets.crawl` 落盘的原始数据（``data/raw/``）读取；
- 抽帧（RGB / 深度）、裁剪、标准化、信号标注（参考
  ``scripts/build_labeled_dataset.py`` 的 compute_frame_signals）；
- 产出训练数据：``metadata.parquet`` + 图像文件 / 视频帧，
  供 :mod:`src.datasets.labeled_dataset` 的 ``LabeledTactileDataset`` 消费。

约定：
- 处理后数据默认存放目录：``data/processed/``（仓库根，已被 .gitignore 忽略）；
- 每个数据集一个子目录，附带 ``dataset_info.json`` 描述字段与统计信息。
"""

from __future__ import annotations

from pathlib import Path


def build_dataset(raw_dir: Path, processed_dir: Path, dataset_name: str) -> Path:
    """处理一个原始数据集，产出训练数据目录。

    Parameters
    ----------
    raw_dir:
        原始数据根目录（``data/raw``）。
    processed_dir:
        处理后数据根目录（``data/processed``）。
    dataset_name:
        数据集名（对应 ``raw_dir / dataset_name``）。

    Returns
    -------
    Path
        处理后数据集目录（含 metadata.parquet 与图像/帧文件）。
    """
    raise NotImplementedError("待实现：抽帧、标注、写 parquet 的处理流水线")


def split_dataset(dataset_dir: Path, *, val_ratio: float = 0.1) -> None:
    """在数据集目录下生成 train / val 划分（写 train.json / val.json 索引）。"""
    raise NotImplementedError("待实现：按样本划分 train / val")
