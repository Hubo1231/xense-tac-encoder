"""Data acquisition, processing and dataset loading package.

本包负责数据全链路：

1. 爬取（:mod:`src.datasets.crawl`）——从外部来源下载 / 爬取原始数据
   （如触觉图像数据集、LeRobot 数据集、设备采集文件的云端备份等）；
2. 处理（:mod:`src.datasets.process`）——将原始数据清洗、转换、标注，
   产出训练数据（如 metadata.parquet、RGB 帧、标注信号等）；
3. 读取（:mod:`src.datasets.labeled_dataset`）——训练/评测时消费
   metadata.parquet + 懒加载 PNG，产出 dict batch（LabeledTactileDataset）；
   或直接消费 xensim 采集的 HDF5 文件（:mod:`src.datasets.h5_dataset`
   的 H5TactileDataset）。

与 ``scripts/`` 的关系：CLI 入口放在 ``scripts/``（如
``scripts/build_labeled_dataset.py``），可复用的逻辑放在本包。
"""

from . import crawl, h5_dataset, labeled_dataset, process
from .h5_dataset import H5TactileDataset, compute_target_stats, list_h5_samples
from .labeled_dataset import LabeledTactileDataset, load_metadata, split_row_indices

__all__ = [
    "crawl",
    "process",
    "labeled_dataset",
    "h5_dataset",
    "LabeledTactileDataset",
    "H5TactileDataset",
    "load_metadata",
    "split_row_indices",
    "list_h5_samples",
    "compute_target_stats",
]
