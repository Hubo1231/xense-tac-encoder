"""数据爬取：从外部来源下载 / 爬取原始数据。

职责：
- 从数据源（HuggingFace Hub、内部服务器、设备采集云端等）下载原始数据；
- 断点续传、去重、完整性校验（checksum）；
- 将原始文件落盘到统一的原始数据目录，供 :mod:`src.datasets.process` 消费。

约定：
- 原始数据默认存放目录：``data/raw/``（仓库根，已被 .gitignore 忽略）；
- 每个数据集一个子目录，附带 ``manifest.json`` 记录来源 URL / 下载时间 /
  checksum，便于追溯与增量更新。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def download(url: str, dest: Path, *, checksum: str | None = None) -> Path:
    """下载单个文件到 ``dest``。

    Parameters
    ----------
    url:
        文件 URL（http(s)）。
    dest:
        目标文件路径。
    checksum:
        可选 sha256 校验值；提供时下载完成后校验，不匹配则删除并报错。

    Returns
    -------
    Path
        下载后的文件路径。
    """
    raise NotImplementedError("待实现：按需接入 urllib / requests / hf_hub_download 等下载逻辑")


def sync_dataset(name: str, remote_root: str, local_root: Path) -> Path:
    """增量同步一个数据集到本地。

    对 ``remote_root`` 下的清单逐项检查本地是否已有（存在 + checksum 一致），
    缺失或损坏的才下载，并在 ``local_root / name / manifest.json`` 记录元信息。

    Returns
    -------
    Path
        数据集本地根目录。
    """
    raise NotImplementedError("待实现：增量同步逻辑")


def write_manifest(dataset_dir: Path, entries: list[dict[str, Any]]) -> None:
    """把下载元信息写入 ``dataset_dir/manifest.json``。"""
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def sha256_of(path: Path) -> str:
    """计算文件的 sha256（分块读取，适合大文件）。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
