"""触觉图像数据集。

注意：触觉传感器的塑料外壳含有重要边界参考信息，
不做 CenterCrop / RandomCrop，仅 Resize + Normalize。
"""
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def default_transform(image_size: int = 224) -> Callable:
    """ImageNet 标准归一化，匹配主流预训练 backbone。"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def list_images(root: Union[str, Path]) -> List[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


class TactileDataset(Dataset):
    """从路径列表或文件夹加载触觉 RGB 图像。"""

    def __init__(
        self,
        source: Union[str, Path, Sequence[Union[str, Path]]],
        image_size: int = 224,
        transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(source, (str, Path)):
            self.image_paths = list_images(source)
        else:
            self.image_paths = [Path(p) for p in source]
        if not self.image_paths:
            raise ValueError(f"未在 {source} 找到图像")
        self.transform = transform or default_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


class RandomTactileDataset(Dataset):
    """伪数据集，便于在没有真实数据时跑通流水线。"""

    def __init__(self, length: int = 64, image_size: int = 224) -> None:
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        # 已归一化的张量（不再走 transform），数值范围近似训练分布
        return torch.randn(3, self.image_size, self.image_size)
