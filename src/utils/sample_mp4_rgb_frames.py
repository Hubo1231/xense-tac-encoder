"""Sample RGB images from an MP4 file using LeRobot's video decoder.

The sampling path mirrors LeRobotDataset's MP4 read path: it builds query
timestamps, calls ``lerobot.datasets.video_utils.decode_video_frames``, and
saves the returned RGB tensors as images.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional display dependency.
    tqdm = None

try:
    from lerobot.datasets.video_utils import (
        decode_video_frames,
        get_safe_default_codec,
        get_video_duration_in_s,
        get_video_info,
    )
except ImportError as exc:  # pragma: no cover - depends on the active env.
    raise SystemExit(
        "未找到 lerobot。请在包含 lerobot 的环境中运行，例如：\n"
        "  conda run -n eval-encoder python src/utils/sample_mp4_rgb_frames.py --video path/to/video.mp4"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data"


class SimpleProgressBar:
    """Small progress fallback when tqdm is not installed."""

    def __init__(self, total: int, desc: str, width: int = 30) -> None:
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0

    def __enter__(self) -> "SimpleProgressBar":
        self._draw()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    def update(self, n: int = 1) -> None:
        self.current = min(self.total, self.current + n)
        self._draw()

    def _draw(self) -> None:
        ratio = self.current / max(self.total, 1)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stdout.write(f"\r{self.desc}: [{bar}] {self.current}/{self.total}")
        sys.stdout.flush()


def progress_bar(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit="img", file=sys.stdout)
    return SimpleProgressBar(total=total, desc=desc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 LeRobotDataset 的 MP4 解码逻辑从视频中采样 RGB PNG 图像。"
    )
    parser.add_argument("--video", required=True, help="输入 MP4 视频路径。")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"输出根目录，默认: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="输出子目录名和文件名前缀；默认使用视频文件名 stem。",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="按原视频帧间隔采样。1 表示逐帧采样；设置 --target-fps 后不使用该参数。",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="按目标 FPS 采样；设置后会忽略 --stride。",
    )
    parser.add_argument("--start-s", type=float, default=0.0, help="采样起始时间，单位秒。")
    parser.add_argument("--end-s", type=float, default=None, help="采样结束时间，单位秒。")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最多保存多少张图像；默认不限制。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="每次调用 LeRobot 解码函数的时间戳数量。",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="与 LeRobotDataset 默认一致的时间戳容差，单位秒。",
    )
    parser.add_argument(
        "--backend",
        choices=("torchcodec", "pyav", "video_reader"),
        default=None,
        help="LeRobot 视频解码 backend；默认使用 LeRobot 的安全默认值。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的 PNG 文件。",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.stride < 1:
        raise SystemExit("--stride 必须 >= 1")
    if args.target_fps is not None and args.target_fps <= 0:
        raise SystemExit("--target-fps 必须 > 0")
    if args.start_s < 0:
        raise SystemExit("--start-s 必须 >= 0")
    if args.end_s is not None and args.end_s <= args.start_s:
        raise SystemExit("--end-s 必须大于 --start-s")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames 必须 >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size 必须 >= 1")
    if args.tolerance_s <= 0:
        raise SystemExit("--tolerance-s 必须 > 0")


def _video_fps(video_path: Path) -> float:
    info = get_video_info(video_path)
    fps = float(info.get("video.fps", 0.0))
    if fps <= 0:
        raise RuntimeError(f"无法从视频元数据中读取 FPS: {video_path}")
    return fps


def build_query_timestamps(
    *,
    fps: float,
    duration_s: float,
    start_s: float,
    end_s: float | None,
    stride: int,
    target_fps: float | None,
    max_frames: int | None,
) -> list[float]:
    """Build timestamps in the same units expected by LeRobot's decoder."""
    stop_s = min(end_s if end_s is not None else duration_s, duration_s)
    if stop_s <= start_s:
        return []

    first_frame = math.ceil(start_s * fps)
    last_frame = math.floor(stop_s * fps)
    if target_fps is not None:
        if target_fps > fps:
            raise ValueError(f"--target-fps 不能大于视频 FPS ({fps:.6g})")
        step = fps / target_fps
        frame_count = max(0, math.ceil((last_frame - first_frame) / step))
        frame_indices = [first_frame + round(i * step) for i in range(frame_count)]
        frame_indices = sorted({idx for idx in frame_indices if first_frame <= idx < last_frame})
    else:
        frame_indices = list(range(first_frame, last_frame, stride))
    timestamps = [idx / fps for idx in frame_indices]

    if max_frames is not None:
        timestamps = timestamps[:max_frames]
    return timestamps


def _to_uint8_hwc_rgb(frame: torch.Tensor) -> torch.Tensor:
    """Convert a LeRobot image tensor to uint8 RGB in HWC order."""
    frame = frame.detach().cpu()
    if frame.ndim != 3:
        raise ValueError(f"期望单帧是 3 维 tensor，实际 shape={tuple(frame.shape)}")

    if frame.shape[0] in (1, 3, 4):
        frame = frame.permute(1, 2, 0)
    elif frame.shape[-1] in (1, 3, 4):
        pass
    else:
        raise ValueError(f"无法识别图像通道维度，shape={tuple(frame.shape)}")

    if frame.shape[-1] == 1:
        frame = frame.repeat(1, 1, 3)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]

    if frame.dtype.is_floating_point:
        frame = frame.clamp(0.0, 1.0).mul(255.0).round()
    else:
        frame = frame.clamp(0, 255)
    return frame.to(torch.uint8).contiguous()


def save_frame(frame: torch.Tensor, output_path: Path) -> None:
    if frame.shape[0] in (1, 3, 4):
        height, width = int(frame.shape[1]), int(frame.shape[2])
    elif frame.shape[-1] in (1, 3, 4):
        height, width = int(frame.shape[0]), int(frame.shape[1])
    else:
        raise ValueError(f"无法识别图像尺寸，shape={tuple(frame.shape)}")

    image = Image.fromarray(_to_uint8_hwc_rgb(frame).numpy(), mode="RGB")
    image.save(output_path)


def sample_video(args: argparse.Namespace) -> int:
    _validate_args(args)

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"输入文件不是 MP4: {video_path}")

    fps = _video_fps(video_path)
    duration_s = get_video_duration_in_s(video_path)
    timestamps = build_query_timestamps(
        fps=fps,
        duration_s=duration_s,
        start_s=args.start_s,
        end_s=args.end_s,
        stride=args.stride,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
    )
    if not timestamps:
        raise RuntimeError("没有生成任何采样时间戳，请检查 start/end/stride/target-fps 参数。")

    prefix = args.prefix or video_path.stem
    output_dir = Path(args.output_dir).expanduser().resolve() / prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = args.backend or get_safe_default_codec()
    print(f"video: {video_path}", flush=True)
    print(f"fps: {fps:.6g}, duration_s: {duration_s:.6g}, backend: {backend}", flush=True)
    print(f"total frames to sample: {len(timestamps)}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)

    saved = 0
    with progress_bar(total=len(timestamps), desc="sampling") as progress:
        for batch_start in range(0, len(timestamps), args.batch_size):
            batch_ts = timestamps[batch_start : batch_start + args.batch_size]
            frames = decode_video_frames(video_path, batch_ts, args.tolerance_s, backend=backend)
            if frames.shape[0] != len(batch_ts):
                raise RuntimeError(f"解码帧数不匹配: got={frames.shape[0]}, expected={len(batch_ts)}")

            for offset, frame in enumerate(frames):
                sample_idx = batch_start + offset
                output_path = output_dir / f"{prefix}_{sample_idx:06d}.png"
                if output_path.exists() and not args.overwrite:
                    raise FileExistsError(f"输出文件已存在，使用 --overwrite 覆盖: {output_path}")
                save_frame(frame, output_path)
                saved += 1
                progress.update(1)

    print(f"saved {saved} RGB frames to {output_dir}")
    return saved


def main() -> None:
    sample_video(parse_args())


if __name__ == "__main__":
    main()
