from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


YOLO_INPUT_SIZE = 320
YOLO_PAD_VALUE = 114


@dataclass(frozen=True)
class LetterboxMeta:
    """记录 Letterbox 的几何参数，供 PyTorch/ONNX 后处理与坐标反变换复用。"""

    original_width: int
    original_height: int
    input_width: int
    input_height: int
    resized_width: int
    resized_height: int
    ratio: float
    pad_x: int
    pad_y: int

    def to_dict(self) -> dict:
        return asdict(self)


def _round_half_up(value: float) -> int:
    """固定取整规则，保证 Python 与 Android Math.round 使用同一语义。"""
    return int(value + 0.5)


def compute_letterbox_meta(width: int, height: int, input_size: int = YOLO_INPUT_SIZE) -> LetterboxMeta:
    if width <= 0 or height <= 0:
        raise ValueError(f"非法图像尺寸：{width}x{height}")
    ratio = min(input_size / width, input_size / height)
    resized_width = max(1, min(input_size, _round_half_up(width * ratio)))
    resized_height = max(1, min(input_size, _round_half_up(height * ratio)))
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    return LetterboxMeta(
        original_width=width,
        original_height=height,
        input_width=input_size,
        input_height=input_size,
        resized_width=resized_width,
        resized_height=resized_height,
        ratio=ratio,
        pad_x=pad_x,
        pad_y=pad_y,
    )


def letterbox_image(image_path: str | Path, input_size: int = YOLO_INPUT_SIZE) -> tuple[np.ndarray, LetterboxMeta]:
    """读取真实图片并执行 YOLOv8 Letterbox，输出 [1, 3, 320, 320] float32 NCHW。"""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    meta = compute_letterbox_meta(image.width, image.height, input_size=input_size)
    resized = image.resize((meta.resized_width, meta.resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), (YOLO_PAD_VALUE,) * 3)
    canvas.paste(resized, (meta.pad_x, meta.pad_y))

    array = np.asarray(canvas, dtype=np.float32) / 255.0  # YOLOv8 只做 [0,1] 归一化
    array = np.ascontiguousarray(np.transpose(array, (2, 0, 1)))  # HWC -> CHW
    return np.expand_dims(array, axis=0), meta


def scale_boxes_to_original(boxes_xyxy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """将 Letterbox 输入坐标反变换到原图坐标，并裁剪到图像边界。"""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    if boxes.size == 0:
        return boxes
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta.pad_x) / meta.ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta.pad_y) / meta.ratio
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(meta.original_width))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(meta.original_height))
    return boxes


def preprocess_image(image_path: str | Path) -> np.ndarray:
    """供校准读取器使用的简洁入口；几何参数可按需通过 letterbox_image 获取。"""
    return letterbox_image(image_path)[0]
