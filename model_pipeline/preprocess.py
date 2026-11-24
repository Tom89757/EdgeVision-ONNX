from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

INPUT_SIZE = (224, 224)
RESIZE_SHORT_SIDE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_size(size: tuple[int, int]) -> tuple[int, int]:
    """短边缩放到 RESIZE_SHORT_SIDE，长边等比放大，保持长宽比。"""
    width, height = size
    scale = RESIZE_SHORT_SIDE / min(width, height)
    return (max(RESIZE_SHORT_SIDE, round(width * scale)),
            max(RESIZE_SHORT_SIDE, round(height * scale)))


def _center_crop(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """中心裁剪到目标尺寸。"""
    width, height = img.size
    left = (width - size[0]) // 2
    top = (height - size[1]) // 2
    return img.crop((left, top, left + size[0], top + size[1]))


def preprocess_image(image_path: str | Path) -> np.ndarray:
    """
    统一图像预处理契约：
    1. 纠正 EXIF 旋转方向，转为 RGB
    2. 保持长宽比：短边双线性缩放到 256，再中心裁剪 224x224
       （与 torchvision 评测标准一致，避免长宽比悬殊的图片被直接拉伸变形）
    3. 归一化到 [0, 1] 后按 ImageNet mean/std 标准化
    4. HWC -> NCHW，输出 shape=(1, 3, 224, 224), dtype=float32
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize(_resize_size(img.size), Image.Resampling.BILINEAR)
    img = _center_crop(img, INPUT_SIZE)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)  # CHW -> NCHW