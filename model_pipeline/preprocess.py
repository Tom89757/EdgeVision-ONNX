from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

INPUT_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_path: str | Path) -> np.ndarray:
    """
    统一图像预处理契约：
    1. 纠正 EXIF 旋转方向，转为 RGB
    2. 双线性缩放到 224x224
    3. 归一化到 [0, 1] 后按 ImageNet mean/std 标准化
    4. HWC -> NCHW，输出 shape=(1, 3, 224, 224), dtype=float32
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize(INPUT_SIZE, Image.Resampling.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)  # CHW -> NCHW