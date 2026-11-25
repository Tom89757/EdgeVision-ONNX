from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from export_onnx import ARTIFACTS_DIR, ONNX_FP32_PATH
from quantize_ptq import ONNX_INT8_PATH
from preprocess import preprocess_image


PIPELINE_DIR = Path(__file__).resolve().parent
CALIB_DIR = PIPELINE_DIR / "calib_images"
TEST_DIR = PIPELINE_DIR / "test_images"
REPORT_PATH = ARTIFACTS_DIR / "alignment_report.json"
CALIB_COUNT = 40
TEST_COUNT = 30
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_holdout_split() -> list[Path]:
    if not CALIB_DIR.is_dir() or not TEST_DIR.is_dir():
        raise FileNotFoundError("calib_images 与 test_images 必须同时存在；请先运行 prepare_datasets.py")
    calibration = list_images(CALIB_DIR)
    test_images = list_images(TEST_DIR)
    if len(calibration) != CALIB_COUNT:
        raise RuntimeError(f"校准集必须恰好 {CALIB_COUNT} 张，当前 {len(calibration)} 张")
    if len(test_images) != TEST_COUNT:
        raise RuntimeError(f"黄金测试集必须恰好 {TEST_COUNT} 张，当前 {len(test_images)} 张")
    overlap = {sha256_file(path) for path in calibration} & {sha256_file(path) for path in test_images}
    if overlap:
        raise RuntimeError(f"校准集与测试集存在 {len(overlap)} 张相同图片，拒绝数据泄漏")
    return test_images


def cosine_sim(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1).astype(np.float64)
    right_flat = right.reshape(-1).astype(np.float64)
    return float(np.dot(left_flat, right_flat) / (np.linalg.norm(left_flat) * np.linalg.norm(right_flat) + 1e-12))


def main() -> None:
    images = validate_holdout_split()
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    sess_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(ONNX_INT8_PATH), providers=["CPUExecutionProvider"])

    fp32_errors, fp32_cos = [], []
    int8_errors, int8_cos = [], []
    fp32_top1 = fp32_top5 = 0
    int8_top1 = int8_top5 = 0

    for image_path in images:
        input_tensor = preprocess_image(image_path)
        with torch.no_grad():
            pt_output = model(torch.from_numpy(input_tensor)).numpy()

        fp32_output = sess_fp32.run(["logits"], {"images": input_tensor})[0]
        int8_output = sess_int8.run(["logits"], {"images": input_tensor})[0]

        pt_top5 = set(np.argsort(pt_output, axis=-1)[0, -5:].tolist())
        fp32_top5 = set(np.argsort(fp32_output, axis=-1)[0, -5:].tolist())
        int8_top5 = set(np.argsort(int8_output, axis=-1)[0, -5:].tolist())
        fp32_top1 += int(np.argmax(pt_output, axis=-1)[0] == np.argmax(fp32_output, axis=-1)[0])
        fp32_top5 += int(pt_top5 == fp32_top5)
        int8_top1 += int(np.argmax(pt_output, axis=-1)[0] == np.argmax(int8_output, axis=-1)[0])
        int8_top5 += int(pt_top5 == int8_top5)

        fp32_errors.append(float(np.max(np.abs(pt_output - fp32_output))))
        int8_errors.append(float(np.max(np.abs(pt_output - int8_output))))
        fp32_cos.append(cosine_sim(pt_output, fp32_output))
        int8_cos.append(cosine_sim(pt_output, int8_output))

    sample_count = len(images)

    def summarise(errors: list[float], cosines: list[float], top1: int, top5: int) -> dict:
        return {
            "numeric_layer": {
                "max_abs_error_global": round(float(np.max(errors)), 6),
                "max_abs_error_mean": round(float(np.mean(errors)), 6),
                "cosine_similarity_mean": round(float(np.mean(cosines)), 6),
            },
            "task_layer": {
                "top1_agreement": f"{top1}/{sample_count} ({top1 / sample_count * 100:.1f}%)",
                "top5_agreement": f"{top5}/{sample_count} ({top5 / sample_count * 100:.1f}%)",
            },
        }

    report = {
        "sample_count": sample_count,
        "test_set": str(TEST_DIR),
        "calibration_set": str(CALIB_DIR),
        "calibration_test_overlap": 0,
        "pytorch_vs_onnx_fp32": summarise(fp32_errors, fp32_cos, fp32_top1, fp32_top5),
        "pytorch_vs_onnx_int8": summarise(int8_errors, int8_cos, int8_top1, int8_top5),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
