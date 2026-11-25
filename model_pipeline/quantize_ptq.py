import json
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from export_onnx import ARTIFACTS_DIR, MANIFEST_PATH, ONNX_FP32_PATH, file_sha256
from preprocess import preprocess_image


PIPELINE_DIR = Path(__file__).resolve().parent
CALIB_DIR = PIPELINE_DIR / "calib_images"
TEST_DIR = PIPELINE_DIR / "test_images"
CALIB_COUNT = 40
TEST_COUNT = 30
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ONNX_PREP_PATH = ARTIFACTS_DIR / "mobilenet_v2_prep.onnx"
ONNX_INT8_PATH = ARTIFACTS_DIR / "mobilenet_v2_int8_qdq.onnx"


class ImageFolderCalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], input_name: str = "images"):
        self.input_name = input_name
        if len(image_paths) != CALIB_COUNT:
            raise RuntimeError(f"校准集必须恰好 {CALIB_COUNT} 张，当前 {len(image_paths)} 张")
        self.image_paths = image_paths
        self._iter = iter(self.image_paths)

    def get_next(self) -> dict[str, np.ndarray] | None:
        path = next(self._iter, None)
        if path is None:
            return None
        return {self.input_name: preprocess_image(path)}

    def rewind(self) -> None:
        self._iter = iter(self.image_paths)


def validate_holdout_split() -> list[Path]:
    if not CALIB_DIR.is_dir():
        raise FileNotFoundError(f"缺少校准集目录：{CALIB_DIR}")
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"缺少黄金测试集目录：{TEST_DIR}；请先运行 prepare_datasets.py")

    calibration = sorted(p for p in CALIB_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    test_images = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if len(calibration) != CALIB_COUNT:
        raise RuntimeError(f"{CALIB_DIR} 必须恰好包含 {CALIB_COUNT} 张校准图片，当前 {len(calibration)} 张")
    if len(test_images) != TEST_COUNT:
        raise RuntimeError(f"{TEST_DIR} 必须恰好包含 {TEST_COUNT} 张黄金测试图片，当前 {len(test_images)} 张")

    calibration_hashes = {file_sha256(path) for path in calibration}
    test_hashes = {file_sha256(path) for path in test_images}
    if len(calibration_hashes) != CALIB_COUNT or len(test_hashes) != TEST_COUNT:
        raise RuntimeError("校准集或测试集存在重复图片")
    overlap = calibration_hashes & test_hashes
    if overlap:
        raise RuntimeError(f"校准集与测试集存在 {len(overlap)} 张相同图片，已拒绝数据泄漏")
    return calibration


def main() -> None:
    calibration_paths = validate_holdout_split()

    # 1. 量化前图规范化与 shape 推断
    quant_pre_process(str(ONNX_FP32_PATH), str(ONNX_PREP_PATH), skip_symbolic_shape=False)

    # 2. 静态 PTQ 量化（QDQ 格式，权重量化对称 INT8，激活非对称 UINT8）
    reader = ImageFolderCalibrationReader(calibration_paths, input_name="images")
    # 找出首层 Conv 和尾层 Gemm，保留 FP32（混合精度回退）
    import onnx

    prep_model = onnx.load(str(ONNX_PREP_PATH))
    conv_gemm_nodes = [n.name for n in prep_model.graph.node if n.op_type in ("Conv", "Gemm")]
    nodes_to_exclude = conv_gemm_nodes[:2] + conv_gemm_nodes[-2:]
    print("Excluding sensitive nodes:", nodes_to_exclude)

    quantize_static(
        model_input=str(ONNX_PREP_PATH),
        model_output=str(ONNX_INT8_PATH),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,  # 防止 x86 非 VNNI CPU 溢出
        op_types_to_quantize=["Conv", "Gemm"],
        # nodes_to_exclude=nodes_to_exclude,  # 首尾敏感层回退 FP32
        calibrate_method=CalibrationMethod.MinMax,
    )

    # 3. 更新资产清单
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["files"]["int8_qdq"] = {
        "filename": ONNX_INT8_PATH.name,
        "bytes": ONNX_INT8_PATH.stat().st_size,
        "sha256": file_sha256(ONNX_INT8_PATH),
        "calibration_samples": len(reader.image_paths),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Quantized INT8 ONNX: {ONNX_INT8_PATH} ({ONNX_INT8_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
