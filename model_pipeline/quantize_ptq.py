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

CALIB_DIR = Path("./calib_images")
# ONNX_PREP_PATH = ARTIFACTS_DIR / "mobilenet_v3_small_prep.onnx"
# ONNX_INT8_PATH = ARTIFACTS_DIR / "mobilenet_v3_small_int8_qdq.onnx"
ONNX_PREP_PATH = ARTIFACTS_DIR / "mobilenet_v2_prep.onnx"
ONNX_INT8_PATH = ARTIFACTS_DIR / "mobilenet_v2_int8_qdq.onnx"

# quantize_static 中删掉 nodes_to_exclude，保留 reduce_range=True 和 op_types_to_quantize=["Conv", "Gemm"]


class ImageFolderCalibrationReader(CalibrationDataReader):
    def __init__(self, image_dir: Path, input_name: str = "images"):
        self.input_name = input_name
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
        if not self.image_paths:
            raise FileNotFoundError(f"请在 {image_dir.resolve()} 中放入 30~50 张校准图片")
        self._iter = iter(self.image_paths)

    def get_next(self) -> dict[str, np.ndarray] | None:
        path = next(self._iter, None)
        if path is None:
            return None
        return {self.input_name: preprocess_image(path)}

    def rewind(self) -> None:
        self._iter = iter(self.image_paths)


def main() -> None:
    # 1. 量化前图规范化与 shape 推断
    quant_pre_process(str(ONNX_FP32_PATH), str(ONNX_PREP_PATH), skip_symbolic_shape=False)

    # 2. 静态 PTQ 量化（QDQ 格式，权重量化对称 INT8，激活非对称 UINT8）
    reader = ImageFolderCalibrationReader(CALIB_DIR, input_name="images")
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
        reduce_range=True,                           # <--- 防止 x86 非 VNNI CPU 溢出
        op_types_to_quantize=["Conv", "Gemm"],
        # nodes_to_exclude=nodes_to_exclude,           # <--- 首尾敏感层回退 FP32
        calibrate_method=CalibrationMethod.MinMax,   # <--- 30张图下 MinMax 比高分位直方图更稳
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