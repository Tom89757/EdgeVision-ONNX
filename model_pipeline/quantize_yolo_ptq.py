from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from export_yolo import (
    ARTIFACTS_DIR,
    ONNX_FP32_PATH,
    ONNX_INT8_PATH,
    file_sha256,
    read_onnx_contract,
    update_manifest,
)
from yolo_preprocess import preprocess_image


PIPELINE_DIR = Path(__file__).resolve().parent
CALIB_DIR = PIPELINE_DIR / "calib_images"
TEST_DIR = PIPELINE_DIR / "test_images"
ONNX_PREP_PATH = ARTIFACTS_DIR / "yolov8n_prep.onnx"

CALIB_COUNT = 40
TEST_COUNT = 30
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split() -> tuple[list[Path], list[Path]]:
    if not CALIB_DIR.is_dir():
        raise FileNotFoundError(f"缺少校准集目录：{CALIB_DIR}")
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"缺少黄金测试集目录：{TEST_DIR}；请先运行 prepare_datasets.py")

    calibration = list_images(CALIB_DIR)
    test = list_images(TEST_DIR)
    if len(calibration) != CALIB_COUNT:
        raise RuntimeError(f"{CALIB_DIR} 必须恰好包含 {CALIB_COUNT} 张校准图，当前 {len(calibration)} 张")
    if len(test) != TEST_COUNT:
        raise RuntimeError(f"{TEST_DIR} 必须恰好包含 {TEST_COUNT} 张黄金测试图，当前 {len(test)} 张")

    calibration_hashes = {sha256_bytes(path) for path in calibration}
    test_hashes = {sha256_bytes(path) for path in test}
    if len(calibration_hashes) != CALIB_COUNT:
        raise RuntimeError("校准集存在重复图片，禁止重复样本参与 PTQ")
    if len(test_hashes) != TEST_COUNT:
        raise RuntimeError("黄金测试集存在重复图片")
    overlap = calibration_hashes & test_hashes
    if overlap:
        raise RuntimeError(f"校准集与测试集存在 {len(overlap)} 张相同图片，已拒绝数据泄漏")
    return calibration, test


class YoloCalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], input_name: str):
        self.input_name = input_name
        self.image_paths = image_paths
        self._iterator = iter(self.image_paths)

    def get_next(self) -> dict[str, np.ndarray] | None:
        path = next(self._iterator, None)
        if path is None:
            return None
        return {self.input_name: preprocess_image(path)}

    def rewind(self) -> None:
        self._iterator = iter(self.image_paths)


def detection_head_nodes(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    """YOLOv8 的 Detect 头位于 /model.22/ 之后，整段尾部保留 FP32。"""
    nodes = list(model.graph.node)
    head_start = next((index for index, node in enumerate(nodes) if "model.22" in node.name), None)
    if head_start is None:
        raise RuntimeError("无法定位 YOLOv8 检测头（未找到 model.22 节点），拒绝在未排除检测头时量化")

    head_nodes = nodes[head_start:]
    if not head_nodes:
        raise RuntimeError("检测头节点列表为空，拒绝继续量化")
    if any(not node.name for node in head_nodes):
        raise RuntimeError("检测头存在未命名节点，无法可靠传给 nodes_to_exclude，拒绝继续量化")
    return head_nodes


def main() -> None:
    if not ONNX_FP32_PATH.is_file():
        raise FileNotFoundError(f"缺少 {ONNX_FP32_PATH}，请先运行 export_yolo.py")

    calibration_paths, test_paths = validate_split()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    input_contract, _ = read_onnx_contract(ONNX_FP32_PATH)

    # 1. 图规范化；随后从第一个 Detect 头节点起整段排除，覆盖分类/回归头和坐标解码分支。
    quant_pre_process(str(ONNX_FP32_PATH), str(ONNX_PREP_PATH), skip_symbolic_shape=False)
    prep_model = onnx.load(str(ONNX_PREP_PATH))
    head_nodes = detection_head_nodes(prep_model)
    nodes_to_exclude = [node.name for node in head_nodes]
    head_conv_count = sum(node.op_type == "Conv" for node in head_nodes)
    print(
        f"[INFO] 检测头保留 FP32：排除 {len(nodes_to_exclude)} 个尾部节点，"
        f"其中 Conv {head_conv_count} 个"
    )

    reader = YoloCalibrationReader(calibration_paths, input_name=input_contract["name"])
    quantize_static(
        model_input=str(ONNX_PREP_PATH),
        model_output=str(ONNX_INT8_PATH),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,  # ARM 端侧以 UInt8 激活为主，不采用 x86 VNNI 的缩小范围
        op_types_to_quantize=["Conv", "MatMul"],
        nodes_to_exclude=nodes_to_exclude,
        calibrate_method=CalibrationMethod.MinMax,
    )

    quantized_input_contract, output_contract = read_onnx_contract(ONNX_INT8_PATH)
    update_manifest(
        "yolo_int8_qdq",
        {
            "filename": ONNX_INT8_PATH.name,
            "bytes": ONNX_INT8_PATH.stat().st_size,
            "sha256": file_sha256(ONNX_INT8_PATH),
            "calibration_samples": len(calibration_paths),
            "test_samples": len(test_paths),
            "excluded_detection_head_nodes": len(nodes_to_exclude),
            "excluded_detection_head_convs": head_conv_count,
        },
        input_contract=quantized_input_contract,
        output_contract=output_contract,
    )
    print(
        f"[OK] 已生成混合精度 QDQ INT8：{ONNX_INT8_PATH} "
        f"({ONNX_INT8_PATH.stat().st_size / 1e6:.2f} MB)"
    )


if __name__ == "__main__":
    main()
