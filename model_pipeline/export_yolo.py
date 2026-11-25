from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import onnx
import onnxruntime as ort


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
WEIGHT_PATH = PIPELINE_DIR / "yolov8n.pt"
ONNX_FP32_PATH = ARTIFACTS_DIR / "yolov8n_fp32.onnx"
ONNX_INT8_PATH = ARTIFACTS_DIR / "yolov8n_int8_qdq.onnx"

YOLO_IMGSZ = 320
YOLO_OPSET = 17
YOLO_CLASS_COUNT = 80
YOLO_FOCUSED_CLASSES = {0: "person", 15: "cat", 16: "dog"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_manifest(
    file_key: str,
    file_entry: dict,
    input_contract: dict | None = None,
    output_contract: dict | None = None,
) -> dict:
    """保留 MobileNetV2 条目，仅增量写入 YOLO 文件与契约。"""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    files = manifest.setdefault("files", {})
    files[file_key] = file_entry
    manifest["yolo_model_name"] = "yolov8n"
    manifest["yolo_opset_version"] = YOLO_OPSET
    manifest["yolo_focused_classes"] = {str(k): v for k, v in YOLO_FOCUSED_CLASSES.items()}
    manifest["yolo_input_contract"] = input_contract or {
        "name": "images",
        "shape": [1, 3, YOLO_IMGSZ, YOLO_IMGSZ],
        "layout": "NCHW",
        "dtype": "float32",
        "color_space": "RGB",
        "normalization": "pixel/255.0",
        "letterbox": {"short_side": YOLO_IMGSZ, "pad_value": 114, "resize": "bilinear"},
    }
    if output_contract is not None:
        manifest["yolo_output_contract"] = output_contract
    write_json_atomic(MANIFEST_PATH, manifest)
    return manifest


def load_yolo_model():
    """加载真实 yolov8n.pt；缺失时由 Ultralytics 下载，失败则明确报错。"""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("缺少 ultralytics，请先执行：pip install -r model_pipeline/requirements.txt") from exc

    if not WEIGHT_PATH.exists():
        print(f"[INFO] 未找到 {WEIGHT_PATH.name}，将由 Ultralytics 从官方 assets 下载到 {WEIGHT_PATH}")
    try:
        model = YOLO(str(WEIGHT_PATH))
    except Exception as exc:
        raise RuntimeError(
            "yolov8n.pt 下载或加载失败：请检查网络后重试，或手动从 Ultralytics 官方 assets 下载 "
            f"yolov8n.pt 并放到 {WEIGHT_PATH}；禁止使用占位文件或伪造权重。"
        ) from exc

    checkpoint_path = Path(getattr(model, "ckpt_path", WEIGHT_PATH)).resolve()
    if checkpoint_path.is_file() and checkpoint_path != WEIGHT_PATH.resolve():
        shutil.copy2(checkpoint_path, WEIGHT_PATH)
    if not WEIGHT_PATH.is_file() or WEIGHT_PATH.stat().st_size < 1_000_000:
        if WEIGHT_PATH.exists() and WEIGHT_PATH.stat().st_size < 1_000_000:
            WEIGHT_PATH.unlink()
        raise RuntimeError(f"权重无效或下载不完整：{WEIGHT_PATH}；请重新下载真实 yolov8n.pt")
    raw_names = getattr(getattr(model, "model", None), "names", None) or getattr(model, "names", None) or {}
    if isinstance(raw_names, dict):
        names_by_index = {int(index): str(name) for index, name in raw_names.items()}
    elif isinstance(raw_names, (list, tuple)):
        names_by_index = {index: str(name) for index, name in enumerate(raw_names)}
    else:
        names_by_index = {}
    if getattr(model, "task", "detect") != "detect" or len(names_by_index) != YOLO_CLASS_COUNT:
        raise RuntimeError("加载的权重不是 COCO 80 类检测模型，拒绝导出；请确认使用官方 yolov8n.pt")
    if [names_by_index.get(index) for index in YOLO_FOCUSED_CLASSES] != list(YOLO_FOCUSED_CLASSES.values()):
        raise RuntimeError("COCO 类别顺序不符合 person/cat/dog 契约，拒绝导出")
    return model


def read_onnx_contract(path: Path) -> tuple[dict, dict]:
    """以 ONNX checker 与 onnxruntime 的实际读取结果固化输入/输出契约。"""
    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(f"YOLO ONNX 必须单输入单输出，实际输入 {len(inputs)}、输出 {len(outputs)}")

    input_info = inputs[0]
    output_info = outputs[0]
    input_shape = [int(dim) for dim in input_info.shape]
    output_shape = [int(dim) for dim in output_info.shape]
    if input_shape != [1, 3, YOLO_IMGSZ, YOLO_IMGSZ]:
        raise RuntimeError(f"YOLO ONNX 输入契约错误：{input_info.name}{input_shape}")
    if input_info.name != "images":
        raise RuntimeError(f"YOLO ONNX 输入名必须为 images，实际为 {input_info.name}")
    if len(output_shape) != 3 or output_shape[0] != 1 or output_shape[1] != 4 + YOLO_CLASS_COUNT:
        raise RuntimeError(f"YOLO ONNX 输出必须为 [1, 84, N]，实际 {output_info.name}{output_shape}")

    input_contract = {
        "name": input_info.name,
        "shape": input_shape,
        "layout": "NCHW",
        "dtype": "float32",
    }
    output_contract = {
        "name": output_info.name,
        "shape": output_shape,
        "layout": "BCN",
        "dtype": "float32",
        "channels": 4 + YOLO_CLASS_COUNT,
        "anchors": output_shape[2],
        "box_encoding": "xywh_center_pixel",
        "class_scores": "80 COCO class scores after sigmoid",
        "nms": False,
    }
    return input_contract, output_contract


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model = load_yolo_model()

    print(f"[INFO] 导出 YOLOv8n FP32 ONNX：imgsz={YOLO_IMGSZ}, opset={YOLO_OPSET}, simplify=True")
    exported = Path(
        model.export(
            format="onnx",
            imgsz=YOLO_IMGSZ,
            opset=YOLO_OPSET,
            simplify=True,
            dynamic=False,
            nms=False,
            batch=1,
            device="cpu",
        )
    ).resolve()
    if not exported.is_file():
        raise RuntimeError(f"Ultralytics 未生成预期的 ONNX 文件：{exported}")

    if exported != ONNX_FP32_PATH.resolve():
        shutil.copy2(exported, ONNX_FP32_PATH)

    input_contract, output_contract = read_onnx_contract(ONNX_FP32_PATH)
    update_manifest(
        "yolo_fp32",
        {
            "filename": ONNX_FP32_PATH.name,
            "bytes": ONNX_FP32_PATH.stat().st_size,
            "sha256": file_sha256(ONNX_FP32_PATH),
            "source": "ultralytics yolov8n.pt",
        },
        input_contract=input_contract,
        output_contract=output_contract,
    )
    print(
        f"[OK] 已导出 {ONNX_FP32_PATH} ({ONNX_FP32_PATH.stat().st_size / 1e6:.2f} MB)，"
        f"输出 {output_contract['name']}{output_contract['shape']}"
    )


if __name__ == "__main__":
    main()
