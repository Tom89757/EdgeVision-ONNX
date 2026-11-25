from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from export_yolo import ARTIFACTS_DIR, ONNX_FP32_PATH, ONNX_INT8_PATH, WEIGHT_PATH, load_yolo_model
from yolo_preprocess import LetterboxMeta, letterbox_image, scale_boxes_to_original


PIPELINE_DIR = Path(__file__).resolve().parent
CALIB_DIR = PIPELINE_DIR / "calib_images"
TEST_DIR = PIPELINE_DIR / "test_images"
REPORT_PATH = ARTIFACTS_DIR / "yolo_alignment_report.json"

TEST_COUNT = 30
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_CLASSES = {0: "person", 15: "cat", 16: "dog"}
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MATCH_IOU_THRESHOLD = 0.50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def validate_test_set() -> list[Path]:
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"缺少黄金测试集目录：{TEST_DIR}；请先运行 prepare_datasets.py")
    images = list_images(TEST_DIR)
    if len(images) != TEST_COUNT:
        raise RuntimeError(f"黄金测试集必须恰好 {TEST_COUNT} 张，当前 {len(images)} 张")
    calibration = list_images(CALIB_DIR) if CALIB_DIR.is_dir() else []
    calibration_hashes = {sha256_file(path) for path in calibration}
    overlap = calibration_hashes & {sha256_file(path) for path in images}
    if overlap:
        raise RuntimeError(f"检测到 {len(overlap)} 张图片同时出现在校准集与测试集，拒绝运行")
    return images


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1).astype(np.float64)
    right_flat = right.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(left_flat) * np.linalg.norm(right_flat) + 1e-12
    return float(np.dot(left_flat, right_flat) / denominator)


def normalise_raw_output(raw: np.ndarray) -> np.ndarray:
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise RuntimeError(f"仅支持 batch=1，实际输出 {array.shape}")
        array = array[0]
    if array.ndim != 2:
        raise RuntimeError(f"YOLO 原始输出应为二维或三维张量，实际 {array.shape}")
    if array.shape[0] == 84:
        return array.T
    if array.shape[1] == 84:
        return array
    raise RuntimeError(f"YOLO 原始输出通道数必须为 84，实际 {array.shape}")


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection_wh = np.clip(right_bottom - left_top, 0.0, None)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    boxes_area = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-12)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        ious = box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return keep


def decode_detections(raw: np.ndarray, meta: LetterboxMeta) -> list[dict]:
    predictions = normalise_raw_output(raw)
    centers = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = np.isin(class_ids, list(ALLOWED_CLASSES)) & (confidences >= CONF_THRESHOLD)
    boxes_xywh = centers[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]
    if boxes_xywh.size == 0:
        return []

    boxes_xyxy = np.empty_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

    kept_indices: list[int] = []
    for class_id in sorted(ALLOWED_CLASSES):
        class_indices = np.flatnonzero(class_ids == class_id)
        local_keep = nms(boxes_xyxy[class_indices], confidences[class_indices], NMS_IOU_THRESHOLD)
        kept_indices.extend(int(class_indices[index]) for index in local_keep)
    if not kept_indices:
        return []

    kept_indices = sorted(kept_indices, key=lambda index: float(confidences[index]), reverse=True)
    kept_boxes = scale_boxes_to_original(boxes_xyxy[kept_indices], meta)
    detections = []
    for box, class_id, confidence in zip(kept_boxes, class_ids[kept_indices], confidences[kept_indices]):
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        detections.append(
            {
                "class_id": int(class_id),
                "class_name": ALLOWED_CLASSES[int(class_id)],
                "confidence": float(confidence),
                "box": [float(value) for value in box],
            }
        )
    return detections


def find_label_path(image_path: Path) -> Path | None:
    candidates = [TEST_DIR / "labels" / f"{image_path.stem}.txt", image_path.with_suffix(".txt")]
    return next((path for path in candidates if path.is_file()), None)


def load_ground_truth(image_path: Path, meta: LetterboxMeta) -> list[dict] | None:
    label_path = find_label_path(image_path)
    if label_path is None:
        return None
    detections = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        class_id = int(float(fields[0]))
        if class_id not in ALLOWED_CLASSES:
            continue
        cx, cy, width, height = (float(value) for value in fields[1:])
        x1 = (cx - width / 2.0) * meta.original_width
        y1 = (cy - height / 2.0) * meta.original_height
        x2 = (cx + width / 2.0) * meta.original_width
        y2 = (cy + height / 2.0) * meta.original_height
        detections.append(
            {
                "class_id": class_id,
                "class_name": ALLOWED_CLASSES[class_id],
                "confidence": 1.0,
                "box": [x1, y1, x2, y2],
            }
        )
    return detections


def match_predictions(predictions: list[dict], references: list[dict], class_aware: bool) -> tuple[int, int, int, list[float], int]:
    if not predictions:
        return 0, 0, len(references), [], 0
    if not references:
        return 0, len(predictions), 0, [], 0

    reference_boxes = np.asarray([reference["box"] for reference in references], dtype=np.float32)
    matched_references: set[int] = set()
    true_positives = 0
    ious: list[float] = []
    class_hits = 0
    geometry_matches = 0

    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        overlaps = box_iou(np.asarray(prediction["box"], dtype=np.float32), reference_boxes)
        best_index = None
        best_iou = 0.0
        for index in np.argsort(overlaps)[::-1]:
            index = int(index)
            if index in matched_references or float(overlaps[index]) < MATCH_IOU_THRESHOLD:
                continue
            if class_aware and references[index]["class_id"] != prediction["class_id"]:
                continue
            best_index = index
            best_iou = float(overlaps[index])
            break
        if best_index is None:
            continue
        matched_references.add(best_index)
        geometry_matches += 1
        class_hits += int(references[best_index]["class_id"] == prediction["class_id"])
        if class_aware:
            true_positives += 1
            ious.append(best_iou)

    if class_aware:
        false_positives = len(predictions) - true_positives
        false_negatives = len(references) - true_positives
        return true_positives, false_positives, false_negatives, ious, class_hits
    return geometry_matches, 0, 0, [], class_hits


def empty_accumulator() -> dict:
    return {"tp": 0, "fp": 0, "fn": 0, "ious": [], "geometry_matches": 0, "class_hits": 0}


def accumulate(accumulator: dict, predictions: list[dict], references: list[dict]) -> None:
    tp, fp, fn, ious, class_hits = match_predictions(predictions, references, class_aware=True)
    geometry_matches, _, _, _, geometry_class_hits = match_predictions(
        predictions, references, class_aware=False
    )
    accumulator["tp"] += tp
    accumulator["fp"] += fp
    accumulator["fn"] += fn
    accumulator["ious"].extend(ious)
    accumulator["geometry_matches"] += geometry_matches
    accumulator["class_hits"] += geometry_class_hits


def summarise(accumulator: dict) -> dict:
    tp = accumulator["tp"]
    fp = accumulator["fp"]
    fn = accumulator["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = float(np.mean(accumulator["ious"])) if accumulator["ious"] else 0.0
    class_agreement = (
        accumulator["class_hits"] / accumulator["geometry_matches"] if accumulator["geometry_matches"] else 0.0
    )
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "mean_iou": round(mean_iou, 6),
        "class_agreement": round(float(class_agreement), 6),
        "geometry_matches": accumulator["geometry_matches"],
    }


def pytorch_raw_output(model, input_tensor: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        output = model.model(torch.from_numpy(input_tensor))
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.detach().cpu().numpy()


def main() -> None:
    test_images = validate_test_set()
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(f"缺少真实权重 {WEIGHT_PATH}，请先运行 export_yolo.py 下载/导出")

    model = load_yolo_model()
    model.model.eval()
    fp32_session = ort.InferenceSession(str(ONNX_FP32_PATH), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(ONNX_INT8_PATH), providers=["CPUExecutionProvider"])
    fp32_input_name = fp32_session.get_inputs()[0].name
    int8_input_name = int8_session.get_inputs()[0].name

    fp32_numeric = {"max_abs_error": [], "cosine": []}
    int8_numeric = {"max_abs_error": [], "cosine": []}
    fp32_vs_pt = empty_accumulator()
    int8_vs_pt = empty_accumulator()
    pt_vs_labels = empty_accumulator()
    fp32_vs_labels = empty_accumulator()
    int8_vs_labels = empty_accumulator()
    labeled_images = 0
    class_coverage = {name: 0 for name in ALLOWED_CLASSES.values()}

    for image_path in test_images:
        input_tensor, meta = letterbox_image(image_path)
        pt_raw = pytorch_raw_output(model, input_tensor)
        fp32_raw = fp32_session.run(None, {fp32_input_name: input_tensor})[0]
        int8_raw = int8_session.run(None, {int8_input_name: input_tensor})[0]

        for target, accumulator in ((fp32_raw, fp32_numeric), (int8_raw, int8_numeric)):
            accumulator["max_abs_error"].append(float(np.max(np.abs(pt_raw.astype(np.float32) - target.astype(np.float32)))))
            accumulator["cosine"].append(cosine_similarity(pt_raw, target))

        pt_detections = decode_detections(pt_raw, meta)
        fp32_detections = decode_detections(fp32_raw, meta)
        int8_detections = decode_detections(int8_raw, meta)
        accumulate(fp32_vs_pt, fp32_detections, pt_detections)
        accumulate(int8_vs_pt, int8_detections, pt_detections)

        labels = load_ground_truth(image_path, meta)
        if labels is not None:
            labeled_images += 1
            for label in labels:
                class_coverage[label["class_name"]] += 1
            accumulate(pt_vs_labels, pt_detections, labels)
            accumulate(fp32_vs_labels, fp32_detections, labels)
            accumulate(int8_vs_labels, int8_detections, labels)

    def numeric_summary(accumulator: dict) -> dict:
        return {
            "max_abs_error_global": round(float(np.max(accumulator["max_abs_error"])), 6),
            "max_abs_error_mean": round(float(np.mean(accumulator["max_abs_error"])), 6),
            "cosine_similarity_mean": round(float(np.mean(accumulator["cosine"])), 6),
        }

    report = {
        "sample_count": len(test_images),
        "test_set": str(TEST_DIR),
        "calibration_set": str(CALIB_DIR),
        "calibration_test_overlap": 0,
        "thresholds": {
            "confidence": CONF_THRESHOLD,
            "nms_iou": NMS_IOU_THRESHOLD,
            "match_iou": MATCH_IOU_THRESHOLD,
            "focused_classes": ALLOWED_CLASSES,
        },
        "numeric_layer": {
            "pytorch_vs_onnx_fp32": numeric_summary(fp32_numeric),
            "pytorch_vs_onnx_int8": numeric_summary(int8_numeric),
        },
        "task_layer": {
            "reference": "pytorch_nms_detections",
            "onnx_fp32_vs_pytorch": summarise(fp32_vs_pt),
            "onnx_int8_vs_pytorch": summarise(int8_vs_pt),
        },
        "golden_labels": {
            "available": labeled_images == len(test_images),
            "labeled_images": labeled_images,
            "class_coverage": class_coverage,
            "pytorch_vs_labels": summarise(pt_vs_labels) if labeled_images else None,
            "onnx_fp32_vs_labels": summarise(fp32_vs_labels) if labeled_images else None,
            "onnx_int8_vs_labels": summarise(int8_vs_labels) if labeled_images else None,
        },
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
