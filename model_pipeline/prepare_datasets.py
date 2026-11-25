"""准备 YOLO 评测数据。

默认来源为 Ultralytics COCO128（真实 COCO train2017 图片及其 YOLO 标注），下载地址：
https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip

下载方式：
    python model_pipeline/prepare_datasets.py --force

也可使用自行采集/下载的真实宠物、家居、监控图片：
    python model_pipeline/prepare_datasets.py --source-dir /path/to/real_images --force

脚本只接受真实图片，拒绝 picsum；按 SHA-256 与 dHash 双重去重，并严格拆分为 40 张校准集和
30 张互不重叠的测试集。图片不足时不会生成占位图，而是明确打印待补充数量。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
CALIB_DIR = PIPELINE_DIR / "calib_images"
TEST_DIR = PIPELINE_DIR / "test_images"
MANIFEST_PATH = PIPELINE_DIR / "dataset_manifest.json"
CALIB_COUNT = 40
TEST_COUNT = 30
SAMPLE_SEED = 20261125
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_CLASSES = {0: "person", 15: "cat", 16: "dog"}
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
EXCLUDED_DIR_NAMES = {".venv", "__pycache__", "legacy_datasets", "artifacts"}


@dataclass
class Sample:
    path: Path
    sha256: str
    dhash: int
    label_path: Path | None
    target_boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    mean_luma: float = 0.0
    tags: set[str] = field(default_factory=set)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash_file(path: Path) -> int:
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(image.getdata())
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return bits


def find_label_path(image_path: Path) -> Path | None:
    candidates = [image_path.with_suffix(".txt"), image_path.parent / "labels" / f"{image_path.stem}.txt"]
    parts = image_path.parts
    if "images" in parts:
        index = parts.index("images")
        relative = Path(*parts[index + 1 :]).with_suffix(".txt")
        candidates.append(Path(*parts[:index]) / "labels" / relative)
    return next((path for path in candidates if path.is_file()), None)


def read_target_boxes(label_path: Path | None) -> list[tuple[int, float, float, float, float]]:
    if label_path is None:
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        class_id = int(float(fields[0]))
        if class_id in TARGET_CLASSES:
            boxes.append((class_id, *(float(value) for value in fields[1:])))
    return boxes


def image_tags(image_path: Path, boxes: list[tuple[int, float, float, float, float]]) -> tuple[float, set[str]]:
    from PIL import Image, ImageOps, ImageStat

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    mean_luma = float(ImageStat.Stat(image.convert("L")).mean[0])
    tags: set[str] = set()
    if mean_luma < 60:
        tags.add("dark")
    if mean_luma > 185:
        tags.add("backlit_or_bright")

    for _class_id, cx, cy, width, height in boxes:
        if cx < 0.25 or cx > 0.75 or cy < 0.25 or cy > 0.75:
            tags.add("target_in_corner")
        if width * height < 0.03:
            tags.add("small_target")
    if len(boxes) >= 3:
        tags.add("multi_target")

    # 两个目标框显著重叠时，图像很可能存在相互遮挡；该标签仅作长尾场景代理指标。
    for left_index, left in enumerate(boxes):
        for right in boxes[left_index + 1 :]:
            left_area = left[3] * left[4]
            right_area = right[3] * right[4]
            overlap_width = max(0.0, min(left[1] + left[3] / 2, right[1] + right[3] / 2) - max(left[1] - left[3] / 2, right[1] - right[3] / 2))
            overlap_height = max(0.0, min(left[2] + left[4] / 2, right[2] + right[4] / 2) - max(left[2] - left[4] / 2, right[2] - right[4] / 2))
            overlap_area = overlap_width * overlap_height
            if overlap_area > 0 and overlap_area / max(min(left_area, right_area), 1e-12) > 0.1:
                tags.add("overlapping_targets")
    return mean_luma, tags


def contains_picsum_marker(path: Path) -> bool:
    if "picsum" in str(path).lower():
        return True
    try:
        from PIL import Image

        with Image.open(path) as image:
            exif_text = " ".join(str(value) for value in image.getexif().values()).lower()
        return "picsum" in exif_text
    except Exception:
        return False


def collect_samples(source: Path) -> list[Sample]:
    paths = sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and CALIB_DIR not in path.parents
        and TEST_DIR not in path.parents
        and not EXCLUDED_DIR_NAMES.intersection(path.parts)
    )
    samples: list[Sample] = []
    seen_sha: set[str] = set()
    seen_dhash: list[int] = []
    skipped_duplicates = 0
    for path in paths:
        if contains_picsum_marker(path):
            continue
        sha = sha256_file(path)
        if sha in seen_sha:
            skipped_duplicates += 1
            continue
        dhash = dhash_file(path)
        if any((dhash ^ previous).bit_count() <= 4 for previous in seen_dhash):
            skipped_duplicates += 1
            continue
        label_path = find_label_path(path)
        boxes = read_target_boxes(label_path)
        mean_luma, tags = image_tags(path, boxes)
        seen_sha.add(sha)
        seen_dhash.append(dhash)
        samples.append(
            Sample(
                path=path,
                sha256=sha,
                dhash=dhash,
                label_path=label_path,
                target_boxes=boxes,
                mean_luma=mean_luma,
                tags=tags,
            )
        )
    print(f"[INFO] 扫描到 {len(paths)} 张图片，去重后 {len(samples)} 张（跳过重复 {skipped_duplicates} 张）")
    return samples


def select_splits(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    if len(samples) < CALIB_COUNT + TEST_COUNT:
        missing_total = CALIB_COUNT + TEST_COUNT - len(samples)
        missing_calib = max(0, CALIB_COUNT - len(samples))
        missing_test = missing_total - missing_calib
        raise RuntimeError(
            f"真实图片不足：需要 {CALIB_COUNT + TEST_COUNT} 张，当前仅 {len(samples)} 张。"
            f"待用户补充 {missing_calib} 张校准图和 {missing_test} 张独立测试图；"
            "不得使用 picsum 随机图，也不得重复使用同一批图片。"
        )

    shuffled = sorted(samples, key=lambda sample: sample.sha256)
    random.Random(SAMPLE_SEED).shuffle(shuffled)

    # 测试集优先覆盖真实长尾场景；标签只来自真实数据集，不使用模型预测冒充真值。
    test: list[Sample] = []
    for tag in ("dark", "backlit_or_bright", "target_in_corner", "small_target", "multi_target", "overlapping_targets"):
        for sample in shuffled:
            if sample not in test and tag in sample.tags:
                test.append(sample)
                break
    for sample in shuffled:
        if len(test) >= TEST_COUNT:
            break
        if sample not in test:
            test.append(sample)

    test_paths = {sample.path for sample in test}
    remaining = [sample for sample in shuffled if sample.path not in test_paths]
    calibration = remaining[:CALIB_COUNT]
    if len(calibration) != CALIB_COUNT:
        raise RuntimeError("无法划分出 40 张校准图，请补充独立真实图片")
    calibration_paths = {sample.path for sample in calibration}
    if calibration_paths & test_paths:
        raise RuntimeError("校准集与测试集出现重叠图片，拒绝生成泄漏数据")
    return calibration, test[:TEST_COUNT]


def prepare_output_dirs(force: bool) -> None:
    existing = [path for path in (CALIB_DIR, TEST_DIR) if path.exists() and any(path.iterdir())]
    if existing and not force:
        raise RuntimeError("calib_images/test_images 已存在内容；确认替换请加 --force，程序会先备份旧数据")
    if existing:
        backup_root = PIPELINE_DIR / "legacy_datasets" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_root.mkdir(parents=True, exist_ok=True)
        for path in existing:
            shutil.move(str(path), str(backup_root / path.name))
        print(f"[INFO] 旧数据已备份到 {backup_root}")
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)


def copy_split(samples: list[Sample], output_dir: Path, prefix: str, copy_labels: bool) -> list[dict]:
    records = []
    label_dir = output_dir / "labels"
    if copy_labels:
        label_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples, start=1):
        suffix = sample.path.suffix.lower()
        destination = output_dir / f"{prefix}_{index:03d}{suffix}"
        shutil.copy2(sample.path, destination)
        label_destination = None
        if copy_labels and sample.label_path is not None:
            label_destination = label_dir / f"{destination.stem}.txt"
            shutil.copy2(sample.label_path, label_destination)
        records.append(
            {
                "filename": destination.name,
                "sha256": sample.sha256,
                "dhash": f"{sample.dhash:016x}",
                "source": str(sample.path),
                "label": str(label_destination) if label_destination else None,
                "target_boxes": len(sample.target_boxes),
                "tags": sorted(sample.tags),
            }
        )
    return records


def resolve_source(source_dir: str | None) -> tuple[Path, dict]:
    if source_dir:
        source = Path(source_dir).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"本地真实图片目录不存在：{source}")
        return source, {"type": "local_directory", "location": str(source), "picsum_allowed": False}

    try:
        from ultralytics.data.utils import check_det_dataset
    except ImportError as exc:
        raise RuntimeError("缺少 ultralytics；请先安装 requirements.txt，或通过 --source-dir 指定已下载的真实数据") from exc

    try:
        data = check_det_dataset("coco128.yaml")
    except Exception as exc:
        raise RuntimeError(
            f"COCO128 下载失败。请检查网络，或手动下载 {COCO128_URL}（真实 COCO 图片），"
            "解压后执行：python prepare_datasets.py --source-dir <coco128目录> --force。"
        ) from exc

    train_path = Path(data["train"])
    source = train_path if train_path.is_dir() else train_path.parent
    return source, {
        "type": "ultralytics_coco128",
        "location": str(source),
        "description": "COCO128：COCO train2017 的前 128 张真实图片，含 YOLO 标注",
        "download": data.get("download", COCO128_URL),
        "picsum_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 40 张校准图与 30 张独立黄金测试图（拒绝 picsum）")
    parser.add_argument("--source-dir", help="本地真实图片/数据集目录；未提供时下载 Ultralytics COCO128")
    parser.add_argument("--force", action="store_true", help="替换已有 calib_images/test_images（先备份）")
    args = parser.parse_args()

    source, source_info = resolve_source(args.source_dir)
    samples = collect_samples(source)
    try:
        calibration, test = select_splits(samples)
    except RuntimeError as exc:
        print(f"[待补充] {exc}")
        raise SystemExit(1) from exc

    prepare_output_dirs(force=args.force)
    calibration_records = copy_split(calibration, CALIB_DIR, "calib", copy_labels=False)
    test_records = copy_split(test, TEST_DIR, "test", copy_labels=True)
    test_labeled = sum(record["label"] is not None for record in test_records)
    test_target_coverage = {name: 0 for name in TARGET_CLASSES.values()}
    for sample in test:
        for class_id, *_ in sample.target_boxes:
            test_target_coverage[TARGET_CLASSES[class_id]] += 1
    tag_counts: dict[str, int] = {}
    for sample in test:
        for tag in sample.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source_info,
        "split_seed": SAMPLE_SEED,
        "deduplication": "sha256_exact_plus_dhash_hamming_le_4",
        "sha256_overlap": 0,
        "calibration_count": len(calibration_records),
        "test_count": len(test_records),
        "test_labeled_count": test_labeled,
        "test_target_coverage": test_target_coverage,
        "test_scene_tags": tag_counts,
        "calibration_images": calibration_records,
        "test_images": test_records,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] calib_images={len(calibration_records)}，test_images={len(test_records)}")
    if test_labeled < TEST_COUNT:
        print(f"[待补充] 黄金测试集仅 {test_labeled}/{TEST_COUNT} 张有真实标注，建议补充独立场景并标注 person/cat/dog")
    print(f"[OK] 数据来源与去重记录：{MANIFEST_PATH}")


if __name__ == "__main__":
    main()
