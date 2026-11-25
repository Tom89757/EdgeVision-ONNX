import hashlib
import json
from pathlib import Path

import onnx
import torch
# from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ONNX_FP32_PATH = ARTIFACTS_DIR / "mobilenet_v3_small_fp32.onnx"
ONNX_FP32_PATH = ARTIFACTS_DIR / "mobilenet_v2_fp32.onnx"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    # 1. 加载预训练模型并切换推理态
    # model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # 2. 导出固定 shape ONNX（opset 17 兼顾算子覆盖与移动端 ORT 兼容性）
    torch.onnx.export(
        model,
        (dummy_input,),
        str(ONNX_FP32_PATH),
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # 导出单文件 FP32，避免外部 .data 拆分
    )

    # 3. 结构与类型校验
    onnx_model = onnx.load(str(ONNX_FP32_PATH))
    onnx.checker.check_model(onnx_model)

    # 4. 写入模型资产契约（版本、shape、字节数、SHA-256）
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    manifest["model_name"] = "mobilenet_v3_small"
    manifest["opset_version"] = 17
    manifest["input_contract"] = {
        "name": "images",
        "shape": [1, 3, 224, 224],
        "layout": "NCHW",
        "dtype": "float32",
    }
    manifest["output_contract"] = {"name": "logits", "shape": [1, 1000], "dtype": "float32"}
    manifest.setdefault("files", {})["fp32"] = {
        "filename": ONNX_FP32_PATH.name,
        "bytes": ONNX_FP32_PATH.stat().st_size,
        "sha256": file_sha256(ONNX_FP32_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Exported FP32 ONNX: {ONNX_FP32_PATH} ({ONNX_FP32_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
