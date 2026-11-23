import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
# from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from export_onnx import ARTIFACTS_DIR, ONNX_FP32_PATH
from quantize_ptq import CALIB_DIR, ONNX_INT8_PATH
from preprocess import preprocess_image

REPORT_PATH = ARTIFACTS_DIR / "alignment_report.json"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.reshape(-1), b.reshape(-1)
    return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12))


def main() -> None:
    # model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval()
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    sess_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(ONNX_INT8_PATH), providers=["CPUExecutionProvider"])

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(p for p in CALIB_DIR.iterdir() if p.suffix.lower() in exts)

    fp32_max_errs, fp32_cos, fp32_top1_match = [], [], 0
    int8_max_errs, int8_cos, int8_top1_match = [], [], 0

    for img_path in images:
        x_np = preprocess_image(img_path)
        with torch.no_grad():
            pt_out = model(torch.from_numpy(x_np)).numpy()

        fp32_out = sess_fp32.run(["logits"], {"images": x_np})[0]
        int8_out = sess_int8.run(["logits"], {"images": x_np})[0]

        pt_cls = int(np.argmax(pt_out, axis=-1)[0])
        fp32_cls = int(np.argmax(fp32_out, axis=-1)[0])
        int8_cls = int(np.argmax(int8_out, axis=-1)[0])

        fp32_max_errs.append(float(np.max(np.abs(pt_out - fp32_out))))
        fp32_cos.append(cosine_sim(pt_out, fp32_out))
        fp32_top1_match += int(pt_cls == fp32_cls)

        int8_max_errs.append(float(np.max(np.abs(pt_out - int8_out))))
        int8_cos.append(cosine_sim(pt_out, int8_out))
        int8_top1_match += int(pt_cls == int8_cls)

    n = len(images)
    report = {
        "sample_count": n,
        "pytorch_vs_onnx_fp32": {
            "max_abs_error_mean": round(float(np.mean(fp32_max_errs)), 6),
            "cosine_similarity_mean": round(float(np.mean(fp32_cos)), 6),
            "top1_agreement": f"{fp32_top1_match}/{n} ({fp32_top1_match / n * 100:.1f}%)",
        },
        "pytorch_vs_onnx_int8": {
            "max_abs_error_mean": round(float(np.mean(int8_max_errs)), 6),
            "cosine_similarity_mean": round(float(np.mean(int8_cos)), 6),
            "top1_agreement": f"{int8_top1_match}/{n} ({int8_top1_match / n * 100:.1f}%)",
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()