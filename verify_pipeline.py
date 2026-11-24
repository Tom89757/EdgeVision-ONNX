#!/usr/bin/env python3
"""EdgeVision-ONNX 阶段 2/3/4 验收脚本。

用法（--model 需 onnxruntime，其余子命令无第三方依赖）：
    python verify_pipeline.py --all
    python verify_pipeline.py --assets
    python verify_pipeline.py --model
    python verify_pipeline.py --android
    python verify_pipeline.py --backend
"""
import argparse
import hashlib
import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "android_app" / "app" / "src" / "main" / "assets"
MANIFEST = ARTIFACTS / "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def check_assets() -> bool:
    if not MANIFEST.exists():
        print(f"[FAIL] 缺少 manifest：{MANIFEST}")
        return False
    manifest = load_manifest()
    ok = True
    for key, meta in manifest.get("files", {}).items():
        p = ARTIFACTS / meta["filename"]
        if not p.exists():
            print(f"[FAIL] 缺少产物：{p}")
            ok = False
            continue
        if p.stat().st_size != meta["bytes"]:
            print(f"[FAIL] 字节数不符：{p.name} 期望 {meta['bytes']} 实际 {p.stat().st_size}")
            ok = False
        if sha256(p) != meta["sha256"]:
            print(f"[FAIL] SHA-256 不符：{p.name}")
            ok = False
        else:
            print(f"[OK] {p.name}：{meta['bytes']} 字节，SHA-256 一致")
    return ok


def check_model() -> bool:
    if not MANIFEST.exists():
        print("[SKIP] 无 manifest，跳过模型契约校验")
        return True
    try:
        import onnxruntime as ort
    except ImportError:
        print("[SKIP] 未安装 onnxruntime，跳过模型契约校验")
        return True
    manifest = load_manifest()
    in_c = manifest["input_contract"]
    out_c = manifest["output_contract"]
    ok = True
    for key in ("fp32", "int8_qdq"):
        meta = manifest.get("files", {}).get(key)
        if not meta:
            continue
        p = ARTIFACTS / meta["filename"]
        try:
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            outp = sess.get_outputs()[0]
            shape_ok = inp.name == in_c["name"] and list(inp.shape) == in_c["shape"]
            out_ok = outp.name == out_c["name"] and list(outp.shape) == out_c["shape"]
            if shape_ok and out_ok:
                print(f"[OK] {p.name}：输入 {inp.name}{inp.shape} / 输出 {outp.name}{outp.shape} 契约一致")
            else:
                print(f"[FAIL] {p.name} 契约不符：输入 {inp.name}{inp.shape}，输出 {outp.name}{outp.shape}")
                ok = False
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {p.name} 加载失败：{e}")
            ok = False
    return ok


def check_android() -> bool:
    if not MANIFEST.exists() or not ASSETS.exists():
        print("[SKIP] 无 manifest 或 android_app assets 尚未生成")
        return True
    manifest = load_manifest()
    ok = True
    for key, meta in manifest.get("files", {}).items():
        src = ARTIFACTS / meta["filename"]
        dst = ASSETS / meta["filename"]
        if not dst.exists():
            print(f"[FAIL] assets 缺少：{dst}")
            ok = False
            continue
        if sha256(dst) != meta["sha256"]:
            print(f"[FAIL] assets SHA-256 不符：{dst.name}")
            ok = False
        else:
            print(f"[OK] assets/{dst.name} 与 artifacts 一致")
    if not (ASSETS / "manifest.json").exists():
        print("[FAIL] assets 缺少 manifest.json")
        ok = False
    return ok


def check_backend() -> bool:
    backend = ROOT / "backend_service"
    files = ["main.py", "schemas.py", "requirements.txt"]
    ok = True
    for f in files:
        p = backend / f
        if not p.exists():
            print(f"[FAIL] 缺少 backend_service/{f}")
            ok = False
    if not ok:
        return False
    routes = ["/api/v1/events/pet", "/api/v1/events", "/api/v1/reports/daily"]
    main_src = (backend / "main.py").read_text(encoding="utf-8")
    for r in routes:
        if r not in main_src:
            print(f"[FAIL] main.py 缺少路由：{r}")
            ok = False
    for f in ("main.py", "schemas.py"):
        try:
            py_compile.compile(str(backend / f), doraise=True)
            print(f"[OK] backend_service/{f} 语法正确")
        except py_compile.PyCompileError as e:
            print(f"[FAIL] backend_service/{f} 语法错误：{e}")
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", action="store_true", help="校验 artifacts 资产 SHA-256")
    ap.add_argument("--model", action="store_true", help="校验 onnx 模型契约（需 onnxruntime）")
    ap.add_argument("--android", action="store_true", help="校验 android assets 与 artifacts 一致")
    ap.add_argument("--backend", action="store_true", help="校验 backend_service 结构与语法")
    ap.add_argument("--all", action="store_true", help="执行全部校验")
    args = ap.parse_args()

    if not any([args.assets, args.model, args.android, args.backend, args.all]):
        ap.print_help()
        return 0

    run_all = args.all
    checks = []
    if run_all or args.assets:
        checks.append(("assets", check_assets))
    if run_all or args.model:
        checks.append(("model", check_model))
    if run_all or args.android:
        checks.append(("android", check_android))
    if run_all or args.backend:
        checks.append(("backend", check_backend))

    failed = 0
    for name, fn in checks:
        print(f"\n=== {name} ===")
        if not fn():
            failed += 1

    print(f"\n{'全部通过' if failed == 0 else f'{failed} 项未通过'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
