#!/usr/bin/env bash
set -euo pipefail

# EdgeVision-ONNX 阶段 2/3/4 自动执行脚本（Codex 后台无交互）
# 运行环境：WSL bash（codex CLI 在 WSL 中可用）
# 用法：bash run_codex_stages.sh
# 幂等：各阶段产物已存在则跳过，支持断点续跑；日志输出到 .codex_logs/

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$ROOT/.codex_logs"
ARTIFACTS="$ROOT/artifacts"
ASSETS="$ROOT/android_app/app/src/main/assets"
mkdir -p "$LOGS" "$ASSETS" "$ROOT/backend_service" "$ROOT/docs"
cd "$ROOT"

run_codex() {
  local prompt="$1" log="$2" attempt
  for attempt in 1 2 3; do
    if codex exec --sandbox workspace-write --skip-git-repo-check "$prompt" </dev/null >"$LOGS/$log" 2>&1; then
      return 0
    fi
    echo "[warn] codex 第 ${attempt} 次异常退出（详见 $LOGS/$log），20 秒后重试"
    sleep 20
  done
  return 1
}

# 0. 资产就位：把阶段 1 产物拷贝进 Android assets（供阶段 2 打包）
echo "[0/4] 拷贝阶段 1 ONNX 资产与 manifest 到 Android assets ..."
cp -f "$ARTIFACTS/mobilenet_v2_fp32.onnx"     "$ASSETS/mobilenet_v2_fp32.onnx"
cp -f "$ARTIFACTS/mobilenet_v2_int8_qdq.onnx" "$ASSETS/mobilenet_v2_int8_qdq.onnx"
cp -f "$ARTIFACTS/manifest.json"              "$ASSETS/manifest.json"

# 阶段 2：Android 端侧推理 App（幂等）
if [[ ! -f "$ROOT/android_app/app/build.gradle" ]]; then
  echo "[1/4] 阶段 2：生成 Android 端侧推理 App（Java）..."
  run_codex "阅读 AGENTS.md 后执行其中「阶段 2：Android ONNX Runtime 端侧推理 App」的全部要求，使用 Java 实现，生成 android_app 完整可编译的 Android 工程。不要修改 model_pipeline/、artifacts/、AGENTS.md、run_codex_stages.sh、verify_pipeline.py。" "stage2.log" || { echo "[error] 阶段 2 失败，详见 $LOGS/stage2.log"; exit 1; }
else
  echo "[1/4] 跳过（已存在）：android_app 工程"
fi

# 阶段 3：Benchmark 与性能剖析（依赖阶段 2，幂等）
if [[ ! -f "$ROOT/docs/benchmark_report.md" ]]; then
  echo "[2/4] 阶段 3：生成 Benchmark 模块与报告模板 ..."
  run_codex "阅读 AGENTS.md 后执行其中「阶段 3：真机多后端 Benchmark 与性能剖析」的全部要求：在既有 android_app 工程上补充 benchmark 模块，并生成 docs/model_contract.md 与 docs/benchmark_report.md（真机实测数据用占位符，严禁编造）。" "stage3.log" || { echo "[error] 阶段 3 失败，详见 $LOGS/stage3.log"; exit 1; }
else
  echo "[2/4] 跳过（已存在）：docs/benchmark_report.md"
fi

# 阶段 4：FastAPI 后端（幂等）
if [[ ! -f "$ROOT/backend_service/main.py" ]]; then
  echo "[3/4] 阶段 4：生成 FastAPI 端云联动服务 ..."
  run_codex "阅读 AGENTS.md 后执行其中「阶段 4：轻量 Python FastAPI 端云联动服务」的全部要求，生成 backend_service/ 完整代码（main.py、schemas.py、requirements.txt）。" "stage4.log" || { echo "[error] 阶段 4 失败，详见 $LOGS/stage4.log"; exit 1; }
else
  echo "[3/4] 跳过（已存在）：backend_service/main.py"
fi

echo "[4/4] 全部阶段完成。验收：python verify_pipeline.py --all"
