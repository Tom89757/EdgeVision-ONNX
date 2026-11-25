#!/usr/bin/env bash
set -euo pipefail

# EdgeVision-ONNX 项目升级自动执行脚本（Codex 后台无交互）
# 运行环境：WSL bash（codex CLI 在 WSL 中可用）
# 用法：bash run_codex_upgrade.sh
# 幂等：各阶段产物已存在则跳过，支持断点续跑；日志输出到 .codex_logs/upgrade_*.log

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$ROOT/.codex_logs"
ARTIFACTS="$ROOT/artifacts"
ASSETS="$ROOT/android_app/app/src/main/assets"
mkdir -p "$LOGS" "$ARTIFACTS" "$ASSETS"
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

# 升级阶段 1：Python 侧 YOLOv8n 导出 + 混合精度量化 + 评测集分离（幂等）
if [[ ! -f "$ROOT/model_pipeline/export_yolo.py" ]]; then
  echo "[1/4] 升级阶段 1：YOLOv8n 导出 / 量化 / 评测集分离 ..."
  run_codex "阅读 UPGRADE_AGENTS.md 后执行其中「升级点 1」的 Python 侧全部要求与「升级点 2」的全部要求：在 model_pipeline 内新增 export_yolo.py、quantize_yolo_ptq.py、yolo_preprocess.py、verify_yolo_alignment.py，用 ultralytics 联网下载 yolov8n.pt 并导出 320x320 单文件 ONNX 到 artifacts（yolov8n_fp32.onnx / yolov8n_int8_qdq.onnx），检测头与坐标解码分支排除量化；扩展 artifacts/manifest.json 新增 yolo 条目与契约；分离 calib_images(40张) 与 test_images(30张) 并禁止用 picsum 充当测试集。严禁伪造权重与图片，严禁修改 UPGRADE_AGENTS.md、run_codex_upgrade.sh、AGENTS.md、run_codex_stages.sh、verify_pipeline.py。" "upgrade_stage1.log" || { echo "[error] 升级阶段 1 失败，详见 $LOGS/upgrade_stage1.log"; exit 1; }
else
  echo "[1/4] 跳过（已存在）：model_pipeline/export_yolo.py"
fi

# 升级阶段 1 后：把 YOLOv8n 资产与更新后 manifest 拷贝进 Android assets（供阶段 2 打包）
echo "[*] 同步 YOLOv8n ONNX 资产与 manifest 到 Android assets ..."
if [[ -f "$ARTIFACTS/yolov8n_fp32.onnx" ]]; then
  cp -f "$ARTIFACTS/yolov8n_fp32.onnx" "$ASSETS/yolov8n_fp32.onnx"
fi
if [[ -f "$ARTIFACTS/yolov8n_int8_qdq.onnx" ]]; then
  cp -f "$ARTIFACTS/yolov8n_int8_qdq.onnx" "$ASSETS/yolov8n_int8_qdq.onnx"
fi
cp -f "$ARTIFACTS/manifest.json" "$ASSETS/manifest.json"

# 升级阶段 2：Android YOLO 检测链路（Letterbox + NMS + 画框，幂等）
if [[ ! -f "$ROOT/android_app/app/src/main/java/com/edgevision/onnx/ml/YoloDetector.java" ]]; then
  echo "[2/4] 升级阶段 2：Android YOLO 检测链路 ..."
  run_codex "阅读 UPGRADE_AGENTS.md 后执行其中「升级点 1」的 Android 侧全部要求：在 com.edgevision.onnx.ml 新增 YoloDetector.java（OrtSession 单例 + Letterbox 预处理 + xywh→xyxy + 置信度过滤 + NMS + 坐标反变换，与 yolo_preprocess.py 1:1 对齐，仅保留 person/cat/dog 三类），改造 ui/MonitorActivity.java 从分类切换为检测画框并复用 EventReporter 上报 cat/dog。必须保留 OnnxClassifier/AlbumActivity 分类链路，不得删除或回退。严禁修改 UPGRADE_AGENTS.md、run_codex_upgrade.sh、AGENTS.md、run_codex_stages.sh、verify_pipeline.py。" "upgrade_stage2.log" || { echo "[error] 升级阶段 2 失败，详见 $LOGS/upgrade_stage2.log"; exit 1; }
else
  echo "[2/4] 跳过（已存在）：ml/YoloDetector.java"
fi

# 升级阶段 3：生产级端侧治理（热降级 + 熔断，幂等）
if [[ ! -f "$ROOT/android_app/app/src/main/java/com/edgevision/onnx/ml/ThermalGuard.java" ]]; then
  echo "[3/4] 升级阶段 3：端侧治理（热状态监听 + 延迟熔断）..."
  run_codex "阅读 UPGRADE_AGENTS.md 后执行其中「升级点 3」的全部要求：在 com.edgevision.onnx.ml 新增 ThermalGuard.java（PowerManager 热状态监听，THERMAL_STATUS_MODERATE 时降频 2FPS→0.5FPS 或暂停上报，回落后设 30 秒冷却窗口；滑动窗口 p95 熔断），并接入 ui/MonitorActivity.java。所有阈值做成可读常量并注释，严禁编造实测数值。严禁修改 UPGRADE_AGENTS.md、run_codex_upgrade.sh、AGENTS.md、run_codex_stages.sh、verify_pipeline.py。" "upgrade_stage3.log" || { echo "[error] 升级阶段 3 失败，详见 $LOGS/upgrade_stage3.log"; exit 1; }
else
  echo "[3/4] 跳过（已存在）：ml/ThermalGuard.java"
fi

# 升级阶段 4：Benchmark 深度剖析报告（ORT Profiling + 报告，幂等）
if ! grep -q "enableProfiling" "$ROOT/android_app/app/src/main/java/com/edgevision/onnx/benchmark/BenchmarkRunner.java" 2>/dev/null; then
  echo "[4/4] 升级阶段 4：Benchmark Profiling 剖析报告 ..."
  run_codex "阅读 UPGRADE_AGENTS.md 后执行其中「升级点 4」的全部要求：在 benchmark/BenchmarkRunner.java 接入 ONNX Runtime enableProfiling() 导出算子耗时 JSON；更新 docs/benchmark_report.md，围绕 NNAPI 劣化与 INT8 NNAPI 初始化失败做根因剖析（算子覆盖→子图切分→CPU 回退+反量化开销），结合 Android 15/16 NNAPI deprecated 给出 EP 选型结论。真机数据一律用【实测 ms】/【实测 MB】占位，严禁编造。严禁修改 UPGRADE_AGENTS.md、run_codex_upgrade.sh、AGENTS.md、run_codex_stages.sh、verify_pipeline.py。" "upgrade_stage4.log" || { echo "[error] 升级阶段 4 失败，详见 $LOGS/upgrade_stage4.log"; exit 1; }
else
  echo "[4/4] 跳过（已存在）：BenchmarkRunner 已接入 enableProfiling"
fi

echo "[完成] 升级各阶段结束。请复核：python verify_pipeline.py --all（如扩展了验收）与 UPGRADE_AGENTS.md 第九节验收标准。"
