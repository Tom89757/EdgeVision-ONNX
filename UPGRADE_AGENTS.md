# EdgeVision-ONNX 项目升级规范（Codex）

> 本文件是 `run_codex_upgrade.sh` 调用 `codex exec` 时的**唯一事实来源与执行规范**。
> 它是对 `AGENTS.md`（阶段 1~4 基线）的**升级叠加层**，目标是把项目从「分类 Demo」升级为「目标检测 + 生产级端侧治理」的硬核工程。
> 你（Codex）负责完成 4 个升级点的代码与文档产出。**已完成的阶段 1~4 代码可读可改，但不得删除或回退其既有能力**（详见下文权限与"并存"约束）。

## 一、升级目标

参考 `upgrade.md`，按面试投入产出比完成以下 4 项，全部落地产出：

1. **监控链路升级为 `YOLOv8n`（320×320）目标检测** —— 从"整图分类"升级为"多目标框选 `person`/`cat`/`dog`"，手写 Letterbox、NMS、坐标反变换，并做检测头敏感的混合精度 PTQ。
2. **修复评测闭环** —— 校准集与黄金测试集严格分离，输出数值层 + 任务层回归报告。
3. **补齐生产级端侧治理** —— 热状态监听 + 动态降级路由 + 延迟预算熔断。
4. **真机 Benchmark 深度剖析报告** —— 接入 ORT Profiling，实证 NNAPI 劣化根因，产出 EP 选型结论。

## 二、读写权限（相对 AGENTS.md 的变更）

| 路径 | 基线状态 | 升级状态 | 说明 |
| :--- | :--- | :--- | :--- |
| `model_pipeline/` | 只读 | **可写** | 新增 YOLOv8n 导出/量化/校验脚本、评测集分离脚本 |
| `artifacts/` | 只读 | **可写** | 新增 `yolov8n_*.onnx` 产物并扩展 `manifest.json` |
| `android_app/` | 可写 | **可写** | 新增检测链路与端侧治理模块 |
| `docs/` | 可写 | **可写** | 新增/更新剖析报告 |
| `backend_service/` | 可写 | 只读（本次不动） | 端云联动已就绪，升级不涉及后端接口 |
| `UPGRADE_AGENTS.md`、`run_codex_upgrade.sh` | — | **只读，禁止修改** | 本规范与编排脚本 |
| `AGENTS.md`、`run_codex_stages.sh`、`verify_pipeline.py` | 只读 | **只读，禁止修改** | 基线规范与验收脚本 |

## 三、并存约束（核心，必须严格遵守）

**保留 MobileNetV2 分类链路，新增 YOLOv8n 检测链路，两者并存，禁止删除或替换既有代码。**

- `ml/OnnxClassifier.java`、`ml/Preprocess.java`、`ml/ImageNetLabels.java`、`ui/AlbumActivity.java` 保持可用。
- 智能相册（`AlbumActivity`）继续用 MobileNetV2 分类。
- 宠物监控（`MonitorActivity`）升级为 YOLOv8n 检测画框，仅在注释中保留分类链路的可选回退说明。
- `MainActivity` 若需要切换，应同时保留「相册分类」与「监控检测」两个入口，不得移除分类入口。

## 四、升级点 1：YOLOv8n（320×320）目标检测

### 4.1 模型与权重

- 模型：`ultralytics` 的 `yolov8n.pt`（COCO 80 类预训练）。
- 权重获取：**由 codex 在 `model_pipeline` 内通过 `ultralytics` 下载**（这是合法下载，非伪造二进制）：
  - `pip install ultralytics` 后，`from ultralytics import YOLO; YOLO("yolov8n.pt")` 会自动下载权重。
  - 下载失败时**必须报错并明确提示**（网络受限 / 需手动下载），**禁止用占位文件冒充权重**。
- 输入尺寸：**`320 × 320`**（固定 shape，`NCHW`，`float32`，RGB）。
- 关注类别：COCO 中 `person`(0)、`cat`(15)、`dog`(16)，其余 77 类在 Android 展示层过滤，仅此三类参与画框与上报。

### 4.2 Python 侧产出（`model_pipeline/`）

新增以下脚本（沿用现有 `export_onnx.py` / `quantize_ptq.py` 的工程风格）：

- `export_yolo.py`：`YOLO("yolov8n.pt")` → `model.export(format="onnx", imgsz=320, opset=17, simplify=True)` 导出单文件 FP32 ONNX，输出 `artifacts/yolov8n_fp32.onnx`；`onnx.checker` 校验；读取输入/输出 shape 固化契约。
- `quantize_yolo_ptq.py`：静态 INT8 PTQ（QDQ），**检测头与坐标解码分支（尾部 Conv/分类与回归头）必须 `nodes_to_exclude` 排除，保留 FP32**；输出 `artifacts/yolov8n_int8_qdq.onnx`。
- `yolo_preprocess.py`：**Letterbox 等比缩放**（短边缩到 320，长边等比，非目标边用 `114` 灰色填充），记录 `ratio` 与 `pad_x/pad_y`，输出 `[1,3,320,320]` NCHW；供校准与对齐复用。
- `verify_yolo_alignment.py`：在黄金样本上对比 PyTorch vs ONNX FP32 vs INT8，**任务层指标必须是 NMS 后的检测框 IoU、Precision、Recall 与类别一致率**（而非只看张量余弦）。

### 4.3 输出契约（YOLOv8n）

- 输出为**不含 NMS 的 raw 检测张量**，shape 以 `export_yolo.py` 用 onnxruntime 实际读取为准（预期 `[1, 84, N]`，`84 = 4 box + 80 cls`，`N` 为 320 输入下三检测头 anchor 总数）。
- 前 4 通道为 `xywh`（中心点 + 宽高，像素坐标），后 80 通道为类别分数（YOLOv8 用 sigmoid 激活）。
- 后处理链（Android 与 Python 1:1 对齐）：`xywh→xyxy` → 置信度过滤 → 按类别 NMS（IoU 阈值）→ 坐标反变换回原图。

### 4.4 Android 侧产出（包 `com.edgevision.onnx`）

新增 `ml/YoloDetector.java`（可拆 `YoloPostprocess.java`），必须包含：

- `OrtEnvironment`/`OrtSession` 单例复用，支持 CPU/XNNPACK/NNAPI 与失败回退（复用 `OnnxClassifier` 的 EP 选择模式）。
- **Letterbox 预处理**：与 `yolo_preprocess.py` 1:1 对齐，等比缩放 + `114` 填充，记录 `ratio`/`pad_x`/`pad_y`。
- **后处理**：`xywh→xyxy`、置信度过滤（阈值可配置，默认 0.25）、NMS（IoU 阈值默认 0.45）、坐标反变换（除以 `ratio` 再减 `pad`）、类别过滤到 `person`/`cat`/`dog`。
- 后台线程推理 + 主线程回调，严禁主线程加载/推理。

改造 `ui/MonitorActivity.java`：从分类切换为检测，对 `person`/`cat`/`dog` 画框；`cat`/`dog` 触发抓拍上报（复用既有 `EventReporter`）。

### 4.5 资产与 manifest

- `artifacts/manifest.json` 在保留既有 MobileNetV2 键（`fp32`/`int8_qdq`）基础上，**新增** `yolo_fp32`、`yolo_int8_qdq` 两个文件条目（`filename`/`bytes`/`sha256`），并新增独立的 `yolo_input_contract` / `yolo_output_contract` 字段描述 YOLOv8n 输入输出。
- Android `AssetVerifier` 按需适配，能同时校验分类与检测两类模型资产；`.onnx` 为二进制，**禁止读取其内容**，只引用文件名/字节数/SHA-256。

## 五、升级点 2：评测闭环（校准集与测试集严格分离）

1. 目录约定：
   - `model_pipeline/calib_images/`：**40 张校准集**，仅用于 PTQ 校准。
   - `model_pipeline/test_images/`：**30 张独立黄金测试集**，仅用于精度验证，**不得用于校准**。
2. 数据来源：优先真实宠物/家居/监控场景图，刻意混入暗光、逆光、遮挡、目标在角落等长尾场景。**禁止伪造图片**：
   - 若目录已有足量真实图 → 直接分离使用。
   - 若数量不足 → 生成 `prepare_datasets.py` 说明来源与下载方式，并明确输出「待用户补充 N 张图」的提示；**不得用 `picsum` 随机图充当测试集，不得重复用同一批图既校准又测试**。
3. `verify_alignment.py` / `verify_yolo_alignment.py` 必须分别输出：
   - **数值层**：最大绝对误差、余弦相似度。
   - **任务层**：分类 Top-1/Top-5 一致率，或检测框 IoU / Precision / Recall。
   把「转换成功不等于精度达标」做成可量化的回归闭环。

## 六、升级点 3：生产级端侧治理

新增 `ml/ThermalGuard.java`（可合并 `LatencyBudget` 逻辑），并接入 `MonitorActivity`：

1. **热状态监听**：用 `PowerManager.addThermalStatusListener` / `getCurrentThermalStatus()` 监听温度。
   - 达到 `THERMAL_STATUS_MODERATE`（中度）或低内存状态时，自动降级：监控分析帧率从 2 FPS 降到 0.5 FPS，或将重任务路由回退（如暂停上报 / 切轻量模型）。
   - 温度回落后设置**冷却窗口（默认 30 秒）**防止频繁抖动。
2. **延迟预算熔断**：统计滑动窗口内单帧 p95 耗时，连续超预算时自动降级推理后端或跳过非关键帧，体现「端侧不能拖垮整机」。
3. 所有阈值/窗口做成可读常量并在注释说明，**严禁编造实测数值**（真机数据用占位符或由运行日志产生）。

## 七、升级点 4：Benchmark 深度剖析报告

1. 在 `BenchmarkRunner` 中接入 ONNX Runtime `SessionOptions.enableProfiling()`，导出单帧算子耗时 JSON（`end_profiling()` 产出的 profiling 文件），定位哪些节点被切回 CPU、引发子图划分与跨设备拷贝开销。
2. 更新 `docs/benchmark_report.md`，围绕已观测到的「FP32 下 NNAPI 明显慢于 XNNPACK、INT8 下 NNAPI 初始化失败并回退」展开：
   - 用 profiling 数据实锤「算子不支持 → 子图切分 → CPU 回退 + 反量化/拷贝开销」的根因链。
   - 结合「Android 15/16 已将 NNAPI deprecated」的平台事实给出 EP 选型结论：通用 ARM 手机优先 INT8 + XNNPACK；特定芯片 NPU 评估厂商 QNN/RKNN 等离线编译工具链。
   - **真机性能数据一律用 `【实测 ms】` / `【实测 MB】` 占位，严禁编造**。

## 八、通用规范（沿用并强化）

1. 全部简体中文注释与文档。
2. **严禁编造**：性能数据、源码路径、库版本号、模型权重、图片。权重必须真实下载，图片必须真实存在。
3. 版本保守兼容：`ultralytics>=8.0.0` 但需与 Python 3.10 / torch 2.x 兼容；Android 侧沿用 `onnxruntime-android:1.22.0`（16KB 对齐）与 Java 11。
4. 只做升级点要求的最小实现，不额外造轮子；**保留全部既有能力，不删除、不重构与升级无关的既有代码**。
5. 完成后自检：Python 侧可独立运行导出/量化/校验；Android 侧 Java 语法正确、Gradle 依赖坐标正确、YOLO 预处理与 `yolo_preprocess.py` 对齐；`docs/benchmark_report.md` 无编造数据。

## 九、验收标准（升级后复核）

- **资产**：`artifacts/yolov8n_fp32.onnx`、`artifacts/yolov8n_int8_qdq.onnx` 存在；`manifest.json` 新增 `yolo_*` 条目且 `bytes`/`sha256` 与实际文件一致；`android_app` assets 同步三份检测资产与更新后的 manifest。
- **模型契约**：onnxruntime 可加载 YOLOv8n，输入 `[1,3,320,320]` float32、输出 `[1,84,N]` float32（N 以实际为准）。
- **检测链路**：`YoloDetector.java`（含 Letterbox/NMS/坐标反变换）存在；`MonitorActivity` 已切换为检测画框且仅 `person`/`cat`/`dog` 三类。
- **评测分离**：`calib_images/` 与 `test_images/` 目录分离，`verify_yolo_alignment.py` 输出数值层 + 任务层（IoU/P/R）。
- **端侧治理**：`ThermalGuard.java`（热监听 + 冷却窗口 + p95 熔断）存在并接入 `MonitorActivity`。
- **剖析报告**：`BenchmarkRunner` 含 `enableProfiling()`；`docs/benchmark_report.md` 含 NNAPI 劣化根因与 EP 选型结论，且无编造数字。
- **并存**：MobileNetV2 分类链路（`OnnxClassifier`/`AlbumActivity`）仍可编译、可运行。
