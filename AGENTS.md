# EdgeVision-ONNX 自动任务规范（Codex）

> 本文件是 `run_codex_stages.sh` 调用 `codex exec` 时的**唯一事实来源与执行规范**。
> 你（Codex）负责完成阶段 2 / 3 / 4 的代码与文档产出；**阶段 1 已完成，只读，禁止修改**。

## 一、项目定位

EdgeVision-ONNX：端侧视觉模型部署闭环（PyTorch → ONNX → INT8 PTQ → Android ONNX Runtime 推理），承载「智能相册离线打标」与「宠物监控端侧检测」两大场景。

## 二、目录与读写权限

| 路径 | 状态 | 说明 |
| :--- | :--- | :--- |
| `model_pipeline/` | **只读，禁止修改** | 阶段 1 已完成：导出 / 量化 / 对齐流水线 |
| `artifacts/` | **只读，禁止修改** | 阶段 1 产物：`mobilenet_v2_fp32.onnx`、`mobilenet_v2_int8_qdq.onnx`、`manifest.json`、`alignment_report.json` |
| `android_app/` | **可写（阶段 2/3 产出）** | Android 端侧 App |
| `backend_service/` | **可写（阶段 4 产出）** | FastAPI 后端 |
| `docs/` | **可写（阶段 3 产出）** | `model_contract.md`、`benchmark_report.md` |
| `AGENTS.md`、`run_codex_stages.sh`、`verify_pipeline.py` | **只读，禁止修改** | 本规范与编排/验收脚本 |

## 三、核心契约（必须严格遵守，禁止改动）

1. **模型**：`MobileNetV2`（torchvision 预训练），ImageNet 1000 类，**本次只做分类，不做 NMS/检测框**（画框检测属后续 YOLOv8n 扩展，不在本次范围，仅可在注释中预留）。
2. **输入**：名称 `images`，shape `[1, 3, 224, 224]`，`NCHW` 布局，`float32`。
3. **归一化**：RGB → `[0,1]` → ImageNet `mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`（见 `model_pipeline/preprocess.py`，Android 端必须 1:1 对齐）。
4. **输出**：名称 `logits`，shape `[1, 1000]`，`float32`。
5. **资产清单**：`artifacts/manifest.json` 含 `fp32` 与 `int8_qdq` 两模型的 `filename`/`bytes`/`sha256`；Android 端启动时校验文件大小与 SHA-256。
6. **二进制资产**：`.onnx` 为二进制，**禁止读取其内容**，只通过 `manifest.json` 引用文件名 / 字节数 / SHA-256。

## 四、阶段 2：Android ONNX Runtime 端侧推理 App

产出：`android_app/` 下**可编译**的完整 Android 工程（Java）。

### 4.1 工程与版本（保守、确定兼容的组合）

- 语言 **Java**；源码目录统一放 `android_app/app/src/main/java/com/edgevision/onnx/`（与蓝图目录树 `java/.../edgevision/` 对齐；AGP 8.x 默认 source/target compatibility 为 Java 11，无需额外配置）。
- `minSdk 26`、`compileSdk 34`、`targetSdk 34`、JDK 17。
- Gradle Wrapper 8.7、Android Gradle Plugin 8.5.2；构建脚本使用 **Groovy DSL**（`settings.gradle` / `build.gradle`），**禁止使用 Kotlin DSL（`.kts`）、禁止引入 Kotlin 插件、禁止生成任何 `.kt` 文件**。
- 依赖：`com.microsoft.onnxruntime:onnxruntime-android:1.22.0`（1.22.0 起支持 Android 15+ 16 KB page size 对齐）；CameraX `androidx.camera:camera-{core,camera2,lifecycle,view}:1.4.0`；`androidx.appcompat`、`androidx.constraintlayout`、`com.google.android.material` 使用兼容 compileSdk 34 的稳定版。
- **禁止生成/伪造二进制文件**（如 `gradle-wrapper.jar`、`.onnx`）。`gradle/wrapper/` 下只写 `gradle-wrapper.properties`；`gradlew` / `gradlew.bat` 脚本照常生成；`gradle-wrapper.jar` 由 Android Studio 或 `gradle wrapper` 命令生成，缺失属预期。

### 4.2 资产打包

- `android_app/app/src/main/assets/` 下应已有（由 `run_codex_stages.sh` 拷贝就位）：`mobilenet_v2_fp32.onnx`、`mobilenet_v2_int8_qdq.onnx`、`manifest.json`。**不要重复拷贝或删除**。

### 4.3 代码结构（包 `com.edgevision.onnx`）

- `MainActivity.java`：入口 Activity，导航到「智能相册」与「宠物监控」两个页面。
- `ml/OnnxClassifier.java`：`OrtEnvironment` / `OrtSession` 单例复用；支持按构造参数选择 EP（`CPUExecutionProvider` / `XNNPACKExecutionProvider` / `NNAPIExecutionProvider`，NNAPI 运行时判 API>=27 并 try/catch 回退）；加载与推理**严禁在主线程**（用 `ExecutorService` 后台 + 主线程回调）。启动时校验 assets 中 onnx 的大小与 SHA-256（对照 `manifest.json`）。
- `ml/Preprocess.java`：与 `model_pipeline/preprocess.py` 1:1 对齐 —— `inJustDecodeBounds` 预读 + 2 的幂降采样防 OOM、`ExifInterface` 校正方向、`Bitmap` 缩放至 `224x224`、转 RGB、归一化 `[0,1]` 后按 ImageNet mean/std 标准化、按 **NCHW** 写入 `FloatBuffer`（`float[1][3][224][224]`，channel 优先）。
- `ml/ImageNetLabels.java`：`Top-K` 后处理输出 `class index` + 置信度 + 中文名；内置约 40 个常见「家居/宠物/人物相关」类别的 `index → 中文名` 映射（ImageNet 1k 标签顺序），未命中显示 `class_{index}`；注释说明数据来源为 ImageNet 1k 标签顺序。
- `ui/AlbumActivity.java`：相册选图（`ActivityResultContracts`）→ 离线 Top-K 打标 → 展示类别与置信度。
- `ui/MonitorActivity.java`：CameraX 实时预览取帧，**控制分析帧率 + 丢帧背压（上一帧未处理完则跳过新帧，避免帧堆积）**，识别到猫/狗相关类别时记录抓拍日志（本阶段仅日志/UI 标记，不上报）。

## 五、阶段 3：真机多后端 Benchmark 与性能剖析

产出：在既有 `android_app` 上补充 benchmark 模块，并生成 `docs/model_contract.md`、`docs/benchmark_report.md`。

### 5.1 benchmark 模块（包 `com.edgevision.onnx.benchmark`）

- `BenchmarkRunner.java`：对比维度 = 模型（FP32 vs INT8）× EP（CPU 1/2/4 线程、XNNPACK、NNAPI）。**预热 5 帧后连续跑 50 帧**；分阶段计时并记录 **预处理耗时 + 纯推理耗时 + 后处理耗时 = 端到端总耗时**；汇总冷启动加载耗时、首帧耗时、稳态 p50 / p95。
- `ui/BenchmarkActivity.java`：一键跑分页面，展示上表；提供"导出 JSON"把结果写到应用私有目录便于 `adb pull`。

### 5.2 文档

- `docs/model_contract.md`：固化输入/输出 shape、NCHW 布局、归一化、ImageNet 1000 类、manifest SHA-256 契约（与 `AGENTS.md` 第三节一致）。
- `docs/benchmark_report.md`：**报告模板 + 占位符**。真机实测数据（机型/芯片/Android 版本/电量散热状态、各 EP 延迟 ms、PSS MB）一律用 `【实测 ms】` / `【实测 MB】` 占位，**严禁编造任何性能数字**；正文需包含「Android 15/16 已将 NNAPI deprecated、算子覆盖差异导致图切分回退、反量化开销」等诚实剖析要点。

## 六、阶段 4：轻量 Python FastAPI 端云联动服务

产出：`backend_service/` 下约 100~150 行的规范后端。

- `main.py`：
  - `POST /api/v1/events/pet`：接收端侧宠物抓拍事件（缩略图、类别、置信度、时间戳、设备 ID），支持 `Idempotency-Key` 幂等防重；SQLite 落库。
  - `GET /api/v1/events`：按时间 / 宠物类型分页查询。
  - `POST /api/v1/reports/daily`：汇总当日检测事件生成日报；调用 OpenAI 兼容 LLM 生成一句摘要，**无 API key 时降级为本地统计文本（不得硬编码 key，不得在无 key 时报错）**。
- `schemas.py`：Pydantic 输入/输出契约；统一错误码信封（`{code, message, data}`）；请求体大小上限校验。
- `requirements.txt`：`fastapi`、`uvicorn`、`pydantic`、`sqlite3`（标准库，不写依赖）、可选 `httpx`。

## 七、通用规范

1. 全部使用简体中文注释与文档。
2. **严禁编造**：真机性能数据、源码路径、库版本号（不确定的版本用已知存在的保守版本，见上文）。
3. **严禁修改只读区**：`model_pipeline/`、`artifacts/`、`AGENTS.md`、`run_codex_stages.sh`、`verify_pipeline.py`。
4. 只做本规范要求的最小实现，不额外造轮子、不加无关功能。
5. 完成后自检：Android 侧确保 Java 语法正确、Gradle 依赖坐标正确、与 `preprocess.py` 归一化对齐；后端侧确保可 `uvicorn` 启动、接口路径正确。

## 八、验收标准（由 `verify_pipeline.py` 复核）

- 资产：`manifest.json` 中每个模型的 `bytes` 与 `sha256` 与 `artifacts/` 下实际文件一致；`android_app` assets 与 `artifacts/` 一致。
- 模型契约：onnxruntime 可加载，输入 `images` [1,3,224,224] float32、输出 `logits` [1,1000] float32。
- Android：工程文件齐全、assets 三文件在位、关键类（OnnxClassifier/Preprocess/BenchmarkRunner）存在。
- 后端：`backend_service/main.py`、`schemas.py`、`requirements.txt` 存在且语法正确，三个路由出现。
