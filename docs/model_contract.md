# EdgeVision-ONNX 模型契约

## 1. 适用范围

本文固化阶段 1 产出的 MobileNetV2 分类模型在 Python 流水线与 Android ONNX Runtime
端侧之间的输入、输出、预处理、标签顺序与资产完整性契约。运行时以本文和
`artifacts/manifest.json` 为准。

当前范围仅包含 ImageNet 1000 类分类，不包含检测框、NMS 或分割输出。宠物监控页面
复用分类 logits 中 ImageNet 猫/犬类别索引，不构成目标检测。

## 2. 模型与资产清单

- 模型结构：MobileNetV2（torchvision 预训练权重）
- ONNX opset：17（来自 `artifacts/manifest.json`）
- 量化方式：INT8 PTQ，QDQ 格式
- 资产来源：`artifacts/`，Android 打包副本：`android_app/app/src/main/assets/`

| Manifest 键 | 文件名 | 字节数 | SHA-256 |
| :--- | :--- | ---: | :--- |
| `fp32` | `mobilenet_v2_fp32.onnx` | 13,992,265 | `ec32b3edb4811e0ca293d5b86d8665d049dbe6d2b9fa9bfbab75d8f80d4b8936` |
| `int8_qdq` | `mobilenet_v2_int8_qdq.onnx` | 3,936,843 | `4c195b8277bb4bf60cfd8c808b58955470590e9f253b8877fd47be61777d0f15` |

> 已知元数据提示：当前 `artifacts/manifest.json` 的 `model_name` 字段为
> `mobilenet_v3_small`，与项目约定、模型文件名不一致。该字段不参与模型加载、
> I/O 契约或 SHA-256 校验；`artifacts/` 为只读区，本文不改写该元数据，建议由阶段 1
> 负责人后续修正。

## 3. 输入契约

| 项目 | 固定值 |
| :--- | :--- |
| 输入名 | `images` |
| Shape | `[1, 3, 224, 224]` |
| 布局 | NCHW |
| 数据类型 | `float32` |
| 颜色空间 | RGB |
| 归一化 | ImageNet mean/std |

预处理顺序必须与 `model_pipeline/preprocess.py` 一致：

1. 读取图片并按 EXIF Orientation 校正方向。
2. 转为 RGB。
3. 保持长宽比缩放：短边双线性缩放到 `256`，长边等比放大。
4. 中心裁剪到 `224 × 224`。
5. 像素值除以 `255.0`，映射到 `[0, 1]`。
6. 按通道执行标准化：

   `x_c = (pixel_c / 255.0 - mean_c) / std_c`

   其中 `mean = [0.485, 0.456, 0.406]`，`std = [0.229, 0.224, 0.225]`。

7. HWC 转 NCHW，写入连续 `float32` 缓冲区：

   `index = c × 224 × 224 + y × 224 + x`

> 第 3、4 步采用 torchvision 评测标准（`Resize(256)` + `CenterCrop(224)`）而非直接拉伸到
> `224 × 224`。直接拉伸会让长宽比悬殊的图片（如竖构图手机照片）严重变形，实测会把无关
> 类别挤进 Top-5。

Android 端由 `ml/Preprocess.java` 执行相同语义；`ml/OnnxClassifier.java` 以
`OnnxTensor.createTensor(..., [1, 3, 224, 224])` 构造输入。

## 4. 输出契约

| 项目 | 固定值 |
| :--- | :--- |
| 输出名 | `logits` |
| Shape | `[1, 1000]` |
| 数据类型 | `float32` |
| 类别空间 | ImageNet 2012 1k 标准顺序 |

模型输出未归一化 logits。Android 展示层对 logits 执行 softmax 后取 Top-K；
`class index` 为 0 到 999 的 ImageNet 顺序索引。中文映射未覆盖的类别显示为
`class_{index}`。

## 5. 资产完整性契约

`manifest.json` 中每个模型都声明 `filename`、`bytes` 与 `sha256`。Android 端必须
在加载前校验文件读取后的字节数与 SHA-256，任一不符即拒绝推理。

- `MainActivity` 启动时调用 `AssetVerifier.verifyAll()` 校验全部模型。
- `OnnxClassifier` 创建 Session 前再次按 manifest 校验目标模型。
- `android_app/app/src/main/assets/` 下三个文件（两个 ONNX 与 `manifest.json`）必须
  与 `artifacts/` 保持一致。

## 6. Execution Provider 契约

- `CPUExecutionProvider`：按 benchmark 配置使用 1、2 或 4 个 intra-op 线程。
- `XNNPACKExecutionProvider`：由 ONNX Runtime 管理内部线程；失败时回退 CPU。
- `NNAPIExecutionProvider`：仅 Android API 27+ 尝试启用；低于 API 27 或初始化失败时
  回退 CPU，并在实际 Provider 描述中标注回退原因。

请求的 Provider 与实际生效的 Provider 不是同一概念。NNAPI 可能只接管部分算子，
图中不支持的节点会回退到 CPU；benchmark JSON 的 `actual_provider` 和报告中的回退
说明必须作为结果解释的一部分。

## 7. 非目标

- 不输出边界框、类别框置信度或 NMS 结果。
- 不在阶段 3 修改模型、量化参数、输入尺寸或类别顺序。
- 不用 benchmark 结果反推或替换阶段 1 的数值对齐结论。
