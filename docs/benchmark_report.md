# EdgeVision-ONNX 真机 Benchmark 与 NNAPI 剖析报告

> 状态：**待真机实测**。本文同时是报告模板与剖析框架。设备信息、延迟、内存和
> profiling 计数必须由 `BenchmarkActivity` 在目标真机运行后填写；未实际运行的数字
> 一律保持 `【实测 ms】` / `【实测 MB】` / `【实测值】` 占位，禁止估算、推测或从其他
> 机型搬运。

## 1. 测试环境

| 项目 | 填写值 |
| :--- | :--- |
| 机型 | 【机型】 |
| SoC / NPU | 【芯片】 |
| RAM | 【内存】 |
| Android 版本 / API | 【Android 版本】 |
| 系统构建号 | 【构建号】 |
| ABI | 【ABI】 |
| App 版本 / Git commit | 【版本】 |
| 电量 | 【电量】 |
| 机身温度 / 散热状态 | 【温度与散热状态】 |
| 是否充电 | 【是/否】 |
| 后台负载 | 【后台负载说明】 |

> 同一机型对比必须保持电量、温度、充电状态和后台负载尽量一致；建议先静置降温，
> 再按固定顺序连续执行至少 3 轮，报告中注明轮次与中位数来源。NNAPI 行为高度依赖
> OEM 驱动与系统版本，任何跨机型结论都必须重新实测。

## 2. 测试方法与 Profiling 接入

### 2.1 基线测量

- 模型：`mobilenet_v2_fp32.onnx` 与 `mobilenet_v2_int8_qdq.onnx`。
- Execution Provider：CPU 1/2/4 线程、XNNPACK、NNAPI。
- 每条配置独立加载 Session，记录冷启动；随后记录首帧。
- 首帧之后预热 **5 帧**，再连续测量 **50 帧**。
- 稳态统计只使用 50 帧，p50 / p95 采用 nearest-rank，不包含冷启动、首帧和预热帧。
- 单帧耗时定义：

  `端到端总耗时 = 预处理耗时 + 纯推理耗时 + 后处理耗时`

  - 预处理：`Preprocess.fromBitmap()`，包含 ORT 输入 Tensor 构造。
  - 纯推理：一次 `OrtSession.run()`。
  - 后处理：提取 logits、执行 Top-K（K=5），并释放 Result/输入 Tensor。
- 输入：App 内生成固定的 640×480 ARGB_8888 合成图，复用同一 Bitmap 以排除相册
  解码和文件 I/O 波动，再由统一预处理缩放到 224×224。

### 2.2 单帧 ONNX Runtime Profiling

`BenchmarkRunner` 在每条配置完成 50 帧稳态测量并释放正式 Session 后，额外创建
一个**独立 Session**，通过 `SessionOptions.enableProfiling(prefix)` 开启 profiling，
只执行一次完整的 `Preprocess -> session.run -> Top-K`，随后调用
`OrtSession.endProfiling()` 取得 trace JSON。

- profiling 配置同样经过 `OnnxClassifier` 的 NNAPI API 检查、初始化异常捕获和
  CPU 回退逻辑，因此 profile 反映的是**实际生效的 Session**，而不是请求的 EP。
- 单独开启 profiling 可以避免把 profiling 本身的记录开销混入 50 帧稳态延迟。
- ORT profile 为 Chrome trace 格式：`traceEvents` 中 `cat=Node` 的事件表示算子节点，
  `args.op_name` 是算子类型，`args.provider` 是该节点实际执行的 EP，`dur` 单位为
  **微秒**。
- `BenchmarkRunner` 会把 profile 压缩成摘要写入导出 JSON 的
  `results[].profiling.summary`，包括：

  - `actual_provider`：实际生效 EP，及是否发生回退；
  - `providers[]`：按 EP 聚合的节点数与耗时占比；
  - `top_nodes[]`：耗时最高的 15 个节点及其 `op_name` / `provider`；
  - `cpu_top_nodes[]`：耗时最高的 15 个 CPU EP 节点，用于定位被切回 CPU 的算子；
  - `cpu_provider_nodes` / `cpu_provider_time_percent`：CPU EP 节点数与耗时占比，
    用于判断 NNAPI/XNNPACK 是否只接管了部分子图；
  - `quantize_linear_nodes` / `dequantize_linear_nodes`：QDQ 图中量化和反量化节点数；
  - `memcpy_nodes` / `memcpy_total_us`：`MemcpyFromHost` / `MemcpyToHost` 等 EP
    边界拷贝节点的数量与耗时，是子图切分和跨设备搬运的直接证据；
  - `session_run_us`：profile 中一次 executor run 的时长，换算为毫秒后并入第 5 节。

- 原始 trace 文件路径见 `results[].profiling.profile_file`。若 profiling 本身失败，
  `results[].profiling.error` 会记录失败原因，主测量结果仍然保留。
- Node 的 `dur` 可能因多线程或异步执行而重叠，`providers[]` 的百分比以节点耗时
  总和为分母，只适合做相对归因；真正的墙钟耗时看 `session_run_us` 与 50 帧稳态统计。

## 3. 总体结果

### 3.1 FP32

| EP / 线程 | 实际生效 EP | 冷启动 | 首帧 | 稳态总耗时 p50 | 稳态总耗时 p95 | PSS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| CPU ×1 | 【实际 EP / 回退状态】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 MB】 |
| CPU ×2 | 【实际 EP / 回退状态】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 MB】 |
| CPU ×4 | 【实际 EP / 回退状态】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 MB】 |
| XNNPACK | 【实际 EP / 回退状态】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 MB】 |
| NNAPI | 【实际 EP / 回退状态】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 MB】 |

### 3.2 INT8 QDQ

| EP / 线程 | 实际生效 EP | 冷启动 | 首帧 | 稳态总耗时 p50 | 稳态总耗时 p95 | PSS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| CPU ×1 | 【实际 EP / 回退状态】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 MB】 |
| CPU ×2 | 【实际 EP / 回退状态】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 MB】 |
| CPU ×4 | 【实际 EP / 回退状态】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 MB】 |
| XNNPACK | 【实际 EP / 回退状态】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 MB】 |
| NNAPI | 【实际 EP / 回退状态】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 ms】 | 【实测 MB】 |

## 4. 分阶段耗时

表中每格填写 `p50 / p95`，单位为 ms。

> 每一帧都满足“总耗时 = 预处理 + 纯推理 + 后处理”，但各阶段的分位数不能直接相加：
> 三个阶段的 p95 可能来自不同帧，因此不要用“预处理 p95 + 推理 p95 + 后处理 p95”
> 代替端到端 p95。总耗时 p50/p95 必须由每帧总耗时单独计算。

| 模型 / EP | 预处理 | 纯推理 | 后处理 | 端到端 |
| :--- | :--- | :--- | :--- | :--- |
| FP32 / CPU ×1 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 |
| FP32 / CPU ×2 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 |
| FP32 / CPU ×4 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 |
| FP32 / XNNPACK | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 |
| FP32 / NNAPI | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 |
| INT8 / CPU ×1 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 |
| INT8 / CPU ×2 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 |
| INT8 / CPU ×4 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 |
| INT8 / XNNPACK | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 |
| INT8 / NNAPI | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 | 【实测 ms】 / 【实测 ms】 |

## 5. Profiling 证据表

本节用于把“体感慢”变成可归因的证据。所有计数和耗时均从
`results[].profiling.summary` 与原始 trace JSON 填写，不得手填估计值。

> CPU-only 配置中的 CPU 节点就是全部节点，不代表“回退”；只有请求 XNNPACK/NNAPI
> 时出现的 CPU 节点，才表示该子图未被目标 EP 接管或初始化失败后回退。

| 模型 / 请求 EP | 实际 Session | 总节点数 | CPU EP 节点数 | CPU 节点耗时占比 | Q 节点数 | DQ 节点数 | 单帧 executor run |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 / CPU ×4 | 【实测值】 | 【实测值】 | 【实测值】 | 【实测值】% | 【实测值】 | 【实测值】 | 【实测 ms】 |
| FP32 / XNNPACK | 【实测值】 | 【实测值】 | 【实测值】 | 【实测值】% | 【实测值】 | 【实测值】 | 【实测 ms】 |
| FP32 / NNAPI | 【实测值】 | 【实测值】 | 【实测值】 | 【实测值】% | 【实测值】 | 【实测值】 | 【实测 ms】 |
| INT8 / XNNPACK | 【实测值】 | 【实测值】 | 【实测值】 | 【实测值】% | 【实测值】 | 【实测值】 | 【实测 ms】 |
| INT8 / NNAPI | 【实测值】 | 【实测值】 | 【实测值】 | 【实测值】% | 【实测值】 | 【实测值】 | 【实测 ms】 |

补充证据：

- FP32 NNAPI 的 `top_nodes` 中，CPU EP 节点及其算子名：`【从 profile 摘录】`。
- INT8 NNAPI 初始化失败时的异常摘要：`【从结果 JSON 的 actual_provider / profiling.error 摘录】`。
- INT8 NNAPI 若回退 CPU，实际生效 EP 与 profile 的 `actual_provider`：`【实测值】`。
- XNNPACK 下仍由 CPU 执行的节点数与算子名：`【从 profile 摘录】`。
- NNAPI 子图边界的 `Memcpy` 节点数：`【实测值】`；拷贝耗时合计：`【实测 ms】`。

## 6. NNAPI 劣化根因剖析

本节解释两类现场现象：**FP32 下 NNAPI 明显慢于 XNNPACK**，以及
**INT8 QDQ 下 NNAPI 初始化失败并回退 CPU**。具体数值以上表实测为准，这里只给出
可从 profiling 验证的因果链。

### 6.1 根因链：算子覆盖 → 子图切分 → CPU 回退 + 跨设备开销

1. **算子覆盖不是“全有或全无”。** NNAPI EP 能把哪些节点交给加速器，取决于
   ORT 版本、Android API level、OEM 驱动和该驱动的算子/数据类型支持。请求 NNAPI
   只代表“允许尝试”，不代表整张 MobileNetV2 都在 NPU/DSP 上执行。
2. **不支持节点导致图切分。** 只要部分算子无法被 NNAPI 接管，ORT 就会把计算图拆成
   多个 NNAPI 子图和 CPU 子图。`profiling.summary.providers[]` 同时出现
   `NNAPIExecutionProvider` 与 `CPUExecutionProvider`，就是图切分的直接证据。
3. **切分产生跨设备数据搬运和同步。** 子图边界处需要把张量从 CPU 内存搬到加速器
   可见内存，必要时还要做 layout/类型转换，等待异步任务完成后再搬回。对小模型而言，
   这些固定开销可能超过算子本身的计算收益。
4. **局部加速可能整体更慢。** 如果 `cpu_provider_time_percent` 很高，说明大量时间
   仍花在 CPU 回退节点上，同时还额外支付了子图切换、拷贝和驱动调度成本；此时 NNAPI
   端到端慢于 XNNPACK/CPU 是合理结果。应以 `top_nodes` 中实际回退的算子为准，而不是
   凭经验预判某个算子一定不支持。
5. **首帧与驱动编译进一步放大差距。** 部分 NNAPI 驱动会在首次执行时编译/调优子图，
   使首帧耗时显著高于稳态；报告必须把冷启动、首帧和稳态 p50/p95 分开看。

### 6.2 为什么 INT8 QDQ 更容易触发 NNAPI 初始化失败

- PTQ 产物是 QDQ 图，图中显式包含 `QuantizeLinear` / `DequantizeLinear` 以及量化
  参数。NNAPI EP 对量化模式、算子版本、张量布局和混合精度的支持范围由驱动决定；
  某些驱动只接受特定量化算子组合，遇到不支持的组合会在 `createSession` 阶段直接失败。
- 失败发生在**初始化**而不是单帧推理：`OnnxClassifier` 捕获异常后回退到 CPU，并把
  `actual_provider` 标记为“CPU（NNAPI 初始化失败，已回退）”。因此结果表中出现
  回退描述时，INT8/NNAPI 的性能数据实际上是 CPU 数据，不能当作 NPU 数据解读。
- 如果 NNAPI 初始化失败但 CPU 回退 Session 成功，profiling 仍会生成；此时
  `providers` 应只有 CPU，`actual_provider` 应显示回退状态。若连回退 Session 也无法
  profiling，则 `profiling.error` 会保留异常信息。
- 失败本身也是有价值的结论：在该机型/系统组合上，INT8 + NNAPI 不是可用加速路径，
  不应为了“用了 NPU”而保留一个必然回退的默认路由。

### 6.3 反量化与拷贝开销：INT8 不一定更快

- QDQ 模型权重更小，磁盘和内存占用可能下降，但这不等于端到端延迟下降。
- 当某个算子没有可用的 int8 kernel 时，ORT 需要先执行 `DequantizeLinear`，以浮点
  执行该算子，必要时再 `QuantizeLinear`；这些节点会计入 `node_total_us`，并在
  `quantize_linear_nodes` / `dequantize_linear_nodes` 中体现。
- XNNPACK 对部分 int8 卷积有专门 kernel，但算子覆盖同样有限；未被接管的节点仍会
  回退 CPU。NNAPI 若发生图切分，还可能叠加量化/布局转换与跨设备拷贝。
- 因此必须同时比较：分阶段耗时、端到端 p50/p95、PSS、profile 中 CPU 节点占比和
  Q/DQ 节点数。只看模型文件大小或单次最好成绩会得出错误结论。

### 6.4 Android 15/16 的平台事实与影响

- Android 15 起 NNAPI 已被标记为 deprecated，Android 16 延续该状态。NNAPI 不再是
  新项目推荐的通用加速入口，后续版本对 HAL/驱动的投入和兼容性存在不确定性。
- 在 Android 15/16 设备上，NNAPI 可能由兼容实现或较旧的驱动路径承接；同一 APK 在
  不同 OEM、不同系统版本上的图切分和回退行为可能不同，必须以当次 profile 为准。
- Android 15/16 引入 16 KB page size 等变化，原生库和驱动都需要匹配。项目使用的
  `onnxruntime-android:1.22.0` 已适配 16 KB page size，但 NNAPI 驱动是否同样兼容不由
  App 决定，仍需真机验证。

## 7. EP 选型结论

### 7.1 决策矩阵

| 场景 | 首选 | 回退顺序 | 结论 |
| :--- | :--- | :--- | :--- |
| 通用 ARM Android 手机 | INT8 QDQ + XNNPACK | FP32 + XNNPACK → CPU ×N | 默认生产路径；以本机 p95 和 profile 为准 |
| XNNPACK 覆盖不足 | FP32/INT8 + CPU ×2/×4 | 按实测 p95 选线程数 | 小模型上多线程未必线性加速，关注 p95 与热稳定性 |
| NNAPI 请求但 CPU 节点占比高 | 不采用 NNAPI | 回到 XNNPACK/CPU | 图切分和拷贝开销可能抵消加速收益 |
| INT8 NNAPI 初始化失败 | 不采用 NNAPI | CPU（记录回退状态） | 该组合不可作为发布路径 |
| 特定旗舰芯片追求 NPU | 厂商离线编译工具链（如 QNN / RKNN 等） | 按厂商 SDK 的量化与算子约束评估 | NNAPI 不再是首选；需单独做模型转换与精度回归 |

### 7.2 结论

- **通用 ARM 手机优先 INT8 QDQ + XNNPACK**。它不依赖单一 OEM 的 NNAPI 驱动，
  算子覆盖和回退行为更可控；但必须用 profile 确认 int8 节点确实被 XNNPACK 接管，
  并以端到端 p95 而不是模型体积作为验收标准。
- **NNAPI 仅作为兼容性/实验路径**。若 FP32 NNAPI 的 CPU 节点耗时占比高，或 INT8
  NNAPI 在初始化阶段失败，应直接从生产路由中移除，而不是依赖运行时“碰运气”回退。
- **特定芯片的 NPU 加速应走厂商 QNN / RKNN 等离线编译工具链**。这些工具链能在
  编译期确定算子支持、量化约束和内存布局，避免 NNAPI 运行时的图切分不确定性；代价
  是需要按芯片维护多套产物和精度回归。
- 本机最终选择：`【填写：例如 INT8 + XNNPACK，或 FP32 + CPU ×N】`；
  依据是稳态 p50 `【实测 ms】`、p95 `【实测 ms】`、PSS `【实测 MB】` 以及
  NNAPI profile 的 CPU 节点占比 `【实测值】%`。

## 8. 必须诚实说明的限制

- **实际 Provider 可能不是完整图执行。** 请求 NNAPI/XNNPACK 不代表整张模型在该 EP
  上运行；`actual_provider` 与 profile 的 `args.provider` 才是事实来源。
- **NNAPI 已 deprecated。** Android 15/16 起不应把 NNAPI 当作唯一或默认加速路径；
  部分设备/系统版本可能行为不同，必须保留 CPU 回退并记录真实状态。
- **INT8 QDQ 存在反量化开销。** 文件更小不等于延迟更低，应同时看预处理、纯推理、
  后处理、端到端 p50/p95 和 PSS。
- **线程越多不一定越快。** 小图 MobileNetV2 可能受调度、功耗墙、热降频和大核唤醒
  影响；重点观察 p95 与连续多轮稳定性。
- **首帧与冷启动受系统状态影响大。** 资产 SHA-256 校验、模型读取、页缓存、CPU 调频
  和后台负载都会影响这两个指标，不能与稳态推理耗时混为一谈。
- **合成图只控制输入波动。** 它避免相册解码差异，适合横向比较，但不代表真实相册
  图片或 CameraX 帧转换的完整成本；CameraX 相关 I/O 需要在业务链路上另行测量。
- **PSS 是进程级观测。** 它包含 Activity、Bitmap、运行时和当前模型 Session，不能
  直接等同于单个模型的权重内存；不同配置间应保持测量时点一致。
- **profiling 会改变执行环境。** 单帧 trace 用于定位节点与 EP，不应把其耗时直接当作
  稳态 p50；正式延迟结论以 50 帧稳态统计为准。
- **本文不得填写估算值。** 未在上述环境中实际运行产生的数字一律保持占位符。

## 9. 导出与复现

1. 打开 App，完成启动资产校验。
2. 进入「多后端 Benchmark」，点击「一键开始跑分」。
3. 跑分完成后点击「导出 JSON」，页面会显示应用私有目录中的绝对路径。
4. 使用 `adb pull <页面显示的绝对路径> .` 取回 benchmark JSON。
5. 从 JSON 的 `results[].profiling.profile_file` 读取每条配置的 ORT trace 路径，
   再用 `adb pull <profile_file> .` 取回原始 profiling JSON。

导出的 benchmark JSON 包含设备信息、测试方法、每条配置的 `actual_provider`、冷启动、
首帧、阶段 p50/p95、端到端 p50/p95、PSS，以及 profiling 的文件路径、EP 聚合、
Top-N 节点和 Q/DQ 节点计数。原始 profile 与 benchmark JSON 应作为本报告附件保存。

## 10. 结论确认

- [ ] 已确认实际生效 EP 与回退状态。
- [ ] 已注明电量、温度和后台负载。
- [ ] 已至少执行 3 轮并保留原始 JSON。
- [ ] 已取回每条配置的 profiling 原始文件。
- [ ] 已用 profile 的 `args.provider` 核实 CPU 回退节点，而非凭经验推断。
- [ ] 未填写任何未实测的延迟或内存数字。
