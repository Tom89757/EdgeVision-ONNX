package com.edgevision.onnx.benchmark;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Debug;
import android.text.TextUtils;

import com.edgevision.onnx.ml.OnnxClassifier;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * 真机多后端 benchmark 执行器。
 *
 * 每条配置先单独加载 Session 并计时冷启动，再测量首帧；随后预热 5 帧，
 * 连续测量 50 帧。每帧耗时严格拆为预处理、纯推理、后处理三部分。
 *
 * 稳态测量结束后，再用独立 Session 开启 ONNX Runtime profiling 跑单帧，
 * 导出包含 Node/EP 耗时的 trace JSON，并把 provider 与 Top-N 节点汇总写入结果。
 */
public final class BenchmarkRunner {
    public static final int WARMUP_FRAMES = 5;
    public static final int MEASURED_FRAMES = 50;
    public static final int TOP_K = 5;

    private final Context appContext;

    public interface ProgressListener {
        void onProgress(String message);
    }

    public BenchmarkRunner(Context context) {
        if (context == null) {
            throw new IllegalArgumentException("Context 不能为空");
        }
        this.appContext = context.getApplicationContext();
    }

    public static final class Report {
        private final long generatedAtMs;
        private final String deviceManufacturer;
        private final String deviceModel;
        private final String androidVersion;
        private final int sdkInt;
        private final String supportedAbis;
        private final List<Result> results;

        private Report(long generatedAtMs, List<Result> results) {
            this.generatedAtMs = generatedAtMs;
            this.deviceManufacturer = Build.MANUFACTURER;
            this.deviceModel = Build.MODEL;
            this.androidVersion = Build.VERSION.RELEASE;
            this.sdkInt = Build.VERSION.SDK_INT;
            this.supportedAbis = TextUtils.join(",", Build.SUPPORTED_ABIS);
            this.results = Collections.unmodifiableList(new ArrayList<>(results));
        }

        public long getGeneratedAtMs() {
            return generatedAtMs;
        }

        public String getDeviceLabel() {
            return deviceManufacturer + " " + deviceModel
                    + " / Android " + androidVersion + " (API " + sdkInt + ")";
        }

        public List<Result> getResults() {
            return results;
        }

        public String getSupportedAbis() {
            return supportedAbis;
        }
    }

    public static final class Result {
        private final OnnxClassifier.Model model;
        private final OnnxClassifier.ExecutionProvider requestedProvider;
        private final int requestedThreads;
        private final String actualProvider;
        private final long coldLoadNanos;
        private final long firstFrameNanos;
        private final double preprocessP50Ms;
        private final double preprocessP95Ms;
        private final double inferenceP50Ms;
        private final double inferenceP95Ms;
        private final double postprocessP50Ms;
        private final double postprocessP95Ms;
        private final double totalP50Ms;
        private final double totalP95Ms;
        private final double pssMb;
        private final String error;
        private final ProfilingCapture profiling;

        private Result(
                OnnxClassifier.Model model,
                OnnxClassifier.ExecutionProvider requestedProvider,
                int requestedThreads,
                String actualProvider,
                long coldLoadNanos,
                long firstFrameNanos,
                double preprocessP50Ms,
                double preprocessP95Ms,
                double inferenceP50Ms,
                double inferenceP95Ms,
                double postprocessP50Ms,
                double postprocessP95Ms,
                double totalP50Ms,
                double totalP95Ms,
                double pssMb,
                String error,
                ProfilingCapture profiling) {
            this.model = model;
            this.requestedProvider = requestedProvider;
            this.requestedThreads = requestedThreads;
            this.actualProvider = actualProvider;
            this.coldLoadNanos = coldLoadNanos;
            this.firstFrameNanos = firstFrameNanos;
            this.preprocessP50Ms = preprocessP50Ms;
            this.preprocessP95Ms = preprocessP95Ms;
            this.inferenceP50Ms = inferenceP50Ms;
            this.inferenceP95Ms = inferenceP95Ms;
            this.postprocessP50Ms = postprocessP50Ms;
            this.postprocessP95Ms = postprocessP95Ms;
            this.totalP50Ms = totalP50Ms;
            this.totalP95Ms = totalP95Ms;
            this.pssMb = pssMb;
            this.error = error;
            this.profiling = profiling;
        }

        private static Result failed(
                Configuration configuration, String actualProvider, String error) {
            return new Result(
                    configuration.model,
                    configuration.provider,
                    configuration.threads,
                    actualProvider,
                    0L,
                    0L,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    error,
                    null);
        }

        public OnnxClassifier.Model getModel() {
            return model;
        }

        public OnnxClassifier.ExecutionProvider getRequestedProvider() {
            return requestedProvider;
        }

        public int getRequestedThreads() {
            return requestedThreads;
        }

        public String getActualProvider() {
            return actualProvider;
        }

        public double getColdLoadMs() {
            return coldLoadNanos / 1_000_000.0;
        }

        public double getFirstFrameMs() {
            return firstFrameNanos / 1_000_000.0;
        }

        public double getPreprocessP50Ms() {
            return preprocessP50Ms;
        }

        public double getPreprocessP95Ms() {
            return preprocessP95Ms;
        }

        public double getInferenceP50Ms() {
            return inferenceP50Ms;
        }

        public double getInferenceP95Ms() {
            return inferenceP95Ms;
        }

        public double getPostprocessP50Ms() {
            return postprocessP50Ms;
        }

        public double getPostprocessP95Ms() {
            return postprocessP95Ms;
        }

        public double getTotalP50Ms() {
            return totalP50Ms;
        }

        public double getTotalP95Ms() {
            return totalP95Ms;
        }

        public double getPssMb() {
            return pssMb;
        }

        public String getError() {
            return error;
        }

        public String getProfilingPath() {
            return profiling == null ? null : profiling.path;
        }

        public String getProfilingError() {
            return profiling == null ? null : profiling.error;
        }

        public boolean isSuccessful() {
            return error == null;
        }

        public String getConfigurationLabel() {
            String providerLabel = requestedProvider == OnnxClassifier.ExecutionProvider.CPU
                    ? "CPU ×" + requestedThreads + " 线程"
                    : requestedProvider.name() + "（线程自动）";
            return model.getDisplayName() + " / " + providerLabel;
        }
    }

    /** 单帧 profiling 的导出文件与汇总；失败时保留错误信息，不影响主测量结果。 */
    private static final class ProfilingCapture {
        private final String path;
        private final JSONObject summary;
        private final String error;

        private ProfilingCapture(String path, JSONObject summary, String error) {
            this.path = path;
            this.summary = summary;
            this.error = error;
        }
    }

    public Report run(Bitmap benchmarkBitmap, ProgressListener listener) {
        if (benchmarkBitmap == null || benchmarkBitmap.isRecycled()) {
            throw new IllegalArgumentException("benchmark Bitmap 不可用");
        }

        List<Configuration> configurations = buildConfigurations();
        List<Result> results = new ArrayList<>(configurations.size());
        for (int index = 0; index < configurations.size(); index++) {
            Configuration configuration = configurations.get(index);
            notifyProgress(listener, "正在测试 " + configuration.label
                    + "（" + (index + 1) + "/" + configurations.size() + "）");
            results.add(runConfiguration(configuration, benchmarkBitmap));
        }
        notifyProgress(listener, "全部配置测试完成");
        return new Report(System.currentTimeMillis(), results);
    }

    public static String toJson(Report report) throws JSONException {
        JSONObject root = new JSONObject();
        root.put("schema_version", 1);
        root.put("generated_at_ms", report.generatedAtMs);

        JSONObject device = new JSONObject();
        device.put("manufacturer", report.deviceManufacturer);
        device.put("model", report.deviceModel);
        device.put("android_version", report.androidVersion);
        device.put("sdk_int", report.sdkInt);
        device.put("supported_abis", report.supportedAbis);
        root.put("device", device);

        JSONObject method = new JSONObject();
        method.put("warmup_frames", WARMUP_FRAMES);
        method.put("measured_frames", MEASURED_FRAMES);
        method.put("top_k", TOP_K);
        method.put("timing_definition", "预处理(含输入 Tensor 构造) + session.run + 后处理");
        method.put("input", "640x480 ARGB_8888 合成图，缩放到 224x224");
        method.put("profiling",
                "稳态测量后用独立 Session 开启 enableProfiling 并执行单帧；"
                        + "profile JSON 的 Node.dur 单位为微秒，args.provider 为实际执行 EP");
        root.put("method", method);

        JSONArray resultArray = new JSONArray();
        for (Result result : report.results) {
            JSONObject item = new JSONObject();
            item.put("model", result.model.getDisplayName());
            item.put("model_key", result.model.getAssetKey());
            item.put("requested_provider", result.requestedProvider.name());
            item.put("requested_threads", result.requestedThreads);
            item.put("actual_provider", result.actualProvider);
            item.put("success", result.isSuccessful());
            if (result.isSuccessful()) {
                item.put("cold_load_ms", round(result.getColdLoadMs()));
                item.put("first_frame_ms", round(result.getFirstFrameMs()));
                item.put("pss_mb", round(result.getPssMb()));
                item.put("p50", stageJson(
                        result.preprocessP50Ms,
                        result.inferenceP50Ms,
                        result.postprocessP50Ms,
                        result.totalP50Ms));
                item.put("p95", stageJson(
                        result.preprocessP95Ms,
                        result.inferenceP95Ms,
                        result.postprocessP95Ms,
                        result.totalP95Ms));
            } else {
                item.put("error", result.error);
            }
            if (result.profiling != null) {
                JSONObject profilingJson = new JSONObject();
                if (result.profiling.path != null) {
                    profilingJson.put("profile_file", result.profiling.path);
                }
                if (result.profiling.summary != null) {
                    profilingJson.put("summary", result.profiling.summary);
                }
                if (result.profiling.error != null) {
                    profilingJson.put("error", result.profiling.error);
                }
                item.put("profiling", profilingJson);
            }
            resultArray.put(item);
        }
        root.put("results", resultArray);
        return root.toString(2);
    }

    private static JSONObject stageJson(
            double preprocessMs, double inferenceMs, double postprocessMs, double totalMs)
            throws JSONException {
        JSONObject stage = new JSONObject();
        stage.put("preprocess_ms", round(preprocessMs));
        stage.put("inference_ms", round(inferenceMs));
        stage.put("postprocess_ms", round(postprocessMs));
        stage.put("total_ms", round(totalMs));
        return stage;
    }

    private Result runConfiguration(Configuration configuration, Bitmap benchmarkBitmap) {
        OnnxClassifier classifier = OnnxClassifier.getInstance(
                appContext,
                configuration.model,
                configuration.provider,
                configuration.threads);
        String actualProvider = "未加载";
        try {
            long loadStart = System.nanoTime();
            classifier.ensureSessionLoaded();
            long coldLoadNanos = System.nanoTime() - loadStart;
            actualProvider = classifier.getActualProvider();

            OnnxClassifier.TimedClassification first =
                    classifier.classifySync(benchmarkBitmap, TOP_K);
            long firstFrameNanos = first.getTotalNanos();

            for (int i = 0; i < WARMUP_FRAMES; i++) {
                classifier.classifySync(benchmarkBitmap, TOP_K);
            }

            long[] preprocess = new long[MEASURED_FRAMES];
            long[] inference = new long[MEASURED_FRAMES];
            long[] postprocess = new long[MEASURED_FRAMES];
            long[] total = new long[MEASURED_FRAMES];
            for (int i = 0; i < MEASURED_FRAMES; i++) {
                OnnxClassifier.TimedClassification timed =
                        classifier.classifySync(benchmarkBitmap, TOP_K);
                preprocess[i] = timed.getPreprocessNanos();
                inference[i] = timed.getInferenceNanos();
                postprocess[i] = timed.getPostprocessNanos();
                total[i] = timed.getTotalNanos();
            }

            double[] preprocessSummary = percentileSummary(preprocess);
            double[] inferenceSummary = percentileSummary(inference);
            double[] postprocessSummary = percentileSummary(postprocess);
            double[] totalSummary = percentileSummary(total);
            double pssMb = currentPssMb();

            classifier.releaseSession();
            ProfilingCapture profiling = captureProfiling(configuration, benchmarkBitmap);

            return new Result(
                    configuration.model,
                    configuration.provider,
                    configuration.threads,
                    actualProvider,
                    coldLoadNanos,
                    firstFrameNanos,
                    preprocessSummary[0],
                    preprocessSummary[1],
                    inferenceSummary[0],
                    inferenceSummary[1],
                    postprocessSummary[0],
                    postprocessSummary[1],
                    totalSummary[0],
                    totalSummary[1],
                    pssMb,
                    null,
                    profiling);
        } catch (Throwable error) {
            actualProvider = classifier.getActualProvider();
            String message = error.getMessage() == null
                    ? error.getClass().getSimpleName() : error.getMessage();
            return Result.failed(configuration, actualProvider, message);
        } finally {
            classifier.releaseSession();
        }
    }

    /**
     * 用独立 Session 跑单帧 profiling。这里显式调用 SessionOptions.enableProfiling，
     * 因此 profile 文件只覆盖一次 session.run，不会混入 50 帧稳态测量的噪声。
     */
    private ProfilingCapture captureProfiling(Configuration configuration, Bitmap benchmarkBitmap) {
        String profilePath = null;
        try {
            File directory = profilingDirectory();
            String prefixName = String.format(
                    Locale.ROOT,
                    "profile_%s_%s_%d_%d",
                    configuration.model.getAssetKey(),
                    configuration.provider.name().toLowerCase(Locale.ROOT),
                    configuration.threads,
                    System.currentTimeMillis());
            String prefixPath = new File(directory, prefixName).getAbsolutePath();

            OnnxClassifier profiler = OnnxClassifier.createTransient(
                    appContext,
                    configuration.model,
                    configuration.provider,
                    configuration.threads,
                    options -> options.enableProfiling(prefixPath));
            try {
                profiler.ensureSessionLoaded();
                profiler.classifySync(benchmarkBitmap, TOP_K);
                profilePath = profiler.endProfiling();
                if (profilePath == null || !new File(profilePath).isFile()) {
                    profilePath = findProfileFile(directory, prefixName);
                }
                JSONObject summary = summarizeProfile(profilePath, profiler.getActualProvider());
                return new ProfilingCapture(profilePath, summary, null);
            } finally {
                profiler.releaseSession();
            }
        } catch (Throwable error) {
            String message = error.getMessage() == null
                    ? error.getClass().getSimpleName() : error.getMessage();
            return new ProfilingCapture(profilePath, null, message);
        }
    }

    private File profilingDirectory() throws IOException {
        File directory = appContext.getExternalFilesDir("profiling");
        if (directory == null) {
            directory = new File(appContext.getFilesDir(), "profiling");
        }
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IOException("无法创建 profiling 目录：" + directory);
        }
        if (!directory.isDirectory()) {
            throw new IOException("profiling 路径不是目录：" + directory);
        }
        return directory;
    }

    /** end_profiling() 未返回可用路径时，按前缀选择目录中最新的 profile 文件。 */
    private static String findProfileFile(File directory, String prefixName) {
        File[] files = directory.listFiles();
        File newest = null;
        if (files != null) {
            for (File file : files) {
                if (!file.isFile() || !file.getName().startsWith(prefixName)) {
                    continue;
                }
                if (newest == null || file.lastModified() > newest.lastModified()) {
                    newest = file;
                }
            }
        }
        return newest == null ? null : newest.getAbsolutePath();
    }

    /**
     * 把 ORT Chrome trace 格式的 profile 压缩成便于报告引用的摘要。
     * Node 事件的 dur 单位为微秒；provider 字段即该节点实际执行的 EP。
     */
    private static JSONObject summarizeProfile(String profilePath, String actualProvider)
            throws IOException, JSONException {
        if (profilePath == null) {
            throw new IOException("ONNX Runtime 未返回 profiling 文件路径");
        }
        File profileFile = new File(profilePath);
        if (!profileFile.isFile()) {
            throw new IOException("profiling 文件不存在：" + profilePath);
        }

        String content = new String(
                Files.readAllBytes(profileFile.toPath()), StandardCharsets.UTF_8);
        JSONObject root = new JSONObject(content);
        JSONArray events = root.optJSONArray("traceEvents");
        if (events == null) {
            throw new JSONException("profiling JSON 缺少 traceEvents");
        }

        Map<String, ProviderTotal> providerTotals = new LinkedHashMap<>();
        List<NodeTiming> nodes = new ArrayList<>();
        long sessionRunUs = 0L;
        long totalNodeUs = 0L;
        int quantizeNodes = 0;
        int dequantizeNodes = 0;
        int memcpyNodes = 0;
        long memcpyUs = 0L;

        for (int index = 0; index < events.length(); index++) {
            JSONObject event = events.optJSONObject(index);
            if (event == null) {
                continue;
            }
            String category = event.optString("cat", "");
            String name = event.optString("name", "");
            long durationUs = Math.max(0L, event.optLong("dur", 0L));

            if ("Session".equals(category) && name.startsWith("sequential_executor")) {
                sessionRunUs += durationUs;
                continue;
            }
            if (!"Node".equals(category)) {
                continue;
            }

            JSONObject args = event.optJSONObject("args");
            String provider = args == null ? "" : args.optString("provider", "");
            String opName = args == null ? "" : args.optString("op_name", "");
            if (provider.isEmpty()) {
                provider = "unknown";
            }
            if (opName.startsWith("QuantizeLinear")) {
                quantizeNodes++;
            } else if (opName.startsWith("DequantizeLinear")) {
                dequantizeNodes++;
            }
            if (opName.contains("Memcpy")) {
                memcpyNodes++;
                memcpyUs += durationUs;
            }

            ProviderTotal providerTotal = providerTotals.get(provider);
            if (providerTotal == null) {
                providerTotal = new ProviderTotal();
                providerTotals.put(provider, providerTotal);
            }
            providerTotal.nodeCount++;
            providerTotal.totalUs += durationUs;
            totalNodeUs += durationUs;
            nodes.add(new NodeTiming(name, opName, provider, durationUs));
        }

        List<Map.Entry<String, ProviderTotal>> orderedProviders =
                new ArrayList<>(providerTotals.entrySet());
        orderedProviders.sort((left, right) ->
                Long.compare(right.getValue().totalUs, left.getValue().totalUs));

        JSONArray providers = new JSONArray();
        long cpuNodeUs = 0L;
        int cpuNodeCount = 0;
        for (Map.Entry<String, ProviderTotal> entry : orderedProviders) {
            ProviderTotal total = entry.getValue();
            JSONObject item = new JSONObject();
            item.put("provider", entry.getKey());
            item.put("node_count", total.nodeCount);
            item.put("total_us", total.totalUs);
            item.put("node_time_percent", totalNodeUs == 0L
                    ? 0.0 : round(total.totalUs * 100.0 / totalNodeUs));
            providers.put(item);

            if (entry.getKey().toUpperCase(Locale.ROOT).contains("CPU")) {
                cpuNodeCount += total.nodeCount;
                cpuNodeUs += total.totalUs;
            }
        }

        nodes.sort(Comparator.comparingLong((NodeTiming node) -> node.durationUs).reversed());
        JSONArray topNodes = new JSONArray();
        int topLimit = Math.min(15, nodes.size());
        for (int index = 0; index < topLimit; index++) {
            NodeTiming node = nodes.get(index);
            JSONObject item = new JSONObject();
            item.put("name", node.name);
            item.put("op_name", node.opName);
            item.put("provider", node.provider);
            item.put("dur_us", node.durationUs);
            topNodes.put(item);
        }

        JSONArray cpuTopNodes = new JSONArray();
        int cpuTopLimit = 0;
        for (NodeTiming node : nodes) {
            if (!node.provider.toUpperCase(Locale.ROOT).contains("CPU")) {
                continue;
            }
            JSONObject item = new JSONObject();
            item.put("name", node.name);
            item.put("op_name", node.opName);
            item.put("dur_us", node.durationUs);
            cpuTopNodes.put(item);
            if (++cpuTopLimit >= 15) {
                break;
            }
        }

        JSONObject summary = new JSONObject();
        summary.put("profile_unit", "us");
        summary.put("actual_provider", actualProvider == null ? "unknown" : actualProvider);
        summary.put("session_run_us", sessionRunUs);
        summary.put("node_count", nodes.size());
        summary.put("node_total_us", totalNodeUs);
        summary.put("cpu_provider_nodes", cpuNodeCount);
        summary.put("cpu_provider_total_us", cpuNodeUs);
        summary.put("cpu_provider_time_percent", totalNodeUs == 0L
                ? 0.0 : round(cpuNodeUs * 100.0 / totalNodeUs));
        summary.put("quantize_linear_nodes", quantizeNodes);
        summary.put("dequantize_linear_nodes", dequantizeNodes);
        summary.put("memcpy_nodes", memcpyNodes);
        summary.put("memcpy_total_us", memcpyUs);
        summary.put("providers", providers);
        summary.put("top_nodes", topNodes);
        summary.put("cpu_top_nodes", cpuTopNodes);
        return summary;
    }

    private static final class ProviderTotal {
        private int nodeCount;
        private long totalUs;
    }

    private static final class NodeTiming {
        private final String name;
        private final String opName;
        private final String provider;
        private final long durationUs;

        private NodeTiming(String name, String opName, String provider, long durationUs) {
            this.name = name;
            this.opName = opName;
            this.provider = provider;
            this.durationUs = durationUs;
        }
    }

    private static List<Configuration> buildConfigurations() {
        List<Configuration> configurations = new ArrayList<>();
        addModelConfigurations(configurations, OnnxClassifier.Model.FP32);
        addModelConfigurations(configurations, OnnxClassifier.Model.INT8_QDQ);
        return configurations;
    }

    private static void addModelConfigurations(
            List<Configuration> configurations, OnnxClassifier.Model model) {
        for (int threads : new int[]{1, 2, 4}) {
            configurations.add(new Configuration(
                    model,
                    OnnxClassifier.ExecutionProvider.CPU,
                    threads,
                    model.getDisplayName() + " / CPU ×" + threads + " 线程"));
        }
        configurations.add(new Configuration(
                model,
                OnnxClassifier.ExecutionProvider.XNNPACK,
                1,
                model.getDisplayName() + " / XNNPACK"));
        configurations.add(new Configuration(
                model,
                OnnxClassifier.ExecutionProvider.NNAPI,
                1,
                model.getDisplayName() + " / NNAPI"));
    }

    private static final class Configuration {
        private final OnnxClassifier.Model model;
        private final OnnxClassifier.ExecutionProvider provider;
        private final int threads;
        private final String label;

        private Configuration(
                OnnxClassifier.Model model,
                OnnxClassifier.ExecutionProvider provider,
                int threads,
                String label) {
            this.model = model;
            this.provider = provider;
            this.threads = threads;
            this.label = label;
        }
    }

    /** 返回 [p50, p95]，单位毫秒，使用 nearest-rank。 */
    private static double[] percentileSummary(long[] samples) {
        long[] sorted = Arrays.copyOf(samples, samples.length);
        Arrays.sort(sorted);
        return new double[]{percentileMs(sorted, 50), percentileMs(sorted, 95)};
    }

    private static double percentileMs(long[] sorted, int percentile) {
        if (sorted.length == 0) {
            return 0.0;
        }
        int rank = (int) Math.ceil(percentile / 100.0 * sorted.length) - 1;
        int index = Math.max(0, Math.min(sorted.length - 1, rank));
        return sorted[index] / 1_000_000.0;
    }

    private static double currentPssMb() {
        Debug.MemoryInfo memoryInfo = new Debug.MemoryInfo();
        Debug.getMemoryInfo(memoryInfo);
        return memoryInfo.getTotalPss() / 1024.0;
    }

    private static double round(double value) {
        return Math.round(value * 1000.0) / 1000.0;
    }

    private static void notifyProgress(ProgressListener listener, String message) {
        if (listener != null) {
            listener.onProgress(message);
        }
    }
}
