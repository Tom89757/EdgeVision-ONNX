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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * 真机多后端 benchmark 执行器。
 *
 * 每条配置先单独加载 Session 并计时冷启动，再测量首帧；随后预热 5 帧，
 * 连续测量 50 帧。每帧耗时严格拆为预处理、纯推理、后处理三部分。
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
                String error) {
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
                    error);
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
                    null);
        } catch (Throwable error) {
            actualProvider = classifier.getActualProvider();
            String message = error.getMessage() == null
                    ? error.getClass().getSimpleName() : error.getMessage();
            return Result.failed(configuration, actualProvider, message);
        } finally {
            classifier.releaseSession();
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
