package com.edgevision.onnx.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * ONNX Runtime 分类器。OrtEnvironment 全局复用，OrtSession 按模型/EP/线程配置缓存；
 * 所有资产读取、Session 创建、预处理、推理与后处理均在后台线程执行，回调切回主线程。
 *
 * 当前模型只输出 ImageNet 分类 logits，不包含检测框/NMS；后续可扩展 YOLOv8n 检测链路。
 */
public final class OnnxClassifier {
    private static final String TAG = "OnnxClassifier";
    public static final String EXTRA_MODEL = "com.edgevision.onnx.extra.MODEL";
    public static final String EXTRA_PROVIDER = "com.edgevision.onnx.extra.PROVIDER";

    private static final String INPUT_NAME = "images";
    private static final long[] INPUT_SHAPE = {1L, 3L, 224L, 224L};
    private static final Map<String, OnnxClassifier> INSTANCES = new HashMap<>();

    private final Context appContext;
    private final Model model;
    private final ExecutionProvider requestedProvider;
    private final int intraOpThreads;
    private final ExecutorService executor;
    private final Handler mainHandler;
    private final Object sessionLock = new Object();

    private volatile OrtSession session;
    private volatile String actualProvider = "未加载";
    private volatile Throwable loadFailure;

    public enum Model {
        FP32("fp32", "FP32", "mobilenet_v2_fp32.onnx"),
        INT8_QDQ("int8_qdq", "INT8 QDQ", "mobilenet_v2_int8_qdq.onnx");

        private final String assetKey;
        private final String displayName;
        private final String filename;

        Model(String assetKey, String displayName, String filename) {
            this.assetKey = assetKey;
            this.displayName = displayName;
            this.filename = filename;
        }

        public String getAssetKey() {
            return assetKey;
        }

        public String getDisplayName() {
            return displayName;
        }

        public String getFilename() {
            return filename;
        }

        public static Model fromKey(String value) {
            if (value != null) {
                for (Model candidate : values()) {
                    if (candidate.name().equalsIgnoreCase(value)
                            || candidate.assetKey.equalsIgnoreCase(value)) {
                        return candidate;
                    }
                }
            }
            return FP32;
        }
    }

    public enum ExecutionProvider {
        CPU,
        XNNPACK,
        NNAPI;

        public static ExecutionProvider fromKey(String value) {
            if (value != null) {
                for (ExecutionProvider candidate : values()) {
                    if (candidate.name().equalsIgnoreCase(value)) {
                        return candidate;
                    }
                }
            }
            return CPU;
        }
    }

    public interface InitCallback {
        void onReady(String providerDescription);

        void onError(Throwable error);
    }

    public interface ClassificationCallback {
        void onResult(ClassificationResult result);

        void onError(Throwable error);
    }

    public static final class ClassificationResult {
        private final List<ImageNetLabels.Prediction> topK;
        private final long preprocessMs;
        private final long inferenceMs;
        private final long postprocessMs;
        private final long totalMs;
        private final String modelName;
        private final String providerName;

        private ClassificationResult(
                List<ImageNetLabels.Prediction> topK,
                long preprocessMs,
                long inferenceMs,
                long postprocessMs,
                long totalMs,
                String modelName,
                String providerName) {
            this.topK = topK;
            this.preprocessMs = preprocessMs;
            this.inferenceMs = inferenceMs;
            this.postprocessMs = postprocessMs;
            this.totalMs = totalMs;
            this.modelName = modelName;
            this.providerName = providerName;
        }

        public List<ImageNetLabels.Prediction> getTopK() {
            return topK;
        }

        public long getPreprocessMs() {
            return preprocessMs;
        }

        public long getInferenceMs() {
            return inferenceMs;
        }

        public long getPostprocessMs() {
            return postprocessMs;
        }

        public long getTotalMs() {
            return totalMs;
        }

        public String getModelName() {
            return modelName;
        }

        public String getProviderName() {
            return providerName;
        }
    }

    /** 同步基准测试结果，保留纳秒精度，供 benchmark 模块分阶段统计。 */
    public static final class TimedClassification {
        private final List<ImageNetLabels.Prediction> topK;
        private final long preprocessNanos;
        private final long inferenceNanos;
        private final long postprocessNanos;

        private TimedClassification(
                List<ImageNetLabels.Prediction> topK,
                long preprocessNanos,
                long inferenceNanos,
                long postprocessNanos) {
            this.topK = topK;
            this.preprocessNanos = preprocessNanos;
            this.inferenceNanos = inferenceNanos;
            this.postprocessNanos = postprocessNanos;
        }

        public List<ImageNetLabels.Prediction> getTopK() {
            return topK;
        }

        public long getPreprocessNanos() {
            return preprocessNanos;
        }

        public long getInferenceNanos() {
            return inferenceNanos;
        }

        public long getPostprocessNanos() {
            return postprocessNanos;
        }

        public long getTotalNanos() {
            return preprocessNanos + inferenceNanos + postprocessNanos;
        }
    }

    private OnnxClassifier(
            Context context, Model model, ExecutionProvider provider, int intraOpThreads) {
        this.appContext = context.getApplicationContext();
        this.model = model;
        this.requestedProvider = provider;
        this.intraOpThreads = intraOpThreads;
        this.mainHandler = new Handler(Looper.getMainLooper());
        this.executor = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(
                    runnable, "onnx-" + model.assetKey + "-" + provider.name().toLowerCase());
            thread.setDaemon(true);
            return thread;
        });
    }

    public static OnnxClassifier getInstance(
            Context context, Model model, ExecutionProvider provider) {
        int threads = Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors()));
        return getInstance(context, model, provider, threads);
    }

    public static OnnxClassifier getInstance(
            Context context, Model model, ExecutionProvider provider, int intraOpThreads) {
        if (context == null) {
            throw new IllegalArgumentException("Context 不能为空");
        }
        Model selectedModel = model == null ? Model.FP32 : model;
        ExecutionProvider selectedProvider = provider == null ? ExecutionProvider.CPU : provider;
        int threads = Math.max(1, Math.min(4, intraOpThreads));
        String key = selectedModel.assetKey + "|" + selectedProvider.name() + "|" + threads;
        synchronized (INSTANCES) {
            OnnxClassifier existing = INSTANCES.get(key);
            if (existing != null) {
                return existing;
            }
            OnnxClassifier created = new OnnxClassifier(
                    context, selectedModel, selectedProvider, threads);
            INSTANCES.put(key, created);
            return created;
        }
    }

    public Model getModel() {
        return model;
    }

    public ExecutionProvider getRequestedProvider() {
        return requestedProvider;
    }

    public int getIntraOpThreads() {
        return intraOpThreads;
    }

    public String getActualProvider() {
        return actualProvider;
    }

    /** 同步加载 Session；仅由后台 benchmark 线程调用。 */
    public void ensureSessionLoaded() throws Exception {
        ensureSession();
    }

    /**
     * 同步执行一次完整分类，仅由 benchmark 后台线程调用。
     *
     * 预处理计时包含输入 Tensor 构造，推理计时仅包含 session.run，
     * 后处理计时包含 logits 提取、Top-K 与 Tensor/Result 释放。
     */
    public TimedClassification classifySync(Bitmap bitmap, int topK) throws Exception {
        return executeClassification(() -> Preprocess.fromBitmap(bitmap), topK);
    }

    /** 释放 Session，供 benchmark 在不同配置间回收内存并重新测量冷启动。 */
    public void releaseSession() {
        synchronized (sessionLock) {
            if (session != null) {
                try {
                    session.close();
                } catch (OrtException error) {
                    Log.w(TAG, "Session 释放失败", error);
                }
                session = null;
            }
            actualProvider = "未加载";
            loadFailure = null;
        }
    }

    /** 显式预加载；即使不调用，classifyAsync 也会在后台完成懒加载。 */
    public void initializeAsync(InitCallback callback) {
        try {
            executor.execute(() -> {
                try {
                    ensureSession();
                    mainHandler.post(() -> callback.onReady(actualProvider));
                } catch (Throwable error) {
                    mainHandler.post(() -> callback.onError(error));
                }
            });
        } catch (RejectedExecutionException error) {
            mainHandler.post(() -> callback.onError(error));
        }
    }

    public void classifyAsync(Uri imageUri, int topK, ClassificationCallback callback) {
        long submittedAt = System.nanoTime();
        try {
            executor.execute(() -> classifyTask(
                    () -> Preprocess.fromUri(appContext, imageUri), topK, submittedAt, callback));
        } catch (RejectedExecutionException error) {
            mainHandler.post(() -> callback.onError(error));
        }
    }

    public void classifyAsync(Bitmap bitmap, int topK, ClassificationCallback callback) {
        long submittedAt = System.nanoTime();
        try {
            executor.execute(() -> classifyTask(
                    () -> Preprocess.fromBitmap(bitmap), topK, submittedAt, callback));
        } catch (RejectedExecutionException error) {
            mainHandler.post(() -> callback.onError(error));
        }
    }

    private void classifyTask(
            InputProducer producer, int topK, long submittedAt, ClassificationCallback callback) {
        try {
            TimedClassification timed = executeClassification(producer, topK);
            long preprocessMs = toMillis(timed.getPreprocessNanos());
            long inferenceMs = toMillis(timed.getInferenceNanos());
            long postprocessMs = toMillis(timed.getPostprocessNanos());
            long totalMs = toMillis(timed.getTotalNanos());
            ClassificationResult classification = new ClassificationResult(
                    timed.getTopK(),
                    preprocessMs,
                    inferenceMs,
                    postprocessMs,
                    totalMs,
                    model.getDisplayName(),
                    actualProvider);
            Log.d(TAG, "分类完成：" + classification.getModelName()
                    + " / " + classification.getProviderName()
                    + "，提交到完成 " + toMillis(System.nanoTime() - submittedAt) + " ms");
            mainHandler.post(() -> callback.onResult(classification));
        } catch (Throwable error) {
            Log.e(TAG, "分类失败", error);
            mainHandler.post(() -> callback.onError(error));
        }
    }

    private TimedClassification executeClassification(InputProducer producer, int topK)
            throws Exception {
        OrtSession activeSession = ensureSession();
        long start = System.nanoTime();
        float[] inputValues = producer.produce();
        OnnxTensor inputTensor = null;
        long preprocessEnd = 0L;
        long inferenceEnd = 0L;
        long postprocessEnd;
        List<ImageNetLabels.Prediction> predictions;
        try {
            inputTensor = OnnxTensor.createTensor(
                    OrtEnvironment.getEnvironment(), FloatBuffer.wrap(inputValues), INPUT_SHAPE);
            preprocessEnd = System.nanoTime();

            OrtSession.Result result = null;
            try {
                result = activeSession.run(Collections.singletonMap(INPUT_NAME, inputTensor));
                inferenceEnd = System.nanoTime();
                predictions = ImageNetLabels.topK(extractLogits(result), topK);
            } finally {
                if (result != null) {
                    result.close();
                }
            }
        } finally {
            if (inputTensor != null) {
                inputTensor.close();
            }
            postprocessEnd = System.nanoTime();
        }

        return new TimedClassification(
                predictions,
                preprocessEnd - start,
                inferenceEnd - preprocessEnd,
                postprocessEnd - inferenceEnd);
    }

    private OrtSession ensureSession() throws Exception {
        OrtSession ready = session;
        if (ready != null) {
            return ready;
        }
        synchronized (sessionLock) {
            if (session != null) {
                return session;
            }
            if (loadFailure != null) {
                throw new IllegalStateException("模型 Session 加载失败", loadFailure);
            }
            try {
                session = createSession();
                return session;
            } catch (Exception error) {
                loadFailure = error;
                throw error;
            }
        }
    }

    private OrtSession createSession() throws IOException, OrtException {
        AssetVerifier.ModelAsset asset;
        try {
            asset = AssetVerifier.findModel(appContext, model.getAssetKey());
        } catch (org.json.JSONException error) {
            throw new IOException("manifest.json 解析失败", error);
        }
        if (!model.getFilename().equals(asset.getFilename())) {
            throw new IOException("manifest 文件名与预期不一致：" + asset.getFilename());
        }

        AssetVerifier.AssetCheck check = AssetVerifier.verifyAsset(appContext, asset);
        if (!check.isPassed()) {
            throw new IOException("模型资产校验失败：" + asset.getFilename()
                    + "（大小 " + check.getActualBytes() + "，SHA-256 " + check.getActualSha256() + "）");
        }

        byte[] modelBytes = readAsset(asset);
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        if (requestedProvider == ExecutionProvider.NNAPI) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O_MR1) {
                return createCpuSession(environment, modelBytes, "CPU（NNAPI 需 API 27+，已回退）");
            }
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.addNnapi();
                OrtSession created = environment.createSession(modelBytes, options);
                actualProvider = "NNAPIExecutionProvider";
                return created;
            } catch (Exception error) {
                Log.w(TAG, "NNAPI 初始化失败，回退 CPU", error);
                return createCpuSession(environment, modelBytes, "CPU（NNAPI 初始化失败，已回退）");
            }
        }

        if (requestedProvider == ExecutionProvider.XNNPACK) {
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.addXnnpack(Collections.emptyMap());
                OrtSession created = environment.createSession(modelBytes, options);
                actualProvider = "XNNPACKExecutionProvider";
                return created;
            } catch (Exception error) {
                Log.w(TAG, "XNNPACK 初始化失败，回退 CPU", error);
                return createCpuSession(environment, modelBytes, "CPU（XNNPACK 初始化失败，已回退）");
            }
        }

        return createCpuSession(environment, modelBytes, null);
    }

    private OrtSession createCpuSession(
            OrtEnvironment environment, byte[] modelBytes, String fallbackDescription)
            throws OrtException {
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(intraOpThreads);
            OrtSession created = environment.createSession(modelBytes, options);
            actualProvider = fallbackDescription != null
                    ? fallbackDescription
                    : "CPUExecutionProvider ×" + intraOpThreads + " 线程";
            return created;
        }
    }

    private byte[] readAsset(AssetVerifier.ModelAsset asset) throws IOException {
        int initialSize = (int) Math.min(Integer.MAX_VALUE - 8L, Math.max(32L, asset.getBytes()));
        try (InputStream input = appContext.getAssets().open(asset.getFilename());
             ByteArrayOutputStream output = new ByteArrayOutputStream(initialSize)) {
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
            return output.toByteArray();
        }
    }

    private static float[] extractLogits(OrtSession.Result result) throws OrtException {
        Object value = ((OnnxTensor) result.get(0)).getValue();
        if (!(value instanceof float[][]) || ((float[][]) value).length == 0) {
            throw new IllegalStateException("输出 logits 不是预期的 float[1][1000]");
        }
        float[] logits = ((float[][]) value)[0];
        if (logits.length != 1000) {
            throw new IllegalStateException("输出类别数为 " + logits.length + "，预期 1000");
        }
        return logits;
    }

    private static long toMillis(long nanos) {
        return TimeUnit.NANOSECONDS.toMillis(Math.max(0L, nanos));
    }

    private interface InputProducer {
        float[] produce() throws IOException;
    }
}
