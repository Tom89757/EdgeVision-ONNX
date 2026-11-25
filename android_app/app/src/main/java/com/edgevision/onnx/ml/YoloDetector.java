package com.edgevision.onnx.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * YOLOv8n 320×320 目标检测器。
 *
 * <p>预处理与 {@code model_pipeline/yolo_preprocess.py} 1:1 对齐：RGB、短边等比缩放、
 * 114 灰色 Letterbox、只做 pixel/255.0 归一化、HWC 转 NCHW。后处理链与
 * {@code verify_yolo_alignment.py} 对齐：xywh→xyxy、置信度过滤、按类别 NMS、坐标反变换。
 * 仅保留 COCO 的 person(0) / cat(15) / dog(16) 三类。</p>
 *
 * <p>OrtEnvironment 全局复用，OrtSession 按“模型 × EP × 线程数”缓存为单例；
 * 模型加载、预处理、推理和后处理全部在后台线程执行，结果回调切回主线程。</p>
 */
public final class YoloDetector {
    private static final String TAG = "YoloDetector";

    public static final int INPUT_SIZE = 320;
    public static final int PAD_VALUE = 114;
    public static final float CONFIDENCE_THRESHOLD = 0.25f;
    public static final float NMS_IOU_THRESHOLD = 0.45f;

    public static final int CLASS_PERSON = 0;
    public static final int CLASS_CAT = 15;
    public static final int CLASS_DOG = 16;

    private static final String INPUT_NAME = "images";
    private static final long[] INPUT_SHAPE = {1L, 3L, INPUT_SIZE, INPUT_SIZE};
    private static final int OUTPUT_CHANNELS = 84;
    private static final int CLASS_COUNT = 80;

    private static final Map<Integer, String> CLASS_NAMES;
    private static final Map<Integer, String> CHINESE_NAMES;
    private static final Map<String, YoloDetector> INSTANCES = new HashMap<>();
    private static volatile OrtEnvironment environment;

    static {
        Map<Integer, String> labels = new LinkedHashMap<>();
        labels.put(CLASS_PERSON, "person");
        labels.put(CLASS_CAT, "cat");
        labels.put(CLASS_DOG, "dog");
        CLASS_NAMES = Collections.unmodifiableMap(labels);

        Map<Integer, String> chinese = new LinkedHashMap<>();
        chinese.put(CLASS_PERSON, "人");
        chinese.put(CLASS_CAT, "猫");
        chinese.put(CLASS_DOG, "狗");
        CHINESE_NAMES = Collections.unmodifiableMap(chinese);
    }

    public enum Model {
        FP32("yolo_fp32", "YOLOv8n FP32", "yolov8n_fp32.onnx"),
        INT8_QDQ("yolo_int8_qdq", "YOLOv8n INT8 QDQ", "yolov8n_int8_qdq.onnx");

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
                            || candidate.assetKey.equalsIgnoreCase(value)
                            || candidate.filename.equalsIgnoreCase(value)) {
                        return candidate;
                    }
                }
                if ("INT8".equalsIgnoreCase(value)
                        || "int8".equalsIgnoreCase(value)
                        || "int8_qdq".equalsIgnoreCase(value)) {
                    return INT8_QDQ;
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

    public interface DetectionCallback {
        void onResult(DetectionResult result);

        void onError(Throwable error);
    }

    /** Letterbox 几何参数，字段语义与 Python 侧 LetterboxMeta 一致。 */
    public static final class LetterboxMeta {
        private final int originalWidth;
        private final int originalHeight;
        private final int resizedWidth;
        private final int resizedHeight;
        private final double ratio;
        private final int padX;
        private final int padY;

        private LetterboxMeta(
                int originalWidth,
                int originalHeight,
                int resizedWidth,
                int resizedHeight,
                double ratio,
                int padX,
                int padY) {
            this.originalWidth = originalWidth;
            this.originalHeight = originalHeight;
            this.resizedWidth = resizedWidth;
            this.resizedHeight = resizedHeight;
            this.ratio = ratio;
            this.padX = padX;
            this.padY = padY;
        }

        public int getOriginalWidth() {
            return originalWidth;
        }

        public int getOriginalHeight() {
            return originalHeight;
        }

        public int getResizedWidth() {
            return resizedWidth;
        }

        public int getResizedHeight() {
            return resizedHeight;
        }

        public double getRatio() {
            return ratio;
        }

        public int getPadX() {
            return padX;
        }

        public int getPadY() {
            return padY;
        }
    }

    public static final class Detection {
        private final int classId;
        private final String className;
        private final String chineseName;
        private final float confidence;
        private final RectF box;

        private Detection(
                int classId,
                String className,
                String chineseName,
                float confidence,
                RectF box) {
            this.classId = classId;
            this.className = className;
            this.chineseName = chineseName;
            this.confidence = confidence;
            this.box = box;
        }

        public int getClassId() {
            return classId;
        }

        public String getClassName() {
            return className;
        }

        public String getChineseName() {
            return chineseName;
        }

        public float getConfidence() {
            return confidence;
        }

        /** 返回原图像素坐标 [left, top, right, bottom] 的副本。 */
        public RectF getBox() {
            return new RectF(box);
        }
    }

    public static final class DetectionResult {
        private final List<Detection> detections;
        private final int sourceWidth;
        private final int sourceHeight;
        private final long preprocessMs;
        private final long inferenceMs;
        private final long postprocessMs;
        private final long totalMs;
        private final String modelName;
        private final String providerName;

        private DetectionResult(
                List<Detection> detections,
                int sourceWidth,
                int sourceHeight,
                long preprocessMs,
                long inferenceMs,
                long postprocessMs,
                long totalMs,
                String modelName,
                String providerName) {
            this.detections = Collections.unmodifiableList(new ArrayList<>(detections));
            this.sourceWidth = sourceWidth;
            this.sourceHeight = sourceHeight;
            this.preprocessMs = preprocessMs;
            this.inferenceMs = inferenceMs;
            this.postprocessMs = postprocessMs;
            this.totalMs = totalMs;
            this.modelName = modelName;
            this.providerName = providerName;
        }

        public List<Detection> getDetections() {
            return detections;
        }

        public int getSourceWidth() {
            return sourceWidth;
        }

        public int getSourceHeight() {
            return sourceHeight;
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

    private final Context appContext;
    private final Model model;
    private final ExecutionProvider requestedProvider;
    private final int intraOpThreads;
    private final float confidenceThreshold;
    private final float nmsIouThreshold;
    private final ExecutorService executor;
    private final Handler mainHandler;
    private final Object sessionLock = new Object();

    private volatile OrtSession session;
    private volatile String actualProvider = "未加载";
    private volatile String inputName = INPUT_NAME;
    private volatile Throwable loadFailure;

    private YoloDetector(
            Context context,
            Model model,
            ExecutionProvider provider,
            int intraOpThreads,
            float confidenceThreshold,
            float nmsIouThreshold) {
        this.appContext = context.getApplicationContext();
        this.model = model;
        this.requestedProvider = provider;
        this.intraOpThreads = intraOpThreads;
        this.confidenceThreshold = confidenceThreshold;
        this.nmsIouThreshold = nmsIouThreshold;
        this.mainHandler = new Handler(Looper.getMainLooper());
        this.executor = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(
                    runnable, "yolo-" + model.assetKey + "-" + provider.name().toLowerCase());
            thread.setDaemon(true);
            return thread;
        });
    }

    public static YoloDetector getInstance(
            Context context, Model model, ExecutionProvider provider) {
        int threads = Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors()));
        return getInstance(context, model, provider, threads);
    }

    public static YoloDetector getInstance(
            Context context, Model model, ExecutionProvider provider, int intraOpThreads) {
        return getInstance(
                context,
                model,
                provider,
                intraOpThreads,
                CONFIDENCE_THRESHOLD,
                NMS_IOU_THRESHOLD);
    }

    /** 允许业务侧覆盖置信度与 NMS 阈值；默认值分别为 0.25 / 0.45。 */
    public static YoloDetector getInstance(
            Context context,
            Model model,
            ExecutionProvider provider,
            int intraOpThreads,
            float confidenceThreshold,
            float nmsIouThreshold) {
        if (context == null) {
            throw new IllegalArgumentException("Context 不能为空");
        }
        Model selectedModel = model == null ? Model.FP32 : model;
        ExecutionProvider selectedProvider = provider == null ? ExecutionProvider.CPU : provider;
        int threads = Math.max(1, Math.min(4, intraOpThreads));
        if (confidenceThreshold < 0.0f || confidenceThreshold > 1.0f) {
            throw new IllegalArgumentException("置信度阈值必须位于 [0,1]");
        }
        if (nmsIouThreshold < 0.0f || nmsIouThreshold > 1.0f) {
            throw new IllegalArgumentException("NMS IoU 阈值必须位于 [0,1]");
        }
        String key = selectedModel.assetKey + "|" + selectedProvider.name() + "|" + threads
                + "|" + confidenceThreshold + "|" + nmsIouThreshold;
        synchronized (INSTANCES) {
            YoloDetector existing = INSTANCES.get(key);
            if (existing != null) {
                return existing;
            }
            YoloDetector created = new YoloDetector(
                    context,
                    selectedModel,
                    selectedProvider,
                    threads,
                    confidenceThreshold,
                    nmsIouThreshold);
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

    public String getActualProvider() {
        return actualProvider;
    }

    public float getConfidenceThreshold() {
        return confidenceThreshold;
    }

    public float getNmsIouThreshold() {
        return nmsIouThreshold;
    }

    /** 在后台线程预热 OrtSession；不调用时 detectAsync 也会懒加载。 */
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

    /** 后台执行完整检测；Bitmap 由调用方在回调后负责回收。 */
    public void detectAsync(Bitmap bitmap, DetectionCallback callback) {
        if (bitmap == null || bitmap.isRecycled()) {
            mainHandler.post(() -> callback.onError(new IllegalArgumentException("Bitmap 不可用")));
            return;
        }
        try {
            executor.execute(() -> {
                try {
                    DetectionResult result = executeDetection(bitmap);
                    mainHandler.post(() -> callback.onResult(result));
                } catch (Throwable error) {
                    Log.e(TAG, "YOLO 检测失败", error);
                    mainHandler.post(() -> callback.onError(error));
                }
            });
        } catch (RejectedExecutionException error) {
            mainHandler.post(() -> callback.onError(error));
        }
    }

    private DetectionResult executeDetection(Bitmap bitmap) throws Exception {
        OrtSession activeSession = ensureSession();
        long startNanos = System.nanoTime();
        LetterboxMeta meta = computeLetterboxMeta(bitmap.getWidth(), bitmap.getHeight());
        float[] inputValues = letterboxToNchw(bitmap, meta);

        OnnxTensor inputTensor = null;
        long preprocessEnd = startNanos;
        long inferenceEnd = startNanos;
        long postprocessEnd = startNanos;
        List<Detection> detections;
        try {
            inputTensor = OnnxTensor.createTensor(
                    getEnvironment(), FloatBuffer.wrap(inputValues), INPUT_SHAPE);
            preprocessEnd = System.nanoTime();

            OrtSession.Result result = null;
            try {
                result = activeSession.run(Collections.singletonMap(inputName, inputTensor));
                inferenceEnd = System.nanoTime();
                Object rawOutput = ((OnnxTensor) result.get(0)).getValue();
                detections = decodeDetections(
                        normaliseRawOutput(rawOutput),
                        meta,
                        confidenceThreshold,
                        nmsIouThreshold);
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

        long totalNanos = postprocessEnd - startNanos;
        return new DetectionResult(
                detections,
                meta.getOriginalWidth(),
                meta.getOriginalHeight(),
                toMillis(preprocessEnd - startNanos),
                toMillis(inferenceEnd - preprocessEnd),
                toMillis(postprocessEnd - inferenceEnd),
                toMillis(totalNanos),
                model.getDisplayName(),
                actualProvider);
    }

    /** 与 yolo_preprocess.py 的 _round_half_up 和 pad 计算保持一致。 */
    public static LetterboxMeta computeLetterboxMeta(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("非法图像尺寸：" + width + "x" + height);
        }
        double ratio = Math.min(INPUT_SIZE / (double) width, INPUT_SIZE / (double) height);
        int resizedWidth = Math.max(
                1, Math.min(INPUT_SIZE, (int) (width * ratio + 0.5)));
        int resizedHeight = Math.max(
                1, Math.min(INPUT_SIZE, (int) (height * ratio + 0.5)));
        int padX = (INPUT_SIZE - resizedWidth) / 2;
        int padY = (INPUT_SIZE - resizedHeight) / 2;
        return new LetterboxMeta(
                width, height, resizedWidth, resizedHeight, ratio, padX, padY);
    }

    /** Letterbox → RGB → [0,1] → NCHW，返回 1×3×320×320 的连续 float 数组。 */
    public static float[] letterboxToNchw(Bitmap bitmap, LetterboxMeta meta) {
        if (bitmap == null || bitmap.isRecycled()) {
            throw new IllegalArgumentException("Bitmap 不可用");
        }

        Bitmap scaled = null;
        Bitmap padded = null;
        try {
            scaled = Bitmap.createScaledBitmap(
                    bitmap, meta.getResizedWidth(), meta.getResizedHeight(), true);
            padded = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(padded);
            canvas.drawColor(Color.rgb(PAD_VALUE, PAD_VALUE, PAD_VALUE));
            Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
            canvas.drawBitmap(scaled, meta.getPadX(), meta.getPadY(), paint);

            int pixelCount = INPUT_SIZE * INPUT_SIZE;
            int[] pixels = new int[pixelCount];
            padded.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

            float[] output = new float[3 * pixelCount];
            int greenOffset = pixelCount;
            int blueOffset = pixelCount * 2;
            for (int index = 0; index < pixelCount; index++) {
                int pixel = pixels[index];
                output[index] = ((pixel >> 16) & 0xff) / 255.0f;
                output[greenOffset + index] = ((pixel >> 8) & 0xff) / 255.0f;
                output[blueOffset + index] = (pixel & 0xff) / 255.0f;
            }
            return output;
        } finally {
            if (scaled != null && scaled != bitmap) {
                scaled.recycle();
            }
            if (padded != null) {
                padded.recycle();
            }
        }
    }

    /** 把 [1,84,N]、[1,N,84] 或对应的二维张量统一成 N×84 的候选矩阵。 */
    private static float[][] normaliseRawOutput(Object rawOutput) {
        if (rawOutput instanceof float[][][]) {
            float[][][] batched = (float[][][]) rawOutput;
            if (batched.length != 1) {
                throw new IllegalStateException("YOLO 输出 batch 必须为 1，实际 " + batched.length);
            }
            return normaliseOutputMatrix(batched[0]);
        }
        if (rawOutput instanceof float[][]) {
            return normaliseOutputMatrix((float[][]) rawOutput);
        }
        throw new IllegalStateException("YOLO 输出不是预期的 float 张量");
    }

    private static float[][] normaliseOutputMatrix(float[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            throw new IllegalStateException("YOLO 输出为空");
        }
        if (matrix.length == OUTPUT_CHANNELS) {
            int candidateCount = matrix[0].length;
            float[][] transposed = new float[candidateCount][OUTPUT_CHANNELS];
            for (int candidate = 0; candidate < candidateCount; candidate++) {
                for (int channel = 0; channel < OUTPUT_CHANNELS; channel++) {
                    transposed[candidate][channel] = matrix[channel][candidate];
                }
            }
            return transposed;
        }
        if (matrix[0].length == OUTPUT_CHANNELS) {
            return matrix;
        }
        throw new IllegalStateException(
                "YOLO 输出通道数必须为 84，实际 " + matrix.length + "x" + matrix[0].length);
    }

    private static List<Detection> decodeDetections(
            float[][] predictions,
            LetterboxMeta meta,
            float confidenceThreshold,
            float nmsIouThreshold) {
        List<Candidate> candidates = new ArrayList<>();
        for (float[] prediction : predictions) {
            int classId = -1;
            float bestScore = -Float.MAX_VALUE;
            for (int index = 0; index < CLASS_COUNT; index++) {
                float score = prediction[4 + index];
                if (!Float.isNaN(score) && score > bestScore) {
                    bestScore = score;
                    classId = index;
                }
            }
            if (!CLASS_NAMES.containsKey(classId) || bestScore < confidenceThreshold) {
                continue;
            }

            float centerX = prediction[0];
            float centerY = prediction[1];
            float width = prediction[2];
            float height = prediction[3];
            float left = centerX - width / 2.0f;
            float top = centerY - height / 2.0f;
            float right = centerX + width / 2.0f;
            float bottom = centerY + height / 2.0f;
            candidates.add(new Candidate(classId, bestScore, left, top, right, bottom));
        }

        List<Candidate> kept = new ArrayList<>();
        for (int classId : new int[]{CLASS_PERSON, CLASS_CAT, CLASS_DOG}) {
            List<Candidate> sameClass = new ArrayList<>();
            for (Candidate candidate : candidates) {
                if (candidate.classId == classId) {
                    sameClass.add(candidate);
                }
            }
            kept.addAll(nms(sameClass, nmsIouThreshold));
        }
        kept.sort((left, right) -> Float.compare(right.confidence, left.confidence));

        List<Detection> detections = new ArrayList<>(kept.size());
        double ratio = meta.getRatio();
        for (Candidate candidate : kept) {
            float left = (float) ((candidate.left - meta.getPadX()) / ratio);
            float top = (float) ((candidate.top - meta.getPadY()) / ratio);
            float right = (float) ((candidate.right - meta.getPadX()) / ratio);
            float bottom = (float) ((candidate.bottom - meta.getPadY()) / ratio);
            left = clamp(left, 0.0f, meta.getOriginalWidth());
            top = clamp(top, 0.0f, meta.getOriginalHeight());
            right = clamp(right, 0.0f, meta.getOriginalWidth());
            bottom = clamp(bottom, 0.0f, meta.getOriginalHeight());
            if (right <= left || bottom <= top) {
                continue;
            }
            detections.add(new Detection(
                    candidate.classId,
                    CLASS_NAMES.get(candidate.classId),
                    CHINESE_NAMES.get(candidate.classId),
                    candidate.confidence,
                    new RectF(left, top, right, bottom)));
        }
        return detections;
    }

    /** 按类别 NMS；IoU 等于阈值时保留，与 Python 侧 ious <= threshold 的语义一致。 */
    private static List<Candidate> nms(List<Candidate> input, float iouThreshold) {
        if (input.isEmpty()) {
            return Collections.emptyList();
        }
        List<Candidate> ordered = new ArrayList<>(input);
        ordered.sort((left, right) -> Float.compare(right.confidence, left.confidence));
        boolean[] suppressed = new boolean[ordered.size()];
        List<Candidate> keep = new ArrayList<>();
        for (int index = 0; index < ordered.size(); index++) {
            if (suppressed[index]) {
                continue;
            }
            Candidate current = ordered.get(index);
            keep.add(current);
            for (int other = index + 1; other < ordered.size(); other++) {
                if (!suppressed[other]
                        && iou(current, ordered.get(other)) > iouThreshold) {
                    suppressed[other] = true;
                }
            }
        }
        return keep;
    }

    private static float iou(Candidate left, Candidate right) {
        float intersectionLeft = Math.max(left.left, right.left);
        float intersectionTop = Math.max(left.top, right.top);
        float intersectionRight = Math.min(left.right, right.right);
        float intersectionBottom = Math.min(left.bottom, right.bottom);
        float intersectionWidth = Math.max(0.0f, intersectionRight - intersectionLeft);
        float intersectionHeight = Math.max(0.0f, intersectionBottom - intersectionTop);
        float intersection = intersectionWidth * intersectionHeight;
        float leftArea = Math.max(0.0f, left.right - left.left)
                * Math.max(0.0f, left.bottom - left.top);
        float rightArea = Math.max(0.0f, right.right - right.left)
                * Math.max(0.0f, right.bottom - right.top);
        return intersection / Math.max(leftArea + rightArea - intersection, 1.0e-12f);
    }

    private static float clamp(float value, float minimum, float maximum) {
        return Math.max(minimum, Math.min(maximum, value));
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
                throw new IllegalStateException("YOLO Session 加载失败", loadFailure);
            }
            try {
                OrtSession created = createSession();
                session = created;
                Set<String> inputNames = created.getInputNames();
                if (inputNames.isEmpty()) {
                    throw new IllegalStateException("YOLO 模型没有输入");
                }
                inputName = inputNames.iterator().next();
                if (!INPUT_NAME.equals(inputName)) {
                    Log.w(TAG, "模型输入名为 " + inputName + "，与契约 " + INPUT_NAME + " 不一致");
                }
                return created;
            } catch (Throwable error) {
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
        if (requestedProvider == ExecutionProvider.NNAPI) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O_MR1) {
                return createCpuSession(modelBytes, "CPU（NNAPI 需 API 27+，已回退）");
            }
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.addNnapi();
                OrtSession created = getEnvironment().createSession(modelBytes, options);
                actualProvider = "NNAPIExecutionProvider";
                return created;
            } catch (Exception error) {
                Log.w(TAG, "NNAPI 初始化失败，回退 CPU", error);
                return createCpuSession(modelBytes, "CPU（NNAPI 初始化失败，已回退）");
            }
        }

        if (requestedProvider == ExecutionProvider.XNNPACK) {
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.addXnnpack(Collections.emptyMap());
                OrtSession created = getEnvironment().createSession(modelBytes, options);
                actualProvider = "XNNPACKExecutionProvider";
                return created;
            } catch (Exception error) {
                Log.w(TAG, "XNNPACK 初始化失败，回退 CPU", error);
                return createCpuSession(modelBytes, "CPU（XNNPACK 初始化失败，已回退）");
            }
        }

        return createCpuSession(modelBytes, null);
    }

    private OrtSession createCpuSession(byte[] modelBytes, String fallbackDescription)
            throws OrtException {
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(intraOpThreads);
            OrtSession created = getEnvironment().createSession(modelBytes, options);
            actualProvider = fallbackDescription != null
                    ? fallbackDescription
                    : "CPUExecutionProvider ×" + intraOpThreads + " 线程";
            return created;
        }
    }

    private byte[] readAsset(AssetVerifier.ModelAsset asset) throws IOException {
        int initialSize = (int) Math.min(
                Integer.MAX_VALUE - 8L, Math.max(32L, asset.getBytes()));
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

    private static OrtEnvironment getEnvironment() {
        OrtEnvironment ready = environment;
        if (ready != null) {
            return ready;
        }
        synchronized (YoloDetector.class) {
            if (environment == null) {
                environment = OrtEnvironment.getEnvironment();
            }
            return environment;
        }
    }

    private static long toMillis(long nanos) {
        return TimeUnit.NANOSECONDS.toMillis(Math.max(0L, nanos));
    }

    private static final class Candidate {
        private final int classId;
        private final float confidence;
        private final float left;
        private final float top;
        private final float right;
        private final float bottom;

        private Candidate(
                int classId,
                float confidence,
                float left,
                float top,
                float right,
                float bottom) {
            this.classId = classId;
            this.confidence = confidence;
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
        }
    }
}
