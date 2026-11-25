package com.edgevision.onnx.ui;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.edgevision.onnx.R;
import com.edgevision.onnx.ml.EventReporter;
import com.edgevision.onnx.ml.OnnxClassifier;
import com.edgevision.onnx.ml.ThermalGuard;
import com.edgevision.onnx.ml.YoloDetector;
import com.google.android.material.button.MaterialButton;
import com.google.common.util.concurrent.ListenableFuture;

import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayDeque;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 宠物监控：CameraX 实时取帧，YOLOv8n 检测 person/cat/dog 并在预览上画框。
 *
 * <p>正常保留 2 FPS 节流与上一帧未完成时丢帧的背压策略；ThermalGuard 在热、低内存或
 * p95 延迟熔断时会把目标帧率降到 0.5 FPS 并暂停上报。MobileNetV2 分类链路继续由
 * {@link OnnxClassifier} 与 {@link AlbumActivity} 提供；本页不删除分类能力，仅以注释预留
 * “检测模型不可用时人工切回分类链路”的运维回退说明，不做自动回退。</p>
 */
public class MonitorActivity extends AppCompatActivity {
    private static final String TAG = "MonitorActivity";
    private static final long PET_EVENT_INTERVAL_MS = 5_000L;
    private static final int MAX_EVENT_LINES = 6;
    private static final int THUMBNAIL_MAX_SIZE = 224;

    private final AtomicBoolean analyzing = new AtomicBoolean(false);
    private final ArrayDeque<String> eventLines = new ArrayDeque<>();

    private ExecutorService analyzerExecutor;
    private YoloDetector detector;
    private EventReporter eventReporter;
    private ThermalGuard thermalGuard;
    private ProcessCameraProvider cameraProvider;
    private ImageAnalysis imageAnalysis;

    private PreviewView previewView;
    private DetectionOverlayView detectionOverlay;
    private TextView monitorStatus;
    private TextView thermalGuardStatus;
    private TextView currentResult;
    private TextView eventLog;

    private long lastAnalysisAtMs;
    private long lastPetEventAtMs;
    private ActivityResultLauncher<String> permissionLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_monitor);

        previewView = findViewById(R.id.previewView);
        detectionOverlay = findViewById(R.id.detectionOverlay);
        monitorStatus = findViewById(R.id.monitorStatus);
        thermalGuardStatus = findViewById(R.id.thermalGuardStatus);
        currentResult = findViewById(R.id.currentResult);
        eventLog = findViewById(R.id.eventLog);
        MaterialButton backButton = findViewById(R.id.backButton);

        YoloDetector.Model model = YoloDetector.Model.fromKey(
                getIntent().getStringExtra(OnnxClassifier.EXTRA_MODEL));
        YoloDetector.ExecutionProvider provider = YoloDetector.ExecutionProvider.fromKey(
                getIntent().getStringExtra(OnnxClassifier.EXTRA_PROVIDER));
        detector = YoloDetector.getInstance(getApplicationContext(), model, provider);
        eventReporter = EventReporter.getInstance(this);
        thermalGuard = new ThermalGuard(this, this::updateThermalGuardStatus);
        updateThermalGuardStatus(thermalGuard.getPolicy());

        detector.initializeAsync(new YoloDetector.InitCallback() {
            @Override
            public void onReady(String providerDescription) {
                Log.i(TAG, "YOLOv8n 已加载：" + providerDescription);
            }

            @Override
            public void onError(Throwable error) {
                Log.e(TAG, "YOLOv8n 加载失败", error);
                if (isFinishing() || isDestroyed()) {
                    return;
                }
                monitorStatus.setText(getString(
                        R.string.monitor_error, errorMessage(error)));
            }
        });

        analyzerExecutor = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, "camera-frame-analyzer");
            thread.setDaemon(true);
            return thread;
        });

        permissionLauncher = registerForActivityResult(
                new ActivityResultContracts.RequestPermission(), granted -> {
                    if (granted) {
                        startCamera();
                    } else {
                        monitorStatus.setText(R.string.monitor_permission_denied);
                    }
                });

        backButton.setOnClickListener(view -> finish());
        requestCameraPermissionOrStart();
    }

    private void requestCameraPermissionOrStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            monitorStatus.setText(R.string.monitor_waiting_permission);
            permissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void startCamera() {
        monitorStatus.setText(R.string.monitor_idle);
        ListenableFuture<ProcessCameraProvider> providerFuture =
                ProcessCameraProvider.getInstance(this);
        providerFuture.addListener(() -> {
            if (isFinishing() || isDestroyed()) {
                return;
            }
            try {
                cameraProvider = providerFuture.get();
                bindCameraUseCases();
            } catch (Exception error) {
                Log.e(TAG, "相机启动失败", error);
                monitorStatus.setText(getString(R.string.camera_error, errorMessage(error)));
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases() {
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        imageAnalysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build();
        imageAnalysis.setAnalyzer(analyzerExecutor, this::onFrame);

        cameraProvider.unbindAll();
        bindPreferredCamera(preview, imageAnalysis);
        detectionOverlay.clear();
        monitorStatus.setText(R.string.monitor_running);
    }

    /** 依次尝试后置 → 外部(USB/UVC) → 前置，覆盖真机与无内置摄像头的 USB 摄像头设备。 */
    private void bindPreferredCamera(Preview preview, ImageAnalysis imageAnalysis) {
        CameraSelector[] candidates = {
                CameraSelector.DEFAULT_BACK_CAMERA,
                new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_EXTERNAL)
                        .build(),
                CameraSelector.DEFAULT_FRONT_CAMERA,
        };
        for (CameraSelector selector : candidates) {
            try {
                cameraProvider.bindToLifecycle(this, selector, preview, imageAnalysis);
                return;
            } catch (IllegalArgumentException error) {
                Log.w(TAG, "该朝向无摄像头，尝试下一个", error);
            }
        }
        throw new IllegalArgumentException("无可用摄像头（后置/外部/前置均不存在）");
    }

    private void onFrame(ImageProxy image) {
        long now = SystemClock.elapsedRealtime();
        if (!analyzing.compareAndSet(false, true)) {
            image.close();
            return;
        }
        if (now - lastAnalysisAtMs < thermalGuard.getAnalysisIntervalMs()) {
            analyzing.set(false);
            image.close();
            return;
        }
        lastAnalysisAtMs = now;

        Bitmap frame = null;
        try {
            frame = imageProxyToBitmap(image);
        } catch (Throwable error) {
            Log.e(TAG, "相机帧转换失败", error);
        } finally {
            image.close();
        }
        if (frame == null) {
            analyzing.set(false);
            return;
        }

        final Bitmap frameForAnalysis = frame;
        try {
            detector.detectAsync(frameForAnalysis, new YoloDetector.DetectionCallback() {
                @Override
                public void onResult(YoloDetector.DetectionResult result) {
                    if (isFinishing() || isDestroyed()) {
                        frameForAnalysis.recycle();
                        analyzing.set(false);
                        return;
                    }
                    try {
                        thermalGuard.recordFrameLatency(result.getTotalMs());
                        renderDetections(result, frameForAnalysis);
                    } catch (Throwable error) {
                        Log.e(TAG, "检测结果渲染失败", error);
                    } finally {
                        frameForAnalysis.recycle();
                        analyzing.set(false);
                    }
                }

                @Override
                public void onError(Throwable error) {
                    frameForAnalysis.recycle();
                    analyzing.set(false);
                    Log.e(TAG, "实时检测失败", error);
                    if (isFinishing() || isDestroyed()) {
                        return;
                    }
                    detectionOverlay.clear();
                    monitorStatus.setText(getString(
                            R.string.monitor_error, errorMessage(error)));
                }
            });
        } catch (RuntimeException error) {
            frameForAnalysis.recycle();
            analyzing.set(false);
            Log.e(TAG, "提交实时检测失败", error);
        }
    }

    private void renderDetections(YoloDetector.DetectionResult result, Bitmap frame) {
        List<YoloDetector.Detection> detections = result.getDetections();
        detectionOverlay.setDetections(
                detections, result.getSourceWidth(), result.getSourceHeight());

        if (detections.isEmpty()) {
            monitorStatus.setText(R.string.monitor_no_detection);
            currentResult.setText(getString(
                    R.string.monitor_detection_latency,
                    result.getPreprocessMs(),
                    result.getInferenceMs(),
                    result.getPostprocessMs(),
                    result.getTotalMs()));
            return;
        }

        int personCount = 0;
        int catCount = 0;
        int dogCount = 0;
        for (YoloDetector.Detection detection : detections) {
            if (detection.getClassId() == YoloDetector.CLASS_PERSON) {
                personCount++;
            } else if (detection.getClassId() == YoloDetector.CLASS_CAT) {
                catCount++;
            } else if (detection.getClassId() == YoloDetector.CLASS_DOG) {
                dogCount++;
            }
        }

        YoloDetector.Detection top = detections.get(0);
        monitorStatus.setText(getString(
                R.string.monitor_detection_summary,
                detections.size(),
                personCount,
                catCount,
                dogCount,
                result.getProviderName()));
        currentResult.setText(getString(
                R.string.monitor_detection_top,
                top.getChineseName(),
                top.getConfidence() * 100.0f,
                result.getPreprocessMs(),
                result.getInferenceMs(),
                result.getPostprocessMs()));

        YoloDetector.Detection pet = findHighestPet(detections);
        if (pet == null) {
            return;
        }
        long now = SystemClock.elapsedRealtime();
        if (now - lastPetEventAtMs < PET_EVENT_INTERVAL_MS) {
            return;
        }
        lastPetEventAtMs = now;
        Bitmap thumbnail = createThumbnail(frame);
        if (thumbnail == null) {
            Log.w(TAG, "抓拍缩略图生成失败，跳过本次上报");
            return;
        }
        recordPetEvent(pet, thumbnail);
    }

    private YoloDetector.Detection findHighestPet(List<YoloDetector.Detection> detections) {
        for (YoloDetector.Detection detection : detections) {
            if (detection.getClassId() == YoloDetector.CLASS_CAT
                    || detection.getClassId() == YoloDetector.CLASS_DOG) {
                return detection;
            }
        }
        return null;
    }

    private Bitmap createThumbnail(Bitmap frame) {
        try {
            int width = frame.getWidth();
            int height = frame.getHeight();
            float scale = Math.min(
                    1.0f, THUMBNAIL_MAX_SIZE / (float) Math.max(width, height));
            int thumbWidth = Math.max(1, Math.round(width * scale));
            int thumbHeight = Math.max(1, Math.round(height * scale));
            Bitmap thumbnail = Bitmap.createScaledBitmap(
                    frame, thumbWidth, thumbHeight, true);
            if (thumbnail == frame) {
                thumbnail = frame.copy(Bitmap.Config.ARGB_8888, false);
            }
            return thumbnail;
        } catch (Throwable error) {
            Log.e(TAG, "生成抓拍缩略图失败", error);
            return null;
        }
    }

    private void recordPetEvent(YoloDetector.Detection pet, Bitmap thumbnail) {
        String time = new SimpleDateFormat("HH:mm:ss", Locale.CHINA).format(new Date());
        String line = getString(
                R.string.monitor_pet_event,
                time,
                pet.getChineseName(),
                pet.getConfidence() * 100.0f);
        eventLines.addFirst(line);
        while (eventLines.size() > MAX_EVENT_LINES) {
            eventLines.removeLast();
        }
        eventLog.setText(String.join("\n", eventLines));

        if (!thermalGuard.isReportingAllowed()) {
            thumbnail.recycle();
            Log.i(TAG, "端侧治理降级，抓拍仅保留本地日志并暂停云端上报");
            return;
        }

        // 复用既有 EventReporter；其内部在后台线程压缩/发送，完成后回收缩略图。
        eventReporter.reportPetEvent(
                pet.getChineseName(),
                pet.getConfidence(),
                thumbnail,
                new EventReporter.Callback() {
                    @Override
                    public void onSuccess(String eventId) {
                        thumbnail.recycle();
                        Log.i(TAG, "事件已上报，event id=" + eventId);
                    }

                    @Override
                    public void onError(Throwable error) {
                        thumbnail.recycle();
                        Log.e(TAG, "事件上报失败", error);
                    }
                });
    }

    private Bitmap imageProxyToBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        int pixelStride = plane.getPixelStride();
        int rowStride = plane.getRowStride();
        int width = image.getWidth();
        int height = image.getHeight();
        int base = buffer.position();
        int limit = buffer.limit();

        if (pixelStride < 4) {
            throw new IllegalStateException("CameraX RGBA 输出 pixelStride 异常：" + pixelStride);
        }

        int[] pixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            int rowStart = base + y * rowStride;
            for (int x = 0; x < width; x++) {
                int offset = rowStart + x * pixelStride;
                if (offset + 2 >= limit) {
                    throw new IllegalStateException("CameraX 帧缓冲区尺寸不足");
                }
                int red = buffer.get(offset) & 0xff;
                int green = buffer.get(offset + 1) & 0xff;
                int blue = buffer.get(offset + 2) & 0xff;
                pixels[y * width + x] = 0xff000000 | (red << 16) | (green << 8) | blue;
            }
        }

        Bitmap raw = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);
        Rect crop = image.getCropRect();
        Matrix matrix = new Matrix();
        matrix.postRotate(image.getImageInfo().getRotationDegrees());
        Bitmap transformed = Bitmap.createBitmap(
                raw, crop.left, crop.top, crop.width(), crop.height(), matrix, true);
        if (transformed != raw) {
            raw.recycle();
        }
        return transformed;
    }

    private static String errorMessage(Throwable error) {
        return error.getMessage() == null
                ? error.getClass().getSimpleName() : error.getMessage();
    }

    private void updateThermalGuardStatus(ThermalGuard.Policy policy) {
        if (isFinishing() || isDestroyed()) {
            return;
        }
        if (policy.isDegraded()) {
            thermalGuardStatus.setText(getString(
                    R.string.monitor_guard_degraded, policy.getReason()));
        } else {
            thermalGuardStatus.setText(R.string.monitor_guard_normal);
        }
        Log.i(TAG, "端侧治理策略：" + policy);
    }

    @Override
    public void onTrimMemory(int level) {
        super.onTrimMemory(level);
        if (thermalGuard != null) {
            thermalGuard.onTrimMemory(level);
        }
    }

    @Override
    public void onLowMemory() {
        super.onLowMemory();
        if (thermalGuard != null) {
            thermalGuard.onLowMemory();
        }
    }

    @Override
    protected void onDestroy() {
        if (thermalGuard != null) {
            thermalGuard.close();
        }
        if (imageAnalysis != null) {
            imageAnalysis.clearAnalyzer();
        }
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        if (detectionOverlay != null) {
            detectionOverlay.clear();
        }
        if (analyzerExecutor != null) {
            analyzerExecutor.shutdownNow();
        }
        super.onDestroy();
    }
}
