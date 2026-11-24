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
import com.edgevision.onnx.ml.ImageNetLabels;
import com.edgevision.onnx.ml.OnnxClassifier;
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

/** 宠物监控：CameraX 实时取帧，2 FPS 节流并在上一帧未完成时主动丢帧。 */
public class MonitorActivity extends AppCompatActivity {
    private static final String TAG = "MonitorActivity";
    private static final int TOP_K = 5;
    private static final long ANALYSIS_INTERVAL_MS = 500L;
    private static final long PET_EVENT_INTERVAL_MS = 5_000L;
    private static final float PET_CONFIDENCE_THRESHOLD = 0.10f;
    private static final int MAX_EVENT_LINES = 6;

    private final AtomicBoolean analyzing = new AtomicBoolean(false);
    private final ArrayDeque<String> eventLines = new ArrayDeque<>();

    private ExecutorService analyzerExecutor;
    private OnnxClassifier classifier;
    private EventReporter eventReporter;
    private ProcessCameraProvider cameraProvider;
    private ImageAnalysis imageAnalysis;

    private PreviewView previewView;
    private TextView monitorStatus;
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
        monitorStatus = findViewById(R.id.monitorStatus);
        currentResult = findViewById(R.id.currentResult);
        eventLog = findViewById(R.id.eventLog);
        MaterialButton backButton = findViewById(R.id.backButton);

        OnnxClassifier.Model model = OnnxClassifier.Model.fromKey(
                getIntent().getStringExtra(OnnxClassifier.EXTRA_MODEL));
        OnnxClassifier.ExecutionProvider provider = OnnxClassifier.ExecutionProvider.fromKey(
                getIntent().getStringExtra(OnnxClassifier.EXTRA_PROVIDER));
        classifier = OnnxClassifier.getInstance(getApplicationContext(), model, provider);
        eventReporter = EventReporter.getInstance(this);

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
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            monitorStatus.setText(R.string.monitor_waiting_permission);
            permissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void startCamera() {
        monitorStatus.setText(R.string.monitor_idle);
        ListenableFuture<ProcessCameraProvider> providerFuture = ProcessCameraProvider.getInstance(this);
        providerFuture.addListener(() -> {
            if (isFinishing() || isDestroyed()) {
                return;
            }
            try {
                cameraProvider = providerFuture.get();
                bindCameraUseCases();
            } catch (Exception error) {
                Log.e(TAG, "相机启动失败", error);
                String message = error.getMessage() == null
                        ? error.getClass().getSimpleName()
                        : error.getMessage();
                monitorStatus.setText(getString(R.string.camera_error, message));
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
        if (now - lastAnalysisAtMs < ANALYSIS_INTERVAL_MS) {
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
            classifier.classifyAsync(frameForAnalysis, TOP_K,
                    new OnnxClassifier.ClassificationCallback() {
                        @Override
                        public void onResult(OnnxClassifier.ClassificationResult result) {
                            // recycle 前降采样为 224x224 小图，供抓拍上报
                            Bitmap thumb = Bitmap.createScaledBitmap(frameForAnalysis, 224, 224, true);
                            frameForAnalysis.recycle();
                            analyzing.set(false);
                            if (isFinishing() || isDestroyed()) {
                                thumb.recycle();
                                return;
                            }
                            renderResult(result, thumb);
                        }

                        @Override
                        public void onError(Throwable error) {
                            frameForAnalysis.recycle();
                            analyzing.set(false);
                            Log.e(TAG, "实时分类失败", error);
                            if (isFinishing() || isDestroyed()) {
                                return;
                            }
                            String message = error.getMessage() == null
                                    ? error.getClass().getSimpleName()
                                    : error.getMessage();
                            monitorStatus.setText(getString(R.string.monitor_error, message));
                        }
                    });
        } catch (RuntimeException error) {
            frameForAnalysis.recycle();
            analyzing.set(false);
            Log.e(TAG, "提交实时分类失败", error);
        }
    }

    private void renderResult(OnnxClassifier.ClassificationResult result, Bitmap thumb) {
        List<ImageNetLabels.Prediction> predictions = result.getTopK();
        if (predictions.isEmpty()) {
            thumb.recycle();
            currentResult.setText(R.string.monitor_no_pet);
            return;
        }

        ImageNetLabels.Prediction top = predictions.get(0);
        currentResult.setText(getString(
                R.string.monitor_current,
                top.getChineseName(),
                top.getClassIndex(),
                top.getConfidence() * 100.0f));

        ImageNetLabels.Prediction pet = findPetPrediction(predictions);
        if (pet == null) {
            thumb.recycle();
            monitorStatus.setText(R.string.monitor_no_pet);
            return;
        }

        monitorStatus.setText(getString(R.string.monitor_pet, pet.getChineseName()));
        long now = SystemClock.elapsedRealtime();
        if (now - lastPetEventAtMs >= PET_EVENT_INTERVAL_MS) {
            lastPetEventAtMs = now;
            recordPetEvent(pet, thumb);
        } else {
            thumb.recycle();
        }
    }

    private ImageNetLabels.Prediction findPetPrediction(
            List<ImageNetLabels.Prediction> predictions) {
        for (ImageNetLabels.Prediction prediction : predictions) {
            if (ImageNetLabels.isPetClass(prediction.getClassIndex())
                    && prediction.getConfidence() >= PET_CONFIDENCE_THRESHOLD) {
                return prediction;
            }
        }
        return null;
    }

    private void recordPetEvent(ImageNetLabels.Prediction pet, Bitmap thumb) {
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

        // 上报到后端（EventReporter 内部在后台线程压缩/发送，完成后回收缩略图）
        eventReporter.reportPetEvent(pet, thumb, new EventReporter.Callback() {
            @Override
            public void onSuccess(String eventId) {
                thumb.recycle();
                Log.i(TAG, "事件已上报，event id=" + eventId);
            }

            @Override
            public void onError(Throwable error) {
                thumb.recycle();
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

    @Override
    protected void onDestroy() {
        if (imageAnalysis != null) {
            imageAnalysis.clearAnalyzer();
        }
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        if (analyzerExecutor != null) {
            analyzerExecutor.shutdownNow();
        }
        super.onDestroy();
    }
}
