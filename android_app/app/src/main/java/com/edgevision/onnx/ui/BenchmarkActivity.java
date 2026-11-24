package com.edgevision.onnx.ui;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Shader;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.edgevision.onnx.R;
import com.edgevision.onnx.benchmark.BenchmarkRunner;
import com.google.android.material.button.MaterialButton;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** 一键执行 FP32/INT8 × CPU/XNNPACK/NNAPI 的真机 benchmark，并导出 JSON。 */
public class BenchmarkActivity extends AppCompatActivity {
    private static final String TAG = "BenchmarkActivity";

    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private ExecutorService executor;

    private MaterialButton runButton;
    private MaterialButton exportButton;
    private MaterialButton backButton;
    private ProgressBar progressBar;
    private TextView statusText;
    private TextView resultText;
    private TextView exportPathText;
    private BenchmarkRunner.Report report;
    private boolean running;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_benchmark);

        runButton = findViewById(R.id.runBenchmarkButton);
        exportButton = findViewById(R.id.exportBenchmarkButton);
        backButton = findViewById(R.id.benchmarkBackButton);
        progressBar = findViewById(R.id.benchmarkProgress);
        statusText = findViewById(R.id.benchmarkStatus);
        resultText = findViewById(R.id.benchmarkResult);
        exportPathText = findViewById(R.id.exportPathText);

        executor = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, "benchmark-runner");
            thread.setDaemon(true);
            return thread;
        });

        runButton.setOnClickListener(view -> startBenchmark());
        exportButton.setOnClickListener(view -> exportJson());
        backButton.setOnClickListener(view -> finish());
        updateControls();
    }

    private void startBenchmark() {
        if (running) {
            return;
        }
        running = true;
        report = null;
        exportPathText.setText("");
        resultText.setText(R.string.benchmark_empty);
        updateControls();

        executor.execute(() -> {
            Bitmap benchmarkBitmap = null;
            try {
                benchmarkBitmap = createBenchmarkBitmap();
                BenchmarkRunner runner = new BenchmarkRunner(getApplicationContext());
                BenchmarkRunner.Report generated = runner.run(
                        benchmarkBitmap,
                        message -> mainHandler.post(() -> {
                            if (!isFinishing() && !isDestroyed()) {
                                statusText.setText(message);
                            }
                        }));
                mainHandler.post(() -> onBenchmarkFinished(generated));
            } catch (Throwable error) {
                Log.e(TAG, "benchmark 执行失败", error);
                String message = error.getMessage() == null
                        ? error.getClass().getSimpleName() : error.getMessage();
                mainHandler.post(() -> onBenchmarkFailed(message));
            } finally {
                if (benchmarkBitmap != null) {
                    benchmarkBitmap.recycle();
                }
            }
        });
    }

    private void onBenchmarkFinished(BenchmarkRunner.Report generated) {
        if (isFinishing() || isDestroyed()) {
            return;
        }
        running = false;
        report = generated;
        statusText.setText(getString(R.string.benchmark_done));
        resultText.setText(renderReport(generated));
        updateControls();
    }

    private void onBenchmarkFailed(String message) {
        if (isFinishing() || isDestroyed()) {
            return;
        }
        running = false;
        report = null;
        statusText.setText(getString(R.string.benchmark_failed, message));
        updateControls();
    }

    private String renderReport(BenchmarkRunner.Report generated) {
        StringBuilder builder = new StringBuilder();
        builder.append(getString(
                R.string.benchmark_method,
                BenchmarkRunner.WARMUP_FRAMES,
                BenchmarkRunner.MEASURED_FRAMES));
        builder.append('\n').append(generated.getDeviceLabel()).append('\n');
        builder.append("ABI: ").append(generated.getSupportedAbis()).append("\n\n");

        for (BenchmarkRunner.Result result : generated.getResults()) {
            builder.append(result.getConfigurationLabel()).append('\n');
            builder.append("实际 EP：").append(result.getActualProvider()).append('\n');
            if (!result.isSuccessful()) {
                builder.append("失败：").append(result.getError()).append("\n\n");
                continue;
            }
            builder.append(String.format(
                    Locale.ROOT,
                    "冷启动 %.1f ms · 首帧 %.1f ms · PSS %.1f MB\n",
                    result.getColdLoadMs(),
                    result.getFirstFrameMs(),
                    result.getPssMb()));
            builder.append(String.format(
                    Locale.ROOT,
                    "稳态总耗时 p50 %.2f ms / p95 %.2f ms\n",
                    result.getTotalP50Ms(),
                    result.getTotalP95Ms()));
            builder.append(String.format(
                    Locale.ROOT,
                    "预处理 %.2f / %.2f · 推理 %.2f / %.2f · 后处理 %.2f / %.2f（p50 / p95 ms）\n\n",
                    result.getPreprocessP50Ms(),
                    result.getPreprocessP95Ms(),
                    result.getInferenceP50Ms(),
                    result.getInferenceP95Ms(),
                    result.getPostprocessP50Ms(),
                    result.getPostprocessP95Ms()));
        }
        return builder.toString().trim();
    }

    private void exportJson() {
        if (report == null) {
            return;
        }
        try {
            String json = BenchmarkRunner.toJson(report);
            File parent = getExternalFilesDir("benchmark");
            if (parent == null) {
                parent = new File(getFilesDir(), "benchmark");
            }
            if (!parent.exists() && !parent.mkdirs()) {
                throw new IOException("无法创建导出目录：" + parent);
            }
            String safeModel = Build.MODEL == null
                    ? "device" : Build.MODEL.replaceAll("[^A-Za-z0-9._-]", "_");
            File output = new File(
                    parent,
                    "benchmark_" + safeModel + "_" + report.getGeneratedAtMs() + ".json");
            try (FileOutputStream stream = new FileOutputStream(output)) {
                stream.write(json.getBytes(StandardCharsets.UTF_8));
                stream.flush();
            }
            exportPathText.setText(output.getAbsolutePath());
            Toast.makeText(this, R.string.benchmark_export_ok, Toast.LENGTH_LONG).show();
        } catch (Exception error) {
            Log.e(TAG, "benchmark JSON 导出失败", error);
            String message = error.getMessage() == null
                    ? error.getClass().getSimpleName() : error.getMessage();
            statusText.setText(getString(R.string.benchmark_export_failed, message));
        }
    }

    private void updateControls() {
        runButton.setEnabled(!running);
        exportButton.setEnabled(!running && report != null);
        progressBar.setVisibility(running ? View.VISIBLE : View.GONE);
        if (running) {
            statusText.setText(R.string.benchmark_running);
        }
    }

    /** 固定 640x480 合成图，避免相册解码和文件 IO 干扰端侧管线的计时。 */
    private static Bitmap createBenchmarkBitmap() {
        int width = 640;
        int height = 480;
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);

        paint.setShader(new LinearGradient(
                0,
                0,
                width,
                height,
                Color.rgb(39, 85, 154),
                Color.rgb(212, 225, 201),
                Shader.TileMode.CLAMP));
        canvas.drawRect(0, 0, width, height, paint);
        paint.setShader(null);

        paint.setColor(Color.rgb(232, 178, 92));
        canvas.drawCircle(180, 210, 92, paint);
        paint.setColor(Color.rgb(74, 62, 54));
        canvas.drawCircle(490, 300, 118, paint);
        paint.setColor(Color.rgb(242, 240, 232));
        canvas.drawRect(70, 370, 570, 410, paint);
        return bitmap;
    }

    @Override
    protected void onDestroy() {
        if (executor != null) {
            executor.shutdownNow();
        }
        super.onDestroy();
    }
}
