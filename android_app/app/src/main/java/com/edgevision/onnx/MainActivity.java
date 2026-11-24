package com.edgevision.onnx;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.RadioGroup;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.edgevision.onnx.ml.AssetVerifier;
import com.edgevision.onnx.ml.OnnxClassifier;
import com.google.android.material.button.MaterialButton;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;

/** 应用入口：启动时校验模型资产，并选择模型 / Execution Provider 后进入业务页面。 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private ExecutorService verificationExecutor;

    private ProgressBar verifyProgress;
    private TextView statusText;
    private MaterialButton retryButton;
    private MaterialButton albumButton;
    private MaterialButton monitorButton;
    private MaterialButton benchmarkButton;
    private RadioGroup modelGroup;
    private RadioGroup providerGroup;
    private boolean assetsVerified;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        verifyProgress = findViewById(R.id.verifyProgress);
        statusText = findViewById(R.id.statusText);
        retryButton = findViewById(R.id.retryButton);
        albumButton = findViewById(R.id.albumButton);
        monitorButton = findViewById(R.id.monitorButton);
        benchmarkButton = findViewById(R.id.benchmarkButton);
        modelGroup = findViewById(R.id.modelGroup);
        providerGroup = findViewById(R.id.providerGroup);

        verificationExecutor = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, "asset-verifier");
            thread.setDaemon(true);
            return thread;
        });

        retryButton.setOnClickListener(view -> verifyAssets());
        albumButton.setOnClickListener(view -> openPage(com.edgevision.onnx.ui.AlbumActivity.class));
        monitorButton.setOnClickListener(view -> openPage(com.edgevision.onnx.ui.MonitorActivity.class));
        benchmarkButton.setOnClickListener(
                view -> openPage(com.edgevision.onnx.ui.BenchmarkActivity.class));

        verifyAssets();
    }

    private void verifyAssets() {
        assetsVerified = false;
        setVerificationBusy(true);
        statusText.setText(R.string.asset_checking);

        try {
            verificationExecutor.execute(() -> {
                try {
                    AssetVerifier.VerificationReport report =
                            AssetVerifier.verifyAll(getApplicationContext());
                    if (report.getChecks().isEmpty()) {
                        throw new IllegalStateException("manifest.json 未声明任何模型文件");
                    }
                    for (AssetVerifier.AssetCheck check : report.getChecks()) {
                        if (!check.isPassed()) {
                            throw new IllegalStateException(
                                    check.getModel().getFilename() + " 大小或 SHA-256 不符");
                        }
                    }
                    int count = report.getChecks().size();
                    long totalBytes = report.getTotalBytes();
                    mainHandler.post(() -> {
                        if (isFinishing() || isDestroyed()) {
                            return;
                        }
                        assetsVerified = true;
                        statusText.setText(getString(R.string.asset_ok, count, totalBytes));
                        setVerificationBusy(false);
                    });
                } catch (Throwable error) {
                    Log.e(TAG, "模型资产校验失败", error);
                    String message = error.getMessage() == null
                            ? error.getClass().getSimpleName() : error.getMessage();
                    mainHandler.post(() -> {
                        if (isFinishing() || isDestroyed()) {
                            return;
                        }
                        statusText.setText(getString(R.string.asset_failed, message));
                        setVerificationBusy(false);
                    });
                }
            });
        } catch (RejectedExecutionException error) {
            statusText.setText(getString(R.string.asset_failed, error.getMessage()));
            setVerificationBusy(false);
        }
    }

    private void setVerificationBusy(boolean busy) {
        verifyProgress.setVisibility(busy ? View.VISIBLE : View.GONE);
        retryButton.setEnabled(!busy);
        albumButton.setEnabled(!busy && assetsVerified);
        monitorButton.setEnabled(!busy && assetsVerified);
        benchmarkButton.setEnabled(!busy && assetsVerified);
    }

    private OnnxClassifier.Model selectedModel() {
        return modelGroup.getCheckedRadioButtonId() == R.id.modelInt8
                ? OnnxClassifier.Model.INT8_QDQ
                : OnnxClassifier.Model.FP32;
    }

    private OnnxClassifier.ExecutionProvider selectedProvider() {
        int checkedId = providerGroup.getCheckedRadioButtonId();
        if (checkedId == R.id.providerXnnpack) {
            return OnnxClassifier.ExecutionProvider.XNNPACK;
        }
        if (checkedId == R.id.providerNnapi) {
            return OnnxClassifier.ExecutionProvider.NNAPI;
        }
        return OnnxClassifier.ExecutionProvider.CPU;
    }

    private void openPage(Class<?> targetActivity) {
        Intent intent = new Intent(this, targetActivity);
        intent.putExtra(OnnxClassifier.EXTRA_MODEL, selectedModel().name());
        intent.putExtra(OnnxClassifier.EXTRA_PROVIDER, selectedProvider().name());
        startActivity(intent);
    }

    @Override
    protected void onDestroy() {
        if (verificationExecutor != null) {
            verificationExecutor.shutdownNow();
        }
        super.onDestroy();
    }
}
