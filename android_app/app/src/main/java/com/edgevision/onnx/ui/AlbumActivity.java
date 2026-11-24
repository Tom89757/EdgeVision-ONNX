package com.edgevision.onnx.ui;

import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import com.edgevision.onnx.R;
import com.edgevision.onnx.ml.ImageNetLabels;
import com.edgevision.onnx.ml.OnnxClassifier;
import com.google.android.material.button.MaterialButton;

import java.util.List;

/** 智能相册：选择本地图片后完全离线执行 Top-K 分类。 */
public class AlbumActivity extends AppCompatActivity {
    private static final String TAG = "AlbumActivity";
    private static final int TOP_K = 5;

    private ActivityResultLauncher<String> imagePicker;
    private OnnxClassifier classifier;

    private MaterialButton pickButton;
    private ProgressBar progressBar;
    private TextView resultText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_album);

        pickButton = findViewById(R.id.pickButton);
        progressBar = findViewById(R.id.progressBar);
        resultText = findViewById(R.id.resultText);
        MaterialButton backButton = findViewById(R.id.backButton);

        OnnxClassifier.Model model = OnnxClassifier.Model.fromKey(
                getIntent().getStringExtra(OnnxClassifier.EXTRA_MODEL));
        OnnxClassifier.ExecutionProvider provider = OnnxClassifier.ExecutionProvider.fromKey(
                getIntent().getStringExtra(OnnxClassifier.EXTRA_PROVIDER));
        classifier = OnnxClassifier.getInstance(getApplicationContext(), model, provider);

        imagePicker = registerForActivityResult(
                new ActivityResultContracts.GetContent(), this::onImageSelected);

        pickButton.setOnClickListener(view -> imagePicker.launch("image/*"));
        backButton.setOnClickListener(view -> finish());
    }

    private void onImageSelected(Uri uri) {
        if (uri == null || isFinishing() || isDestroyed()) {
            return;
        }

        setRunning(true);
        resultText.setText(R.string.recognizing);
        classifier.classifyAsync(uri, TOP_K, new OnnxClassifier.ClassificationCallback() {
            @Override
            public void onResult(OnnxClassifier.ClassificationResult result) {
                if (isFinishing() || isDestroyed()) {
                    return;
                }
                setRunning(false);
                renderResult(result);
            }

            @Override
            public void onError(Throwable error) {
                Log.e(TAG, "相册分类失败", error);
                if (isFinishing() || isDestroyed()) {
                    return;
                }
                setRunning(false);
                String message = error.getMessage() == null
                        ? error.getClass().getSimpleName() : error.getMessage();
                resultText.setText(getString(R.string.album_error, message));
            }
        });
    }

    private void renderResult(OnnxClassifier.ClassificationResult result) {
        List<ImageNetLabels.Prediction> predictions = result.getTopK();
        if (predictions.isEmpty()) {
            resultText.setText(R.string.album_no_result);
            return;
        }

        StringBuilder builder = new StringBuilder();
        builder.append(getString(
                R.string.album_result_header,
                result.getModelName(),
                result.getProviderName(),
                result.getPreprocessMs(),
                result.getInferenceMs(),
                result.getPostprocessMs()));
        builder.append('\n').append('\n');
        for (int i = 0; i < predictions.size(); i++) {
            ImageNetLabels.Prediction prediction = predictions.get(i);
            builder.append(getString(
                    R.string.album_result_line,
                    i + 1,
                    prediction.getChineseName(),
                    prediction.getClassIndex(),
                    prediction.getConfidence() * 100.0f));
            if (i < predictions.size() - 1) {
                builder.append('\n');
            }
        }
        resultText.setText(builder.toString());
    }

    private void setRunning(boolean running) {
        pickButton.setEnabled(!running);
        progressBar.setVisibility(running ? View.VISIBLE : View.GONE);
        if (running) {
            resultText.setText(R.string.recognizing);
        }
    }
}
