package com.edgevision.onnx.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import com.edgevision.onnx.ml.YoloDetector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/** 在 PreviewView 上方按 FILL_CENTER 规则绘制检测框。 */
public final class DetectionOverlayView extends View {
    private static final int COLOR_PERSON = 0xFFFFB300;
    private static final int COLOR_CAT = 0xFF4CAF50;
    private static final int COLOR_DOG = 0xFF2196F3;

    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private List<YoloDetector.Detection> detections = Collections.emptyList();
    private int sourceWidth;
    private int sourceHeight;

    public DetectionOverlayView(Context context) {
        this(context, null);
    }

    public DetectionOverlayView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public DetectionOverlayView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        float density = getResources().getDisplayMetrics().density;
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(2.5f * density);
        fillPaint.setStyle(Paint.Style.FILL);
        textPaint.setTextSize(12.0f * density);
        textPaint.setColor(Color.WHITE);
        setClickable(false);
        setFocusable(false);
    }

    public void setDetections(
            List<YoloDetector.Detection> updated, int frameWidth, int frameHeight) {
        detections = updated == null
                ? Collections.emptyList()
                : new ArrayList<>(updated);
        sourceWidth = frameWidth;
        sourceHeight = frameHeight;
        invalidate();
    }

    public void clear() {
        detections = Collections.emptyList();
        sourceWidth = 0;
        sourceHeight = 0;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (detections.isEmpty() || sourceWidth <= 0 || sourceHeight <= 0) {
            return;
        }

        // PreviewView 默认 FILL_CENTER：等比放大至铺满，再居中裁剪。
        float scale = Math.max(
                getWidth() / (float) sourceWidth, getHeight() / (float) sourceHeight);
        float offsetX = (getWidth() - sourceWidth * scale) / 2.0f;
        float offsetY = (getHeight() - sourceHeight * scale) / 2.0f;

        float density = getResources().getDisplayMetrics().density;
        for (YoloDetector.Detection detection : detections) {
            RectF sourceBox = detection.getBox();
            RectF box = new RectF(
                    sourceBox.left * scale + offsetX,
                    sourceBox.top * scale + offsetY,
                    sourceBox.right * scale + offsetX,
                    sourceBox.bottom * scale + offsetY);
            if (!box.intersect(0.0f, 0.0f, getWidth(), getHeight())) {
                continue;
            }

            int color = colorFor(detection.getClassId());
            boxPaint.setColor(color);
            canvas.drawRect(box, boxPaint);

            String label = String.format(
                    Locale.ROOT,
                    "%s %s %.0f%%",
                    detection.getChineseName(),
                    detection.getClassName(),
                    detection.getConfidence() * 100.0f);
            float textHeight = textPaint.descent() - textPaint.ascent();
            float textWidth = textPaint.measureText(label);
            float labelTop = box.top - textHeight - 6.0f * density;
            if (labelTop < 0.0f) {
                labelTop = box.top + 2.0f * density;
            }
            RectF labelBackground = new RectF(
                    box.left,
                    labelTop,
                    box.left + textWidth + 8.0f * density,
                    labelTop + textHeight + 4.0f * density);
            fillPaint.setColor((color & 0x00FFFFFF) | 0xCC000000);
            canvas.drawRect(labelBackground, fillPaint);
            canvas.drawText(
                    label,
                    labelBackground.left + 4.0f * density,
                    labelTop - textPaint.ascent() + 2.0f * density,
                    textPaint);
        }
    }

    private static int colorFor(int classId) {
        if (classId == YoloDetector.CLASS_CAT) {
            return COLOR_CAT;
        }
        if (classId == YoloDetector.CLASS_DOG) {
            return COLOR_DOG;
        }
        return COLOR_PERSON;
    }
}
