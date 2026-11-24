package com.edgevision.onnx.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;

import androidx.exifinterface.media.ExifInterface;

import java.io.IOException;
import java.io.InputStream;

/**
 * 与 model_pipeline/preprocess.py 对齐的图像预处理。
 *
 * Python 侧流程：EXIF 校正 -> RGB -> 短边缩放到 256 -> 中心裁剪 224x224 -> [0,1]
 * -> ImageNet mean/std 标准化 -> HWC 转 NCHW。本类逐步复刻该流程。
 */
public final class Preprocess {
    public static final int INPUT_SIZE = 224;
    public static final int RESIZE_SHORT_SIDE = 256;
    public static final int PIXEL_COUNT = INPUT_SIZE * INPUT_SIZE;
    public static final int FLOAT_COUNT = 3 * PIXEL_COUNT;

    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};

    private Preprocess() {
    }

    /** 从相册 Uri 解码并生成 NCHW float 输入，解码阶段先降采样以控制内存峰值。 */
    public static float[] fromUri(Context context, Uri uri) throws IOException {
        if (uri == null) {
            throw new IOException("图片 Uri 为空");
        }

        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream input = openInput(context, uri)) {
            BitmapFactory.decodeStream(input, null, bounds);
        }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw new IOException("无法识别图片尺寸，文件可能不是有效图片");
        }

        BitmapFactory.Options decodeOptions = new BitmapFactory.Options();
        decodeOptions.inSampleSize = calculateInSampleSize(bounds.outWidth, bounds.outHeight);
        decodeOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;

        Bitmap decoded;
        try (InputStream input = openInput(context, uri)) {
            decoded = BitmapFactory.decodeStream(input, null, decodeOptions);
        }
        if (decoded == null) {
            throw new IOException("图片解码失败");
        }

        Bitmap oriented = decoded;
        try {
            oriented = applyExifOrientation(decoded, readExifOrientation(context, uri));
            return fromBitmap(oriented);
        } finally {
            if (oriented != decoded) {
                oriented.recycle();
            }
            decoded.recycle();
        }
    }

    /** 将已校正方向的 Bitmap 缩放裁剪并转换为 NCHW float32。 */
    public static float[] fromBitmap(Bitmap bitmap) {
        if (bitmap == null || bitmap.isRecycled()) {
            throw new IllegalArgumentException("Bitmap 不可用");
        }

        Bitmap cropped = centerCropToInputSize(bitmap);
        try {
            int[] pixels = new int[PIXEL_COUNT];
            cropped.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

            float[] output = new float[FLOAT_COUNT];
            int greenOffset = PIXEL_COUNT;
            int blueOffset = PIXEL_COUNT * 2;
            for (int i = 0; i < PIXEL_COUNT; i++) {
                int pixel = pixels[i];
                float red = ((pixel >> 16) & 0xff) / 255.0f;
                float green = ((pixel >> 8) & 0xff) / 255.0f;
                float blue = (pixel & 0xff) / 255.0f;

                output[i] = (red - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                output[greenOffset + i] = (green - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                output[blueOffset + i] = (blue - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
            }
            return output;
        } finally {
            if (cropped != bitmap) {
                cropped.recycle();
            }
        }
    }

    /**
     * 短边等比缩放到 256，再中心裁剪 224x224。
     * 与 preprocess.py 及 torchvision 评测标准一致，避免长宽比悬殊的图片被拉伸变形。
     */
    private static Bitmap centerCropToInputSize(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float scale = (float) RESIZE_SHORT_SIDE / Math.min(width, height);
        int resizedWidth = Math.max(RESIZE_SHORT_SIDE, Math.round(width * scale));
        int resizedHeight = Math.max(RESIZE_SHORT_SIDE, Math.round(height * scale));

        Bitmap resized = (resizedWidth == width && resizedHeight == height)
                ? bitmap
                : Bitmap.createScaledBitmap(bitmap, resizedWidth, resizedHeight, true);

        int left = (resized.getWidth() - INPUT_SIZE) / 2;
        int top = (resized.getHeight() - INPUT_SIZE) / 2;
        Bitmap cropped = Bitmap.createBitmap(resized, left, top, INPUT_SIZE, INPUT_SIZE);
        if (cropped == null) {
            cropped = resized;
        }
        if (resized != bitmap && resized != cropped) {
            resized.recycle();
        }
        return cropped;
    }

    private static InputStream openInput(Context context, Uri uri) throws IOException {
        InputStream input = context.getContentResolver().openInputStream(uri);
        if (input == null) {
            throw new IOException("无法打开图片输入流");
        }
        return input;
    }

    private static int calculateInSampleSize(int width, int height) {
        int sampleSize = 1;
        while (Math.max(width / (sampleSize * 2), height / (sampleSize * 2)) >= INPUT_SIZE) {
            sampleSize *= 2;
        }
        return sampleSize;
    }

    private static int readExifOrientation(Context context, Uri uri) {
        try (InputStream input = openInput(context, uri)) {
            ExifInterface exif = new ExifInterface(input);
            return exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
        } catch (IOException | RuntimeException ignored) {
            return ExifInterface.ORIENTATION_NORMAL;
        }
    }

    private static Bitmap applyExifOrientation(Bitmap source, int orientation) {
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.setScale(-1.0f, 1.0f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180.0f);
                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.setScale(1.0f, -1.0f);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.setRotate(90.0f);
                matrix.postScale(-1.0f, 1.0f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90.0f);
                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.setRotate(-90.0f);
                matrix.postScale(-1.0f, 1.0f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.setRotate(-90.0f);
                break;
            case ExifInterface.ORIENTATION_NORMAL:
            case ExifInterface.ORIENTATION_UNDEFINED:
            default:
                return source;
        }

        Bitmap transformed = Bitmap.createBitmap(
                source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
        return transformed == null ? source : transformed;
    }
}
