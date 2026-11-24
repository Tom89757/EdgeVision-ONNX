package com.edgevision.onnx.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.Base64;
import android.util.Log;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;

/**
 * 端云联动上报器：把端侧宠物抓拍事件 POST 到 FastAPI 后端。
 * 网络请求在单线程 ExecutorService 后台执行，回调切回主线程。
 */
public final class EventReporter {
    private static final String TAG = "EventReporter";

    /** 后端所在电脑的局域网 IP（真机访问不到 localhost）。 */
    private static final String BACKEND_BASE_URL = "http://192.168.1.2:8000";

    private static final int CONNECT_TIMEOUT_MS = 5_000;
    private static final int READ_TIMEOUT_MS = 8_000;
    private static final int MAX_RETRIES = 2;

    private final String deviceId;
    private final ExecutorService executor;
    private final Handler mainHandler;

    private static volatile EventReporter instance;

    public interface Callback {
        void onSuccess(String eventId);

        void onError(Throwable error);
    }

    private EventReporter(Context context) {
        this.deviceId = resolveDeviceId(context);
        this.mainHandler = new Handler(Looper.getMainLooper());
        this.executor = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, "event-reporter");
            thread.setDaemon(true);
            return thread;
        });
    }

    public static EventReporter getInstance(Context context) {
        if (instance == null) {
            synchronized (EventReporter.class) {
                if (instance == null) {
                    instance = new EventReporter(context.getApplicationContext());
                }
            }
        }
        return instance;
    }

    private static String resolveDeviceId(Context context) {
        String androidId = Settings.Secure.getString(
                context.getContentResolver(), Settings.Secure.ANDROID_ID);
        return androidId == null ? "unknown-device" : androidId;
    }

    /** 异步上报；幂等 key 在内部生成并在重试时复用。 */
    public void reportPetEvent(final ImageNetLabels.Prediction pet,
                               final Bitmap thumbnail,
                               final Callback callback) {
        final String idempotencyKey = UUID.randomUUID().toString();
        try {
            executor.execute(() -> {
                try {
                    String eventId = postWithRetry(pet, thumbnail, idempotencyKey);
                    mainHandler.post(() -> callback.onSuccess(eventId));
                } catch (Throwable error) {
                    Log.e(TAG, "事件上报失败", error);
                    mainHandler.post(() -> callback.onError(error));
                }
            });
        } catch (RejectedExecutionException error) {
            mainHandler.post(() -> callback.onError(error));
        }
    }

    private String postWithRetry(ImageNetLabels.Prediction pet, Bitmap thumbnail, String key)
            throws Exception {
        Exception last = null;
        for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            try {
                return post(pet, thumbnail, key);
            } catch (Exception error) {
                last = error;
                Log.w(TAG, "上报第 " + attempt + " 次失败：" + error.getMessage());
            }
        }
        throw last == null ? new IllegalStateException("上报失败") : last;
    }

    private String post(ImageNetLabels.Prediction pet, Bitmap thumbnail, String key)
            throws Exception {
        JSONObject body = new JSONObject();
        body.put("thumbnail", bitmapToBase64(thumbnail));
        body.put("category", pet.getChineseName());
        body.put("confidence", pet.getConfidence());
        body.put("timestamp", nowUtcIso());
        body.put("device_id", deviceId);

        HttpURLConnection connection = null;
        try {
            URL url = new URL(BACKEND_BASE_URL + "/api/v1/events/pet");
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            connection.setRequestProperty("Content-Type", "application/json; charset=utf-8");
            connection.setRequestProperty("Idempotency-Key", key);
            connection.setDoOutput(true);

            byte[] payload = body.toString().getBytes(StandardCharsets.UTF_8);
            try (OutputStream output = connection.getOutputStream()) {
                output.write(payload);
            }

            int status = connection.getResponseCode();
            if (status != 200 && status != 201) {
                throw new IllegalStateException("后端返回 HTTP " + status);
            }
            return parseEventId(connection);
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private String parseEventId(HttpURLConnection connection) throws Exception {
        try (InputStream input = connection.getInputStream();
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
            JSONObject json = new JSONObject(output.toString("UTF-8"));
            JSONObject data = json.optJSONObject("data");
            return data == null ? "unknown" : String.valueOf(data.opt("id"));
        }
    }

    /** 后台线程内压缩为 JPEG 再转 Base64；入参应已是降采样后的小图。 */
    private static String bitmapToBase64(Bitmap bitmap) {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 70, output);
        return Base64.encodeToString(output.toByteArray(), Base64.NO_WRAP);
    }

    /** 生成带 +00:00 偏移的 UTC 时间，兼容后端 fromisoformat（不支持 Z 后缀）。 */
    private static String nowUtcIso() {
        return OffsetDateTime.now(ZoneOffset.UTC)
                .format(DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSxxx"));
    }
}
