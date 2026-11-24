package com.edgevision.onnx.ml;

import android.content.Context;
import android.content.res.AssetManager;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;

/** 读取 manifest.json 并校验打包进 assets 的模型文件大小与 SHA-256。 */
public final class AssetVerifier {
    public static final String MANIFEST_ASSET = "manifest.json";

    private AssetVerifier() {
    }

    public static final class ModelAsset {
        private final String key;
        private final String filename;
        private final long bytes;
        private final String sha256;

        private ModelAsset(String key, String filename, long bytes, String sha256) {
            this.key = key;
            this.filename = filename;
            this.bytes = bytes;
            this.sha256 = sha256;
        }

        public String getKey() {
            return key;
        }

        public String getFilename() {
            return filename;
        }

        public long getBytes() {
            return bytes;
        }

        public String getSha256() {
            return sha256;
        }
    }

    public static final class AssetCheck {
        private final ModelAsset model;
        private final long actualBytes;
        private final String actualSha256;
        private final boolean passed;

        private AssetCheck(ModelAsset model, long actualBytes, String actualSha256, boolean passed) {
            this.model = model;
            this.actualBytes = actualBytes;
            this.actualSha256 = actualSha256;
            this.passed = passed;
        }

        public ModelAsset getModel() {
            return model;
        }

        public long getActualBytes() {
            return actualBytes;
        }

        public String getActualSha256() {
            return actualSha256;
        }

        public boolean isPassed() {
            return passed;
        }
    }

    public static final class VerificationReport {
        private final List<AssetCheck> checks;
        private final long totalBytes;

        private VerificationReport(List<AssetCheck> checks, long totalBytes) {
            this.checks = Collections.unmodifiableList(checks);
            this.totalBytes = totalBytes;
        }

        public List<AssetCheck> getChecks() {
            return checks;
        }

        public long getTotalBytes() {
            return totalBytes;
        }
    }

    public static VerificationReport verifyAll(Context context) throws IOException, JSONException {
        List<ModelAsset> models = readModels(context);
        List<AssetCheck> checks = new ArrayList<>(models.size());
        long totalBytes = 0L;
        for (ModelAsset model : models) {
            AssetCheck check = verifyAsset(context, model);
            checks.add(check);
            totalBytes += model.getBytes();
        }
        return new VerificationReport(checks, totalBytes);
    }

    public static ModelAsset findModel(Context context, String key) throws IOException, JSONException {
        JSONObject files = readManifest(context).getJSONObject("files");
        if (!files.has(key)) {
            throw new IOException("manifest.json 中缺少模型键：" + key);
        }
        return parseModel(key, files.getJSONObject(key));
    }

    public static AssetCheck verifyAsset(Context context, ModelAsset model) throws IOException {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("当前运行环境不支持 SHA-256", e);
        }

        long actualBytes = 0L;
        byte[] buffer = new byte[64 * 1024];
        try (InputStream input = context.getAssets().open(model.getFilename())) {
            int read;
            while ((read = input.read(buffer)) != -1) {
                if (read == 0) {
                    continue;
                }
                digest.update(buffer, 0, read);
                actualBytes += read;
            }
        }

        String actualSha256 = toHex(digest.digest());
        boolean passed = actualBytes == model.getBytes()
                && actualSha256.equalsIgnoreCase(model.getSha256());
        return new AssetCheck(model, actualBytes, actualSha256, passed);
    }

    private static List<ModelAsset> readModels(Context context) throws IOException, JSONException {
        JSONObject files = readManifest(context).getJSONObject("files");
        List<ModelAsset> models = new ArrayList<>();
        Iterator<String> keys = files.keys();
        while (keys.hasNext()) {
            String key = keys.next();
            models.add(parseModel(key, files.getJSONObject(key)));
        }
        models.sort(Comparator.comparing(ModelAsset::getKey));
        return models;
    }

    private static ModelAsset parseModel(String key, JSONObject metadata) throws JSONException {
        return new ModelAsset(
                key,
                metadata.getString("filename"),
                metadata.getLong("bytes"),
                metadata.getString("sha256"));
    }

    private static JSONObject readManifest(Context context) throws IOException, JSONException {
        AssetManager assets = context.getAssets();
        try (InputStream input = assets.open(MANIFEST_ASSET);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
            String json = output.toString(StandardCharsets.UTF_8.name());
            return new JSONObject(json);
        }
    }

    private static String toHex(byte[] bytes) {
        StringBuilder builder = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            builder.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return builder.toString();
    }
}
