package com.edgevision.onnx.ml;

import android.annotation.TargetApi;
import android.content.ComponentCallbacks2;
import android.content.Context;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.os.SystemClock;
import android.util.Log;

import java.util.Arrays;
import java.util.Objects;

/**
 * 端侧运行治理：热状态监听、低内存降级与滑动窗口 p95 延迟熔断。
 *
 * <p>正常监控目标为 2 FPS；任一治理条件触发后切换到 0.5 FPS，并暂停云端上报等非关键任务。
 * 温度回落或延迟熔断解除后仍保持 {@link #COOLDOWN_MS} 的冷却窗口，避免状态频繁抖动。</p>
 */
public final class ThermalGuard {
    private static final String TAG = "ThermalGuard";

    /** 正常分析目标间隔：500 ms，即 2 FPS。 */
    public static final long NORMAL_ANALYSIS_INTERVAL_MS = 500L;

    /** 降级分析目标间隔：2000 ms，即 0.5 FPS。 */
    public static final long DEGRADED_ANALYSIS_INTERVAL_MS = 2_000L;

    /** 温度回落或 p95 熔断后继续降级的冷却窗口，默认 30 秒。 */
    public static final long COOLDOWN_MS = 30_000L;

    /** p95 滑动窗口样本数；以 2 FPS 计约覆盖 15 秒，仅为窗口配置而非实测值。 */
    public static final int LATENCY_WINDOW_SIZE = 30;

    /** 使用 95 分位耗时判断长尾延迟，不采用单帧最大值以免偶发抖动直接熔断。 */
    public static final double LATENCY_PERCENTILE = 0.95d;

    /**
     * 端到端单帧 p95 延迟预算。该值是为 2 FPS 目标预留余量的工程预算，
     * 不是任何机型或 Execution Provider 的实测性能数字；真机部署时应按实测分布调整。
     */
    public static final long LATENCY_P95_BUDGET_MS = 450L;

    /** 窗口填满后逐帧复核，连续 3 次 p95 超预算即熔断。 */
    public static final int LATENCY_BREACH_STREAK_LIMIT = 3;

    /** 治理策略不可变快照，供 UI 读取当前帧率与上报开关。 */
    public static final class Policy {
        private final boolean degraded;
        private final long analysisIntervalMs;
        private final boolean reportingAllowed;
        private final String reason;

        private Policy(
                boolean degraded,
                long analysisIntervalMs,
                boolean reportingAllowed,
                String reason) {
            this.degraded = degraded;
            this.analysisIntervalMs = analysisIntervalMs;
            this.reportingAllowed = reportingAllowed;
            this.reason = reason;
        }

        public boolean isDegraded() {
            return degraded;
        }

        public long getAnalysisIntervalMs() {
            return analysisIntervalMs;
        }

        public boolean isReportingAllowed() {
            return reportingAllowed;
        }

        /** 降级原因；正常状态返回“正常”。 */
        public String getReason() {
            return reason;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof Policy)) {
                return false;
            }
            Policy that = (Policy) other;
            return degraded == that.degraded
                    && analysisIntervalMs == that.analysisIntervalMs
                    && reportingAllowed == that.reportingAllowed
                    && Objects.equals(reason, that.reason);
        }

        @Override
        public int hashCode() {
            return Objects.hash(degraded, analysisIntervalMs, reportingAllowed, reason);
        }

        @Override
        public String toString() {
            return "Policy{degraded=" + degraded
                    + ", intervalMs=" + analysisIntervalMs
                    + ", reportingAllowed=" + reportingAllowed
                    + ", reason='" + reason + "'}";
        }
    }

    public interface Listener {
        void onPolicyChanged(Policy policy);
    }

    /** 隔离 API 29+ 热状态类型，确保 minSdk 26 设备不会提前解析新 API。 */
    private interface ThermalMonitor {
        void close();
    }

    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final PowerManager powerManager;
    private final Listener listener;
    private final Object lock = new Object();
    private final long[] latencyWindow = new long[LATENCY_WINDOW_SIZE];

    private final ThermalMonitor thermalMonitor;
    private int latencySampleCount;
    private int latencyWriteIndex;
    private int consecutiveOverBudgetWindows;
    private long currentP95Ms;
    private boolean thermalDegraded;
    private long thermalCooldownUntilMs;
    private boolean memoryDegraded;
    private boolean latencyCircuitOpen;
    private long latencyCircuitUntilMs;
    private boolean closed;
    private Policy lastNotifiedPolicy;

    public ThermalGuard(Context context, Listener listener) {
        if (context == null) {
            throw new IllegalArgumentException("Context 不能为空");
        }
        Context appContext = context.getApplicationContext();
        this.powerManager = (PowerManager) appContext.getSystemService(Context.POWER_SERVICE);
        this.listener = listener;

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && powerManager != null) {
            thermalMonitor = new Api29ThermalMonitor(powerManager, mainHandler, this);
        } else {
            thermalMonitor = null;
            Log.i(TAG, "当前系统不支持热状态监听，仅启用低内存与延迟熔断治理");
        }
        mainHandler.post(this::notifyPolicyIfChanged);
    }

    @TargetApi(Build.VERSION_CODES.Q)
    private static final class Api29ThermalMonitor
            implements ThermalMonitor, PowerManager.OnThermalStatusChangedListener {
        private final PowerManager manager;
        private final Handler handler;
        private final ThermalGuard guard;
        private boolean registered;

        Api29ThermalMonitor(PowerManager manager, Handler handler, ThermalGuard guard) {
            this.manager = manager;
            this.handler = handler;
            this.guard = guard;
            start();
        }

        private void start() {
            try {
                manager.addThermalStatusListener(this);
                registered = true;
                int initialStatus = manager.getCurrentThermalStatus();
                Log.i(TAG, "热状态监听已注册，当前状态=" + initialStatus);
                handler.post(() -> dispatch(initialStatus));
            } catch (RuntimeException error) {
                Log.w(TAG, "热状态监听注册失败，继续运行其他治理策略", error);
                if (registered) {
                    try {
                        manager.removeThermalStatusListener(this);
                    } catch (RuntimeException ignored) {
                        Log.w(TAG, "移除热状态监听失败", ignored);
                    }
                    registered = false;
                }
            }
        }

        @Override
        public void onThermalStatusChanged(int status) {
            handler.post(() -> dispatch(status));
        }

        private void dispatch(int status) {
            guard.handleThermalStatus(status >= PowerManager.THERMAL_STATUS_MODERATE);
        }

        @Override
        public void close() {
            if (!registered) {
                return;
            }
            try {
                manager.removeThermalStatusListener(this);
            } catch (RuntimeException error) {
                Log.w(TAG, "移除热状态监听失败", error);
            }
            registered = false;
        }
    }

    private void handleThermalStatus(boolean hot) {
        synchronized (lock) {
            if (closed) {
                return;
            }
            long now = SystemClock.elapsedRealtime();
            if (hot) {
                thermalDegraded = true;
                thermalCooldownUntilMs = 0L;
            } else if (thermalDegraded) {
                thermalDegraded = false;
                thermalCooldownUntilMs = now + COOLDOWN_MS;
                Log.i(TAG, "热状态回落，进入 " + COOLDOWN_MS + " ms 冷却窗口");
            }
        }
        notifyPolicyIfChanged();
    }

    /** 接收 Activity 转发的 onTrimMemory；低或严重内存压力立即降级。 */
    public void onTrimMemory(int level) {
        boolean changed = false;
        synchronized (lock) {
            if (closed) {
                return;
            }
            if (level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW
                    || level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL) {
                if (!memoryDegraded) {
                    memoryDegraded = true;
                    changed = true;
                }
            } else if (level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_MODERATE
                    && memoryDegraded) {
                memoryDegraded = false;
                changed = true;
            }
        }
        if (changed) {
            Log.i(TAG, "低内存治理状态已更新");
            notifyPolicyIfChanged();
        }
    }

    /** 兼容旧版 onLowMemory 回调；无法获知具体压力等级时按最保守策略降级。 */
    public void onLowMemory() {
        synchronized (lock) {
            if (closed || memoryDegraded) {
                return;
            }
            memoryDegraded = true;
        }
        Log.i(TAG, "收到 onLowMemory，进入降级");
        notifyPolicyIfChanged();
    }

    /**
     * 记录一帧检测的端到端耗时（预处理 + 纯推理 + 后处理）。
     * 窗口填满后计算 p95，连续超预算达到阈值时开启熔断。
     */
    public void recordFrameLatency(long frameLatencyMs) {
        if (frameLatencyMs < 0L) {
            return;
        }
        synchronized (lock) {
            if (closed) {
                return;
            }
            refreshExpiredState(SystemClock.elapsedRealtime());

            latencyWindow[latencyWriteIndex] = frameLatencyMs;
            latencyWriteIndex = (latencyWriteIndex + 1) % LATENCY_WINDOW_SIZE;
            if (latencySampleCount < LATENCY_WINDOW_SIZE) {
                latencySampleCount++;
            }
            if (latencySampleCount >= LATENCY_WINDOW_SIZE) {
                currentP95Ms = percentile95(latencyWindow, latencySampleCount);
                if (currentP95Ms > LATENCY_P95_BUDGET_MS) {
                    consecutiveOverBudgetWindows++;
                } else {
                    consecutiveOverBudgetWindows = 0;
                }

                if (!latencyCircuitOpen
                        && consecutiveOverBudgetWindows >= LATENCY_BREACH_STREAK_LIMIT) {
                    latencyCircuitOpen = true;
                    latencyCircuitUntilMs = SystemClock.elapsedRealtime() + COOLDOWN_MS;
                    Log.w(TAG, "p95=" + currentP95Ms + " ms 连续超预算，开启 "
                            + COOLDOWN_MS + " ms 延迟熔断");
                }
            }
        }
        notifyPolicyIfChanged();
    }

    public Policy getPolicy() {
        synchronized (lock) {
            refreshExpiredState(SystemClock.elapsedRealtime());
            return buildPolicy();
        }
    }

    public long getAnalysisIntervalMs() {
        return getPolicy().getAnalysisIntervalMs();
    }

    public boolean isReportingAllowed() {
        return getPolicy().isReportingAllowed();
    }

    /** 当前滑动窗口 p95；窗口未填满时返回 0。 */
    public long getCurrentP95Ms() {
        synchronized (lock) {
            return currentP95Ms;
        }
    }

    private void refreshExpiredState(long now) {
        if (thermalCooldownUntilMs > 0L && now >= thermalCooldownUntilMs) {
            thermalCooldownUntilMs = 0L;
        }
        if (latencyCircuitOpen && now >= latencyCircuitUntilMs) {
            latencyCircuitOpen = false;
            latencyCircuitUntilMs = 0L;
            consecutiveOverBudgetWindows = 0;
            latencySampleCount = 0;
            latencyWriteIndex = 0;
            currentP95Ms = 0L;
            Log.i(TAG, "延迟熔断冷却结束，重新开始采样");
        }
    }

    private Policy buildPolicy() {
        boolean thermalCooling = !thermalDegraded
                && thermalCooldownUntilMs > SystemClock.elapsedRealtime();
        boolean latencyTripped = latencyCircuitOpen;
        boolean degraded = thermalDegraded || thermalCooling || memoryDegraded || latencyTripped;

        String reason = "正常";
        if (thermalDegraded) {
            reason = "热状态中度及以上";
        } else if (thermalCooling) {
            reason = "温度回落冷却窗口";
        } else if (memoryDegraded) {
            reason = "低内存";
        } else if (latencyTripped) {
            reason = "p95 延迟熔断";
        }

        return new Policy(
                degraded,
                degraded ? DEGRADED_ANALYSIS_INTERVAL_MS : NORMAL_ANALYSIS_INTERVAL_MS,
                !degraded,
                reason);
    }

    private void notifyPolicyIfChanged() {
        Policy snapshot;
        synchronized (lock) {
            if (closed) {
                return;
            }
            refreshExpiredState(SystemClock.elapsedRealtime());
            snapshot = buildPolicy();
            if (snapshot.equals(lastNotifiedPolicy)) {
                return;
            }
            lastNotifiedPolicy = snapshot;
        }
        mainHandler.post(() -> {
            synchronized (lock) {
                if (closed) {
                    return;
                }
            }
            if (listener != null) {
                listener.onPolicyChanged(snapshot);
            }
        });
    }

    private static long percentile95(long[] samples, int count) {
        long[] sorted = Arrays.copyOf(samples, count);
        Arrays.sort(sorted);
        int index = Math.max(
                0, (int) Math.ceil(sorted.length * LATENCY_PERCENTILE) - 1);
        return sorted[index];
    }

    public void close() {
        synchronized (lock) {
            if (closed) {
                return;
            }
            closed = true;
            lastNotifiedPolicy = null;
        }
        mainHandler.removeCallbacksAndMessages(null);
        if (thermalMonitor != null) {
            thermalMonitor.close();
        }
    }
}
