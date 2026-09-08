package com.idr.navigation.sensor

import android.content.Context
import kotlinx.coroutines.*
import org.json.JSONArray
import kotlin.math.min

/**
 * Replays pre-recorded calibrated IO-VNBD trip sequences bundled in assets/demo_trip.json.
 * Enables deterministic indoor testing and evaluation without needing an active moving vehicle.
 *
 * ### Tunnel Auto-Drive Mode
 * When a simulated GNSS outage is active (tunnel entered), the provider switches to
 * **TunnelAutoDrive** — it emits synthetic [SensorFrame]s with a gradually ramping
 * forward speed ([SensorFrame.autoDriveSpeedMs]) that starts from the last known
 * replay speed and accelerates up to [TUNNEL_MAX_SPEED_MS] (≈ 60 km/h) over
 * [TUNNEL_RAMP_DURATION_S] seconds, then cruises at that speed until GNSS is restored.
 */
class ReplaySensorProvider(
    private val context: Context,
    private val assetPath: String = "demo_trip.json",
    private val playbackIntervalMs: Long = 100L // 10 Hz playback
) : SensorDataProvider {

    companion object {
        /** Maximum cruise speed during tunnel auto-drive (≈ 60 km/h). */
        private const val TUNNEL_MAX_SPEED_MS = 16.67f  // 60 km/h in m/s
        /** Time to ramp from starting speed to cruise speed (seconds). */
        private const val TUNNEL_RAMP_DURATION_S = 8.0f
        /** Acceleration applied each 100 ms tick (m/s per tick). */
        private val RAMP_STEP_PER_TICK_MS = TUNNEL_MAX_SPEED_MS / (TUNNEL_RAMP_DURATION_S * 10f)
        /** Synthetic forward acceleration magnitude felt in vehicle frame (m/s²) during ramp. */
        private const val SYNTHETIC_AX_RAMP = 0.6f
        /** Synthetic forward acceleration during cruise (near-zero, slight road vibration). */
        private const val SYNTHETIC_AX_CRUISE = 0.05f
        /** Tiny realistic yaw rate to create a gentle curve during tunnel (rad/s). */
        private const val SYNTHETIC_GZ_TUNNEL = 0.004f
    }

    private var onFrame: ((SensorFrame) -> Unit)? = null
    private var simulatedOutage = false
    private var replayJob: Job? = null
    private val scope = CoroutineScope(Dispatchers.Default)

    private val samples = mutableListOf<SensorFrame>()
    private var currentIndex = 0

    // --- Tunnel Auto-Drive State ---
    /** Current synthetic forward speed in m/s (maintained across frames). */
    @Volatile private var tunnelSpeedMs: Float = 0f

    init {
        loadSamples()
    }

    private fun loadSamples() {
        try {
            val jsonStr = context.assets.open(assetPath).bufferedReader().use { it.readText() }
            val array = JSONArray(jsonStr)
            for (i in 0 until array.length()) {
                val obj = array.getJSONObject(i)
                samples.add(
                    SensorFrame(
                        timestampMs = obj.optLong("timestamp_ms", i * 100L),
                        ax = obj.optDouble("ax", 0.0).toFloat(),
                        ay = obj.optDouble("ay", 0.0).toFloat(),
                        az = obj.optDouble("az", 9.81).toFloat(),
                        gx = obj.optDouble("gx", 0.0).toFloat(),
                        gy = obj.optDouble("gy", 0.0).toFloat(),
                        gz = obj.optDouble("gz", 0.0).toFloat(),
                        gnssLat = if (obj.has("gnss_lat")) obj.getDouble("gnss_lat") else null,
                        gnssLon = if (obj.has("gnss_lon")) obj.getDouble("gnss_lon") else null,
                        gnssSpeedMs = if (obj.has("gnss_speed")) obj.getDouble("gnss_speed") else null,
                        gnssCourseDeg = if (obj.has("gnss_course")) obj.getDouble("gnss_course") else null,
                        refLat = if (obj.has("ref_lat")) obj.getDouble("ref_lat") else null,
                        refLon = if (obj.has("ref_lon")) obj.getDouble("ref_lon") else null
                    )
                )
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun start(onFrameReceived: (SensorFrame) -> Unit) {
        this.onFrame = onFrameReceived
        if (samples.isEmpty()) return

        replayJob?.cancel()
        replayJob = scope.launch {
            while (isActive) {
                if (currentIndex >= samples.size) {
                    currentIndex = 0 // loop continuously
                }

                val currentTs = System.currentTimeMillis()
                val frameToEmit: SensorFrame

                if (simulatedOutage) {
                    // ---- Tunnel Auto-Drive Mode ----
                    // Ramp speed up to cruise limit each tick
                    tunnelSpeedMs = min(TUNNEL_MAX_SPEED_MS, tunnelSpeedMs + RAMP_STEP_PER_TICK_MS)
                    val isRamping = tunnelSpeedMs < TUNNEL_MAX_SPEED_MS * 0.95f
                    val syntheticAx = if (isRamping) SYNTHETIC_AX_RAMP else SYNTHETIC_AX_CRUISE

                    // Derive IMU values from last replay frame for realism
                    val base = samples[currentIndex]
                    frameToEmit = SensorFrame(
                        timestampMs = currentTs,
                        ax = syntheticAx,
                        ay = base.ay * 0.1f,  // minimal lateral
                        az = base.az,          // gravity component unchanged
                        gx = 0f,
                        gy = 0f,
                        gz = SYNTHETIC_GZ_TUNNEL,  // gentle curve
                        gnssLat = null,
                        gnssLon = null,
                        gnssSpeedMs = null,
                        gnssCourseDeg = null,
                        autoDriveSpeedMs = tunnelSpeedMs
                    )
                    currentIndex++
                } else {
                    // ---- Normal GNSS Replay Mode ----
                    // Reset tunnel speed to last known replay speed for smooth re-entry
                    val original = samples[currentIndex]
                    tunnelSpeedMs = original.gnssSpeedMs?.toFloat() ?: tunnelSpeedMs

                    frameToEmit = original.copy(timestampMs = currentTs)
                    currentIndex++
                }

                withContext(Dispatchers.Main) {
                    onFrame?.invoke(frameToEmit)
                }

                delay(playbackIntervalMs)
            }
        }
    }

    override fun stop() {
        replayJob?.cancel()
        replayJob = null
        onFrame = null
    }

    override fun setSimulatedOutage(isOutage: Boolean) {
        if (isOutage && !simulatedOutage) {
            // Entering tunnel: seed speed from last replay sample speed or 0
            val lastSpeed = if (currentIndex > 0 && currentIndex <= samples.size) {
                samples.getOrNull(currentIndex - 1)?.gnssSpeedMs?.toFloat() ?: 0f
            } else 0f
            tunnelSpeedMs = lastSpeed.coerceAtLeast(0f)
        }
        this.simulatedOutage = isOutage
    }

    override fun isOutageSimulated(): Boolean = simulatedOutage

    fun resetReplay() {
        currentIndex = 0
        tunnelSpeedMs = 0f
    }
}
