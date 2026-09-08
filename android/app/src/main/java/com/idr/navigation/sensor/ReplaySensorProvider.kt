package com.idr.navigation.sensor

import android.content.Context
import kotlinx.coroutines.*
import org.json.JSONArray

/**
 * Replays pre-recorded calibrated IO-VNBD trip sequences bundled in assets/demo_trip.json.
 * Enables deterministic indoor testing and evaluation without needing an active moving vehicle.
 *
 * During a simulated GNSS outage, GNSS fields are stripped from each frame but the
 * real recorded IMU data (accel/gyro) continues flowing — dead reckoning is driven by
 * the recorded inertial data, not by synthetic movement.
 */
class ReplaySensorProvider(
    private val context: Context,
    private val assetPath: String = "demo_trip.json",
    private val playbackIntervalMs: Long = 100L // 10 Hz playback
) : SensorDataProvider {

    private var onFrame: ((SensorFrame) -> Unit)? = null
    private var simulatedOutage = false
    private var replayJob: Job? = null
    private val scope = CoroutineScope(Dispatchers.Default)

    private val samples = mutableListOf<SensorFrame>()
    private var currentIndex = 0

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
                        az = obj.optDouble("az", 0.0).toFloat(),
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

                val original = samples[currentIndex]
                val currentTs = System.currentTimeMillis()

                // Apply simulated tunnel blackout if active — strip GNSS but keep real IMU data
                val frameToEmit = if (simulatedOutage) {
                    original.copy(
                        timestampMs = currentTs,
                        gnssLat = null,
                        gnssLon = null,
                        gnssSpeedMs = null,
                        gnssCourseDeg = null
                    )
                } else {
                    original.copy(timestampMs = currentTs)
                }

                withContext(Dispatchers.Main) {
                    onFrame?.invoke(frameToEmit)
                }

                currentIndex++
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
        this.simulatedOutage = isOutage
    }

    override fun isOutageSimulated(): Boolean = simulatedOutage

    fun resetReplay() {
        currentIndex = 0
    }
}
