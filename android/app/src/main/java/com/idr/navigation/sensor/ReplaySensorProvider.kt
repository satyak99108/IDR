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

            // If recorded gnss_course is missing or flat 0.0 across all frames,
            // compute the ground-truth course bearing from consecutive reference / GNSS coordinates
            val needsCourseComputation = samples.all { (it.gnssCourseDeg ?: 0.0) == 0.0 }
            if (needsCourseComputation && samples.isNotEmpty()) {
                var lastBearing = 0.0
                // Find initial bearing from first points with displacement
                for (i in 0 until samples.size) {
                    val current = samples[i]
                    val lat1 = current.refLat ?: current.gnssLat
                    val lon1 = current.refLon ?: current.gnssLon

                    // Look ahead 5-10 frames (~0.5-1s) for clear displacement
                    val aheadIdx = kotlin.math.min(samples.size - 1, i + 10)
                    val ahead = samples[aheadIdx]
                    val lat2 = ahead.refLat ?: ahead.gnssLat
                    val lon2 = ahead.refLon ?: ahead.gnssLon

                    if (lat1 != null && lon1 != null && lat2 != null && lon2 != null) {
                        val dLat = Math.toRadians(lat2 - lat1)
                        val dLon = Math.toRadians(lon2 - lon1)
                        val dNorth = dLat * 6378137.0
                        val dEast = dLon * 6378137.0 * kotlin.math.cos(Math.toRadians(lat1))
                        val dist = kotlin.math.sqrt(dNorth * dNorth + dEast * dEast)
                        if (dist > 1.0) {
                            lastBearing = (Math.toDegrees(kotlin.math.atan2(dEast, dNorth)) + 360.0) % 360.0
                        }
                    }

                    samples[i] = current.copy(gnssCourseDeg = lastBearing)
                }
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
