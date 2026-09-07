package com.idr.navigation.core

import kotlin.math.*

/**
 * Android Navigation Core Engine.
 * Combines AI forward velocity estimation, gyroscope heading integration,
 * Non-Holonomic Constraints (NHC), GNSS blending, and smooth jump mitigation.
 */
class DeadReckoningEngine(
    val modeManager: NavigationModeManager = NavigationModeManager()
) {
    // Reference Origin for Flat-Earth NED
    var originLatDeg: Double? = null
        private set
    var originLonDeg: Double? = null
        private set

    // Navigation state
    private var pNorthM: Double = 0.0
    private var pEastM: Double = 0.0
    private var pDownM: Double = 0.0

    private var headingDeg: Double = 0.0
    private var speedMs: Double = 0.0

    private var lastTimestampMs: Long = 0
    private var totalDistanceM: Double = 0.0
    private var outageDistanceM: Double = 0.0
    private var outageStartMs: Long = 0L

    // Gyro bias compensation (estimated during stationary periods)
    private var gyroBiasZ: Double = 0.0

    /**
     * Process one incoming sensor/navigation sample.
     *
     * @param timestampMs sample timestamp in epoch millis
     * @param axVeh acceleration forward (vehicle frame, m/s^2)
     * @param ayVeh acceleration lateral (vehicle frame, m/s^2)
     * @param azVeh acceleration vertical (vehicle frame, m/s^2)
     * @param gzVeh yaw rate (vehicle frame, rad/s)
     * @param aiForwardSpeedMs velocity estimated by TFLite model (m/s)
     * @param gnssLat GNSS latitude if available, else null
     * @param gnssLon GNSS longitude if available, else null
     * @param gnssSpeedMs GNSS speed if available, else null
     * @param gnssCourseDeg GNSS course over ground if available, else null
     * @param isSimulatedOutage true if GNSS is intentionally cut off (tunnel simulation)
     */
    fun processSample(
        timestampMs: Long,
        axVeh: Float,
        ayVeh: Float,
        azVeh: Float,
        gzVeh: Float,
        aiForwardSpeedMs: Float,
        gnssLat: Double?,
        gnssLon: Double?,
        gnssSpeedMs: Double?,
        gnssCourseDeg: Double?,
        isSimulatedOutage: Boolean
    ): NavigationOutput {
        val timestampS = timestampMs / 1000.0

        // Calculate dt
        val dt = if (lastTimestampMs > 0 && timestampMs > lastTimestampMs) {
            min(0.2, (timestampMs - lastTimestampMs) / 1000.0)
        } else {
            0.1 // default 10 Hz
        }
        lastTimestampMs = timestampMs

        // 1. Initialize origin if not yet established
        val hasGnss = gnssLat != null && gnssLon != null && !gnssLat.isNaN() && !gnssLon.isNaN()
        if (originLatDeg == null && hasGnss) {
            originLatDeg = gnssLat
            originLonDeg = gnssLon
            pNorthM = 0.0
            pEastM = 0.0
            if (gnssCourseDeg != null && !gnssCourseDeg.isNaN()) {
                headingDeg = gnssCourseDeg
            }
        }

        val origLat = originLatDeg ?: 0.0
        val origLon = originLonDeg ?: 0.0

        // 2. Integrate Heading from Gyroscope (yaw rate gz)
        val gzCompensated = gzVeh.toDouble() - gyroBiasZ
        val dHeadingDeg = Math.toDegrees(gzCompensated * dt)
        headingDeg = CoordinatesNED.wrapAngle360(headingDeg + dHeadingDeg)

        // 3. Determine Forward Velocity (AI model + NHC)
        speedMs = if (aiForwardSpeedMs > 0.1f) {
            aiForwardSpeedMs.toDouble()
        } else {
            // Kinematic fallback
            max(0.0, speedMs + axVeh * dt)
        }

        // Apply stationary zero-velocity update (ZUPT)
        val isStationary = abs(axVeh) < 0.15f && abs(ayVeh) < 0.15f && abs(gzVeh) < 0.05f && speedMs < 0.5
        if (isStationary) {
            speedMs = 0.0
        }

        // 4. Dead Reckoning Position Propagation with NHC
        // In vehicle frame: v_forward = speedMs, v_lateral = 0 (NHC), v_vertical = 0
        val headingRad = Math.toRadians(headingDeg)
        val vNorth = speedMs * cos(headingRad)
        val vEast = speedMs * sin(headingRad)

        val deltaM = speedMs * dt
        pNorthM += vNorth * dt
        pEastM += vEast * dt
        totalDistanceM += deltaM

        // 5. Update Navigation Mode Manager
        val mode = modeManager.update(
            timestampS = timestampS,
            hasGnssSample = hasGnss,
            isSimulatedOutage = isSimulatedOutage,
            currentNorthM = pNorthM,
            currentEastM = pEastM
        )

        // 6. GNSS Blending (when fix is valid and mode allows)
        if (hasGnss && !isSimulatedOutage && origLat != 0.0) {
            val gnssNed = CoordinatesNED.geodeticToNed(
                latDeg = gnssLat!!,
                lonDeg = gnssLon!!,
                originLatDeg = origLat,
                originLonDeg = origLon
            )

            val alpha = when (mode) {
                NavigationMode.GNSS_INS -> 0.35 // Active fusion blending
                NavigationMode.RECOVERY -> 0.15 // Gentle blending during recovery
                else -> 0.0
            }

            pNorthM = (1.0 - alpha) * pNorthM + alpha * gnssNed.northM
            pEastM = (1.0 - alpha) * pEastM + alpha * gnssNed.eastM

            // Align heading with GNSS course when moving fast enough
            if (gnssCourseDeg != null && !gnssCourseDeg.isNaN() && speedMs > 3.0) {
                val beta = 0.08
                val dCourse = CoordinatesNED.wrapAngle180(gnssCourseDeg - headingDeg)
                headingDeg = CoordinatesNED.wrapAngle360(headingDeg + beta * dCourse)
            }
        }

        // 7. Apply Smooth Jump Mitigation Offset during Recovery
        val (dnSmooth, deSmooth) = modeManager.getJumpOffsets(timestampS)
        val outNorth = pNorthM + dnSmooth
        val outEast = pEastM + deSmooth

        // 8. Convert output NED back to Geodetic (Lat, Lon)
        val geodetic = CoordinatesNED.nedToGeodetic(
            northM = outNorth,
            eastM = outEast,
            originLatDeg = origLat,
            originLonDeg = origLon
        )

        // 9. Outage telemetry tracking
        val isOutage = (mode == NavigationMode.DEAD_RECKONING)
        if (isOutage) {
            if (outageStartMs == 0L) outageStartMs = timestampMs
            outageDistanceM += deltaM
        } else {
            outageStartMs = 0L
            outageDistanceM = 0.0
        }
        val outageDurationS = if (isOutage && outageStartMs > 0) {
            (timestampMs - outageStartMs) / 1000.0
        } else 0.0

        // 10. Estimate confidence score (10-99%)
        val confidencePct = when (mode) {
            NavigationMode.GNSS_INS -> 95
            NavigationMode.RECOVERY -> 85
            NavigationMode.DEGRADED_GNSS -> 70
            NavigationMode.DEAD_RECKONING -> {
                // Confidence gently decays with outage duration
                val decay = min(50.0, outageDurationS * 0.8)
                max(35, (85.0 - decay).toInt())
            }
        }

        val speedKmh = speedMs * 3.6

        return NavigationOutput(
            timestampMs = timestampMs,
            latDeg = geodetic.latDeg,
            lonDeg = geodetic.lonDeg,
            speedKmh = round(speedKmh * 10.0) / 10.0,
            headingDeg = round(headingDeg * 10.0) / 10.0,
            mode = mode,
            confidencePct = confidencePct,
            isOutageActive = isOutage,
            outageDurationS = round(outageDurationS * 10.0) / 10.0,
            distanceTravelledM = totalDistanceM
        )
    }

    fun reset() {
        pNorthM = 0.0
        pEastM = 0.0
        pDownM = 0.0
        headingDeg = 0.0
        speedMs = 0.0
        lastTimestampMs = 0
        totalDistanceM = 0.0
        outageDistanceM = 0.0
        outageStartMs = 0L
        originLatDeg = null
        originLonDeg = null
        modeManager.reset()
    }
}
