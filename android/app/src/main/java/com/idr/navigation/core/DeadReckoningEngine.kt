package com.idr.navigation.core

import android.util.Log
import kotlin.math.*

/**
 * Android Navigation Core Engine.
 *
 * Combines GNSS speed, AI forward velocity estimation, gyroscope heading integration,
 * Non-Holonomic Constraints (NHC), GNSS blending, and smooth jump mitigation.
 *
 * ### Corrected Speed Data Flow
 * ```
 * GNSS speed (primary when available, m/s)
 *      ↓ fallback
 * AI TFLite velocity prediction (primary during DR, m/s)
 *      ↓ fallback
 * Exponential decay toward zero (prevents accumulation)
 *      ↓
 * Stationary ZUPT override (consecutive samples, accelMag < threshold)
 *      ↓
 * EMA smoothing (α=0.3)
 *      ↓
 * speedMs × 3.6 → speedKmh (single canonical conversion)
 * ```
 */
class DeadReckoningEngine(
    val modeManager: NavigationModeManager = NavigationModeManager()
) {
    companion object {
        private const val TAG = "IDR_DREngine"

        /** Enable/disable verbose velocity debug logging. Set false for production. */
        var DEBUG_VELOCITY = true

        // ---- Stationary detection thresholds (designed for gravity-free linear accel) ----
        /** Linear acceleration magnitude below this → likely stationary (m/s²). */
        private const val STATIONARY_ACCEL_THRESHOLD = 0.5
        /** Yaw rate magnitude below this → likely stationary (rad/s). */
        private const val STATIONARY_GYRO_THRESHOLD = 0.08
        /** Speed must be below this for ZUPT to trigger (m/s ≈ 5.4 km/h). */
        private const val STATIONARY_SPEED_THRESHOLD = 1.5
        /** Consecutive stationary samples required before ZUPT fires. */
        private const val STATIONARY_CONFIRM_SAMPLES = 3

        // ---- Velocity filtering ----
        /** EMA smoothing factor (0 = no change, 1 = no smoothing). */
        private const val VELOCITY_EMA_ALPHA = 0.3
        /** Decay factor per step when neither GNSS nor AI provides velocity. */
        private const val VELOCITY_DECAY_FACTOR = 0.85
        /** Maximum physically reasonable vehicle speed (m/s) ≈ 200 km/h. */
        private const val MAX_SPEED_MS = 55.0
    }

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

    // Stationary detector: consecutive-sample counter
    private var stationaryCount: Int = 0

    /**
     * Process one incoming sensor/navigation sample.
     *
     * @param timestampMs sample timestamp in epoch millis
     * @param axVeh linear acceleration forward (vehicle frame, m/s², **gravity removed**)
     * @param ayVeh linear acceleration lateral (vehicle frame, m/s², **gravity removed**)
     * @param azVeh linear acceleration vertical (vehicle frame, m/s², **gravity removed**)
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

        // ── Step 1: Calculate dt from actual timestamps ──
        // No hardcoded rate assumption. First sample gets dt=0 (no integration).
        val dt = if (lastTimestampMs > 0 && timestampMs > lastTimestampMs) {
            val rawDt = (timestampMs - lastTimestampMs) / 1000.0
            // Clamp to prevent spikes from lifecycle pauses (max 200ms = 5 Hz floor)
            min(0.2, rawDt)
        } else {
            0.0 // First sample: no integration, no speed kick
        }
        lastTimestampMs = timestampMs

        // ── Step 2: Initialize origin if not yet established ──
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

        // ── Step 3: Integrate heading from gyroscope (yaw rate gz) ──
        if (dt > 0.0) {
            val gzCompensated = gzVeh.toDouble() - gyroBiasZ
            val dHeadingDeg = Math.toDegrees(gzCompensated * dt)
            headingDeg = CoordinatesNED.wrapAngle360(headingDeg + dHeadingDeg)
        }

        // ── Step 4: Determine forward velocity — corrected priority chain ──
        // Priority: GNSS speed → AI prediction → decay toward zero
        val gnssSpeedValid = hasGnss && !isSimulatedOutage &&
                gnssSpeedMs != null && !gnssSpeedMs.isNaN() && gnssSpeedMs >= 0.0

        val rawSpeedMs: Double = when {
            // Priority 1: GNSS speed when available — most accurate source
            gnssSpeedValid -> gnssSpeedMs!!

            // Priority 2: AI model prediction during DR or when GNSS speed unavailable
            aiForwardSpeedMs > 0.05f -> aiForwardSpeedMs.toDouble()

            // Priority 3: Decay toward zero — prevents unbounded accumulation
            // (replaces the broken `max(0, speed + ax*dt)` fallback)
            else -> speedMs * VELOCITY_DECAY_FACTOR
        }

        // ── Step 5: Stationary detection (ZUPT) ──
        // Uses gravity-free linear acceleration magnitude + gyro magnitude.
        // Requires consecutive stationary samples to avoid false triggers from single noisy readings.
        val accelMag = sqrt(
            (axVeh.toDouble()).pow(2) + (ayVeh.toDouble()).pow(2) + (azVeh.toDouble()).pow(2)
        )
        val isLikelyStationary =
            accelMag < STATIONARY_ACCEL_THRESHOLD && abs(gzVeh) < STATIONARY_GYRO_THRESHOLD

        if (isLikelyStationary) {
            stationaryCount++
        } else {
            stationaryCount = 0
        }

        val isStationary =
            stationaryCount >= STATIONARY_CONFIRM_SAMPLES && rawSpeedMs < STATIONARY_SPEED_THRESHOLD

        if (isStationary) {
            speedMs = 0.0
            // Estimate gyro bias during stationary periods for heading drift compensation
            gyroBiasZ = gyroBiasZ * 0.95 + gzVeh.toDouble() * 0.05
        } else if (dt > 0.0) {
            // EMA smoothing — reduces jitter while maintaining responsiveness
            speedMs = (1.0 - VELOCITY_EMA_ALPHA) * speedMs + VELOCITY_EMA_ALPHA * rawSpeedMs
        }

        // Clamp to physically reasonable range [0, MAX_SPEED_MS]
        speedMs = speedMs.coerceIn(0.0, MAX_SPEED_MS)

        // ── Step 6: Dead Reckoning position propagation with NHC ──
        // v_lateral = 0 (NHC), v_vertical = 0 (NHC)
        if (dt > 0.0) {
            val headingRad = Math.toRadians(headingDeg)
            val vNorth = speedMs * cos(headingRad)
            val vEast = speedMs * sin(headingRad)

            pNorthM += vNorth * dt
            pEastM += vEast * dt
            totalDistanceM += speedMs * dt
        }

        // ── Step 7: Update Navigation Mode Manager ──
        val mode = modeManager.update(
            timestampS = timestampS,
            hasGnssSample = hasGnss,
            isSimulatedOutage = isSimulatedOutage,
            currentNorthM = pNorthM,
            currentEastM = pEastM
        )

        // ── Step 8: GNSS Blending (when fix is valid and mode allows) ──
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

        // ── Step 9: Smooth Jump Mitigation during Recovery ──
        val (dnSmooth, deSmooth) = modeManager.getJumpOffsets(timestampS)
        val outNorth = pNorthM + dnSmooth
        val outEast = pEastM + deSmooth

        // ── Step 10: Convert NED → Geodetic ──
        val geodetic = CoordinatesNED.nedToGeodetic(
            northM = outNorth,
            eastM = outEast,
            originLatDeg = origLat,
            originLonDeg = origLon
        )

        // ── Step 11: Outage telemetry ──
        val isOutage = (mode == NavigationMode.DEAD_RECKONING)
        if (isOutage) {
            if (outageStartMs == 0L) outageStartMs = timestampMs
            if (dt > 0.0) outageDistanceM += speedMs * dt
        } else {
            outageStartMs = 0L
            outageDistanceM = 0.0
        }
        val outageDurationS = if (isOutage && outageStartMs > 0) {
            (timestampMs - outageStartMs) / 1000.0
        } else 0.0

        // ── Step 12: Confidence score ──
        val confidencePct = when (mode) {
            NavigationMode.GNSS_INS -> 95
            NavigationMode.RECOVERY -> 85
            NavigationMode.DEGRADED_GNSS -> 70
            NavigationMode.DEAD_RECKONING -> {
                val decay = min(50.0, outageDurationS * 0.8)
                max(35, (85.0 - decay).toInt())
            }
        }

        // ── Single canonical conversion: m/s → km/h ──
        val speedKmh = speedMs * 3.6

        // ── Debug logging ──
        if (DEBUG_VELOCITY) {
            Log.d(TAG, String.format(
                "dt=%.3f mode=%s stat=%b accelMag=%.3f " +
                        "gnssSpd=%.1f aiSpd=%.2f rawSpd=%.2f smoothSpd=%.2f kmh=%.1f",
                dt, mode.name, isStationary, accelMag,
                gnssSpeedMs ?: -1.0, aiForwardSpeedMs.toDouble(),
                rawSpeedMs, speedMs, speedKmh
            ))
        }

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
        stationaryCount = 0
        gyroBiasZ = 0.0
        modeManager.reset()
    }
}
