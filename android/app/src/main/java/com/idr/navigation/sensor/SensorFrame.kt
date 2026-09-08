package com.idr.navigation.sensor

/**
 * Single synchronized time-stamped sensor observation.
 *
 * @param autoDriveSpeedMs optional override forward speed (m/s) injected during
 *   tunnel auto-drive simulation. When non-null, the navigation core uses this
 *   value directly as the AI forward velocity estimate instead of the TFLite model.
 */
data class SensorFrame(
    val timestampMs: Long,
    val ax: Float,
    val ay: Float,
    val az: Float,
    val gx: Float,
    val gy: Float,
    val gz: Float,
    val gnssLat: Double? = null,
    val gnssLon: Double? = null,
    val gnssSpeedMs: Double? = null,
    val gnssCourseDeg: Double? = null,
    val refLat: Double? = null,
    val refLon: Double? = null,
    val autoDriveSpeedMs: Float? = null
)
