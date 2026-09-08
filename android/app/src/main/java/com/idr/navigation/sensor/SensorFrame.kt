package com.idr.navigation.sensor

/**
 * Single synchronized time-stamped sensor observation.
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
    val refLon: Double? = null
)
