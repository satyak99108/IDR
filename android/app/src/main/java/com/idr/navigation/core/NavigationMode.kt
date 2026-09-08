package com.idr.navigation.core

/**
 * Operating modes of the IDR navigation state machine.
 */
enum class NavigationMode(val displayName: String, val badgeColorHex: String) {
    GNSS_INS("GNSS + INS", "#10B981"),        // Emerald green
    DEAD_RECKONING("DEAD RECKONING", "#F59E0B"), // Amber / warning
    RECOVERY("GNSS RECOVERY", "#06B6D4"),     // Cyan / blending
    DEGRADED_GNSS("DEGRADED GNSS", "#EAB308") // Yellow
}

/**
 * Real-time navigation estimate emitted at 10-20 Hz.
 */
data class NavigationOutput(
    val timestampMs: Long,
    val latDeg: Double,
    val lonDeg: Double,
    val speedKmh: Double,
    val headingDeg: Double,
    val mode: NavigationMode,
    val confidencePct: Int,
    val isOutageActive: Boolean,
    val outageDurationS: Double = 0.0,
    val distanceTravelledM: Double = 0.0,
    val estimatedDriftM: Double = 0.0,
    val estimatedDriftPct: Double = 0.0,
    val hasValidFix: Boolean = true
)
