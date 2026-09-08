package com.idr.navigation.core

import kotlin.math.exp
import kotlin.math.max

/**
 * Manages transitions between GNSS_INS, DEAD_RECKONING, and RECOVERY modes.
 * Implements anti-chatter debouncing and smooth jump mitigation on GNSS recovery.
 */
class NavigationModeManager(
    val dropoutDebounceSamples: Int = 3,
    val recoveryConfirmSamples: Int = 3,
    val recoveryDurationS: Double = 3.0
) {
    var currentMode: NavigationMode = NavigationMode.GNSS_INS
        private set

    private var consecutiveDrops = 0
    private var consecutiveFixes = 0

    // Recovery & jump mitigation state
    private var recoveryStartTimeS: Double = 0.0
    private var dn0: Double = 0.0
    private var de0: Double = 0.0
    private var lastDrNorth: Double = 0.0
    private var lastDrEast: Double = 0.0
    var isJustEnteredRecovery: Boolean = false
        private set

    /**
     * Update state machine with current sensor/GNSS availability.
     */
    fun update(
        timestampS: Double,
        hasGnssSample: Boolean,
        isSimulatedOutage: Boolean,
        currentNorthM: Double,
        currentEastM: Double
    ): NavigationMode {
        isJustEnteredRecovery = false
        val gnssValid = hasGnssSample && !isSimulatedOutage

        if (gnssValid) {
            consecutiveFixes++
            consecutiveDrops = 0
        } else {
            consecutiveDrops++
            consecutiveFixes = 0
        }

        when (currentMode) {
            NavigationMode.GNSS_INS -> {
                if (consecutiveDrops >= dropoutDebounceSamples) {
                    currentMode = NavigationMode.DEAD_RECKONING
                    lastDrNorth = currentNorthM
                    lastDrEast = currentEastM
                }
            }

            NavigationMode.DEAD_RECKONING -> {
                if (consecutiveFixes >= recoveryConfirmSamples) {
                    currentMode = NavigationMode.RECOVERY
                    recoveryStartTimeS = timestampS
                    isJustEnteredRecovery = true

                    // Discrepancy between the dead-reckoned trajectory point and returning GNSS fix.
                    // Blending this offset over recoveryDurationS ensures zero abrupt teleportation.
                    dn0 = lastDrNorth - currentNorthM
                    de0 = lastDrEast - currentEastM
                } else {
                    // Update dead reckoning coordinates while still in blackout
                    lastDrNorth = currentNorthM
                    lastDrEast = currentEastM
                }
            }

            NavigationMode.RECOVERY -> {
                if (consecutiveDrops >= dropoutDebounceSamples) {
                    // Outage resumed during recovery
                    currentMode = NavigationMode.DEAD_RECKONING
                    lastDrNorth = currentNorthM
                    lastDrEast = currentEastM
                } else {
                    val elapsed = timestampS - recoveryStartTimeS
                    if (elapsed >= recoveryDurationS) {
                        currentMode = NavigationMode.GNSS_INS
                        dn0 = 0.0
                        de0 = 0.0
                    }
                }
            }

            NavigationMode.DEGRADED_GNSS -> {
                if (consecutiveDrops >= dropoutDebounceSamples) {
                    currentMode = NavigationMode.DEAD_RECKONING
                } else if (consecutiveFixes >= recoveryConfirmSamples) {
                    currentMode = NavigationMode.GNSS_INS
                }
            }
        }

        return currentMode
    }

    /**
     * Calculate smooth jump offset (dn, de) at current time to eliminate position teleportation.
     */
    fun getJumpOffsets(timestampS: Double): Pair<Double, Double> {
        if (currentMode != NavigationMode.RECOVERY || recoveryDurationS <= 0.0) {
            return Pair(0.0, 0.0)
        }

        val elapsed = max(0.0, timestampS - recoveryStartTimeS)
        // Exponential decay: e^(-3 * t / T), decays to ~5% at t=T
        val blendWeight = exp(-3.0 * (elapsed / recoveryDurationS))
        return Pair(dn0 * blendWeight, de0 * blendWeight)
    }

    fun reset() {
        currentMode = NavigationMode.GNSS_INS
        consecutiveDrops = 0
        consecutiveFixes = 0
        recoveryStartTimeS = 0.0
        dn0 = 0.0
        de0 = 0.0
        isJustEnteredRecovery = false
    }
}
