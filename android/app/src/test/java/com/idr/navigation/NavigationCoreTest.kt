package com.idr.navigation

import com.idr.navigation.ai.ScalerNormalizer
import com.idr.navigation.core.CoordinatesNED
import com.idr.navigation.core.DeadReckoningEngine
import com.idr.navigation.core.NavigationMode
import com.idr.navigation.core.NavigationModeManager
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs

class NavigationCoreTest {

    @Test
    fun testCoordinatesRoundTrip() {
        val origLat = 37.7749
        val origLon = -122.4194

        val targetLat = 37.7760
        val targetLon = -122.4180

        val ned = CoordinatesNED.geodeticToNed(
            latDeg = targetLat,
            lonDeg = targetLon,
            originLatDeg = origLat,
            originLonDeg = origLon
        )

        val backGeo = CoordinatesNED.nedToGeodetic(
            northM = ned.northM,
            eastM = ned.eastM,
            originLatDeg = origLat,
            originLonDeg = origLon
        )

        assertEquals(targetLat, backGeo.latDeg, 1e-6)
        assertEquals(targetLon, backGeo.lonDeg, 1e-6)
    }

    @Test
    fun testHaversineDistanceKnownValues() {
        val lat1 = 0.0
        val lon1 = 0.0
        val lat2 = 1.0
        val lon2 = 0.0

        val distM = CoordinatesNED.haversineDistance(lat1, lon1, lat2, lon2)
        // 1 degree along meridian is ~111.3 km
        assertTrue("Expected ~111.3 km, got $distM", distM in 111000.0..111500.0)
    }

    @Test
    fun testScalerNormalizerMatchesFormula() {
        val normalizer = ScalerNormalizer()
        val raw = floatArrayOf(
            -0.2702922f, // should become 0.0
            -0.0296167f, // should become 0.0
            0.4300335f,  // mean + 1 std
            0.0011831f,
            0.0011792f,
            0.0015184f
        )

        val norm = normalizer.normalizeSample(raw)
        assertEquals(0.0f, norm[0], 1e-4f)
        assertEquals(0.0f, norm[1], 1e-4f)
        assertEquals(1.0f, norm[2], 1e-3f)
    }

    @Test
    fun testNavigationModeDebounce() {
        val manager = NavigationModeManager(
            dropoutDebounceSamples = 3,
            recoveryConfirmSamples = 3,
            recoveryDurationS = 3.0
        )

        assertEquals(NavigationMode.GNSS_INS, manager.currentMode)

        // 1 drop: stays GNSS_INS
        manager.update(1.0, hasGnssSample = false, isSimulatedOutage = false, 0.0, 0.0)
        assertEquals(NavigationMode.GNSS_INS, manager.currentMode)

        // 2 drops: stays GNSS_INS
        manager.update(1.1, hasGnssSample = false, isSimulatedOutage = false, 0.0, 0.0)
        assertEquals(NavigationMode.GNSS_INS, manager.currentMode)

        // 3 drops: transitions to DEAD_RECKONING
        manager.update(1.2, hasGnssSample = false, isSimulatedOutage = false, 0.0, 0.0)
        assertEquals(NavigationMode.DEAD_RECKONING, manager.currentMode)
    }

    @Test
    fun testNavigationModeSmoothJumpMitigation() {
        val manager = NavigationModeManager(
            dropoutDebounceSamples = 2,
            recoveryConfirmSamples = 2,
            recoveryDurationS = 2.0
        )

        // Enter dead reckoning
        manager.update(1.0, hasGnssSample = false, isSimulatedOutage = false, 100.0, 50.0)
        manager.update(1.1, hasGnssSample = false, isSimulatedOutage = false, 100.0, 50.0)
        assertEquals(NavigationMode.DEAD_RECKONING, manager.currentMode)

        // GNSS fix returns (say at 90.0, 45.0, so discrepancy dn0 = 10, de0 = 5)
        manager.update(2.0, hasGnssSample = true, isSimulatedOutage = false, 90.0, 45.0)
        manager.update(2.1, hasGnssSample = true, isSimulatedOutage = false, 90.0, 45.0)
        assertEquals(NavigationMode.RECOVERY, manager.currentMode)

        // At initiation of recovery (t = 2.1), offset is near 10m and 5m
        val (dnStart, deStart) = manager.getJumpOffsets(2.1)
        assertEquals(10.0, dnStart, 0.1)
        assertEquals(5.0, deStart, 0.1)

        // As recovery progresses (t = 3.1, 1s in), offset decays smoothly
        val (dnMid, deMid) = manager.getJumpOffsets(3.1)
        assertTrue(dnMid < dnStart && dnMid > 0.0)

        // After recovery duration (t >= 4.1), transitions to GNSS_INS
        manager.update(4.2, hasGnssSample = true, isSimulatedOutage = false, 90.0, 45.0)
        assertEquals(NavigationMode.GNSS_INS, manager.currentMode)
        val (dnEnd, deEnd) = manager.getJumpOffsets(4.2)
        assertEquals(0.0, dnEnd, 1e-6)
        assertEquals(0.0, deEnd, 1e-6)
    }

    @Test
    fun testDeadReckoningEngineStraightNorth() {
        val engine = DeadReckoningEngine()

        // 1st sample establishes origin and heading 0 (North)
        val o1 = engine.processSample(
            timestampMs = 1000L,
            axVeh = 0f, ayVeh = 0f, azVeh = 0f, gzVeh = 0f,
            aiForwardSpeedMs = 10.0f, // 10 m/s = 36 km/h
            gnssLat = 51.5074, gnssLon = -0.1278,
            gnssSpeedMs = 10.0, gnssCourseDeg = 0.0,
            isSimulatedOutage = false
        )
        assertEquals(51.5074, o1.latDeg, 1e-4)

        // 2nd sample: forward north, no outage
        val o2 = engine.processSample(
            timestampMs = 1100L, // dt = 0.1s
            axVeh = 0f, ayVeh = 0f, azVeh = 0f, gzVeh = 0f,
            aiForwardSpeedMs = 10.0f,
            gnssLat = 51.50741, gnssLon = -0.1278,
            gnssSpeedMs = 10.0, gnssCourseDeg = 0.0,
            isSimulatedOutage = false
        )
        assertTrue("Vehicle should move North", o2.latDeg >= o1.latDeg)
        assertEquals(36.0, o2.speedKmh, 1.0)
    }

    @Test
    fun testDeadReckoningContinuesDuringOutage() {
        val engine = DeadReckoningEngine()

        // Initialize with GNSS
        engine.processSample(
            timestampMs = 1000L,
            axVeh = 0f, ayVeh = 0f, azVeh = 0f, gzVeh = 0f,
            aiForwardSpeedMs = 15.0f,
            gnssLat = 51.5074, gnssLon = -0.1278,
            gnssSpeedMs = 15.0, gnssCourseDeg = 90.0, // Heading East
            isSimulatedOutage = false
        )

        // Simulate 5 consecutive outage steps (tunnel blackout)
        var lastLon = -0.1278
        for (step in 1..5) {
            val out = engine.processSample(
                timestampMs = 1000L + step * 100L,
                axVeh = 0f, ayVeh = 0f, azVeh = 0f, gzVeh = 0f,
                aiForwardSpeedMs = 15.0f,
                gnssLat = null, gnssLon = null,
                gnssSpeedMs = null, gnssCourseDeg = null,
                isSimulatedOutage = true
            )
            assertTrue("Vehicle position should continue advancing East during outage", out.lonDeg > lastLon)
            lastLon = out.lonDeg
        }
    }
}
