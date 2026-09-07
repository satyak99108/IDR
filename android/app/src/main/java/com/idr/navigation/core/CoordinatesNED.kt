package com.idr.navigation.core

import kotlin.math.*

/**
 * Coordinate transformations between WGS-84 Geodetic (Lat, Lon, Alt) and
 * Local Flat-Earth North-East-Down (NED) frame in metres.
 */
object CoordinatesNED {
    private const val EARTH_RADIUS_METRES = 6378137.0

    data class NedPoint(val northM: Double, val eastM: Double, val downM: Double)
    data class GeodeticPoint(val latDeg: Double, val lonDeg: Double, val altM: Double)

    /**
     * Convert Geodetic (latitude, longitude, altitude) to local NED coordinates.
     */
    fun geodeticToNed(
        latDeg: Double,
        lonDeg: Double,
        altM: Double = 0.0,
        originLatDeg: Double,
        originLonDeg: Double,
        originAltM: Double = 0.0
    ): NedPoint {
        val latRad = Math.toRadians(latDeg)
        val lonRad = Math.toRadians(lonDeg)
        val origLatRad = Math.toRadians(originLatDeg)
        val origLonRad = Math.toRadians(originLonDeg)

        val dLat = latRad - origLatRad
        val dLon = lonRad - origLonRad

        val northM = dLat * EARTH_RADIUS_METRES
        val eastM = dLon * EARTH_RADIUS_METRES * cos(origLatRad)
        val downM = originAltM - altM

        return NedPoint(northM, eastM, downM)
    }

    /**
     * Convert local NED coordinates back to Geodetic (latitude, longitude, altitude).
     */
    fun nedToGeodetic(
        northM: Double,
        eastM: Double,
        downM: Double = 0.0,
        originLatDeg: Double,
        originLonDeg: Double,
        originAltM: Double = 0.0
    ): GeodeticPoint {
        val origLatRad = Math.toRadians(originLatDeg)
        val origLonRad = Math.toRadians(originLonDeg)

        val dLat = northM / EARTH_RADIUS_METRES
        val dLon = eastM / (EARTH_RADIUS_METRES * cos(origLatRad))

        val latRad = origLatRad + dLat
        val lonRad = origLonRad + dLon

        return GeodeticPoint(
            latDeg = Math.toDegrees(latRad),
            lonDeg = Math.toDegrees(lonRad),
            altM = originAltM - downM
        )
    }

    /**
     * Great-circle Haversine distance in metres between two points.
     */
    fun haversineDistance(
        lat1Deg: Double,
        lon1Deg: Double,
        lat2Deg: Double,
        lon2Deg: Double
    ): Double {
        val phi1 = Math.toRadians(lat1Deg)
        val phi2 = Math.toRadians(lat2Deg)
        val dPhi = Math.toRadians(lat2Deg - lat1Deg)
        val dLambda = Math.toRadians(lon2Deg - lon1Deg)

        val a = sin(dPhi / 2.0).pow(2.0) +
                cos(phi1) * cos(phi2) * sin(dLambda / 2.0).pow(2.0)
        val c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
        return EARTH_RADIUS_METRES * c
    }

    /**
     * Wrap heading angle to [-180, +180] degrees.
     */
    fun wrapAngle180(angleDeg: Double): Double {
        var a = angleDeg % 360.0
        if (a > 180.0) a -= 360.0
        if (a <= -180.0) a += 360.0
        return a
    }

    /**
     * Normalize heading angle to [0, 360) degrees.
     */
    fun wrapAngle360(angleDeg: Double): Double {
        var a = angleDeg % 360.0
        if (a < 0.0) a += 360.0
        return a
    }
}
