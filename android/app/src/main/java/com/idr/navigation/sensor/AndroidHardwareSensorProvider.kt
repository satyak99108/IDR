package com.idr.navigation.sensor

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper

/**
 * Live hardware sensor provider reading Android SensorManager (Accel/Gyro)
 * and LocationManager (GNSS) at 10-50 Hz.
 */
class AndroidHardwareSensorProvider(
    private val context: Context
) : SensorDataProvider, SensorEventListener, LocationListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    private var onFrame: ((SensorFrame) -> Unit)? = null
    private var simulatedOutage = false

    // Sensor buffers
    private var lastAx = 0f
    private var lastAy = 0f
    private var lastAz = 0f

    private var lastGx = 0f
    private var lastGy = 0f
    private var lastGz = 0f

    // GNSS cache
    private var latestLocation: Location? = null

    // Periodic emitter loop (20 Hz = 50ms)
    private val handler = Handler(Looper.getMainLooper())
    private val emitRunnable = object : Runnable {
        override fun run() {
            val nowMs = System.currentTimeMillis()
            val loc = if (simulatedOutage) null else latestLocation

            val frame = SensorFrame(
                timestampMs = nowMs,
                ax = lastAx,
                ay = lastAy,
                az = lastAz,
                gx = lastGx,
                gy = lastGy,
                gz = lastGz,
                gnssLat = loc?.latitude,
                gnssLon = loc?.longitude,
                gnssSpeedMs = loc?.speed?.toDouble(),
                gnssCourseDeg = loc?.bearing?.toDouble()
            )
            onFrame?.invoke(frame)
            handler.postDelayed(this, 50)
        }
    }

    @SuppressLint("MissingPermission")
    override fun start(onFrameReceived: (SensorFrame) -> Unit) {
        this.onFrame = onFrameReceived

        val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        accel?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        gyro?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }

        try {
            locationManager.requestLocationUpdates(
                LocationManager.GPS_PROVIDER,
                500L,
                0f,
                this
            )
        } catch (e: SecurityException) {
            e.printStackTrace()
        }

        handler.post(emitRunnable)
    }

    override fun stop() {
        sensorManager.unregisterListener(this)
        locationManager.removeUpdates(this)
        handler.removeCallbacks(emitRunnable)
        onFrame = null
    }

    override fun setSimulatedOutage(isOutage: Boolean) {
        this.simulatedOutage = isOutage
    }

    override fun isOutageSimulated(): Boolean = simulatedOutage

    // SensorEventListener
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                lastAx = event.values[0]
                lastAy = event.values[1]
                lastAz = event.values[2]
            }
            Sensor.TYPE_GYROSCOPE -> {
                lastGx = event.values[0]
                lastGy = event.values[1]
                lastGz = event.values[2]
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // LocationListener
    override fun onLocationChanged(location: Location) {
        latestLocation = location
    }

    @Deprecated("Deprecated in Java")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}
}
