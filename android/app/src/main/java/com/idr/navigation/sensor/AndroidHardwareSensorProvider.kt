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
import android.util.Log

/**
 * Live hardware sensor provider reading Android SensorManager and LocationManager.
 *
 * Uses [Sensor.TYPE_LINEAR_ACCELERATION] (gravity-removed) rather than
 * [Sensor.TYPE_ACCELEROMETER] because the full navigation pipeline — TFLite model,
 * stationary detector, kinematic fallback — expects gravity-free linear acceleration
 * matching the IO-VNBD training features (`ax_veh_lin, ay_veh_lin, az_veh_lin`).
 *
 * Falls back to `TYPE_ACCELEROMETER − TYPE_GRAVITY` when `TYPE_LINEAR_ACCELERATION`
 * is unavailable.
 */
class AndroidHardwareSensorProvider(
    private val context: Context
) : SensorDataProvider, SensorEventListener, LocationListener {

    companion object {
        private const val TAG = "IDR_HWSensor"
        /** Emitter interval — approximately 20 Hz navigation output. */
        private const val EMIT_INTERVAL_MS = 50L
    }

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    private var onFrame: ((SensorFrame) -> Unit)? = null
    private var simulatedOutage = false

    /**
     * Guards against duplicate registration caused by Activity lifecycle
     * (onCreate → start, then onResume → start again before onPause → stop).
     */
    private var isRunning = false

    // Linear acceleration buffers (gravity REMOVED)
    private var lastAx = 0f
    private var lastAy = 0f
    private var lastAz = 0f

    // Gyroscope buffers
    private var lastGx = 0f
    private var lastGy = 0f
    private var lastGz = 0f

    // Gravity vector (used only when TYPE_LINEAR_ACCELERATION is unavailable)
    private var gravX = 0f
    private var gravY = 0f
    private var gravZ = 9.81f // default until first gravity reading

    // Whether we are using TYPE_LINEAR_ACCELERATION (true) or manual subtraction (false)
    private var usingLinearAccel = false

    // GNSS cache
    private var latestLocation: Location? = null

    // Periodic emitter loop (~20 Hz)
    private val handler = Handler(Looper.getMainLooper())
    private val emitRunnable = object : Runnable {
        override fun run() {
            if (!isRunning) return

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
            handler.postDelayed(this, EMIT_INTERVAL_MS)
        }
    }

    @SuppressLint("MissingPermission")
    override fun start(onFrameReceived: (SensorFrame) -> Unit) {
        // Prevent duplicate registration from Activity lifecycle (onCreate + onResume)
        if (isRunning) {
            // Update callback reference only (may be a new lambda after config change)
            this.onFrame = onFrameReceived
            return
        }

        this.onFrame = onFrameReceived
        isRunning = true

        // --- IMU Registration ---
        // Prefer TYPE_LINEAR_ACCELERATION: hardware sensor-fusion that removes gravity.
        // This matches the training features (ax_veh_lin, ay_veh_lin, az_veh_lin).
        val linearAccel = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        if (linearAccel != null) {
            usingLinearAccel = true
            sensorManager.registerListener(this, linearAccel, SensorManager.SENSOR_DELAY_GAME)
            Log.i(TAG, "Using TYPE_LINEAR_ACCELERATION (gravity removed by Android)")
        } else {
            // Fallback: raw accelerometer + gravity sensor → manual subtraction
            usingLinearAccel = false
            val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
            val gravity = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
            accel?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            gravity?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
            Log.i(TAG, "Fallback: TYPE_ACCELEROMETER minus TYPE_GRAVITY")
        }

        gyro?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }

        // --- GNSS Registration ---
        // 1. Immediately seed latestLocation from last known position (GPS or Network)
        // This avoids starting with null coordinates or defaulting to (0,0) before first live fix.
        try {
            val lastGps = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            val lastNet = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            latestLocation = when {
                lastGps != null && lastNet != null -> {
                    if (lastGps.time >= lastNet.time) lastGps else lastNet
                }
                lastGps != null -> lastGps
                else -> lastNet
            }
            if (latestLocation != null) {
                Log.i(TAG, "Seeded initial location from last known: lat=${latestLocation?.latitude}, lon=${latestLocation?.longitude}")
            }
        } catch (e: SecurityException) {
            Log.w(TAG, "GNSS permission not granted for last known location", e)
        }

        // 2. Request updates from GPS_PROVIDER
        try {
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    500L,   // min interval ms
                    0f,     // min distance m
                    this
                )
            }
        } catch (e: SecurityException) {
            Log.w(TAG, "GPS permission not granted", e)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to request GPS updates", e)
        }

        // 3. Fallback to NETWORK_PROVIDER for fast indoor fix / rough initial lock
        try {
            if (locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.NETWORK_PROVIDER,
                    1000L,
                    0f,
                    this
                )
            }
        } catch (e: SecurityException) {
            Log.w(TAG, "Network location permission not granted", e)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to request Network location updates", e)
        }

        handler.post(emitRunnable)
    }

    override fun stop() {
        isRunning = false
        sensorManager.unregisterListener(this)
        locationManager.removeUpdates(this)
        handler.removeCallbacks(emitRunnable)
        onFrame = null
    }

    override fun setSimulatedOutage(isOutage: Boolean) {
        this.simulatedOutage = isOutage
    }

    override fun isOutageSimulated(): Boolean = simulatedOutage

    // ---- SensorEventListener ----

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                // Already gravity-free — use directly
                lastAx = event.values[0]
                lastAy = event.values[1]
                lastAz = event.values[2]
            }

            Sensor.TYPE_ACCELEROMETER -> {
                // Manual gravity subtraction (only when TYPE_LINEAR_ACCELERATION unavailable)
                if (!usingLinearAccel) {
                    lastAx = event.values[0] - gravX
                    lastAy = event.values[1] - gravY
                    lastAz = event.values[2] - gravZ
                }
            }

            Sensor.TYPE_GRAVITY -> {
                gravX = event.values[0]
                gravY = event.values[1]
                gravZ = event.values[2]
            }

            Sensor.TYPE_GYROSCOPE -> {
                lastGx = event.values[0]
                lastGy = event.values[1]
                lastGz = event.values[2]
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // ---- LocationListener ----

    override fun onLocationChanged(location: Location) {
        latestLocation = location
    }

    @Deprecated("Deprecated in Java")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}
}
