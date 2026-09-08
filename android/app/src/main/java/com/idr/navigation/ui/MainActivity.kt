package com.idr.navigation.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.view.MotionEvent
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import android.content.Context
import com.idr.navigation.R
import com.idr.navigation.ai.TFLiteVelocityEstimator
import com.idr.navigation.core.DeadReckoningEngine
import com.idr.navigation.core.NavigationMode
import com.idr.navigation.core.NavigationOutput
import com.idr.navigation.databinding.ActivityMainBinding
import com.idr.navigation.sensor.AndroidHardwareSensorProvider
import com.idr.navigation.sensor.ReplaySensorProvider
import com.idr.navigation.sensor.SensorDataProvider
import com.idr.navigation.sensor.SensorFrame
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline

/**
 * Main Activity providing full OpenStreetMap visualization via OSMDroid,
 * heads-up live telemetry, and interactive judge demonstration controls.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    // Core engines
    private lateinit var velocityEstimator: TFLiteVelocityEstimator
    private lateinit var deadReckoningEngine: DeadReckoningEngine

    // Sensor providers
    private lateinit var liveProvider: AndroidHardwareSensorProvider
    private lateinit var replayProvider: ReplaySensorProvider
    private var activeProvider: SensorDataProvider? = null

    // Map overlays
    private var vehicleMarker: Marker? = null
    private val routePolyline = Polyline()
    private var isMapCenteredOnVehicle = true
    private var lastGeoPoint: GeoPoint? = null

    // Permission request launcher
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val fineGranted = permissions[Manifest.permission.ACCESS_FINE_LOCATION] ?: false
        if (fineGranted) {
            Toast.makeText(this, "GNSS Location permission granted", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize OSMDroid configuration
        Configuration.getInstance().load(this, getSharedPreferences("idr_prefs", Context.MODE_PRIVATE))
        Configuration.getInstance().userAgentValue = packageName

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Check location permissions
        checkLocationPermissions()

        // Initialize navigation engines
        deadReckoningEngine = DeadReckoningEngine()
        velocityEstimator = TFLiteVelocityEstimator(this)

        // Initialize sensor providers
        liveProvider = AndroidHardwareSensorProvider(this)
        replayProvider = ReplaySensorProvider(this)

        // Set up MapView
        setupMapView()

        // Set up Judge Controls & Buttons
        setupJudgeControls()

        // Default to IO-VNBD Replay mode for reliable instant demonstration
        setSensorSource(isReplay = true)
    }

    private fun setupMapView() {
        binding.mapView.apply {
            setTileSource(TileSourceFactory.MAPNIK)
            setMultiTouchControls(true)
            controller.setZoom(17.5)

            // Add route polyline overlay
            routePolyline.outlinePaint.color = Color.parseColor("#10B981") // Default emerald
            routePolyline.outlinePaint.strokeWidth = 10f
            overlayManager.add(routePolyline)

            // Vehicle marker overlay 🚗
            vehicleMarker = Marker(this).apply {
                icon = ContextCompat.getDrawable(this@MainActivity, R.drawable.ic_car_marker)
                setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
                title = "Vehicle Position"
            }
            overlayManager.add(vehicleMarker)

            // Detect user manual touch drag to immediately release vehicle auto-centering
            setOnTouchListener { _, event ->
                if (event.actionMasked == MotionEvent.ACTION_MOVE || event.actionMasked == MotionEvent.ACTION_DOWN) {
                    if (isMapCenteredOnVehicle) {
                        setMapAutoCentering(false)
                    }
                }
                false
            }

            // Detect user manual panning/scrolling to release lock on vehicle
            addMapListener(object : org.osmdroid.events.MapListener {
                override fun onScroll(event: org.osmdroid.events.ScrollEvent?): Boolean {
                    if (isMapCenteredOnVehicle) {
                        setMapAutoCentering(false)
                    }
                    return false
                }

                override fun onZoom(event: org.osmdroid.events.ZoomEvent?): Boolean {
                    return false
                }
            })
        }
    }

    private fun setupJudgeControls() {
        // Toggle Outage (Simulate Tunnel Blackout)
        binding.btnToggleOutage.setOnClickListener {
            val provider = activeProvider ?: return@setOnClickListener
            val isNowOutage = !provider.isOutageSimulated()
            provider.setSimulatedOutage(isNowOutage)

            if (isNowOutage) {
                // In Tunnel / Blackout mode
                binding.btnToggleOutage.text = getString(R.string.btn_restore_gnss)
                binding.btnToggleOutage.setBackgroundColor(ContextCompat.getColor(this, R.color.btn_restore_bg))
                binding.tvGnssStatus.text = getString(R.string.gnss_lost)
                binding.tvGnssStatus.setTextColor(ContextCompat.getColor(this, R.color.status_outage))
                binding.tvOutageTimer.visibility = View.VISIBLE
            } else {
                // Restored GNSS mode
                binding.btnToggleOutage.text = getString(R.string.btn_kill_gnss)
                binding.btnToggleOutage.setBackgroundColor(ContextCompat.getColor(this, R.color.btn_tunnel_bg))
                binding.tvGnssStatus.text = getString(R.string.gnss_active)
                binding.tvGnssStatus.setTextColor(Color.parseColor("#34D399"))
                binding.tvOutageTimer.visibility = View.GONE
            }
        }

        // Toggle Live vs Replay source
        binding.btnToggleSource.setOnClickListener {
            val isCurrentlyReplay = (activeProvider === replayProvider)
            setSensorSource(!isCurrentlyReplay)
        }

        // Reset Trajectory
        binding.btnReset.setOnClickListener {
            deadReckoningEngine.reset()
            replayProvider.resetReplay()
            routePolyline.actualPoints.clear()
            binding.mapView.invalidate()
            Toast.makeText(this, "Trajectory reset", Toast.LENGTH_SHORT).show()
        }

        // Recenter Map
        binding.fabRecenter.setOnClickListener {
            setMapAutoCentering(true)
            lastGeoPoint?.let { binding.mapView.controller.animateTo(it) }
        }
    }

    private fun setMapAutoCentering(centered: Boolean) {
        isMapCenteredOnVehicle = centered
        if (centered) {
            binding.fabRecenter.imageTintList = ContextCompat.getColorStateList(this, R.color.fab_active_tint)
        } else {
            binding.fabRecenter.imageTintList = ContextCompat.getColorStateList(this, R.color.fab_inactive_tint)
        }
    }

    private fun setSensorSource(isReplay: Boolean) {
        activeProvider?.stop()

        if (isReplay) {
            activeProvider = replayProvider
            binding.btnToggleSource.text = getString(R.string.btn_mode_live)
            binding.tvSourceBadge.text = "IO-VNBD REPLAY"
            binding.tvSourceBadge.setTextColor(Color.parseColor("#38BDF8"))
        } else {
            activeProvider = liveProvider
            binding.btnToggleSource.text = getString(R.string.btn_mode_replay)
            binding.tvSourceBadge.text = "LIVE HARDWARE SENSORS"
            binding.tvSourceBadge.setTextColor(Color.parseColor("#34D399"))
        }

        activeProvider?.start { frame ->
            onSensorFrameReceived(frame)
        }
    }

    private fun onSensorFrameReceived(frame: SensorFrame) {
        // Run AI forward velocity estimation
        val estimatedSpeedMs = velocityEstimator.estimateVelocity(
            ax = frame.ax, ay = frame.ay, az = frame.az,
            gx = frame.gx, gy = frame.gy, gz = frame.gz,
            fallbackKinematicSpeedMs = (frame.gnssSpeedMs ?: 0.0).toFloat()
        )

        // Process dead reckoning step
        val isOutage = activeProvider?.isOutageSimulated() ?: false
        val navOutput = deadReckoningEngine.processSample(
            timestampMs = frame.timestampMs,
            axVeh = frame.ax,
            ayVeh = frame.ay,
            azVeh = frame.az,
            gzVeh = frame.gz,
            aiForwardSpeedMs = estimatedSpeedMs,
            gnssLat = frame.gnssLat,
            gnssLon = frame.gnssLon,
            gnssSpeedMs = frame.gnssSpeedMs,
            gnssCourseDeg = frame.gnssCourseDeg,
            isSimulatedOutage = isOutage
        )

        // Update UI
        updateNavigationUI(navOutput)
    }

    private fun updateNavigationUI(output: NavigationOutput) {
        runOnUiThread {
            // Speedometer
            binding.tvSpeed.text = String.format("%.1f", output.speedKmh)

            // Navigation Mode
            binding.tvNavMode.text = output.mode.displayName
            binding.tvNavMode.setTextColor(Color.parseColor(output.mode.badgeColorHex))

            // Confidence Score
            binding.tvConfidence.text = "Confidence: ${output.confidencePct}%"

            // Outage duration timer
            if (output.isOutageActive) {
                binding.tvOutageTimer.visibility = View.VISIBLE
                binding.tvOutageTimer.text = String.format("Tunnel Outage: %.1fs", output.outageDurationS)
            } else {
                binding.tvOutageTimer.visibility = View.GONE
            }

            // Map update
            val currentPoint = GeoPoint(output.latDeg, output.lonDeg)
            lastGeoPoint = currentPoint

            // Update Vehicle Marker position & heading rotation
            vehicleMarker?.position = currentPoint
            vehicleMarker?.rotation = output.headingDeg.toFloat()

            // Update polyline color by navigation mode
            when (output.mode) {
                NavigationMode.GNSS_INS -> routePolyline.outlinePaint.color = Color.parseColor("#10B981")
                NavigationMode.DEAD_RECKONING -> routePolyline.outlinePaint.color = Color.parseColor("#F59E0B")
                NavigationMode.RECOVERY -> routePolyline.outlinePaint.color = Color.parseColor("#06B6D4")
                NavigationMode.DEGRADED_GNSS -> routePolyline.outlinePaint.color = Color.parseColor("#EAB308")
            }

            routePolyline.addPoint(currentPoint)
            if (isMapCenteredOnVehicle) {
                binding.mapView.controller.setCenter(currentPoint)
            }
            binding.mapView.invalidate()
        }
    }

    private fun checkLocationPermissions() {
        val permissions = arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )
        val notGranted = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (notGranted.isNotEmpty()) {
            requestPermissionLauncher.launch(permissions)
        }
    }

    override fun onResume() {
        super.onResume()
        binding.mapView.onResume()
        activeProvider?.start { onSensorFrameReceived(it) }
    }

    override fun onPause() {
        super.onPause()
        binding.mapView.onPause()
        activeProvider?.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        activeProvider?.stop()
        velocityEstimator.close()
    }
}
