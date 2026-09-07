package com.idr.navigation.sensor

/**
 * Common interface for live hardware sensors or offline trip replay streamer.
 */
interface SensorDataProvider {
    fun start(onFrameReceived: (SensorFrame) -> Unit)
    fun stop()
    fun setSimulatedOutage(isOutage: Boolean)
    fun isOutageSimulated(): Boolean
}
