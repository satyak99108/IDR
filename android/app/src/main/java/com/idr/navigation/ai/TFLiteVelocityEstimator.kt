package com.idr.navigation.ai

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max

/**
 * On-device AI Forward Velocity Estimator.
 * Feeds a sliding 50-sample window of 6-axis IMU features into velocity_cnn.tflite.
 */
class TFLiteVelocityEstimator(
    private val context: Context,
    private val modelFileName: String = "velocity_cnn.tflite",
    scalerFileName: String = "scaler_params.json"
) {
    private var interpreter: Interpreter? = null
    private val normalizer: ScalerNormalizer

    // Sliding window buffer (50 samples x 6 features)
    private val windowSize = 50
    private val numFeatures = 6
    private val buffer = Array(windowSize) { FloatArray(numFeatures) }
    private var bufferIdx = 0
    private var sampleCount = 0

    // Direct ByteBuffers for zero-copy TFLite execution
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * windowSize * numFeatures * 4).apply {
        order(ByteOrder.nativeOrder())
    }
    private val outputBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 1 * 4).apply {
        order(ByteOrder.nativeOrder())
    }

    init {
        // Load scaler params
        normalizer = try {
            val jsonStr = context.assets.open(scalerFileName).bufferedReader().use { it.readText() }
            ScalerNormalizer.fromJson(jsonStr)
        } catch (e: Exception) {
            ScalerNormalizer()
        }

        // Load TFLite model
        try {
            val modelBuffer = loadModelFile(modelFileName)
            val options = Interpreter.Options().apply {
                setNumThreads(2)
            }
            interpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            e.printStackTrace()
            interpreter = null
        }
    }

    private fun loadModelFile(filename: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Ingest one 6-axis IMU sample [ax, ay, az, gx, gy, gz] and predict forward velocity (m/s).
     */
    fun estimateVelocity(
        ax: Float, ay: Float, az: Float,
        gx: Float, gy: Float, gz: Float,
        fallbackKinematicSpeedMs: Float = 0.0f
    ): Float {
        // Add to ring buffer
        val slot = buffer[bufferIdx]
        slot[0] = ax
        slot[1] = ay
        slot[2] = az
        slot[3] = gx
        slot[4] = gy
        slot[5] = gz

        bufferIdx = (bufferIdx + 1) % windowSize
        sampleCount++

        val interp = interpreter
        if (interp == null || sampleCount < windowSize) {
            return fallbackKinematicSpeedMs
        }

        // Fill input tensor in chronological order
        inputBuffer.rewind()
        for (i in 0 until windowSize) {
            val idx = (bufferIdx + i) % windowSize
            val raw = buffer[idx]
            val norm = normalizer.normalizeSample(raw)
            for (f in 0 until numFeatures) {
                inputBuffer.putFloat(norm[f])
            }
        }

        outputBuffer.rewind()
        interp.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val predictedSpeedMs = outputBuffer.float

        // Clamp to non-negative vehicle forward speed
        return max(0.0f, predictedSpeedMs)
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
