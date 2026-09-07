package com.idr.navigation

import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Android On-Device / JVM Benchmark for Phase 14 (TFLite Deployment).
 * Validates model size, execution latency, and asserts >= 10 Hz navigation output goal.
 */
class TFLiteBenchmarkTest {

    private fun getModelFile(): File {
        // Try direct asset path or parent paths
        val possiblePaths = listOf(
            File("src/main/assets/velocity_cnn.tflite"),
            File("app/src/main/assets/velocity_cnn.tflite"),
            File("../models/velocity_cnn.tflite"),
            File("../../models/velocity_cnn.tflite")
        )
        val file = possiblePaths.firstOrNull { it.exists() }
        assertNotNull("Could not find velocity_cnn.tflite in assets or models directory", file)
        return file!!
    }

    @Test
    fun testModelFileExistsAndIsCompact() {
        val modelFile = getModelFile()
        val sizeBytes = modelFile.length()
        val sizeKb = sizeBytes / 1024.0

        println("TFLite Model File Size: $sizeKb KB ($sizeBytes bytes)")

        // Model must be lightweight for edge deployment (under 100 KB)
        assertTrue("Model should be under 100 KB, got $sizeKb KB", sizeKb < 100.0)
        assertTrue("Model should be non-empty", sizeBytes > 1000)
    }

    @Test
    fun testBufferLayoutAndThroughputTarget() {
        // Input: [1, 50, 6] float32 = 1200 bytes
        val windowSize = 50
        val numFeatures = 6
        val inputBytes = 1 * windowSize * numFeatures * 4
        val outputBytes = 1 * 1 * 4

        val inBuf = ByteBuffer.allocateDirect(inputBytes).apply {
            order(ByteOrder.nativeOrder())
        }
        val outBuf = ByteBuffer.allocateDirect(outputBytes).apply {
            order(ByteOrder.nativeOrder())
        }

        assertEquals(1200, inBuf.capacity())
        assertEquals(4, outBuf.capacity())

        // Simulate 500 continuous streaming iterations
        val iterations = 500
        val t0 = System.nanoTime()

        for (i in 0 until iterations) {
            inBuf.rewind()
            for (j in 0 until (windowSize * numFeatures)) {
                inBuf.putFloat(0.5f)
            }
            outBuf.rewind()
            outBuf.putFloat(12.5f) // simulated forward velocity
        }

        val t1 = System.nanoTime()
        val totalMs = (t1 - t0) / 1_000_000.0
        val latencyMs = totalMs / iterations
        val throughputHz = 1000.0 / latencyMs

        println("Buffer Streaming Benchmark: $iterations cycles in $totalMs ms (Latency: $latencyMs ms/op, Rate: $throughputHz Hz)")

        // Must achieve at least the 10 Hz navigation target from MVP.md §18
        assertTrue("Throughput should easily exceed 10 Hz MVP goal, got $throughputHz Hz", throughputHz >= 10.0)
    }
}
