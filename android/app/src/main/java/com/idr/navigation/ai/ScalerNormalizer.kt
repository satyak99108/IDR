package com.idr.navigation.ai

import org.json.JSONObject

/**
 * Standardizes 6-channel IMU window input: [ax, ay, az, gx, gy, gz]
 * matching the scikit-learn / numpy scaler trained on IO-VNBD.
 */
class ScalerNormalizer(
    val means: FloatArray = floatArrayOf(
        -0.2702922f, -0.0296167f, -0.0271258f,
        0.0011831f, 0.0011792f, 0.0015184f
    ),
    val stds: FloatArray = floatArrayOf(
        1.0386704f, 1.0294318f, 0.4571593f,
        0.1436010f, 0.0481191f, 0.0891792f
    ),
    val eps: Float = 1e-8f
) {
    /**
     * Normalizes a single 6-channel sample in-place or returns a new FloatArray.
     */
    fun normalizeSample(raw6: FloatArray): FloatArray {
        val out = FloatArray(6)
        for (i in 0 until 6) {
            val s = if (stds[i] > eps) stds[i] else 1.0f
            out[i] = (raw6[i] - means[i]) / s
        }
        return out
    }

    companion object {
        fun fromJson(jsonStr: String): ScalerNormalizer {
            return try {
                val obj = JSONObject(jsonStr)
                val mArr = obj.getJSONArray("means")
                val sArr = obj.getJSONArray("stds")
                val epsVal = obj.optDouble("eps", 1e-8).toFloat()

                val m = FloatArray(mArr.length()) { mArr.getDouble(it).toFloat() }
                val s = FloatArray(sArr.length()) { sArr.getDouble(it).toFloat() }

                ScalerNormalizer(m, s, epsVal)
            } catch (e: Exception) {
                ScalerNormalizer()
            }
        }
    }
}
