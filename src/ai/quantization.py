"""
IDR MVP -- TFLite Quantization and On-Device Deployment Engine (Phase 14)
========================================================================
Implements systematic post-training optimization and quantization strategies:
1. Baseline Float32
2. Dynamic-Range Quantization (INT8 weights, Float32 activations)
3. Float16 Quantization (FP16 weights for GPU/NNAPI acceleration)
4. Full Integer Quantization (INT8 calibrated with representative IMU windows)

Also provides latency profiling, throughput benchmarking, accuracy evaluation,
and model card metadata generation per MVP.md §18 and TECH_STACK.md §5.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, Any, Generator, List, Optional, Tuple

import numpy as np


@dataclass
class QuantizationBenchmarkResult:
    """Benchmark metrics for a single TFLite model variant."""
    variant_name: str
    file_path: str
    file_size_bytes: int
    file_size_kb: float
    compression_ratio_vs_keras: float

    # Latency statistics (milliseconds per inference)
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    std_latency_ms: float

    # Throughput
    throughput_hz: float
    passes_10hz_goal: bool

    # Accuracy metrics (vs reference velocity in m/s)
    mae_ms: float
    rmse_ms: float
    max_error_ms: float
    r2_score: float

    # Discrepancy vs unquantized Float32 Keras model
    max_delta_vs_keras_ms: float = 0.0
    mean_delta_vs_keras_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TFLiteQuantizer:
    """
    Exports Keras models to multiple TFLite optimization candidates.
    """

    @staticmethod
    def export_float32(keras_model, output_path: Path) -> Path:
        """Export standard Float32 TFLite model without quantization."""
        import tensorflow as tf
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_bytes = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_bytes)
        return output_path

    @staticmethod
    def export_dynamic_range(keras_model, output_path: Path) -> Path:
        """Export Dynamic-Range Post-Training Quantized model (INT8 weights)."""
        import tensorflow as tf
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_bytes)
        return output_path

    @staticmethod
    def export_float16(keras_model, output_path: Path) -> Path:
        """Export Float16 quantized model for mobile GPU / NNAPI acceleration."""
        import tensorflow as tf
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_bytes = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_bytes)
        return output_path

    @staticmethod
    def export_full_integer(
        keras_model,
        representative_dataset_gen: Callable[[], Generator[List[np.ndarray], None, None]],
        output_path: Path
    ) -> Path:
        """Export full integer quantized model calibrated on representative sensor windows."""
        import tensorflow as tf
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        # Keep float fallback for I/O layers to avoid requiring integer-scaling callers
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_bytes = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_bytes)
        return output_path


class TFLiteBenchmarkEngine:
    """
    Benchmarks TFLite models on realistic driving sensor windows.
    Measures latency distribution, throughput, memory size, and prediction accuracy.
    """

    def __init__(self, keras_model_size_bytes: int = 0):
        self.keras_size_bytes = keras_model_size_bytes

    def benchmark_variant(
        self,
        variant_name: str,
        tflite_path: Path,
        test_windows: np.ndarray,
        reference_velocities: np.ndarray,
        keras_predictions: Optional[np.ndarray] = None,
        num_warmup: int = 50,
        num_timing_runs: int = 500,
    ) -> QuantizationBenchmarkResult:
        """
        Benchmark a single TFLite model on test data.

        Parameters
        ----------
        variant_name : str
            Display name (e.g., 'Dynamic INT8', 'Float32').
        tflite_path : Path
            Path to .tflite model file.
        test_windows : np.ndarray
            Input windows of shape [N, window_size, features].
        reference_velocities : np.ndarray
            Ground truth forward velocities [N].
        keras_predictions : Optional[np.ndarray]
            Baseline Keras predictions for delta comparison.
        """
        import tensorflow as tf

        tflite_path = Path(tflite_path)
        file_size_bytes = tflite_path.stat().st_size
        file_size_kb = round(file_size_bytes / 1024.0, 2)
        compression_ratio = round(
            (self.keras_size_bytes / file_size_bytes) if file_size_bytes > 0 and self.keras_size_bytes > 0 else 1.0,
            2
        )

        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_idx = input_details[0]["index"]
        output_idx = output_details[0]["index"]

        n_samples = len(test_windows)
        predictions = np.zeros(n_samples, dtype=np.float32)

        # 1. Full inference run over test dataset
        for i in range(n_samples):
            inp = test_windows[i:i + 1].astype(input_details[0]["dtype"])
            interpreter.set_tensor(input_idx, inp)
            interpreter.invoke()
            out = interpreter.get_tensor(output_idx)
            predictions[i] = float(out.flatten()[0])

        # 2. Timing benchmark runs for accurate latency profiling
        timing_indices = np.random.choice(n_samples, size=min(n_samples, num_timing_runs), replace=True)
        sample_inp = test_windows[0:1].astype(input_details[0]["dtype"])

        # Warm-up
        for _ in range(num_warmup):
            interpreter.set_tensor(input_idx, sample_inp)
            interpreter.invoke()

        # Measured latencies
        latencies_ms = []
        for idx in timing_indices:
            inp = test_windows[idx:idx + 1].astype(input_details[0]["dtype"])
            interpreter.set_tensor(input_idx, inp)
            t0 = time.perf_counter()
            interpreter.invoke()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        latencies_ms = np.array(latencies_ms)
        mean_lat = float(np.mean(latencies_ms))
        median_lat = float(np.median(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        p99_lat = float(np.percentile(latencies_ms, 99))
        std_lat = float(np.std(latencies_ms))
        throughput_hz = round(1000.0 / mean_lat, 1) if mean_lat > 0 else 0.0

        # 3. Accuracy metrics
        ref = reference_velocities.flatten()
        pred = predictions.flatten()
        errors = np.abs(pred - ref)
        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean((pred - ref) ** 2)))
        max_err = float(np.max(errors))

        # R^2 score
        ss_res = np.sum((ref - pred) ** 2)
        ss_tot = np.sum((ref - np.mean(ref)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-6 else 0.0

        # 4. Keras parity delta
        max_delta_keras = 0.0
        mean_delta_keras = 0.0
        if keras_predictions is not None:
            deltas = np.abs(pred - keras_predictions.flatten())
            max_delta_keras = float(np.max(deltas))
            mean_delta_keras = float(np.mean(deltas))

        return QuantizationBenchmarkResult(
            variant_name=variant_name,
            file_path=str(tflite_path),
            file_size_bytes=file_size_bytes,
            file_size_kb=file_size_kb,
            compression_ratio_vs_keras=compression_ratio,
            mean_latency_ms=round(mean_lat, 3),
            median_latency_ms=round(median_lat, 3),
            p95_latency_ms=round(p95_lat, 3),
            p99_latency_ms=round(p99_lat, 3),
            std_latency_ms=round(std_lat, 3),
            throughput_hz=throughput_hz,
            passes_10hz_goal=(throughput_hz >= 10.0),
            mae_ms=round(mae, 4),
            rmse_ms=round(rmse, 4),
            max_error_ms=round(max_err, 4),
            r2_score=round(r2, 4),
            max_delta_vs_keras_ms=round(max_delta_keras, 4),
            mean_delta_vs_keras_ms=round(mean_delta_keras, 4),
        )


def generate_model_card(
    best_variant: QuantizationBenchmarkResult,
    input_shape: List[int],
    output_shape: List[int],
    feature_names: List[str],
    hardware_target: str = "Android ARM64 / x86_64",
) -> Dict[str, Any]:
    """Generate structured Model Card metadata for production mobile deployment."""
    return {
        "model_name": "IDR Forward Velocity 1D-CNN",
        "version": "1.0.0-mvp",
        "framework": "TensorFlow Lite",
        "optimization": best_variant.variant_name,
        "model_file_size_kb": best_variant.file_size_kb,
        "input_tensor": {
            "name": "imu_sliding_window",
            "shape": input_shape,
            "dtype": "float32",
            "features": feature_names,
            "sampling_frequency_hz": 100,
            "window_duration_seconds": 0.5,
        },
        "output_tensor": {
            "name": "forward_velocity",
            "shape": output_shape,
            "dtype": "float32",
            "units": "m/s",
        },
        "performance_profile": {
            "mean_latency_ms": best_variant.mean_latency_ms,
            "p95_latency_ms": best_variant.p95_latency_ms,
            "throughput_hz": best_variant.throughput_hz,
            "minimum_target_hz": 10.0,
            "target_met": best_variant.passes_10hz_goal,
        },
        "accuracy_profile": {
            "mae_ms": best_variant.mae_ms,
            "rmse_ms": best_variant.rmse_ms,
            "r2_score": best_variant.r2_score,
            "parity_delta_vs_keras_ms": best_variant.mean_delta_vs_keras_ms,
        },
        "deployment_target": hardware_target,
    }
