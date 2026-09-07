"""
IDR MVP -- Unit Tests for Phase 14: TFLite Deployment and Optimization
======================================================================
Validates quantization strategies, tensor dimensions, latency profiling,
and throughput targets (>= 10 Hz navigation goal) per MVP.md §18.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.ai.model import build_1d_cnn_velocity_model
from src.ai.quantization import (
    TFLiteQuantizer,
    TFLiteBenchmarkEngine,
    QuantizationBenchmarkResult,
    generate_model_card,
)


@pytest.fixture(scope="module")
def sample_keras_model():
    """Create a lightweight trained Keras model for testing export pipelines."""
    model = build_1d_cnn_velocity_model(input_shape=(50, 6))
    # Train 1 epoch on dummy data to ensure weights are fully instantiated
    x = np.random.randn(20, 50, 6).astype(np.float32)
    y = np.random.uniform(5.0, 20.0, size=(20, 1)).astype(np.float32)
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=1, verbose=0)
    return model


class TestTFLiteQuantizer:
    def test_export_float32(self, sample_keras_model, tmp_path):
        out_path = tmp_path / "test_float32.tflite"
        res = TFLiteQuantizer.export_float32(sample_keras_model, out_path)
        assert res.exists()
        assert res.stat().st_size > 0

        # Verify tensor shapes
        interp = tf.lite.Interpreter(model_path=str(res))
        interp.allocate_tensors()
        inp_shape = interp.get_input_details()[0]["shape"].tolist()
        out_shape = interp.get_output_details()[0]["shape"].tolist()
        assert inp_shape == [1, 50, 6]
        assert out_shape == [1, 1]

    def test_export_dynamic_range(self, sample_keras_model, tmp_path):
        out_path = tmp_path / "test_dynamic_range.tflite"
        res = TFLiteQuantizer.export_dynamic_range(sample_keras_model, out_path)
        assert res.exists()
        size_kb = res.stat().st_size / 1024.0
        # Dynamic range INT8 should be lightweight (< 100 KB)
        assert size_kb < 100.0

    def test_export_float16(self, sample_keras_model, tmp_path):
        out_path = tmp_path / "test_float16.tflite"
        res = TFLiteQuantizer.export_float16(sample_keras_model, out_path)
        assert res.exists()
        assert res.stat().st_size > 0


class TestTFLiteBenchmarkEngine:
    def test_benchmark_variant_throughput(self, sample_keras_model, tmp_path):
        model_path = tmp_path / "bench_model.tflite"
        TFLiteQuantizer.export_float32(sample_keras_model, model_path)

        n_samples = 30
        test_windows = np.random.randn(n_samples, 50, 6).astype(np.float32)
        ref_vels = np.random.uniform(5.0, 15.0, size=n_samples).astype(np.float32)

        engine = TFLiteBenchmarkEngine(keras_model_size_bytes=100000)
        res = engine.benchmark_variant(
            variant_name="Test Float32",
            tflite_path=model_path,
            test_windows=test_windows,
            reference_velocities=ref_vels,
            num_warmup=5,
            num_timing_runs=20,
        )

        assert isinstance(res, QuantizationBenchmarkResult)
        assert res.variant_name == "Test Float32"
        assert res.file_size_kb > 0
        assert res.mean_latency_ms >= 0.0
        assert res.throughput_hz > 0.0
        # Validates MVP.md §18 target
        assert res.passes_10hz_goal is True
        assert res.mae_ms >= 0.0

    def test_to_dict_serializable(self, sample_keras_model, tmp_path):
        model_path = tmp_path / "serial_model.tflite"
        TFLiteQuantizer.export_float32(sample_keras_model, model_path)

        engine = TFLiteBenchmarkEngine()
        res = engine.benchmark_variant(
            variant_name="SerialTest",
            tflite_path=model_path,
            test_windows=np.random.randn(10, 50, 6).astype(np.float32),
            reference_velocities=np.ones(10, dtype=np.float32),
            num_warmup=2,
            num_timing_runs=10,
        )

        d = res.to_dict()
        assert isinstance(d, dict)
        assert d["variant_name"] == "SerialTest"
        assert "throughput_hz" in d
        assert "passes_10hz_goal" in d


class TestModelCard:
    def test_generate_model_card(self):
        result = QuantizationBenchmarkResult(
            variant_name="Dynamic INT8",
            file_path="models/velocity_cnn.tflite",
            file_size_bytes=16000,
            file_size_kb=15.6,
            compression_ratio_vs_keras=3.8,
            mean_latency_ms=1.2,
            median_latency_ms=1.1,
            p95_latency_ms=1.8,
            p99_latency_ms=2.2,
            std_latency_ms=0.2,
            throughput_hz=833.3,
            passes_10hz_goal=True,
            mae_ms=0.18,
            rmse_ms=0.25,
            max_error_ms=0.85,
            r2_score=0.92,
        )

        card = generate_model_card(
            best_variant=result,
            input_shape=[1, 50, 6],
            output_shape=[1, 1],
            feature_names=["ax", "ay", "az", "gx", "gy", "gz"],
        )

        assert card["model_name"] == "IDR Forward Velocity 1D-CNN"
        assert card["performance_profile"]["target_met"] is True
        assert card["performance_profile"]["throughput_hz"] == 833.3
        assert card["input_tensor"]["shape"] == [1, 50, 6]
