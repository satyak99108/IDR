"""
Unit Tests -- AI Package (Phase 5)
===================================
Tests IMUScaler, IMUSequenceDataset, and VelocityEvaluator.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.ai import (
    IMUScaler,
    IMUSequenceDataset,
    VelocityEvaluator,
    VelocityMetrics,
)


class TestAIComponents(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 500  # 500 timesteps
        # Synthetic 6-axis IMU features: [ax, ay, az, gx, gy, gz]
        features = np.random.normal(loc=1.0, scale=2.0, size=(n, 6)).astype(np.float32)
        # Synthetic forward velocity smoothly varying from 0 to 15 m/s
        target = np.abs(np.sin(np.linspace(0, 3 * np.pi, n)) * 15.0).astype(np.float32)

        self.df = pd.DataFrame({
            "timestamp": np.arange(0, n * 100, 100),
            "ax_veh_lin": features[:, 0],
            "ay_veh_lin": features[:, 1],
            "az_veh_lin": features[:, 2],
            "gx_veh": features[:, 3],
            "gy_veh": features[:, 4],
            "gz_veh": features[:, 5],
            "ref_speed": target * 3.6,  # in km/h
        })

    def test_imu_scaler(self):
        scaler = IMUScaler()
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
        scaled = scaler.fit_transform(data)
        self.assertTrue(scaler.is_fitted)
        self.assertAlmostEqual(float(np.mean(scaled[:, 0])), 0.0, places=5)
        self.assertAlmostEqual(float(np.std(scaled[:, 0])), 1.0, places=4)

        # Test save & load
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        try:
            scaler.save(temp_path)
            loaded_scaler = IMUScaler.load(temp_path)
            self.assertTrue(loaded_scaler.is_fitted)
            re_scaled = loaded_scaler.transform(data)
            np.testing.assert_allclose(scaled, re_scaled, atol=1e-5)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_sliding_window_creation(self):
        features, _ = IMUSequenceDataset.extract_features(self.df)
        targets = IMUSequenceDataset.extract_target_velocity(self.df)

        window_size = 50
        X, y = IMUSequenceDataset.create_sliding_windows(features, targets, window_size=window_size, step_size=1)

        expected_windows = len(self.df) - window_size + 1
        self.assertEqual(X.shape, (expected_windows, window_size, 6))
        self.assertEqual(y.shape, (expected_windows, 1))
        # Verify endpoint target alignment
        self.assertAlmostEqual(float(y[0, 0]), float(targets[window_size - 1]), places=4)

    def test_dataset_partitioning(self):
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = (
            IMUSequenceDataset.partition_trip_dataset(
                self.df,
                window_size=50,
                step_size=1,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
            )
        )
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_val), 0)
        self.assertGreater(len(X_test), 0)
        self.assertTrue(scaler.is_fitted)

    def test_velocity_evaluator(self):
        y_true = np.array([0.0, 2.0, 5.0, 10.0, 15.0])
        # Prediction with small offset
        y_pred = y_true + np.array([0.1, -0.2, 0.3, -0.1, 0.2])

        metrics = VelocityEvaluator.evaluate(y_true, y_pred)
        self.assertLess(metrics.mae_ms, 0.3)
        self.assertLess(metrics.rmse_ms, 0.3)
        self.assertGreater(metrics.r2_score, 0.95)
    def test_build_and_forward_1d_cnn(self):
        from src.ai import build_1d_cnn_velocity_model, NumpyCNNInference
        model = build_1d_cnn_velocity_model(input_shape=(50, 6))
        self.assertIsNotNone(model)
        
        # Test forward pass with Keras
        dummy_X = np.random.normal(size=(4, 50, 6)).astype(np.float32)
        preds = model.predict(dummy_X, verbose=0)
        self.assertEqual(preds.shape, (4, 1))

        # Test NumpyCNNInference
        np_engine = NumpyCNNInference.from_keras_model(model)
        np_preds = np_engine.predict(dummy_X)
        self.assertEqual(np_preds.shape, (4, 1))
        np.testing.assert_allclose(preds, np_preds, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
