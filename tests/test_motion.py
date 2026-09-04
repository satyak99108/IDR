"""
Unit Tests -- Motion / Vibration Filtering (Phase 6)
=====================================================
Tests MotionFeatureExtractor, MotionClassifier, and MotionLabel on synthetic
IMU signals with known characteristics.
"""

import unittest
import numpy as np
import pandas as pd

from src.motion import MotionFeatureExtractor, MotionClassifier, MotionLabel


class TestMotionFeatureExtractor(unittest.TestCase):
    """Tests for MotionFeatureExtractor."""

    def setUp(self):
        np.random.seed(42)
        self.fs = 10.0
        self.extractor = MotionFeatureExtractor(
            sampling_rate_hz=self.fs,
            window_size_sec=1.0,
            step_size_sec=0.5,
        )

    def _make_constant_gravity_data(self, n=100):
        """Constant gravity + zero gyro → stationary signal."""
        accel = np.zeros((n, 3), dtype=np.float64)
        accel[:, 2] = 9.81  # gravity along z
        gyro = np.zeros((n, 3), dtype=np.float64)
        return accel, gyro

    def test_feature_extraction_shapes(self):
        """Output DataFrame should have correct shape and columns."""
        n = 100
        accel = np.random.normal(0, 1, (n, 3))
        gyro = np.random.normal(0, 0.1, (n, 3))

        result = self.extractor.extract(accel, gyro)

        # Check that we get a DataFrame with feature columns
        self.assertIsInstance(result, pd.DataFrame)
        for feat in [
            "accel_mean", "accel_var", "accel_rms", "accel_peak",
            "gyro_rms", "gyro_peak", "jerk_rms",
            "freq_energy_ratio", "spectral_entropy",
        ]:
            self.assertIn(feat, result.columns, f"Missing feature column: {feat}")

        # Check metadata columns
        self.assertIn("window_start_idx", result.columns)
        self.assertIn("window_end_idx", result.columns)
        self.assertIn("timestamp", result.columns)

        # Should have at least 1 window
        self.assertGreater(len(result), 0)

    def test_stationary_features(self):
        """Constant-gravity signal should yield low variance, low gyro RMS."""
        accel, gyro = self._make_constant_gravity_data(n=50)
        result = self.extractor.extract(accel, gyro)

        # All windows should have very low accel variance
        self.assertTrue((result["accel_var"] < 0.01).all(),
                        f"Stationary accel_var too high: {result['accel_var'].values}")

        # Gyro RMS should be ~0
        self.assertTrue((result["gyro_rms"] < 0.01).all(),
                        f"Stationary gyro_rms too high: {result['gyro_rms'].values}")

        # Accel mean should be ~9.81
        self.assertTrue((result["accel_mean"] > 9.5).all(),
                        f"Stationary accel_mean too low: {result['accel_mean'].values}")

    def test_shock_features(self):
        """Injected acceleration spike should produce high jerk_rms and accel_peak."""
        n = 50
        accel = np.zeros((n, 3), dtype=np.float64)
        accel[:, 2] = 9.81
        gyro = np.zeros((n, 3), dtype=np.float64)

        # Inject a sharp spike at sample 25
        accel[25, 0] = 30.0   # 30 m/s² spike in x
        accel[25, 2] = 30.0   # also in z

        result = self.extractor.extract(accel, gyro)

        # At least one window should have elevated jerk_rms and accel_peak
        self.assertTrue((result["jerk_rms"].max() > 10.0),
                        f"Shock jerk_rms too low: {result['jerk_rms'].max()}")
        self.assertTrue((result["accel_peak"].max() > 20.0),
                        f"Shock accel_peak too low: {result['accel_peak'].max()}")

    def test_vibration_features(self):
        """High-frequency noise should produce high freq_energy_ratio and spectral_entropy."""
        n = 100
        t = np.arange(n) / self.fs

        # High-frequency sinusoid (4 Hz, near Nyquist for 10 Hz sampling)
        accel = np.zeros((n, 3), dtype=np.float64)
        accel[:, 0] = 5.0 * np.sin(2 * np.pi * 4.0 * t)
        accel[:, 1] = 3.0 * np.sin(2 * np.pi * 3.5 * t)
        accel[:, 2] = 9.81 + 4.0 * np.sin(2 * np.pi * 4.5 * t)
        gyro = np.random.normal(0, 0.01, (n, 3))

        result = self.extractor.extract(accel, gyro)

        # Should have elevated high-frequency energy
        self.assertTrue(result["freq_energy_ratio"].max() > 0.2,
                        f"Vibration freq_energy_ratio too low: {result['freq_energy_ratio'].max()}")
        # Accel variance should be elevated
        self.assertTrue(result["accel_var"].max() > 1.0,
                        f"Vibration accel_var too low: {result['accel_var'].max()}")

    def test_dataframe_interface(self):
        """extract_from_dataframe should work with standard column names."""
        n = 50
        df = pd.DataFrame({
            "timestamp": np.arange(n),
            "ax_veh_lin": np.random.normal(0, 1, n),
            "ay_veh_lin": np.random.normal(0, 1, n),
            "az_veh_lin": np.random.normal(0, 1, n) + 9.81,
            "gx_veh": np.random.normal(0, 0.05, n),
            "gy_veh": np.random.normal(0, 0.05, n),
            "gz_veh": np.random.normal(0, 0.05, n),
        })

        result = self.extractor.extract_from_dataframe(df)
        self.assertGreater(len(result), 0)

    def test_too_short_raises(self):
        """Data shorter than window should raise ValueError."""
        accel = np.zeros((3, 3))
        gyro = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            self.extractor.extract(accel, gyro)


class TestMotionClassifier(unittest.TestCase):
    """Tests for MotionClassifier."""

    def setUp(self):
        self.classifier = MotionClassifier()

    def test_normal_driving(self):
        """Features resembling normal driving should → NORMAL."""
        features = {
            "accel_mean": 9.85,
            "accel_var": 0.5,
            "accel_rms": 10.0,
            "accel_peak": 12.0,
            "gyro_rms": 0.15,
            "gyro_peak": 0.5,
            "jerk_rms": 5.0,
            "freq_energy_ratio": 0.2,
            "spectral_entropy": 0.5,
        }
        self.assertEqual(self.classifier.classify_window(features), MotionLabel.NORMAL)

    def test_stationary_detection(self):
        """Low-variance gravity-only features should → STATIONARY."""
        features = {
            "accel_mean": 9.81,
            "accel_var": 0.001,
            "accel_rms": 9.81,
            "accel_peak": 9.82,
            "gyro_rms": 0.01,
            "gyro_peak": 0.02,
            "jerk_rms": 0.1,
            "freq_energy_ratio": 0.1,
            "spectral_entropy": 0.3,
        }
        self.assertEqual(self.classifier.classify_window(features), MotionLabel.STATIONARY)

    def test_shock_detection(self):
        """High jerk + high peak accel should → SHOCK."""
        features = {
            "accel_mean": 12.0,
            "accel_var": 15.0,
            "accel_rms": 15.0,
            "accel_peak": 25.0,
            "gyro_rms": 0.3,
            "gyro_peak": 0.8,
            "jerk_rms": 50.0,
            "freq_energy_ratio": 0.3,
            "spectral_entropy": 0.6,
        }
        self.assertEqual(self.classifier.classify_window(features), MotionLabel.SHOCK)

    def test_vibration_detection(self):
        """High freq energy + high entropy + high variance should → VIBRATION."""
        features = {
            "accel_mean": 10.5,
            "accel_var": 5.0,
            "accel_rms": 11.0,
            "accel_peak": 15.0,
            "gyro_rms": 0.2,
            "gyro_peak": 0.5,
            "jerk_rms": 10.0,
            "freq_energy_ratio": 0.7,
            "spectral_entropy": 0.9,
        }
        self.assertEqual(self.classifier.classify_window(features), MotionLabel.VIBRATION)

    def test_abnormal_detection(self):
        """Extreme gyro + extreme accel should → ABNORMAL."""
        features = {
            "accel_mean": 15.0,
            "accel_var": 20.0,
            "accel_rms": 18.0,
            "accel_peak": 30.0,
            "gyro_rms": 2.0,
            "gyro_peak": 5.0,
            "jerk_rms": 60.0,
            "freq_energy_ratio": 0.4,
            "spectral_entropy": 0.7,
        }
        self.assertEqual(self.classifier.classify_window(features), MotionLabel.ABNORMAL)

    def test_trust_weights(self):
        """Trust weight mapping should return expected values."""
        self.assertAlmostEqual(self.classifier.get_trust_weight(MotionLabel.NORMAL), 1.0)
        self.assertAlmostEqual(self.classifier.get_trust_weight(MotionLabel.STATIONARY), 1.0)
        self.assertAlmostEqual(self.classifier.get_trust_weight(MotionLabel.SHOCK), 0.1)
        self.assertAlmostEqual(self.classifier.get_trust_weight(MotionLabel.VIBRATION), 0.4)
        self.assertAlmostEqual(self.classifier.get_trust_weight(MotionLabel.ABNORMAL), 0.0)

    def test_classify_feature_dataframe(self):
        """Bulk classification on a feature DataFrame should return correct types."""
        rows = [
            {"accel_mean": 9.81, "accel_var": 0.001, "accel_rms": 9.81,
             "accel_peak": 9.82, "gyro_rms": 0.01, "gyro_peak": 0.02,
             "jerk_rms": 0.1, "freq_energy_ratio": 0.1, "spectral_entropy": 0.3,
             "window_start_idx": 0, "window_end_idx": 9, "timestamp": 0.5},
            {"accel_mean": 9.85, "accel_var": 0.5, "accel_rms": 10.0,
             "accel_peak": 12.0, "gyro_rms": 0.15, "gyro_peak": 0.5,
             "jerk_rms": 5.0, "freq_energy_ratio": 0.2, "spectral_entropy": 0.5,
             "window_start_idx": 5, "window_end_idx": 14, "timestamp": 1.0},
        ]
        feat_df = pd.DataFrame(rows)
        labels = self.classifier.classify_feature_dataframe(feat_df)

        self.assertEqual(len(labels), 2)
        self.assertEqual(int(labels.iloc[0]), int(MotionLabel.STATIONARY))
        self.assertEqual(int(labels.iloc[1]), int(MotionLabel.NORMAL))

    def test_label_distribution(self):
        """Label distribution should sum to total count."""
        labels = pd.Series([0, 0, 1, 1, 2, 3, 0, 0, 4, 0])
        dist = MotionClassifier.label_distribution(labels)
        self.assertEqual(sum(dist.values()), len(labels))
        self.assertEqual(dist["NORMAL"], 5)
        self.assertEqual(dist["STATIONARY"], 2)
        self.assertEqual(dist["SHOCK"], 1)


class TestEndToEnd(unittest.TestCase):
    """Integration test: extractor → classifier pipeline."""

    def test_stationary_pipeline(self):
        """Constant gravity → features → classifier → STATIONARY."""
        n = 50
        accel = np.zeros((n, 3), dtype=np.float64)
        accel[:, 2] = 9.81
        gyro = np.zeros((n, 3), dtype=np.float64)

        extractor = MotionFeatureExtractor(
            sampling_rate_hz=10.0,
            window_size_sec=1.0,
            step_size_sec=1.0,
        )
        features = extractor.extract(accel, gyro)

        classifier = MotionClassifier()
        labels = classifier.classify_feature_dataframe(features)

        # All windows should be STATIONARY
        for lbl in labels:
            self.assertEqual(int(lbl), int(MotionLabel.STATIONARY),
                             f"Expected STATIONARY, got {MotionLabel(lbl).name}")

    def test_normal_driving_pipeline(self):
        """Smooth acceleration profile → features → NORMAL."""
        np.random.seed(123)
        n = 100
        t = np.arange(n) / 10.0

        accel = np.zeros((n, 3), dtype=np.float64)
        accel[:, 0] = 0.5 * np.sin(0.5 * t)   # gentle forward acceleration
        accel[:, 2] = 9.81 + 0.1 * np.random.randn(n)
        gyro = np.zeros((n, 3), dtype=np.float64)
        gyro[:, 2] = 0.1 * np.sin(0.3 * t)  # gentle yaw rate

        extractor = MotionFeatureExtractor(sampling_rate_hz=10.0)
        features = extractor.extract(accel, gyro)

        classifier = MotionClassifier()
        labels = classifier.classify_feature_dataframe(features)

        # Majority should be NORMAL
        normal_pct = (labels == int(MotionLabel.NORMAL)).mean()
        self.assertGreater(normal_pct, 0.5,
                           f"Expected majority NORMAL, got {normal_pct:.1%}")

    def test_vectorized_classify_matches_scalar(self):
        """Vectorized classify_feature_dataframe must match scalar classify_window exactly."""
        np.random.seed(42)
        n = 500
        classifier = MotionClassifier()
        t = classifier.thresholds

        # Generate wide feature ranges to hit all motion categories
        feat_df = pd.DataFrame({
            "accel_var": np.random.uniform(0, t.vibration_accel_var_min * 3, n),
            "accel_mean": np.random.uniform(t.stationary_accel_mean_min - 2, t.stationary_accel_mean_max + 2, n),
            "accel_peak": np.random.uniform(0, t.abnormal_accel_peak_min * 1.5, n),
            "gyro_rms": np.random.uniform(0, t.stationary_gyro_rms_max * 3, n),
            "gyro_peak": np.random.uniform(0, t.abnormal_gyro_peak_min * 1.5, n),
            "jerk_rms": np.random.uniform(0, t.shock_jerk_rms_min * 2, n),
            "freq_energy_ratio": np.random.uniform(0, 1.0, n),
            "spectral_entropy": np.random.uniform(0, 1.0, n),
        })

        vectorized_labels = classifier.classify_feature_dataframe(feat_df)
        scalar_labels = [int(classifier.classify_window(row.to_dict())) for _, row in feat_df.iterrows()]

        self.assertEqual(list(vectorized_labels), scalar_labels)

    def test_label_source_dataframe_reused_window_labels(self):
        """label_source_dataframe with precomputed labels matches default execution."""
        extractor = MotionFeatureExtractor(sampling_rate_hz=10.0, window_size_sec=1.0, step_size_sec=0.5)
        accel = np.random.randn(100, 3)
        gyro = np.random.randn(100, 3) * 0.1
        feat_df = extractor.extract(accel, gyro)
        dummy_source = pd.DataFrame(index=np.arange(100))

        classifier = MotionClassifier()
        win_labels = classifier.classify_feature_dataframe(feat_df)

        labels_default = classifier.label_source_dataframe(dummy_source, feat_df)
        labels_reused = classifier.label_source_dataframe(dummy_source, feat_df, window_labels=win_labels)

        pd.testing.assert_series_equal(labels_default, labels_reused)

    def test_spectral_features_consistency(self):
        """_spectral_features must yield identical results to helper functions."""
        extractor = MotionFeatureExtractor(sampling_rate_hz=10.0)
        sig = np.random.randn(50)
        ratio, entropy = extractor._spectral_features(sig)
        self.assertAlmostEqual(ratio, extractor._freq_energy_ratio(sig))
        self.assertAlmostEqual(entropy, extractor._spectral_entropy(sig))


if __name__ == "__main__":
    unittest.main()
