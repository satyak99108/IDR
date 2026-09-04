import unittest
import numpy as np
import pandas as pd
from src.ins.integration import haversine_distance
from scripts.run_phase3 import compute_drift_metrics as compute_drift_metrics_p3, compute_distance_series as compute_distance_series_p3
from scripts.run_phase7 import compute_position_errors as compute_position_errors_p7, compute_drift_metrics as compute_drift_metrics_p7


class TestVectorizedMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        self.traj_df = pd.DataFrame({
            "lat_deg": 37.7749 + np.cumsum(np.random.randn(n) * 1e-4),
            "lon_deg": -122.4194 + np.cumsum(np.random.randn(n) * 1e-4),
            "speed_ms": np.random.uniform(5, 20, n),
        })
        self.ref_df = pd.DataFrame({
            "ref_lat": self.traj_df["lat_deg"] + np.random.randn(n) * 1e-5,
            "ref_lon": self.traj_df["lon_deg"] + np.random.randn(n) * 1e-5,
        })
        # Add some NaNs to test robustness
        self.ref_df.loc[10:12, "ref_lat"] = np.nan
        self.ref_df.loc[10:12, "ref_lon"] = np.nan

    def test_phase3_drift_metrics(self):
        metrics, errors_m = compute_drift_metrics_p3(self.traj_df, self.ref_df)
        self.assertIn("rmse_m", metrics)
        self.assertIn("mae_m", metrics)
        self.assertIn("drift_percent", metrics)
        self.assertTrue(np.isnan(errors_m[10]))
        self.assertFalse(np.isnan(errors_m[0]))
        self.assertEqual(len(errors_m), len(self.traj_df))
        self.assertGreater(metrics["total_distance_m"], 0.0)

    def test_phase3_distance_series(self):
        dist_series = compute_distance_series_p3(self.ref_df)
        self.assertEqual(len(dist_series), len(self.ref_df))
        self.assertEqual(dist_series[0], 0.0)
        # Monotonically non-decreasing
        self.assertTrue(np.all(np.diff(dist_series) >= 0.0))

    def test_phase7_position_errors(self):
        errors = compute_position_errors_p7(self.traj_df, self.ref_df)
        self.assertEqual(len(errors), len(self.traj_df))
        self.assertTrue(np.isnan(errors[10]))
        self.assertFalse(np.isnan(errors[0]))

    def test_phase7_drift_metrics(self):
        errors = compute_position_errors_p7(self.traj_df, self.ref_df)
        metrics = compute_drift_metrics_p7(errors, self.ref_df)
        self.assertIn("rmse_m", metrics)
        self.assertIn("mae_m", metrics)
        self.assertIn("drift_percent", metrics)
        self.assertGreater(metrics["total_distance_m"], 0.0)

    def test_phase8_vectorized_calculations(self):
        n_samples = len(self.traj_df)
        fused_lat = np.radians(self.traj_df["lat_deg"].to_numpy(dtype=float)[:n_samples])
        fused_lon = np.radians(self.traj_df["lon_deg"].to_numpy(dtype=float)[:n_samples])
        ref_lat = self.ref_df["ref_lat"].fillna(37.77).to_numpy(dtype=float)
        ref_lon = self.ref_df["ref_lon"].fillna(-122.41).to_numpy(dtype=float)
        r_lat_rad = np.radians(ref_lat[:n_samples])
        r_lon_rad = np.radians(ref_lon[:n_samples])
        full_errors = haversine_distance(fused_lat, fused_lon, r_lat_rad, r_lon_rad)
        self.assertEqual(len(full_errors), n_samples)
        self.assertFalse(np.any(np.isnan(full_errors)))

        start_idx, end_idx = 10, 50
        outage_ref_lat = r_lat_rad[start_idx : end_idx + 1]
        outage_ref_lon = r_lon_rad[start_idx : end_idx + 1]
        outage_step_dists = haversine_distance(
            outage_ref_lat[:-1], outage_ref_lon[:-1],
            outage_ref_lat[1:],  outage_ref_lon[1:]
        )
        outage_dist = float(np.sum(outage_step_dists))
        self.assertGreater(outage_dist, 0.0)


if __name__ == "__main__":
    unittest.main()
