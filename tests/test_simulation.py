"""
Unit Tests -- Simulation Package (Phase 4)
===========================================
Tests GNSSOutageSimulator, OutageWindow, OutageScenario, and OutageEvaluator.
"""

import unittest
import numpy as np
import pandas as pd

from src.simulation import (
    OutageWindow,
    OutageScenario,
    GNSSOutageSimulator,
    OutageEvaluator,
    OutageMetrics,
    ScenarioMetrics,
)


class TestGNSSOutageSimulator(unittest.TestCase):

    def setUp(self):
        # Create synthetic 10 Hz dataset for 150 seconds (1500 samples)
        n_samples = 1500
        time_ms = np.arange(0, n_samples * 100, 100, dtype=np.int64)
        
        # Vehicle moving north at ~10 m/s (~0.00009 deg lat per second)
        base_lat = 45.0
        base_lon = 7.0
        lat_step = (10.0 / 111319.5) / 10.0  # 10 m/s at 10 Hz
        
        ref_lat = base_lat + np.arange(n_samples) * lat_step
        ref_lon = np.full(n_samples, base_lon)
        ref_speed = np.full(n_samples, 10.0)
        
        self.df = pd.DataFrame({
            "timestamp_ms": time_ms,
            "ref_lat": ref_lat,
            "ref_lon": ref_lon,
            "ref_speed": ref_speed,
            "ref_heading": np.zeros(n_samples),
            "gnss_lat": ref_lat + np.random.normal(0, 1e-6, n_samples),
            "gnss_lon": ref_lon + np.random.normal(0, 1e-6, n_samples),
            "gnss_speed": ref_speed + np.random.normal(0, 0.1, n_samples),
            "gnss_heading": np.zeros(n_samples),
            "gnss_accuracy": np.full(n_samples, 3.0),
        })
        self.simulator = GNSSOutageSimulator()

    def test_create_standard_mvp_scenario(self):
        scenario = self.simulator.create_standard_mvp_scenario(start_sec=30.0, duration_sec=30.0)
        self.assertEqual(scenario.name, "mvp_standard_30s_tunnel")
        self.assertEqual(scenario.num_outages, 1)
        self.assertEqual(scenario.windows[0].start_time_s, 30.0)
        self.assertEqual(scenario.windows[0].end_time_s, 60.0)
        self.assertEqual(scenario.windows[0].duration_s, 30.0)

    def test_apply_outage_mask(self):
        scenario = self.simulator.create_standard_mvp_scenario(start_sec=30.0, duration_sec=30.0)
        df_masked, resolved_scenario = self.simulator.apply_outage_mask(self.df, scenario)

        # Check new columns
        self.assertIn("is_outage", df_masked.columns)
        self.assertIn("gnss_available", df_masked.columns)
        self.assertIn("outage_id", df_masked.columns)
        self.assertIn("outage_label", df_masked.columns)
        self.assertIn("time_elapsed_s", df_masked.columns)

        # Check outside blackout (e.g. t = 10s -> sample 100)
        self.assertFalse(df_masked.loc[100, "is_outage"])
        self.assertEqual(df_masked.loc[100, "gnss_available"], 1)
        self.assertEqual(df_masked.loc[100, "outage_id"], 0)
        self.assertFalse(np.isnan(df_masked.loc[100, "gnss_lat"]))

        # Check inside blackout (e.g. t = 45s -> sample 450)
        self.assertTrue(df_masked.loc[450, "is_outage"])
        self.assertEqual(df_masked.loc[450, "gnss_available"], 0)
        self.assertEqual(df_masked.loc[450, "outage_id"], 1)
        self.assertTrue(np.isnan(df_masked.loc[450, "gnss_lat"]))
        self.assertTrue(np.isnan(df_masked.loc[450, "gnss_speed"]))

        # Check ground truth is preserved
        self.assertFalse(np.isnan(df_masked.loc[450, "ref_lat"]))
        self.assertFalse(np.isnan(df_masked.loc[450, "ref_speed"]))

        # Check resolved window has indices and distances
        w = resolved_scenario.windows[0]
        self.assertIsNotNone(w.start_idx)
        self.assertIsNotNone(w.end_idx)
        self.assertIsNotNone(w.start_dist_m)
        self.assertIsNotNone(w.end_dist_m)
        self.assertGreater(w.end_dist_m, w.start_dist_m)

    def test_multi_outage_scenario(self):
        scenario = self.simulator.create_multi_outage_scenario()
        df_masked, resolved = self.simulator.apply_outage_mask(self.df, scenario)
        self.assertEqual(resolved.num_outages, 3)
        self.assertEqual(len(df_masked["outage_id"].unique()), 4)  # 0, 1, 2, 3

    def test_periodic_scenario(self):
        scenario = self.simulator.create_periodic_scenario(
            total_duration_s=100.0,
            outage_duration_s=10.0,
            gap_duration_s=20.0,
            initial_offset_s=10.0,
        )
        self.assertGreater(scenario.num_outages, 1)

    def test_stochastic_scenario(self):
        scenario = self.simulator.create_stochastic_scenario(
            total_duration_s=100.0,
            num_outages=3,
            seed=123,
        )
        self.assertGreaterEqual(scenario.num_outages, 1)


class TestOutageEvaluator(unittest.TestCase):

    def setUp(self):
        n = 300  # 30 seconds at 10 Hz
        time_s = np.linspace(0, 30, n)
        ref_lat = 45.0 + np.linspace(0, 0.0027, n)  # ~300 meters
        ref_lon = np.full(n, 7.0)

        # Simulated prediction drifting linearly up to 15 meters at the end
        drift_lat = np.linspace(0, 15.0 / 111319.5, n)
        pred_lat = ref_lat + drift_lat
        pred_lon = ref_lon.copy()

        self.df_slice = pd.DataFrame({
            "time_elapsed_s": time_s,
            "ref_lat": ref_lat,
            "ref_lon": ref_lon,
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "ref_speed": np.full(n, 10.0),
            "pred_speed": np.full(n, 10.2),
            "ref_heading": np.zeros(n),
            "pred_heading": np.full(n, 1.5),
            "outage_id": np.ones(n, dtype=int),
            "outage_label": ["30s Tunnel"] * n,
        })

    def test_evaluate_outage_window(self):
        metrics = OutageEvaluator.evaluate_outage_window(
            self.df_slice,
            window_id=1,
            label="30s Tunnel",
        )
        self.assertAlmostEqual(metrics.duration_s, 30.0, places=1)
        self.assertAlmostEqual(metrics.distance_travelled_m, 300.5, delta=5.0)
        self.assertAlmostEqual(metrics.endpoint_error_m, 15.0, delta=1.0)
        # Drift % = 15m / 300m * 100 ≈ 5%
        self.assertAlmostEqual(metrics.drift_percent, 5.0, delta=1.0)
        self.assertGreater(metrics.rmse_error_m, 0.0)
        self.assertAlmostEqual(metrics.velocity_mae_ms, 0.2, places=2)
        self.assertAlmostEqual(metrics.heading_mae_deg, 1.5, places=2)

    def test_evaluate_scenario(self):
        scenario_metrics = OutageEvaluator.evaluate_scenario(
            self.df_slice,
            scenario_name="Test Scenario",
            description="Test Description",
        )
        self.assertEqual(scenario_metrics.num_outages, 1)
        self.assertAlmostEqual(scenario_metrics.mean_drift_percent, 5.0, delta=1.0)
        self.assertAlmostEqual(scenario_metrics.weighted_drift_percent, 5.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
