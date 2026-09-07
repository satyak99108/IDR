"""
IDR MVP -- Unit Tests for Evaluation Package (Phase 12)
========================================================
Tests ResourceProfiler, RecoveryAnalyzer, and BenchmarkRunner with synthetic data.
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.resource_profiler import ResourceProfiler, ResourceProfile
from src.evaluation.recovery_analyzer import RecoveryAnalyzer, RecoveryMetrics
from src.evaluation.benchmark import (
    BenchmarkRunner,
    StageMetrics,
    OutageResult,
    TripResult,
    BenchmarkSummary,
    _compute_position_metrics,
    _compute_velocity_metrics,
    _compute_heading_metrics,
    _compute_outage_distance,
)


class TestResourceProfiler(unittest.TestCase):
    """Tests for the ResourceProfiler context manager."""

    def test_context_manager_records_wall_time(self):
        """Profiler should capture positive wall-clock time."""
        profiler = ResourceProfiler()
        with profiler:
            time.sleep(0.05)  # 50ms
        profiler.set_samples_processed(100)
        profile = profiler.get_profile()

        self.assertGreater(profile.wall_time_s, 0.01)
        self.assertEqual(profile.samples_processed, 100)
        self.assertGreater(profile.throughput_hz, 0)
        self.assertGreater(profile.latency_per_sample_ms, 0)

    def test_model_file_sizes(self):
        """Profiler should measure model file sizes if models_dir exists."""
        models_dir = PROJECT_ROOT / "models"
        if not models_dir.exists():
            self.skipTest("models/ directory not found")

        profiler = ResourceProfiler(models_dir=models_dir)
        with profiler:
            pass
        profile = profiler.get_profile()

        # Should find at least one model file
        self.assertGreater(len(profile.model_files), 0)
        for fname, size_kb in profile.model_files.items():
            self.assertGreater(size_kb, 0)

    def test_to_dict(self):
        """ResourceProfile.to_dict() should return a serializable dict."""
        profile = ResourceProfile(
            wall_time_s=1.234,
            peak_ram_mb=256.5,
            mean_cpu_pct=45.2,
            samples_processed=1000,
            throughput_hz=810.0,
            latency_per_sample_ms=1.23,
            model_files={"test.npz": 100.5},
        )
        d = profile.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["samples_processed"], 1000)
        # Must be JSON serializable
        json.dumps(d)


class TestRecoveryAnalyzer(unittest.TestCase):
    """Tests for the GNSS RecoveryAnalyzer."""

    def test_perfect_recovery(self):
        """Analyzer should detect immediate convergence when error drops to 0."""
        n = 500
        pred_lats = np.full(n, 28.6139)
        pred_lons = np.full(n, 77.2090)
        ref_lats = np.full(n, 28.6139)
        ref_lons = np.full(n, 77.2090)

        analyzer = RecoveryAnalyzer(
            convergence_threshold_m=10.0,
            jump_limit_m=50.0,
            recovery_window_s=10.0,
            sampling_rate_hz=10.0,
        )
        result = analyzer.analyze(pred_lats, pred_lons, ref_lats, ref_lons, outage_end_idx=200)

        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.convergence_time_s, 0.0, places=1)
        self.assertTrue(result.is_jump_free)
        self.assertAlmostEqual(result.error_at_outage_end_m, 0.0, places=1)

    def test_no_convergence(self):
        """Analyzer should report non-convergence when error stays high."""
        n = 500
        pred_lats = np.full(n, 28.6139)
        pred_lons = np.full(n, 77.2090)
        ref_lats = np.full(n, 28.6139)
        ref_lons = np.full(n, 77.2090)

        # Offset prediction by ~100m after outage
        pred_lats[200:] += 0.001  # ~111m offset

        analyzer = RecoveryAnalyzer(
            convergence_threshold_m=10.0,
            recovery_window_s=5.0,
            sampling_rate_hz=10.0,
        )
        result = analyzer.analyze(pred_lats, pred_lons, ref_lats, ref_lons, outage_end_idx=200)

        self.assertFalse(result.converged)
        self.assertGreater(result.error_at_outage_end_m, 50.0)

    def test_gradual_convergence(self):
        """Analyzer should measure convergence time for gradually improving error."""
        n = 500
        ref_lats = np.full(n, 28.6139)
        ref_lons = np.full(n, 77.2090)
        pred_lats = ref_lats.copy()
        pred_lons = ref_lons.copy()

        # Start with offset, linearly converge over 3s (30 samples at 10Hz)
        outage_end = 200
        for i in range(30):
            frac = 1.0 - (i / 30.0)
            pred_lats[outage_end + i] += 0.0005 * frac  # ~55m → 0m over 30 samples

        analyzer = RecoveryAnalyzer(
            convergence_threshold_m=10.0,
            recovery_window_s=10.0,
            sampling_rate_hz=10.0,
        )
        result = analyzer.analyze(pred_lats, pred_lons, ref_lats, ref_lons, outage_end_idx=outage_end)

        self.assertTrue(result.converged)
        self.assertGreater(result.convergence_time_s, 0.0)
        self.assertLess(result.convergence_time_s, 5.0)

    def test_to_dict_serializable(self):
        """RecoveryMetrics.to_dict() should be JSON serializable."""
        metrics = RecoveryMetrics(
            outage_end_idx=100,
            error_at_outage_end_m=25.3,
            convergence_time_s=2.5,
            converged=True,
            convergence_threshold_m=10.0,
            max_single_step_jump_m=5.0,
            is_jump_free=True,
            jump_limit_m=50.0,
            error_after_5s_m=8.0,
            error_after_10s_m=3.0,
        )
        d = metrics.to_dict()
        json.dumps(d)


class TestMetricHelpers(unittest.TestCase):
    """Tests for the benchmark metric computation functions."""

    def test_compute_position_metrics_zero_error(self):
        """Position metrics should be zero for identical predictions."""
        n = 100
        lats = np.full(n, 28.6139)
        lons = np.full(n, 77.2090)
        m = _compute_position_metrics(lats, lons, lats, lons, 10, 50, 500.0)
        self.assertAlmostEqual(m["final_error_m"], 0.0, places=1)
        self.assertAlmostEqual(m["drift_pct"], 0.0, places=1)

    def test_compute_position_metrics_nonzero(self):
        """Position metrics should be positive for offset predictions."""
        n = 100
        ref_lats = np.full(n, 28.6139)
        ref_lons = np.full(n, 77.2090)
        pred_lats = ref_lats.copy()
        pred_lons = ref_lons.copy()
        pred_lats[10:51] += 0.0001  # ~11m offset

        m = _compute_position_metrics(pred_lats, pred_lons, ref_lats, ref_lons, 10, 50, 500.0)
        self.assertGreater(m["final_error_m"], 5.0)
        self.assertGreater(m["drift_pct"], 0.0)

    def test_compute_velocity_metrics(self):
        """Velocity metrics should compute MAE and RMSE."""
        pred = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        ref = np.array([10.0, 10.5, 12.0, 13.5, 14.0])
        mae, rmse = _compute_velocity_metrics(pred, ref, 0, 4)
        self.assertIsNotNone(mae)
        self.assertIsNotNone(rmse)
        self.assertGreater(mae, 0.0)

    def test_compute_velocity_metrics_nan_handling(self):
        """Velocity metrics should handle NaN values gracefully."""
        pred = np.array([10.0, np.nan, 12.0])
        ref = np.array([10.0, 10.5, np.nan])
        mae, rmse = _compute_velocity_metrics(pred, ref, 0, 2)
        # Should still compute for the one valid pair
        self.assertIsNotNone(mae)

    def test_compute_heading_metrics(self):
        """Heading metrics should handle wraparound correctly."""
        pred = np.array([5.0, 355.0, 180.0])
        ref = np.array([0.0, 0.0, 170.0])
        mae = _compute_heading_metrics(pred, ref, 0, 2)
        self.assertIsNotNone(mae)
        # 5deg, 5deg (355→0 is 5deg diff), 10deg → mean = 6.67
        self.assertLess(mae, 15.0)

    def test_compute_outage_distance(self):
        """Outage distance should be positive for a moving trajectory."""
        n = 100
        lats = np.linspace(28.6139, 28.6200, n)
        lons = np.full(n, 77.2090)
        dist = _compute_outage_distance(lats, lons, 0, n - 1)
        self.assertGreater(dist, 100.0)  # ~678m for 0.0061 deg latitude


class TestStageMetrics(unittest.TestCase):
    """Tests for StageMetrics dataclass."""

    def test_to_dict(self):
        sm = StageMetrics(
            stage_name="Test",
            final_error_m=10.5,
            max_error_m=20.3,
            mean_error_m=8.2,
            rmse_error_m=9.1,
            drift_pct=5.5,
            velocity_mae_ms=1.2,
            heading_mae_deg=3.5,
        )
        d = sm.to_dict()
        self.assertEqual(d["stage_name"], "Test")
        self.assertEqual(d["drift_pct"], 5.5)
        self.assertIn("velocity_mae_ms", d)
        self.assertIn("heading_mae_deg", d)
        json.dumps(d)


class TestTripResult(unittest.TestCase):
    """Tests for TripResult dataclass."""

    def test_all_passed_true(self):
        trip = TripResult(trip_name="test", n_samples=100, duration_s=10.0)
        trip.outage_results = [
            OutageResult(outage_duration_s=60.0, outage_start_s=300.0, outage_distance_m=500.0, target_met=True),
            OutageResult(outage_duration_s=30.0, outage_start_s=300.0, outage_distance_m=250.0, target_met=True),
        ]
        self.assertTrue(trip.all_passed)

    def test_all_passed_false(self):
        trip = TripResult(trip_name="test", n_samples=100, duration_s=10.0)
        trip.outage_results = [
            OutageResult(outage_duration_s=60.0, outage_start_s=300.0, outage_distance_m=500.0, target_met=True),
            OutageResult(outage_duration_s=120.0, outage_start_s=300.0, outage_distance_m=1000.0, target_met=False),
        ]
        self.assertFalse(trip.all_passed)


class TestBenchmarkSummary(unittest.TestCase):
    """Tests for BenchmarkSummary serialization."""

    def test_empty_summary_serializable(self):
        summary = BenchmarkSummary()
        d = summary.to_dict()
        json.dumps(d)
        self.assertEqual(d["num_trips"], 0)

    def test_summary_with_data(self):
        summary = BenchmarkSummary(
            num_trips=2,
            num_outages_total=8,
            num_passed=6,
            num_failed=2,
            mean_drift_pct_60s=7.5,
            best_drift_pct_60s=5.2,
            worst_drift_pct_60s=12.1,
            mean_throughput_hz=350.0,
            mean_latency_ms=2.86,
            peak_ram_mb=450.0,
            model_total_kb=800.0,
        )
        d = summary.to_dict()
        json.dumps(d)
        self.assertEqual(d["num_passed"], 6)


if __name__ == "__main__":
    unittest.main()
