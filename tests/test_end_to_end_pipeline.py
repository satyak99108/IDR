"""
IDR MVP -- Unit Tests for End-to-End Navigation Pipeline (Phase 11)
===================================================================
Tests IDRNavigationEngine, PipelineConfig, SensorInput, and NavigationOutput.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    IDRNavigationEngine,
    PipelineConfig,
    SensorInput,
    NavigationOutput,
)
from src.map_matching import RoadGraph, RoadSegment
from src.fusion import NavigationMode


def _build_synthetic_road_graph() -> RoadGraph:
    """Builds a simple synthetic road graph for testing."""
    rg = RoadGraph(cache_dir=None)

    # Road 1: E-W road along lat=28.6139 from lon 77.200 to 77.220
    def make_seg(edge_id, lat_s, lon_s, lat_e, lon_e, name="Test Rd"):
        lat_m = (lat_s + lat_e) / 2.0
        lon_m = (lon_s + lon_e) / 2.0
        bearing = RoadGraph._compute_bearing(lat_s, lon_s, lat_e, lon_e)
        dlat_m = (lat_e - lat_s) * 111_320.0
        dlon_m = (lon_e - lon_s) * 111_320.0 * np.cos(np.radians(lat_m))
        length = float(np.sqrt(dlat_m ** 2 + dlon_m ** 2))
        return RoadSegment(
            edge_id=edge_id,
            u_node=edge_id[0],
            v_node=edge_id[1],
            lat_start=lat_s,
            lon_start=lon_s,
            lat_end=lat_e,
            lon_end=lon_e,
            lat_mid=lat_m,
            lon_mid=lon_m,
            bearing_deg=bearing,
            length_m=length,
            road_type="primary",
            name=name,
            oneway=False,
        )

    rg.segments = [
        make_seg((1, 2, 0), 28.6139, 77.2000, 28.6139, 77.2200, "Rajpath"),
        make_seg((2, 1, 0), 28.6139, 77.2200, 28.6139, 77.2000, "Rajpath (Rev)"),
    ]
    rg._build_spatial_index()
    rg._loaded = True
    return rg


def _create_synthetic_trip_df(n_samples: int = 50) -> pd.DataFrame:
    """Creates a realistic synthetic vehicle trip DataFrame."""
    t_ms = np.arange(1000, 1000 + n_samples * 100, 100)  # 10 Hz
    # Vehicle moving East along lat=28.61395 (~5m north of Road 1)
    lons = np.linspace(77.2050, 77.2150, n_samples)
    lats = np.full(n_samples, 28.61395)
    speeds = np.full(n_samples, 12.0)  # ~43 km/h
    headings = np.full(n_samples, 90.0)

    # Accelerations & Gyros
    ax_lin = np.zeros(n_samples)
    ay_lin = np.zeros(n_samples)
    az_lin = np.zeros(n_samples)
    gx = np.zeros(n_samples)
    gy = np.zeros(n_samples)
    gz = np.zeros(n_samples)

    return pd.DataFrame({
        "timestamp": t_ms,
        "ax_veh_lin": ax_lin,
        "ay_veh_lin": ay_lin,
        "az_veh_lin": az_lin,
        "ax_veh": ax_lin,
        "ay_veh": ay_lin,
        "az_veh": az_lin + 9.81,
        "gx_veh": gx,
        "gy_veh": gy,
        "gz_veh": gz,
        "gnss_lat": lats,
        "gnss_lon": lons,
        "gnss_speed": speeds,
        "heading": headings,
        "ref_heading": headings,
        "ALTITUDE (m)": np.full(n_samples, 215.0),
        "_extra_GPS ACCURACY (m)": np.full(n_samples, 3.5),
        "_extra_GPS SATELLITES IN RANGE": np.full(n_samples, 12),
    })


class TestPipelineInitialization(unittest.TestCase):
    """Tests IDRNavigationEngine instantiation and configuration."""

    def test_default_initialization(self):
        engine = IDRNavigationEngine()
        self.assertIsNotNone(engine.config)
        self.assertTrue(engine.config.use_ai_velocity)
        self.assertTrue(engine.config.use_map_matching)
        self.assertEqual(engine.config.window_size, 50)

    def test_custom_config(self):
        cfg = PipelineConfig(
            use_ai_velocity=False,
            use_map_matching=False,
            recovery_window_s=5.0,
            road_search_radius_m=100.0,
        )
        engine = IDRNavigationEngine(config=cfg)
        self.assertFalse(engine.config.use_ai_velocity)
        self.assertFalse(engine.config.use_map_matching)
        self.assertEqual(engine.config.recovery_window_s, 5.0)

    def test_reset(self):
        engine = IDRNavigationEngine()
        engine._prev_timestamp_ms = 5000
        engine._origin_initialized = True
        engine._imu_window_buffer.append(np.zeros(6))

        engine.reset()
        self.assertIsNone(engine._prev_timestamp_ms)
        self.assertFalse(engine._origin_initialized)
        self.assertEqual(len(engine._imu_window_buffer), 0)


class TestPipelineBatchProcessing(unittest.TestCase):
    """Tests batch trip processing through the end-to-end pipeline."""

    def setUp(self):
        self.rg = _build_synthetic_road_graph()
        self.trip_df = _create_synthetic_trip_df(n_samples=40)
        self.engine = IDRNavigationEngine(road_graph=self.rg)

    def test_batch_output_schema_and_columns(self):
        result_df = self.engine.process_trip(self.trip_df)

        self.assertFalse(result_df.empty)
        self.assertEqual(len(result_df), len(self.trip_df))

        required_cols = [
            "timestamp_ms", "lat_deg", "lon_deg", "speed_ms", "heading_deg",
            "mode", "confidence", "final_lat", "final_lon", "is_map_matched",
            "mm_lat", "mm_lon", "mm_road_id", "mm_snap_dist_m",
        ]
        for col in required_cols:
            self.assertIn(col, result_df.columns, f"Missing required output column: {col}")

    def test_map_matching_snapping(self):
        # The vehicle travels 5m north of Road 1 (lat=28.61395 vs road lat=28.61390)
        # Search radius is 80m so it should snap onto the road
        cfg = PipelineConfig(road_search_radius_m=800.0, map_matching_confidence_threshold=0.1)
        engine = IDRNavigationEngine(config=cfg, road_graph=self.rg)
        result_df = engine.process_trip(self.trip_df)

        self.assertTrue(result_df["is_map_matched"].any())
        # Snapped latitude should be close to road latitude (28.61390)
        matched_lats = result_df.loc[result_df["is_map_matched"], "final_lat"]
        self.assertAlmostEqual(float(matched_lats.iloc[-1]), 28.61390, places=4)

    def test_outage_handling_and_mode_transitions(self):
        # Create a blackout mask for the middle 20 samples (index 10 to 30)
        n = len(self.trip_df)
        outage_mask = np.zeros(n, dtype=bool)
        outage_mask[10:30] = True

        cfg = PipelineConfig(
            road_search_radius_m=800.0,
            dropout_debounce_samples=3,
            recovery_confirm_samples=2,
            recovery_window_s=1.0,
        )
        engine = IDRNavigationEngine(config=cfg, road_graph=self.rg)
        result_df = engine.process_trip(self.trip_df, outage_mask=outage_mask)

        # Confirm DEAD_RECKONING was activated
        modes = result_df["mode"].tolist()
        self.assertIn(NavigationMode.DEAD_RECKONING.value, modes)

        # Check that after outage ends, it transitions into RECOVERY or GNSS_INS
        post_outage_modes = modes[32:]
        self.assertTrue(
            any(m in [NavigationMode.RECOVERY.value, NavigationMode.GNSS_INS.value] for m in post_outage_modes)
        )

        # Output coordinates must be valid (not NaN or Inf) throughout outage
        self.assertFalse(result_df["final_lat"].isna().any())
        self.assertFalse(result_df["final_lon"].isna().any())

    def test_fallback_without_map_matching(self):
        cfg = PipelineConfig(use_map_matching=False)
        engine = IDRNavigationEngine(config=cfg)
        result_df = engine.process_trip(self.trip_df)

        self.assertIn("final_lat", result_df.columns)
        self.assertIn("final_lon", result_df.columns)
        # Without map matching, final_lat should equal lat_deg
        np.testing.assert_allclose(result_df["final_lat"].values, result_df["lat_deg"].values)


class TestPipelineStreaming(unittest.TestCase):
    """Tests real-time step-by-step streaming execution."""

    def setUp(self):
        self.rg = _build_synthetic_road_graph()
        self.cfg = PipelineConfig(road_search_radius_m=800.0, window_size=5)
        self.engine = IDRNavigationEngine(config=self.cfg, road_graph=self.rg)

    def test_step_streaming_output(self):
        # Step 0: First sample with GNSS to initialize origin
        s0 = SensorInput(
            timestamp_ms=1000,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            ax_lin=0.0, ay_lin=0.0, az_lin=0.0,
            gnss_lat=28.61395,
            gnss_lon=77.2100,
            gnss_alt=215.0,
            gnss_speed=12.0,
            gnss_course=90.0,
            gnss_accuracy=2.5,
            gnss_sats=12,
        )
        out0 = self.engine.step(s0)
        self.assertIsInstance(out0, NavigationOutput)
        self.assertEqual(out0.timestamp_ms, 1000)
        self.assertAlmostEqual(out0.lat_deg, 28.61395, places=5)
        self.assertAlmostEqual(out0.lon_deg, 77.2100, places=4)

        # Stream successive samples
        for i in range(1, 10):
            ts = 1000 + i * 100
            s_i = SensorInput(
                timestamp_ms=ts,
                ax=0.0, ay=0.0, az=9.81,
                gx=0.0, gy=0.0, gz=0.0,
                ax_lin=0.0, ay_lin=0.0, az_lin=0.0,
                gnss_lat=28.61395,
                gnss_lon=77.2100 + i * 0.0001,
                gnss_alt=215.0,
                gnss_speed=12.0,
                gnss_course=90.0,
                gnss_accuracy=2.5,
                gnss_sats=12,
            )
            out_i = self.engine.step(s_i)
            self.assertIsInstance(out_i, NavigationOutput)
            self.assertGreater(out_i.lat_deg, 0.0)
            self.assertGreater(out_i.lon_deg, 0.0)
            self.assertIsNotNone(out_i.mode)
            self.assertGreaterEqual(out_i.confidence, 0.0)

    def test_streaming_to_dict_schema(self):
        s = SensorInput(
            timestamp_ms=1000,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            gnss_lat=28.6139, gnss_lon=77.2100,
        )
        out = self.engine.step(s)
        d = out.to_dict()
        self.assertIn("timestamp_ms", d)
        self.assertIn("lat_deg", d)
        self.assertIn("lon_deg", d)
        self.assertIn("mode", d)
        self.assertIn("confidence", d)
        self.assertIn("final_lat", d)
        self.assertIn("final_lon", d)


class TestEdgeCasesAndDegradation(unittest.TestCase):
    """Tests edge cases, empty datasets, and graceful degradation."""

    def test_empty_dataframe(self):
        engine = IDRNavigationEngine()
        empty_res = engine.process_trip(pd.DataFrame())
        self.assertTrue(empty_res.empty)

    def test_missing_gnss_initialization(self):
        engine = IDRNavigationEngine()
        s_no_gnss = SensorInput(
            timestamp_ms=1000,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
        )
        out = engine.step(s_no_gnss)
        # Should return uninitialized baseline without crashing
        self.assertEqual(out.lat_deg, 0.0)
        self.assertEqual(out.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
