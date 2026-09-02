"""
Unit Tests -- GNSS + INS Fusion Package (Phase 8)
===================================================
Tests NavigationEKF, GNSSINSFusionEngine, coordinate conversions, and mode transitions.
"""

import unittest
import numpy as np
import pandas as pd

from src.fusion import NavigationEKF, GNSSINSFusionEngine, FusionState, NavigationMode


class TestNavigationEKF(unittest.TestCase):
    """Tests for the 10-state Navigation EKF."""

    def setUp(self):
        self.ekf = NavigationEKF()
        self.ekf.initialize_state(
            p_n=0.0, p_e=0.0, p_d=0.0,
            v_n=10.0, v_e=0.0, v_d=0.0,
            heading_rad=0.0, # Facing North
            timestamp_s=0.0,
        )

    def test_initialization(self):
        """State vector and covariance should match expected dimensions."""
        self.assertEqual(len(self.ekf.x), 10)
        self.assertEqual(self.ekf.P.shape, (10, 10))
        self.assertAlmostEqual(self.ekf.speed_ms, 10.0, places=4)
        self.assertAlmostEqual(self.ekf.heading_deg, 0.0, places=4)

    def test_imu_prediction(self):
        """Forward acceleration should increase North velocity and advance position."""
        dt = 0.1
        # Accelerate at 2 m/s^2 forward (North)
        self.ekf.predict_imu(dt=dt, ax_lin=2.0, ay_lin=0.0, az_lin=0.0, gz_veh=0.0)

        # v_n should be 10.0 + 2.0 * 0.1 = 10.2
        self.assertAlmostEqual(self.ekf.x[3], 10.2, places=3)
        # Position should have moved forward: p_n ~ 10.1 * 0.1 = 1.01 m
        self.assertGreater(self.ekf.x[0], 1.0)
        # Lateral velocity v_e should remain ~0
        self.assertAlmostEqual(self.ekf.x[4], 0.0, places=3)

    def test_gnss_pv_update(self):
        """GNSS measurement update should reduce uncertainty and correct state."""
        initial_trace = self.ekf.state_uncertainty_trace

        # GNSS says we are at (5.0, 2.0, 0.0) with vel (10.0, 0.5)
        self.ekf.update_gnss_pv(
            p_n_meas=5.0, p_e_meas=2.0, p_d_meas=0.0,
            v_n_meas=10.0, v_e_meas=0.5,
            pos_std=1.0, vel_std=0.1,
        )

        # Position should be corrected towards measurement
        self.assertGreater(self.ekf.x[0], 0.0)
        self.assertGreater(self.ekf.x[1], 0.0)

        # State uncertainty should decrease
        self.assertLess(self.ekf.state_uncertainty_trace, initial_trace)

    def test_ai_velocity_update(self):
        """AI forward velocity update should correct speed along heading."""
        # Facing North (heading=0), true speed 15 m/s
        self.ekf.update_ai_velocity(v_forward_meas=15.0, vel_std=0.5)

        # North velocity should increase towards 15 m/s
        self.assertGreater(self.ekf.x[3], 10.0)
        self.assertLessEqual(self.ekf.x[3], 15.1)

    def test_nhc_update(self):
        """NHC update should suppress lateral and vertical velocity."""
        # Perturb lateral velocity v_e
        self.ekf.x[4] = 4.0  # 4 m/s sideways
        self.ekf.x[5] = 2.0  # 2 m/s downwards

        self.ekf.update_nhc(lateral_std=0.1, vertical_std=0.1)

        # Both should be pulled towards 0
        self.assertLess(abs(self.ekf.x[4]), 4.0)
        self.assertLess(abs(self.ekf.x[5]), 2.0)


class TestFusionEngine(unittest.TestCase):
    """Tests for the GNSSINSFusionEngine."""

    def setUp(self):
        self.engine = GNSSINSFusionEngine()
        self.engine.set_origin(lat_deg=52.4, lon_deg=-1.5, alt_m=100.0)

    def test_coordinate_transforms_roundtrip(self):
        """WGS-84 to NED and back should be consistent within millimeters."""
        target_lat = 52.405
        target_lon = -1.492
        target_alt = 115.0

        p_n, p_e, p_d = self.engine.geodetic_to_ned(target_lat, target_lon, target_alt)
        lat_back, lon_back, alt_back = self.engine.ned_to_geodetic(p_n, p_e, p_d)

        self.assertAlmostEqual(lat_back, target_lat, places=6)
        self.assertAlmostEqual(lon_back, target_lon, places=6)
        self.assertAlmostEqual(alt_back, target_alt, places=2)

    def test_mode_transitions(self):
        """Engine should transition GNSS_INS -> DEAD_RECKONING -> RECOVERY -> GNSS_INS."""
        # 1. Normal step
        s1 = self.engine.step(
            dt=0.1, ax_lin=0.0, ay_lin=0.0, az_lin=0.0, gz_veh=0.0,
            gnss_lat=52.4001, gnss_lon=-1.4999, gnss_alt=100.0, gnss_speed=10.0,
            is_outage=False, timestamp_ms=100,
        )
        self.assertEqual(s1.mode, NavigationMode.GNSS_INS.value)

        # 2. Blackout step
        s2 = self.engine.step(
            dt=0.1, ax_lin=0.0, ay_lin=0.0, az_lin=0.0, gz_veh=0.0,
            gnss_lat=None, gnss_lon=None, gnss_speed=None,
            ai_speed=10.0, is_outage=True, timestamp_ms=200,
        )
        self.assertEqual(s2.mode, NavigationMode.DEAD_RECKONING.value)

        # 3. GNSS returns
        s3 = self.engine.step(
            dt=0.1, ax_lin=0.0, ay_lin=0.0, az_lin=0.0, gz_veh=0.0,
            gnss_lat=52.4003, gnss_lon=-1.4997, gnss_alt=100.0, gnss_speed=10.0,
            is_outage=False, timestamp_ms=300,
        )
        self.assertEqual(s3.mode, NavigationMode.RECOVERY.value)

    def test_outage_dead_reckoning_stability(self):
        """During a 30s blackout, AI velocity + NHC should keep trajectory moving and stable."""
        dt = 0.1
        steps = 300  # 30 seconds at 10 Hz

        for i in range(steps):
            fstate = self.engine.step(
                dt=dt, ax_lin=0.0, ay_lin=0.0, az_lin=0.0, gz_veh=0.0,
                gnss_lat=None, gnss_lon=None, gnss_speed=None,
                ai_speed=10.0, # 10 m/s forward
                is_outage=True, timestamp_ms=i * 100,
            )

        # Total distance should be approximately 10 m/s * 30s = 300m
        self.assertAlmostEqual(fstate.p_n, 300.0, delta=20.0)
        self.assertAlmostEqual(fstate.p_e, 0.0, delta=5.0)


if __name__ == "__main__":
    unittest.main()
