"""
Unit Tests -- AI-Assisted INS & NHC (Phase 7)
================================================
Tests AIAssistedINS, NHCCorrector, and their integration.
"""

import unittest
import numpy as np
import pandas as pd

from src.ins import AIAssistedINS, NHCCorrector, INSState


class TestNHCCorrector(unittest.TestCase):
    """Tests for the Non-Holonomic Constraint corrector."""

    def setUp(self):
        self.nhc = NHCCorrector(lateral_weight=1.0, vertical_weight=1.0)

    def test_preserves_pure_forward_north(self):
        """Pure northward velocity should be unchanged."""
        vn, ve, vd = self.nhc.correct_ned_velocity(
            v_north=10.0, v_east=0.0, v_down=0.0, heading_rad=0.0
        )
        self.assertAlmostEqual(vn, 10.0, places=5)
        self.assertAlmostEqual(ve, 0.0, places=5)
        self.assertAlmostEqual(vd, 0.0, places=5)

    def test_preserves_pure_forward_east(self):
        """Pure eastward velocity with heading=90° should be unchanged."""
        heading = np.pi / 2  # 90 degrees = East
        vn, ve, vd = self.nhc.correct_ned_velocity(
            v_north=0.0, v_east=10.0, v_down=0.0, heading_rad=heading
        )
        self.assertAlmostEqual(vn, 0.0, places=4)
        self.assertAlmostEqual(ve, 10.0, places=4)
        self.assertAlmostEqual(vd, 0.0, places=5)

    def test_zeroes_lateral_velocity(self):
        """Lateral velocity should be removed."""
        # Heading = North, but vehicle has eastward (lateral) component
        vn, ve, vd = self.nhc.correct_ned_velocity(
            v_north=10.0, v_east=5.0, v_down=0.0, heading_rad=0.0
        )
        # Forward component (along North) should be preserved
        self.assertAlmostEqual(vn, 10.0, places=4)
        # Lateral (East when heading North) should be zeroed
        self.assertAlmostEqual(ve, 0.0, places=4)

    def test_zeroes_vertical_velocity(self):
        """Vertical velocity should be removed."""
        vn, ve, vd = self.nhc.correct_ned_velocity(
            v_north=10.0, v_east=0.0, v_down=3.0, heading_rad=0.0
        )
        self.assertAlmostEqual(vd, 0.0, places=5)
        # Forward unchanged
        self.assertAlmostEqual(vn, 10.0, places=4)

    def test_correction_magnitude(self):
        """Correction magnitude should match the removed velocity."""
        mag = self.nhc.correction_magnitude(
            v_north=10.0, v_east=5.0, v_down=3.0, heading_rad=0.0
        )
        # Removed: lateral=5.0, vertical=3.0 → magnitude = sqrt(25+9) ≈ 5.83
        expected = np.sqrt(5.0**2 + 3.0**2)
        self.assertAlmostEqual(mag, expected, places=3)

    def test_soft_nhc(self):
        """Partial suppression should reduce but not zero lateral velocity."""
        soft_nhc = NHCCorrector(lateral_weight=0.5, vertical_weight=0.5)
        vn, ve, vd = soft_nhc.correct_ned_velocity(
            v_north=10.0, v_east=4.0, v_down=2.0, heading_rad=0.0
        )
        # Lateral should be halved (heading=North, so East = lateral)
        self.assertAlmostEqual(ve, 2.0, places=3)
        self.assertAlmostEqual(vd, 1.0, places=3)


class TestAIAssistedINS(unittest.TestCase):
    """Tests for the AI-Assisted INS engine."""

    def _make_ins(self, enable_nhc=False):
        """Create and initialize an AIAssistedINS at the origin heading North."""
        ins = AIAssistedINS(enable_nhc=enable_nhc)
        ins.initialize(
            lat_deg=0.0, lon_deg=0.0, heading_deg=0.0,
            v_north=0.0, v_east=0.0,
        )
        return ins

    def test_initialization(self):
        """Should initialize with correct state."""
        ins = self._make_ins()
        self.assertIsNotNone(ins.state)
        self.assertAlmostEqual(ins.state.lat_deg, 0.0, places=5)
        self.assertAlmostEqual(ins.state.lon_deg, 0.0, places=5)

    def test_straight_line_north(self):
        """Constant forward velocity + zero gyro → straight northward trajectory."""
        ins = self._make_ins()
        dt = 0.1  # 10 Hz
        v_forward = 10.0  # 10 m/s forward

        # Run 100 steps = 10 seconds
        for _ in range(100):
            ins.update_ai(
                dt=dt,
                gx_veh=0.0, gy_veh=0.0, gz_veh=0.0,
                ai_forward_velocity=v_forward,
            )

        state = ins.state
        # Should have moved north (~100m)
        self.assertGreater(state.lat_deg, 0.0)
        # Should not have moved east significantly
        self.assertAlmostEqual(state.lon_deg, 0.0, places=5)
        # Speed should be ~10 m/s
        self.assertAlmostEqual(state.speed_ms, 10.0, places=2)

    def test_circular_motion(self):
        """Constant velocity + constant yaw rate → circular arc."""
        ins = self._make_ins()
        dt = 0.1
        v_forward = 5.0   # 5 m/s
        yaw_rate = 0.1     # rad/s → ~5.7 deg/s

        # Run a quarter circle: π/2 / 0.1 = ~15.7 seconds → ~157 steps
        for _ in range(157):
            ins.update_ai(
                dt=dt,
                gx_veh=0.0, gy_veh=0.0, gz_veh=yaw_rate,
                ai_forward_velocity=v_forward,
            )

        state = ins.state
        # Heading should be ~90° (π/2 rad)
        heading_deg = np.degrees(state.heading_rad) % 360
        self.assertAlmostEqual(heading_deg, 90.0, delta=10.0)
        # Should have moved in both lat and lon
        self.assertGreater(state.lat_deg, 0.0)
        self.assertGreater(state.lon_deg, 0.0)

    def test_stationary(self):
        """Zero velocity → no position change."""
        ins = self._make_ins()
        dt = 0.1

        for _ in range(50):
            ins.update_ai(
                dt=dt,
                gx_veh=0.0, gy_veh=0.0, gz_veh=0.0,
                ai_forward_velocity=0.0,
            )

        state = ins.state
        self.assertAlmostEqual(state.lat_deg, 0.0, places=8)
        self.assertAlmostEqual(state.lon_deg, 0.0, places=8)
        self.assertAlmostEqual(state.speed_ms, 0.0, places=5)

    def test_nhc_enabled(self):
        """With NHC enabled, result should still produce valid trajectory."""
        ins = self._make_ins(enable_nhc=True)
        dt = 0.1

        for _ in range(100):
            ins.update_ai(
                dt=dt,
                gx_veh=0.0, gy_veh=0.0, gz_veh=0.0,
                ai_forward_velocity=10.0,
            )

        state = ins.state
        self.assertGreater(state.lat_deg, 0.0)
        self.assertAlmostEqual(state.v_down, 0.0, places=5)

    def test_batch_processing(self):
        """run_batch_ai should produce a trajectory DataFrame."""
        ins = self._make_ins()
        n = 50
        df = pd.DataFrame({
            "timestamp": np.arange(0, n * 100, 100),
            "gx_veh": np.zeros(n),
            "gy_veh": np.zeros(n),
            "gz_veh": np.zeros(n),
        })
        ai_vel = np.full(n, 10.0)  # Constant 10 m/s

        traj = ins.run_batch_ai(df, ai_vel)
        self.assertEqual(len(traj), n)
        self.assertIn("lat_deg", traj.columns)
        self.assertIn("lon_deg", traj.columns)
        self.assertIn("speed_ms", traj.columns)
        # Should have moved north
        self.assertGreater(traj["lat_deg"].iloc[-1], traj["lat_deg"].iloc[0])

    def test_nan_velocity_treated_as_zero(self):
        """NaN in ai_velocity should be treated as zero (no movement)."""
        ins = self._make_ins()
        n = 20
        df = pd.DataFrame({
            "timestamp": np.arange(0, n * 100, 100),
            "gx_veh": np.zeros(n),
            "gy_veh": np.zeros(n),
            "gz_veh": np.zeros(n),
        })
        ai_vel = np.full(n, np.nan)

        traj = ins.run_batch_ai(df, ai_vel)
        # With all NaN velocity, position should not change
        self.assertAlmostEqual(traj["lat_deg"].iloc[-1], 0.0, places=6)


class TestAIvsRawDrift(unittest.TestCase):
    """Integration test: AI-assisted INS should drift less than raw INS
    on synthetic data with a known velocity profile."""

    def test_ai_better_than_naive_acceleration(self):
        """Given perfect AI velocity, AI-INS should have zero drift
        compared to raw INS which double-integrates noisy acceleration."""
        from src.ins import StrapdownINS

        np.random.seed(42)
        n = 200
        dt = 0.1
        true_velocity = 10.0  # m/s northward

        # -- Pipeline A: Raw INS with noisy acceleration --
        raw_ins = StrapdownINS()
        raw_ins.initialize(lat_deg=0.0, lon_deg=0.0, heading_deg=0.0)

        for i in range(n):
            # Noisy acceleration (should integrate to ~10 m/s but will drift)
            noisy_ax = 0.0 + np.random.normal(0, 0.5)
            raw_ins.update(
                dt=dt,
                ax_lin=noisy_ax, ay_lin=np.random.normal(0, 0.3), az_lin=np.random.normal(0, 0.3),
                gx_veh=0.0, gy_veh=0.0, gz_veh=0.0,
            )

        raw_final = raw_ins.state

        # -- Pipeline B: AI-assisted INS with perfect velocity --
        ai_ins = AIAssistedINS(enable_nhc=False)
        ai_ins.initialize(lat_deg=0.0, lon_deg=0.0, heading_deg=0.0)

        for i in range(n):
            ai_ins.update_ai(
                dt=dt,
                gx_veh=0.0, gy_veh=0.0, gz_veh=0.0,
                ai_forward_velocity=true_velocity,
            )

        ai_final = ai_ins.state

        # AI-assisted should be very close to the expected position
        # Expected: 200 * 0.1s * 10 m/s = 200m north
        expected_lat_change = 200.0 / 111_320.0  # ~degrees for 200m at equator

        ai_error = abs(ai_final.lat_deg - expected_lat_change)
        raw_error = abs(raw_final.lat_deg - expected_lat_change)

        # AI should be much more accurate
        self.assertLess(ai_error, 0.001,
                        f"AI error too large: {ai_error:.6f}°")
        # Raw INS should be worse (or at least not better)
        self.assertGreater(raw_error, ai_error * 0.5,
                           "Raw INS unexpectedly more accurate than AI")


if __name__ == "__main__":
    unittest.main()
