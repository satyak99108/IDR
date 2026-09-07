"""
Unit Tests -- Navigation Mode Manager (Phase 9)
=================================================
Tests NavigationModeManager, SmoothJumpMitigator, anti-chatter debounce,
degraded GNSS assessment, and smooth position jump mitigation.
"""

import unittest
import numpy as np

from src.fusion.mode_manager import (
    NavigationMode,
    NavigationModeManager,
    SmoothJumpMitigator,
    SignalQualityStatus,
)
from src.fusion.fusion_engine import GNSSINSFusionEngine


class TestSmoothJumpMitigator(unittest.TestCase):
    """Tests for the continuous position jump mitigator."""

    def setUp(self):
        self.mitigator = SmoothJumpMitigator(recovery_duration_s=3.0)

    def test_zero_jump_at_initiation(self):
        """At t=0, smoothed offset must exactly cancel the position jump."""
        last_dr = (100.0, 50.0, 0.0)
        new_ekf = (100.0, 10.0, 0.0) # 40m discrepancy East

        jump_mag = self.mitigator.start_recovery(current_time_s=10.0, last_dr_pos_ned=last_dr, new_ekf_pos_ned=new_ekf)
        self.assertAlmostEqual(jump_mag, 40.0, places=3)

        dn, de, weight = self.mitigator.compute_smoothed_offset(current_time_s=10.0)
        self.assertAlmostEqual(dn, 0.0, places=3)
        self.assertAlmostEqual(de, 40.0, places=3)
        self.assertAlmostEqual(weight, 1.0, places=3)

        # Added to new_ekf: (10.0 + 40.0) == 50.0 (matches last_dr exactly)
        self.assertAlmostEqual(new_ekf[1] + de, last_dr[1], places=3)

    def test_smooth_decay(self):
        """Offset must smoothly diminish over the recovery window."""
        self.mitigator.start_recovery(
            current_time_s=0.0,
            last_dr_pos_ned=(0.0, 30.0, 0.0),
            new_ekf_pos_ned=(0.0, 0.0, 0.0),
        )

        _, de_1s, w_1s = self.mitigator.compute_smoothed_offset(1.0)
        _, de_2s, w_2s = self.mitigator.compute_smoothed_offset(2.0)
        _, de_3s, w_3s = self.mitigator.compute_smoothed_offset(3.0)

        # Monotonic decay
        self.assertGreater(30.0, de_1s)
        self.assertGreater(de_1s, de_2s)
        self.assertGreater(de_2s, de_3s)

        # Exactly 0 at t=3.0s
        self.assertAlmostEqual(de_3s, 0.0, places=3)
        self.assertAlmostEqual(w_3s, 0.0, places=3)


class TestNavigationModeManager(unittest.TestCase):
    """Tests for anti-chatter hysteresis and mode transitions."""

    def setUp(self):
        self.manager = NavigationModeManager(
            dropout_debounce_samples=5,
            recovery_confirm_samples=5,
            recovery_duration_s=3.0,
            max_acceptable_accuracy_m=15.0,
            min_acceptable_satellites=4,
        )

    def test_anti_chatter_suppression(self):
        """1 to 4 missing GNSS samples should NOT trigger DEAD_RECKONING."""
        # 10 nominal samples
        for i in range(10):
            mode, telem = self.manager.update(
                timestamp_s=i * 0.1,
                has_gnss_sample=True,
            )
            self.assertEqual(mode, NavigationMode.GNSS_INS)

        # 4 dropouts (e.g. overhead bridge, tree canopy)
        for i in range(10, 14):
            mode, telem = self.manager.update(
                timestamp_s=i * 0.1,
                has_gnss_sample=False,
            )
            # Must remain in GNSS_INS
            self.assertEqual(mode, NavigationMode.GNSS_INS)
            self.assertGreater(telem["chatter_suppressed_count"], 0)

        # 5th dropout triggers DEAD_RECKONING
        mode, telem = self.manager.update(
            timestamp_s=1.4,
            has_gnss_sample=False,
        )
        self.assertEqual(mode, NavigationMode.DEAD_RECKONING)

    def test_anti_flicker_recovery_confirmation(self):
        """Intermittent single-sample GNSS during blackout must NOT trigger premature recovery."""
        # Enter DR
        for i in range(5):
            self.manager.update(timestamp_s=i * 0.1, has_gnss_sample=False)
        self.assertEqual(self.manager.current_mode, NavigationMode.DEAD_RECKONING)

        # 2 momentary flickers of GNSS
        for i in range(5, 7):
            mode, _ = self.manager.update(timestamp_s=i * 0.1, has_gnss_sample=True)
            self.assertEqual(mode, NavigationMode.DEAD_RECKONING)

        # 5 consecutive valid samples confirm RECOVERY
        for i in range(7, 12):
            mode, _ = self.manager.update(timestamp_s=i * 0.1, has_gnss_sample=True)

        self.assertEqual(mode, NavigationMode.RECOVERY)

    def test_degraded_gnss_detection(self):
        """Poor accuracy (>15m) or low satellites (<4) enters DEGRADED_GNSS."""
        mode, telem = self.manager.update(
            timestamp_s=0.1,
            has_gnss_sample=True,
            gnss_accuracy=25.0, # Exceeds 15m
            satellites=3,       # Below 4
        )
        self.assertEqual(mode, NavigationMode.DEGRADED_GNSS)
        self.assertTrue(telem["is_degraded"])
        self.assertLess(telem["confidence"], 90.0)

    def test_jump_mitigation_integration_with_fusion_engine(self):
        """
        Verify that GNSSINSFusionEngine applies smooth jump mitigation,
        preventing step-to-step position teleportation upon GNSS re-acquisition.
        """
        engine = GNSSINSFusionEngine(recovery_window_s=3.0)
        engine.set_origin(lat_deg=52.4, lon_deg=-1.5, alt_m=100.0)

        # 1. Normal cruising
        for i in range(20):
            engine.step(
                dt=0.1, ax_lin=0.0, ay_lin=0.0, az_lin=0.0, gz_veh=0.0,
                gnss_lat=52.4001, gnss_lon=-1.4999, gnss_alt=100.0, gnss_speed=10.0,
                timestamp_ms=i * 100, is_outage=False,
            )

        # 2. Simulated 30s blackout with dead reckoning
        for i in range(20, 320):
            engine.step(
                dt=0.1, ax_lin=0.0, ay_lin=0.0, az_lin=0.0, gz_veh=0.0,
                ai_speed=10.0, timestamp_ms=i * 100, is_outage=True,
            )

        dr_exit_state = engine.step(
            dt=0.1, ax_lin=0.0, ay_lin=0.0, az_lin=0.0, gz_veh=0.0,
            ai_speed=10.0, timestamp_ms=32000, is_outage=True,
        )

        # 3. First sample of re-acquired GNSS
        # GNSS fix differs from DR position by ~30 meters
        recovered_state = engine.step(
            dt=0.1, ax_lin=0.0, ay_lin=0.0, az_lin=0.0, gz_veh=0.0,
            gnss_lat=52.4025, gnss_lon=-1.4975, gnss_alt=100.0, gnss_speed=10.0,
            timestamp_ms=32100, is_outage=False,
        )

        self.assertEqual(recovered_state.mode, NavigationMode.RECOVERY.value)

        # The step-to-step position delta across the boundary must be smooth (< 2.5m)
        step_delta_m = np.sqrt(
            (recovered_state.p_n - dr_exit_state.p_n)**2 +
            (recovered_state.p_e - dr_exit_state.p_e)**2
        )
        self.assertLess(step_delta_m, 2.5, f"Step jump was {step_delta_m:.2f}m, expected < 2.5m")


if __name__ == "__main__":
    unittest.main()
