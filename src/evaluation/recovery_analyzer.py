"""
IDR MVP -- GNSS Recovery Analyzer (Phase 12)
==============================================
Analyzes how smoothly and quickly the navigation system converges back to
accurate positioning after a GNSS outage ends.

Metrics:
    - Recovery convergence time (seconds until error < threshold)
    - Max single-step position jump during recovery
    - Error trajectory during the recovery window
    - Whether recovery was jump-free (no teleportation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from src.ins.integration import haversine_distance_deg


@dataclass
class RecoveryMetrics:
    """Metrics describing GNSS recovery behavior after an outage."""
    outage_end_idx: int
    error_at_outage_end_m: float
    convergence_time_s: float              # Time to reach < threshold after outage
    converged: bool                        # Did it converge within the analysis window?
    convergence_threshold_m: float         # Threshold used
    max_single_step_jump_m: float          # Largest position step during recovery
    is_jump_free: bool                     # True if max_jump < jump_limit
    jump_limit_m: float                    # Limit used
    error_after_5s_m: float                # Error 5s after outage end
    error_after_10s_m: float               # Error 10s after outage end
    recovery_window_errors: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outage_end_idx": self.outage_end_idx,
            "error_at_outage_end_m": round(self.error_at_outage_end_m, 2),
            "convergence_time_s": round(self.convergence_time_s, 2),
            "converged": self.converged,
            "convergence_threshold_m": self.convergence_threshold_m,
            "max_single_step_jump_m": round(self.max_single_step_jump_m, 2),
            "is_jump_free": self.is_jump_free,
            "jump_limit_m": self.jump_limit_m,
            "error_after_5s_m": round(self.error_after_5s_m, 2),
            "error_after_10s_m": round(self.error_after_10s_m, 2),
        }


class RecoveryAnalyzer:
    """
    Analyzes GNSS recovery behavior after each outage window.

    Parameters
    ----------
    convergence_threshold_m : float
        Position error threshold (metres) below which recovery is considered complete.
    jump_limit_m : float
        Maximum allowed single-step position change (metres) during recovery.
        Steps exceeding this are considered "teleport/jump" events.
    recovery_window_s : float
        Duration (seconds) after outage end to analyze for recovery.
    sampling_rate_hz : float
        Data sampling rate in Hz.
    """

    def __init__(
        self,
        convergence_threshold_m: float = 10.0,
        jump_limit_m: float = 50.0,
        recovery_window_s: float = 30.0,
        sampling_rate_hz: float = 10.0,
    ):
        self.convergence_threshold_m = convergence_threshold_m
        self.jump_limit_m = jump_limit_m
        self.recovery_window_s = recovery_window_s
        self.sampling_rate_hz = sampling_rate_hz

    def analyze(
        self,
        pred_lats: np.ndarray,
        pred_lons: np.ndarray,
        ref_lats: np.ndarray,
        ref_lons: np.ndarray,
        outage_end_idx: int,
    ) -> RecoveryMetrics:
        """
        Analyze recovery behavior after a single outage.

        Parameters
        ----------
        pred_lats, pred_lons : np.ndarray
            Predicted positions (degrees) for the full trip.
        ref_lats, ref_lons : np.ndarray
            Ground-truth positions (degrees) for the full trip.
        outage_end_idx : int
            Index at which the outage ends (first sample with GNSS available again).

        Returns
        -------
        RecoveryMetrics
        """
        n_total = len(pred_lats)
        dt = 1.0 / self.sampling_rate_hz
        recovery_samples = int(self.recovery_window_s * self.sampling_rate_hz)
        end_analysis_idx = min(n_total, outage_end_idx + recovery_samples)

        # Slice the recovery window
        r_start = outage_end_idx
        r_end = end_analysis_idx

        if r_start >= n_total or r_start >= r_end:
            return RecoveryMetrics(
                outage_end_idx=outage_end_idx,
                error_at_outage_end_m=0.0,
                convergence_time_s=0.0,
                converged=True,
                convergence_threshold_m=self.convergence_threshold_m,
                max_single_step_jump_m=0.0,
                is_jump_free=True,
                jump_limit_m=self.jump_limit_m,
                error_after_5s_m=0.0,
                error_after_10s_m=0.0,
            )

        # Position errors in recovery window
        recovery_pred_lat = pred_lats[r_start:r_end]
        recovery_pred_lon = pred_lons[r_start:r_end]
        recovery_ref_lat = ref_lats[r_start:r_end]
        recovery_ref_lon = ref_lons[r_start:r_end]

        errors = haversine_distance_deg(
            recovery_pred_lat, recovery_pred_lon,
            recovery_ref_lat, recovery_ref_lon,
        )
        errors = np.nan_to_num(errors, nan=0.0)

        # Error at outage end
        error_at_outage_end = float(errors[0]) if len(errors) > 0 else 0.0

        # Convergence time: first sample where error drops below threshold
        converged = False
        convergence_time = self.recovery_window_s  # Default: did not converge
        for i, err in enumerate(errors):
            if err < self.convergence_threshold_m:
                convergence_time = i * dt
                converged = True
                break

        # Max single-step jump during recovery
        if len(recovery_pred_lat) > 1:
            step_dists = haversine_distance_deg(
                recovery_pred_lat[:-1], recovery_pred_lon[:-1],
                recovery_pred_lat[1:], recovery_pred_lon[1:],
            )
            max_jump = float(np.max(step_dists)) if len(step_dists) > 0 else 0.0
        else:
            max_jump = 0.0

        is_jump_free = max_jump < self.jump_limit_m

        # Error at 5s and 10s after outage end
        idx_5s = min(int(5.0 * self.sampling_rate_hz), len(errors) - 1)
        idx_10s = min(int(10.0 * self.sampling_rate_hz), len(errors) - 1)
        error_after_5s = float(errors[idx_5s]) if idx_5s < len(errors) else 0.0
        error_after_10s = float(errors[idx_10s]) if idx_10s < len(errors) else 0.0

        return RecoveryMetrics(
            outage_end_idx=outage_end_idx,
            error_at_outage_end_m=error_at_outage_end,
            convergence_time_s=convergence_time,
            converged=converged,
            convergence_threshold_m=self.convergence_threshold_m,
            max_single_step_jump_m=max_jump,
            is_jump_free=is_jump_free,
            jump_limit_m=self.jump_limit_m,
            error_after_5s_m=error_after_5s,
            error_after_10s_m=error_after_10s,
            recovery_window_errors=errors.tolist(),
        )
