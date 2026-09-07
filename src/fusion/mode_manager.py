"""
IDR MVP -- Navigation Mode Manager (Phase 9)
==============================================
Manages explicit navigation operating states, prevents mode chattering from
transient GNSS fluctuations, and suppresses position jumps upon signal
re-acquisition (MVP.md §13, TECH_STACK.md §9, §10, §14).

Supported States:
    GNSS_INS       -- Normal operational state with high-quality GNSS + IMU fusion.
    DEGRADED_GNSS  -- Low satellite count, poor accuracy, or high DOP; cautionary state.
    DEAD_RECKONING -- Complete GNSS blackout / tunnel outage; IMU + AI + NHC driven.
    RECOVERY       -- Smooth re-acquisition phase; jump mitigation & progressive convergence.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


class NavigationMode(enum.Enum):
    """Explicit navigation states per MVP.md §13."""
    GNSS_INS = "GNSS_INS"
    DEGRADED_GNSS = "DEGRADED_GNSS"
    DEAD_RECKONING = "DEAD_RECKONING"
    RECOVERY = "RECOVERY"


@dataclass
class ModeTransitionEvent:
    """Record of a discrete mode transition."""
    timestamp_s: float
    from_mode: NavigationMode
    to_mode: NavigationMode
    reason: str
    position_jump_m: float = 0.0


@dataclass
class SignalQualityStatus:
    """Assessment of incoming GNSS signal reliability."""
    is_available: bool
    is_degraded: bool
    accuracy_m: Optional[float] = None
    satellites: Optional[int] = None
    reason: str = ""


class SmoothJumpMitigator:
    """
    Mitigates instantaneous position jumps ('teleportation') when re-acquiring
    GNSS fixes following extended Dead Reckoning.

    Blends the spatial discrepancy vector between the dead-reckoning trajectory
    and the re-acquired GNSS/EKF position using a continuous decay function:
        offset(t) = offset_0 * exp(-3.0 * t / T) * (1 - t / T)
    Ensuring:
        1. Zero jump at the exact transition boundary (offset(0) = offset_0).
        2. Smooth, continuous derivative (no abrupt velocity spike).
        3. Exact convergence to true EKF state at t >= T (offset(T) = 0).
    """

    def __init__(self, recovery_duration_s: float = 3.0):
        self.recovery_duration_s = max(0.5, float(recovery_duration_s))
        self.is_active = False
        self.start_time_s: Optional[float] = None
        self.offset_n_m: float = 0.0
        self.offset_e_m: float = 0.0
        self.initial_jump_mag_m: float = 0.0

    def start_recovery(
        self,
        current_time_s: float,
        last_dr_pos_ned: Tuple[float, float, float],
        new_ekf_pos_ned: Tuple[float, float, float],
    ) -> float:
        """
        Initiates smooth jump mitigation.
        Returns the initial jump magnitude in meters.
        """
        self.start_time_s = current_time_s
        self.is_active = True

        # Offset needed to make new EKF position match the last DR position at t=0
        self.offset_n_m = float(last_dr_pos_ned[0] - new_ekf_pos_ned[0])
        self.offset_e_m = float(last_dr_pos_ned[1] - new_ekf_pos_ned[1])
        self.initial_jump_mag_m = float(np.sqrt(self.offset_n_m**2 + self.offset_e_m**2))
        return self.initial_jump_mag_m

    def compute_smoothed_offset(self, current_time_s: float) -> Tuple[float, float, float]:
        """
        Returns (dn_smooth, de_smooth, blend_weight) for current timestamp.
        blend_weight goes from 1.0 (at start) down to 0.0 (at completion).
        """
        if not self.is_active or self.start_time_s is None:
            return 0.0, 0.0, 0.0

        elapsed_s = current_time_s - self.start_time_s
        if elapsed_s >= self.recovery_duration_s or elapsed_s < 0.0:
            self.reset()
            return 0.0, 0.0, 0.0

        tau = elapsed_s / self.recovery_duration_s
        # Smooth cubic hermite / exponential decay function
        weight = float((1.0 - tau)**2 * (1.0 + 2.0 * tau))

        dn = self.offset_n_m * weight
        de = self.offset_e_m * weight
        return dn, de, weight

    def reset(self) -> None:
        """Resets mitigator state."""
        self.is_active = False
        self.start_time_s = None
        self.offset_n_m = 0.0
        self.offset_e_m = 0.0


class NavigationModeManager:
    """
    Intelligent Navigation Mode Manager with Anti-Chatter Hysteresis and
    Smooth GNSS Re-acquisition Blending.

    Prevents:
        - Mode chattering on transient dropouts (e.g. overhead bridges, tree canopies).
        - Sudden position teleportation when GNSS returns after dead reckoning.
        - Unnecessary covariance resets or trajectory discontinuities.
    """

    def __init__(
        self,
        dropout_debounce_samples: int = 5,       # 0.5s at 10 Hz before declaring DR
        recovery_confirm_samples: int = 5,       # 0.5s at 10 Hz before confirming recovery
        recovery_duration_s: float = 3.0,        # 3.0s smooth blending window
        max_acceptable_accuracy_m: float = 15.0, # Degraded GNSS threshold
        min_acceptable_satellites: int = 4,      # Min satellites for high-accuracy fix
    ):
        self.dropout_debounce_samples = max(1, int(dropout_debounce_samples))
        self.recovery_confirm_samples = max(1, int(recovery_confirm_samples))
        self.recovery_duration_s = float(recovery_duration_s)
        self.max_acceptable_accuracy_m = float(max_acceptable_accuracy_m)
        self.min_acceptable_satellites = int(min_acceptable_satellites)

        # Active state
        self.current_mode: NavigationMode = NavigationMode.GNSS_INS
        self.previous_mode: NavigationMode = NavigationMode.GNSS_INS

        # Hysteresis counters
        self.consecutive_invalid: int = 0
        self.consecutive_valid: int = 0

        # Jump mitigation
        self.jump_mitigator = SmoothJumpMitigator(recovery_duration_s=self.recovery_duration_s)

        # Event telemetry
        self.transition_history: List[ModeTransitionEvent] = []
        self.chatter_suppressed_count: int = 0
        self.last_dr_pos_ned: Optional[Tuple[float, float, float]] = None
        self.last_timestamp_s: Optional[float] = None
        self.time_in_mode_s: Dict[NavigationMode, float] = {m: 0.0 for m in NavigationMode}
        self._was_simulated_outage: bool = False
        self.just_entered_recovery: bool = False

    def init_jump_mitigation(
        self,
        current_time_s: float,
        new_ekf_pos_ned: Tuple[float, float, float],
    ) -> float:
        """
        Initializes smooth jump mitigation with the post-measurement EKF position.
        """
        self.just_entered_recovery = False
        if self.last_dr_pos_ned is not None:
            return self.jump_mitigator.start_recovery(
                current_time_s=current_time_s,
                last_dr_pos_ned=self.last_dr_pos_ned,
                new_ekf_pos_ned=new_ekf_pos_ned,
            )
        return 0.0

    def get_jump_offsets(self, timestamp_s: float) -> Tuple[float, float, float]:
        """Returns (dn_smooth_m, de_smooth_m, blend_weight) for current time."""
        return self.jump_mitigator.compute_smoothed_offset(timestamp_s)

    def assess_signal_quality(
        self,
        has_gnss_sample: bool,
        is_simulated_outage: bool = False,
        gnss_accuracy: Optional[float] = None,
        satellites: Optional[int] = None,
    ) -> SignalQualityStatus:
        """Assesses the physical/simulated reliability of the GNSS fix."""
        if is_simulated_outage or not has_gnss_sample:
            return SignalQualityStatus(
                is_available=False,
                is_degraded=False,
                accuracy_m=gnss_accuracy,
                satellites=satellites,
                reason="Signal unavailable or blackout",
            )

        # Check for signal degradation
        is_degraded = False
        reasons = []

        if gnss_accuracy is not None and not np.isnan(gnss_accuracy):
            if gnss_accuracy > self.max_acceptable_accuracy_m:
                is_degraded = True
                reasons.append(f"Low accuracy ({gnss_accuracy:.1f}m > {self.max_acceptable_accuracy_m:.1f}m)")

        if satellites is not None and not np.isnan(satellites):
            if satellites < self.min_acceptable_satellites:
                is_degraded = True
                reasons.append(f"Low satellite count ({satellites} < {self.min_acceptable_satellites})")

        return SignalQualityStatus(
            is_available=True,
            is_degraded=is_degraded,
            accuracy_m=gnss_accuracy,
            satellites=satellites,
            reason="; ".join(reasons) if reasons else "Nominal GNSS fix",
        )

    def update(
        self,
        timestamp_s: float,
        has_gnss_sample: bool,
        is_simulated_outage: bool = False,
        gnss_accuracy: Optional[float] = None,
        satellites: Optional[int] = None,
        current_ekf_pos_ned: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[NavigationMode, Dict[str, Any]]:
        """
        Executes a state machine step with hysteresis and jump mitigation.

        Returns:
            (active_mode, telemetry_dict)
        """
        dt = (timestamp_s - self.last_timestamp_s) if self.last_timestamp_s is not None else 0.1
        dt = max(0.001, min(1.0, dt))
        self.last_timestamp_s = timestamp_s

        signal_status = self.assess_signal_quality(
            has_gnss_sample=has_gnss_sample,
            is_simulated_outage=is_simulated_outage,
            gnss_accuracy=gnss_accuracy,
            satellites=satellites,
        )

        initial_jump_m = 0.0

        # -------------------------------------------------------------
        # State Machine Transitions
        # -------------------------------------------------------------
        if signal_status.is_available and not signal_status.is_degraded:
            if self._was_simulated_outage and self.current_mode == NavigationMode.DEAD_RECKONING:
                # Simulated outage ended explicitly
                self.consecutive_valid = max(self.consecutive_valid + 1, self.recovery_confirm_samples)
                self._was_simulated_outage = False
            else:
                self.consecutive_valid += 1
            self.consecutive_invalid = 0

            if self.current_mode == NavigationMode.DEAD_RECKONING:
                # GNSS returned! Apply hysteresis confirmation before exiting DR
                if self.consecutive_valid >= self.recovery_confirm_samples:
                    self._transition_to(
                        NavigationMode.RECOVERY,
                        timestamp_s,
                        reason=f"GNSS confirmed valid for {self.consecutive_valid} samples",
                    )
                    self.just_entered_recovery = True

            elif self.current_mode == NavigationMode.RECOVERY:
                # Check if recovery blending duration elapsed
                elapsed_recovery = timestamp_s - (self.jump_mitigator.start_time_s or timestamp_s)
                if elapsed_recovery >= self.recovery_duration_s:
                    self._transition_to(
                        NavigationMode.GNSS_INS,
                        timestamp_s,
                        reason=f"Recovery window ({self.recovery_duration_s:.1f}s) complete",
                    )
                    self.jump_mitigator.reset()

            elif self.current_mode == NavigationMode.DEGRADED_GNSS:
                self._transition_to(
                    NavigationMode.GNSS_INS,
                    timestamp_s,
                    reason="Signal quality restored to nominal",
                )

        elif signal_status.is_available and signal_status.is_degraded:
            # Degraded signal
            self.consecutive_valid += 1
            self.consecutive_invalid = 0

            if self.current_mode == NavigationMode.GNSS_INS:
                self._transition_to(
                    NavigationMode.DEGRADED_GNSS,
                    timestamp_s,
                    reason=f"Signal degraded: {signal_status.reason}",
                )

        else:
            # Signal unavailable (outage or missing fix)
            if is_simulated_outage:
                self.consecutive_invalid = max(self.consecutive_invalid + 1, self.dropout_debounce_samples)
                self._was_simulated_outage = True
            else:
                self.consecutive_invalid += 1
            self.consecutive_valid = 0

            if self.current_mode in (NavigationMode.GNSS_INS, NavigationMode.DEGRADED_GNSS, NavigationMode.RECOVERY):
                # Anti-chatter: requires debounce count before declaring DEAD_RECKONING
                if self.consecutive_invalid >= self.dropout_debounce_samples:
                    self._transition_to(
                        NavigationMode.DEAD_RECKONING,
                        timestamp_s,
                        reason="Simulated outage active" if is_simulated_outage else f"GNSS missing for {self.consecutive_invalid} consecutive samples",
                    )
                    self.jump_mitigator.reset()
                else:
                    # Anti-chatter suppression active!
                    self.chatter_suppressed_count += 1

        # Track dwell time
        self.time_in_mode_s[self.current_mode] += dt

        # If in Dead Reckoning, record current position for jump mitigator
        if self.current_mode == NavigationMode.DEAD_RECKONING and current_ekf_pos_ned is not None:
            self.last_dr_pos_ned = current_ekf_pos_ned

        # Compute smoothed jump offsets if recovery is active
        dn_offset, de_offset, blend_weight = self.jump_mitigator.compute_smoothed_offset(timestamp_s)

        # Dynamic confidence score (0 - 100%)
        confidence = self._compute_confidence(signal_status, blend_weight)

        # Adaptive EKF position measurement noise scaling during recovery
        # Progressively tightens R from 10.0m -> 3.0m as blending completes
        if self.current_mode == NavigationMode.RECOVERY:
            ekf_pos_std = 3.0 + 7.0 * blend_weight
        elif self.current_mode == NavigationMode.DEGRADED_GNSS:
            ekf_pos_std = max(6.0, float(signal_status.accuracy_m or 8.0))
        elif self.current_mode == NavigationMode.GNSS_INS:
            ekf_pos_std = 3.0
        else:
            ekf_pos_std = 15.0

        telemetry = {
            "mode": self.current_mode.value,
            "confidence": confidence,
            "blend_weight": blend_weight,
            "dn_smooth_m": dn_offset,
            "de_smooth_m": de_offset,
            "initial_jump_m": initial_jump_m,
            "ekf_pos_std": ekf_pos_std,
            "consecutive_valid": self.consecutive_valid,
            "consecutive_invalid": self.consecutive_invalid,
            "chatter_suppressed_count": self.chatter_suppressed_count,
            "is_degraded": signal_status.is_degraded,
        }

        return self.current_mode, telemetry

    def _transition_to(self, new_mode: NavigationMode, timestamp_s: float, reason: str) -> None:
        """Records a state change."""
        if new_mode != self.current_mode:
            event = ModeTransitionEvent(
                timestamp_s=timestamp_s,
                from_mode=self.current_mode,
                to_mode=new_mode,
                reason=reason,
            )
            self.transition_history.append(event)
            self.previous_mode = self.current_mode
            self.current_mode = new_mode

    def _compute_confidence(self, signal_status: SignalQualityStatus, blend_weight: float) -> float:
        """Calculates a normalized 0-100% navigation confidence rating."""
        if self.current_mode == NavigationMode.GNSS_INS:
            return 98.0
        elif self.current_mode == NavigationMode.DEGRADED_GNSS:
            acc = signal_status.accuracy_m if signal_status.accuracy_m is not None else 15.0
            return float(np.clip(85.0 - (acc - 10.0) * 2.0, 50.0, 85.0))
        elif self.current_mode == NavigationMode.RECOVERY:
            # Ramps up smoothly from 70% to 95%
            return float(70.0 + 25.0 * (1.0 - blend_weight))
        elif self.current_mode == NavigationMode.DEAD_RECKONING:
            # Slowly decays as DR duration increases
            dr_time = self.time_in_mode_s[NavigationMode.DEAD_RECKONING]
            return float(np.clip(90.0 - 0.5 * dr_time, 35.0, 90.0))
        return 75.0

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Returns diagnostic metrics for evaluation reporting."""
        total_time = sum(self.time_in_mode_s.values())
        return {
            "total_transitions": len(self.transition_history),
            "chatter_suppressed_samples": self.chatter_suppressed_count,
            "time_in_mode_s": {m.value: round(t, 2) for m, t in self.time_in_mode_s.items()},
            "time_in_mode_pct": {
                m.value: round((t / total_time * 100.0), 2) if total_time > 0 else 0.0
                for m, t in self.time_in_mode_s.items()
            },
            "transitions": [
                {
                    "timestamp_s": round(e.timestamp_s, 2),
                    "from_mode": e.from_mode.value,
                    "to_mode": e.to_mode.value,
                    "reason": e.reason,
                }
                for e in self.transition_history
            ],
        }
