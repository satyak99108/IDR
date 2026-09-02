"""
IDR MVP -- AI-Assisted Dead Reckoning
=======================================
Combines the Phase 5 AI velocity estimator with the Phase 3 strapdown INS
to produce a position trajectory with significantly less drift (MVP.md §11).

Pipeline:
    AI Velocity (scalar forward speed in m/s)
        +
    Gyroscope / Heading (quaternion integration from StrapdownINS)
        ↓
    NED Velocity = v_forward * [cos(heading), sin(heading), 0]
        ↓
    Position Integration (WGS-84)
        ↓
    Optional NHC Correction

Key difference from raw StrapdownINS:
    - Raw INS double-integrates noisy acceleration → velocity → position.
    - AI-assisted INS uses the CNN-predicted forward speed directly,
      bypassing the acceleration double-integration entirely.
    - Heading is still derived from gyroscope integration (same as Phase 3).

Classes:
    AIAssistedINS -- AI velocity + gyro heading → position trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .strapdown import INSState
from .nhc import NHCCorrector
from .integration import (
    normalize_quaternion,
    quaternion_from_euler,
    quaternion_to_euler,
    angular_velocity_to_quaternion_rate,
    latlon_update_ned,
)


class AIAssistedINS:
    """
    AI-Assisted Dead Reckoning engine.

    Uses an externally provided forward velocity estimate (from the Phase 5
    AI model) instead of double-integrating IMU acceleration. Heading is
    still computed by integrating gyroscope measurements via quaternions.

    Optionally applies NHC (Non-Holonomic Constraints) after each step.

    Usage:
        ai_ins = AIAssistedINS(enable_nhc=True)
        ai_ins.initialize(lat_deg, lon_deg, heading_deg)
        trajectory_df = ai_ins.run_batch_ai(df, ai_velocity_series)
    """

    def __init__(
        self,
        enable_nhc: bool = False,
        nhc_lateral_weight: float = 1.0,
        nhc_vertical_weight: float = 1.0,
    ):
        """
        Args:
            enable_nhc: Whether to apply NHC after each integration step.
            nhc_lateral_weight: NHC lateral suppression (1.0 = full).
            nhc_vertical_weight: NHC vertical suppression (1.0 = full).
        """
        self.enable_nhc = enable_nhc
        self._nhc = NHCCorrector(
            lateral_weight=nhc_lateral_weight,
            vertical_weight=nhc_vertical_weight,
        ) if enable_nhc else None
        self._state: Optional[INSState] = None

    # ------------------------------------------------------------------
    # Initialization (same interface as StrapdownINS)
    # ------------------------------------------------------------------

    def initialize(
        self,
        lat_deg: float,
        lon_deg: float,
        heading_deg: float,
        alt_m: float = 0.0,
        v_north: float = 0.0,
        v_east: float = 0.0,
        v_down: float = 0.0,
        timestamp_ms: int = 0,
    ) -> None:
        """Sets the initial navigation state (identical to StrapdownINS)."""
        yaw_rad = np.radians(heading_deg)
        q0 = quaternion_from_euler(0.0, 0.0, yaw_rad)

        self._state = INSState(
            timestamp_ms=timestamp_ms,
            lat_rad=np.radians(lat_deg),
            lon_rad=np.radians(lon_deg),
            alt_m=alt_m,
            v_north=v_north,
            v_east=v_east,
            v_down=v_down,
            q=q0,
            heading_rad=yaw_rad,
            roll_rad=0.0,
            pitch_rad=0.0,
        )

    # ------------------------------------------------------------------
    # Single-step update (AI velocity mode)
    # ------------------------------------------------------------------

    def update_ai(
        self,
        dt: float,
        gx_veh: float,
        gy_veh: float,
        gz_veh: float,
        ai_forward_velocity: float,
        timestamp_ms: int = 0,
    ) -> INSState:
        """
        Advance the navigation state by one timestep using AI-predicted
        forward velocity.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        gx_veh, gy_veh, gz_veh : float
            Gyroscope measurements in vehicle frame (rad/s).
        ai_forward_velocity : float
            AI-predicted scalar forward speed (m/s, ≥ 0).
        timestamp_ms : int
            Current epoch timestamp (ms).

        Returns
        -------
        INSState
            Updated navigation state.
        """
        if self._state is None:
            raise RuntimeError(
                "AIAssistedINS not initialized. Call initialize() first."
            )
        if dt <= 0.0:
            return self._state

        s = self._state
        q = s.q.copy()

        # ------------------------------------------------------------------
        # 1. Attitude update — integrate gyroscope (identical to StrapdownINS)
        # ------------------------------------------------------------------
        omega = np.array([gx_veh, gy_veh, gz_veh])
        dq_dt = angular_velocity_to_quaternion_rate(q, omega)
        q_new = normalize_quaternion(q + dq_dt * dt)

        roll_new, pitch_new, yaw_new = quaternion_to_euler(q_new)

        # ------------------------------------------------------------------
        # 2. AI velocity → NED velocity
        #    Forward velocity is projected along the heading vector.
        #    v_north = v_fwd * cos(heading)
        #    v_east  = v_fwd * sin(heading)
        #    v_down  = 0 (ground vehicle assumption)
        # ------------------------------------------------------------------
        v_fwd = max(0.0, ai_forward_velocity)  # Clamp to non-negative
        heading = yaw_new % (2.0 * np.pi)

        v_north_new = v_fwd * np.cos(heading)
        v_east_new = v_fwd * np.sin(heading)
        v_down_new = 0.0  # Ground vehicle — no vertical velocity

        # ------------------------------------------------------------------
        # 3. Optional NHC correction
        # ------------------------------------------------------------------
        if self._nhc is not None:
            v_north_new, v_east_new, v_down_new = self._nhc.correct_ned_velocity(
                v_north_new, v_east_new, v_down_new, heading,
            )

        # ------------------------------------------------------------------
        # 4. Position integration (midpoint velocity for accuracy)
        # ------------------------------------------------------------------
        v_n_mid = 0.5 * (s.v_north + v_north_new)
        v_e_mid = 0.5 * (s.v_east + v_east_new)

        lat_new, lon_new = latlon_update_ned(
            s.lat_rad, s.lon_rad, v_n_mid, v_e_mid, dt,
        )

        v_d_mid = 0.5 * (s.v_down + v_down_new)
        alt_new = s.alt_m - v_d_mid * dt

        # ------------------------------------------------------------------
        # 5. Store new state
        # ------------------------------------------------------------------
        self._state = INSState(
            timestamp_ms=timestamp_ms,
            lat_rad=lat_new,
            lon_rad=lon_new,
            alt_m=alt_new,
            v_north=v_north_new,
            v_east=v_east_new,
            v_down=v_down_new,
            q=q_new,
            heading_rad=heading,
            roll_rad=roll_new,
            pitch_rad=pitch_new,
        )
        return self._state

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def run_batch_ai(
        self,
        df: pd.DataFrame,
        ai_velocity: np.ndarray,
        gyro_cols: Tuple[str, str, str] = ("gx_veh", "gy_veh", "gz_veh"),
        timestamp_col: str = "timestamp",
        timestamp_unit: str = "ms",
    ) -> pd.DataFrame:
        """
        Run AI-assisted dead reckoning over an entire calibrated DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Calibrated IMU data.
        ai_velocity : ndarray of shape (N,)
            AI-predicted forward velocity for each sample (m/s).
            Use NaN or 0.0 for samples without a prediction (e.g. the
            initial ramp-up window).
        gyro_cols : tuple of str
            Column names for gyroscope axes.
        timestamp_col : str
            Timestamp column name.
        timestamp_unit : str
            'ms' or 's'.

        Returns
        -------
        pd.DataFrame
            Trajectory with lat_deg, lon_deg, speed_ms, heading_deg, etc.
        """
        if self._state is None:
            raise RuntimeError(
                "AIAssistedINS not initialized. Call initialize() first."
            )

        gx_col, gy_col, gz_col = gyro_cols

        records: List[dict] = []
        records.append(self._state.to_dict())

        timestamps = df[timestamp_col].values
        gx_arr = df[gx_col].values
        gy_arr = df[gy_col].values
        gz_arr = df[gz_col].values

        # Ensure ai_velocity has the right length
        vel = np.asarray(ai_velocity, dtype=np.float64)
        if len(vel) < len(df):
            # Pad with zeros at the start
            vel = np.concatenate([np.zeros(len(df) - len(vel)), vel])

        prev_ts = timestamps[0]

        for i in range(1, len(df)):
            ts = timestamps[i]

            if timestamp_unit == "ms":
                dt = (ts - prev_ts) / 1000.0
            else:
                dt = ts - prev_ts

            # Guard against bad timesteps
            if dt <= 0.0 or dt > 5.0:
                dt = 0.1  # fallback: 10 Hz

            v_fwd = vel[i]
            if np.isnan(v_fwd):
                v_fwd = 0.0

            state = self.update_ai(
                dt=dt,
                gx_veh=gx_arr[i],
                gy_veh=gy_arr[i],
                gz_veh=gz_arr[i],
                ai_forward_velocity=v_fwd,
                timestamp_ms=int(ts),
            )
            records.append(state.to_dict())
            prev_ts = ts

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> Optional[INSState]:
        """Returns the current navigation state."""
        return self._state

    def reset(self) -> None:
        """Clears the navigation state."""
        self._state = None
