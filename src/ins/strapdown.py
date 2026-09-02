"""
IDR MVP -- Strapdown INS
=========================
Implements a baseline strapdown Inertial Navigation System (INS) for Phase 3.

Pipeline:
    Vehicle-frame IMU (gravity-subtracted)
        ↓
    Quaternion orientation integration (gyroscope)
        ↓
    Body → NED rotation
        ↓
    NED linear acceleration
        ↓
    NED velocity integration
        ↓
    NED position integration (WGS-84 lat/lon)

This is deliberately a non-AI, physics-only baseline.
Its purpose is to demonstrate raw IMU drift — motivating Phases 5–8.

Classes:
    INSState      -- Frozen navigation state at one timestep.
    StrapdownINS  -- Full strapdown INS pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .integration import (
    normalize_quaternion,
    quaternion_from_euler,
    quaternion_to_rotation_matrix,
    quaternion_to_euler,
    angular_velocity_to_quaternion_rate,
    latlon_update_ned,
    haversine_distance,
)


# ---------------------------------------------------------------------------
# Navigation State
# ---------------------------------------------------------------------------

@dataclass
class INSState:
    """
    Represents the full navigation state at a single timestep.

    Position is stored in radians (lat, lon) and metres (alt).
    Velocity is in the NED frame (m/s).
    Orientation is stored as a unit quaternion [w, x, y, z].
    """
    timestamp_ms: int            # Epoch milliseconds
    lat_rad: float               # Geodetic latitude (rad)
    lon_rad: float               # Geodetic longitude (rad)
    alt_m: float                 # Altitude above WGS-84 ellipsoid (m)
    v_north: float               # North velocity (m/s)
    v_east: float                # East velocity (m/s)
    v_down: float                # Down velocity (m/s)
    q: np.ndarray                # Attitude quaternion [w, x, y, z]
    heading_rad: float = 0.0     # Derived yaw (rad, 0=North, CW positive)
    roll_rad: float = 0.0
    pitch_rad: float = 0.0

    @property
    def lat_deg(self) -> float:
        return np.degrees(self.lat_rad)

    @property
    def lon_deg(self) -> float:
        return np.degrees(self.lon_rad)

    @property
    def speed_ms(self) -> float:
        """Horizontal ground speed in m/s."""
        return np.sqrt(self.v_north**2 + self.v_east**2)

    def to_dict(self) -> dict:
        return {
            "timestamp_ms": self.timestamp_ms,
            "lat_deg": self.lat_deg,
            "lon_deg": self.lon_deg,
            "alt_m": self.alt_m,
            "v_north": self.v_north,
            "v_east": self.v_east,
            "v_down": self.v_down,
            "speed_ms": self.speed_ms,
            "heading_deg": np.degrees(self.heading_rad) % 360.0,
            "roll_deg": np.degrees(self.roll_rad),
            "pitch_deg": np.degrees(self.pitch_rad),
        }


# ---------------------------------------------------------------------------
# Strapdown INS
# ---------------------------------------------------------------------------

class StrapdownINS:
    """
    Baseline strapdown INS for the IDR MVP Phase 3.

    Integrates vehicle-frame IMU measurements (already gravity-compensated
    by Phase 2 calibration) to produce a dead-reckoned position trajectory.

    Usage:
        ins = StrapdownINS()
        ins.initialize(lat_deg, lon_deg, heading_deg, alt_m)
        trajectory_df = ins.run_batch(df_calibrated)
    """

    # Gravity magnitude used for sanity check; Phase 2 value is preferred.
    _G0 = 9.81  # m/s²

    def __init__(self, gravity_magnitude: float = 9.81):
        """
        Args:
            gravity_magnitude: Calibrated gravity magnitude from Phase 2
                               (loaded from calibration_params.json).
        """
        self._g = gravity_magnitude
        self._state: Optional[INSState] = None

    # ------------------------------------------------------------------
    # Initialization
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
        """
        Sets the initial INS navigation state.

        Args:
            lat_deg:       Initial geodetic latitude (degrees).
            lon_deg:       Initial geodetic longitude (degrees).
            heading_deg:   Initial vehicle heading (degrees, 0=North, CW).
            alt_m:         Initial altitude above WGS-84 ellipsoid (metres).
            v_north:       Initial north velocity (m/s).
            v_east:        Initial east velocity (m/s).
            v_down:        Initial down velocity (m/s).
            timestamp_ms:  Initial epoch timestamp (milliseconds).
        """
        yaw_rad = np.radians(heading_deg)
        # Assume level vehicle at initialization (roll=0, pitch=0)
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
    # Single Step
    # ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        ax_lin: float,
        ay_lin: float,
        az_lin: float,
        gx_veh: float,
        gy_veh: float,
        gz_veh: float,
        timestamp_ms: int = 0,
    ) -> INSState:
        """
        Advances the INS state by one timestep.

        Args:
            dt:           Time step in seconds.
            ax_lin:       Forward linear acceleration (gravity subtracted, m/s²).
            ay_lin:       Lateral linear acceleration (m/s²).
            az_lin:       Vertical linear acceleration (m/s², down positive in NED).
            gx_veh:       Gyroscope X (vehicle forward axis, rad/s).
            gy_veh:       Gyroscope Y (vehicle lateral axis, rad/s).
            gz_veh:       Gyroscope Z (vehicle vertical/down axis, rad/s).
            timestamp_ms: Current epoch timestamp (ms).

        Returns:
            Updated INSState.
        """
        if self._state is None:
            raise RuntimeError("StrapdownINS not initialized. Call initialize() first.")
        if dt <= 0.0:
            return self._state

        s = self._state
        q = s.q.copy()

        # ------------------------------------------------------------------
        # 1. Attitude update via quaternion integration (first-order Euler)
        #    dq/dt = 0.5 * Omega(ω) * q
        # ------------------------------------------------------------------
        omega = np.array([gx_veh, gy_veh, gz_veh])
        dq_dt = angular_velocity_to_quaternion_rate(q, omega)
        q_new = normalize_quaternion(q + dq_dt * dt)

        # Extract Euler from updated quaternion
        roll_new, pitch_new, yaw_new = quaternion_to_euler(q_new)

        # ------------------------------------------------------------------
        # 2. Body → NED rotation of linear acceleration
        #    a_body = [ax_lin, ay_lin, az_lin]  (gravity already removed)
        #    a_NED  = R_bn @ a_body
        # ------------------------------------------------------------------
        R_bn = quaternion_to_rotation_matrix(q_new)  # body → NED
        a_body = np.array([ax_lin, ay_lin, az_lin])
        a_ned = R_bn @ a_body  # [a_north, a_east, a_down]

        # ------------------------------------------------------------------
        # 3. Velocity integration (Euler)
        # ------------------------------------------------------------------
        v_north_new = s.v_north + a_ned[0] * dt
        v_east_new  = s.v_east  + a_ned[1] * dt
        v_down_new  = s.v_down  + a_ned[2] * dt

        # ------------------------------------------------------------------
        # 4. Position integration (WGS-84 NED → lat/lon)
        #    Use midpoint velocity for better accuracy
        # ------------------------------------------------------------------
        v_n_mid = 0.5 * (s.v_north + v_north_new)
        v_e_mid = 0.5 * (s.v_east  + v_east_new)

        lat_new, lon_new = latlon_update_ned(
            s.lat_rad, s.lon_rad, v_n_mid, v_e_mid, dt
        )

        # Altitude update (down positive → subtract from altitude)
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
            heading_rad=yaw_new % (2.0 * np.pi),
            roll_rad=roll_new,
            pitch_rad=pitch_new,
        )
        return self._state

    # ------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------

    def run_batch(
        self,
        df: pd.DataFrame,
        accel_cols: Tuple[str, str, str] = ("ax_veh_lin", "ay_veh_lin", "az_veh_lin"),
        gyro_cols: Tuple[str, str, str] = ("gx_veh", "gy_veh", "gz_veh"),
        timestamp_col: str = "timestamp",
        timestamp_unit: str = "ms",
    ) -> pd.DataFrame:
        """
        Runs strapdown INS integration over an entire calibrated DataFrame.

        Args:
            df:             Calibrated DataFrame from Phase 2.
            accel_cols:     Column names for gravity-subtracted vehicle-frame acceleration.
            gyro_cols:      Column names for vehicle-frame gyroscope.
            timestamp_col:  Column with epoch timestamps.
            timestamp_unit: 'ms' (milliseconds) or 's' (seconds).

        Returns:
            DataFrame with one row per IMU sample, containing:
                timestamp_ms, lat_deg, lon_deg, alt_m,
                v_north, v_east, v_down, speed_ms,
                heading_deg, roll_deg, pitch_deg
        """
        if self._state is None:
            raise RuntimeError("StrapdownINS not initialized. Call initialize() first.")

        ax_col, ay_col, az_col = accel_cols
        gx_col, gy_col, gz_col = gyro_cols

        records: List[dict] = []

        # Record the initial state
        records.append(self._state.to_dict())

        timestamps = df[timestamp_col].values
        ax_arr = df[ax_col].values
        ay_arr = df[ay_col].values
        az_arr = df[az_col].values
        gx_arr = df[gx_col].values
        gy_arr = df[gy_col].values
        gz_arr = df[gz_col].values

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

            state = self.update(
                dt=dt,
                ax_lin=ax_arr[i],
                ay_lin=ay_arr[i],
                az_lin=az_arr[i],
                gx_veh=gx_arr[i],
                gy_veh=gy_arr[i],
                gz_veh=gz_arr[i],
                timestamp_ms=int(ts),
            )
            records.append(state.to_dict())
            prev_ts = ts

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # State Accessor
    # ------------------------------------------------------------------

    @property
    def state(self) -> Optional[INSState]:
        """Returns the current navigation state."""
        return self._state

    def reset(self) -> None:
        """Clears the navigation state. Call initialize() before running again."""
        self._state = None
