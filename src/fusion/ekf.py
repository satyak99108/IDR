"""
IDR MVP -- Extended Kalman Filter (EKF) for GNSS + INS Fusion
==============================================================
Implements a 10-state Extended Kalman Filter using FilterPy per MVP.md §12
and TECH_STACK.md §9.

State Vector (10 dimensions):
    x = [p_n, p_e, p_d, v_n, v_e, v_d, psi, b_ax, b_ay, b_gz]^T

Where:
    p_n, p_e, p_d : Position in local NED frame (meters)
    v_n, v_e, v_d : Velocity in local NED frame (m/s)
    psi           : Heading angle / yaw (radians, 0 = North, CW positive)
    b_ax, b_ay    : Accelerometer bias along body forward and lateral axes (m/s²)
    b_gz          : Gyroscope bias along vehicle yaw axis (rad/s)

Supported Measurement Updates:
    1. GNSS Position & Velocity (p_n, p_e, p_d, v_n, v_e)
    2. GNSS Heading / Course (psi)
    3. AI Forward Velocity (v_fwd = v_n*cos(psi) + v_e*sin(psi))
    4. Non-Holonomic Constraints (v_lat ≈ 0, v_down ≈ 0)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Dict, Any
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter


def wrap_angle(angle_rad: float) -> float:
    """Wraps angle to [-pi, pi)."""
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


from src.ins.integration import (
    angular_velocity_to_quaternion_rate,
    normalize_quaternion,
    quaternion_to_euler,
    quaternion_from_euler,
)


class NavigationEKF(ExtendedKalmanFilter):
    """
    10-State Extended Kalman Filter for ground vehicle positioning.

    Couples IMU strapdown dynamics with GNSS observations and AI / NHC
    pseudo-measurements.
    """

    STATE_DIM = 10

    def __init__(
        self,
        q_pos: float = 0.05,       # Position process noise (m)
        q_vel: float = 0.20,       # Velocity process noise (m/s)
        q_psi: float = 0.02,       # Heading process noise (rad) — increased for MEMS gyro drift
        q_acc_bias: float = 1e-4,  # Accel bias random walk (m/s²)
        q_gyro_bias: float = 5e-5, # Gyro bias random walk (rad/s) — increased for MEMS gyro
    ):
        super().__init__(dim_x=self.STATE_DIM, dim_z=5)

        # Initial state estimate
        self.x = np.zeros(self.STATE_DIM, dtype=np.float64)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Initial state covariance (P)
        self.P = np.diag([
            10.0**2, 10.0**2, 20.0**2,   # Pos (p_n, p_e, p_d) in m
            2.0**2, 2.0**2, 2.0**2,      # Vel (v_n, v_e, v_d) in m/s
            (np.radians(10.0))**2,       # Heading (psi) in rad
            0.1**2, 0.1**2,              # Accel biases (m/s²)
            (np.radians(1.0))**2,        # Gyro bias (rad/s)
        ]).astype(np.float64)

        # Process noise components
        self._q_pos = q_pos
        self._q_vel = q_vel
        self._q_psi = q_psi
        self._q_acc_bias = q_acc_bias
        self._q_gyro_bias = q_gyro_bias

        # Current epoch time
        self.last_timestamp_s: Optional[float] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_state(
        self,
        p_n: float = 0.0,
        p_e: float = 0.0,
        p_d: float = 0.0,
        v_n: float = 0.0,
        v_e: float = 0.0,
        v_d: float = 0.0,
        heading_rad: float = 0.0,
        timestamp_s: float = 0.0,
    ) -> None:
        """Initializes the filter state vector and attitude quaternion."""
        psi_init = wrap_angle(heading_rad)
        self.x = np.array([
            p_n, p_e, p_d,
            v_n, v_e, v_d,
            psi_init,
            0.0, 0.0, 0.0,
        ], dtype=np.float64)

        self.q = quaternion_from_euler(0.0, 0.0, psi_init)
        self.last_timestamp_s = timestamp_s

    # ------------------------------------------------------------------
    # Prediction Step (IMU Integration)
    # ------------------------------------------------------------------

    def predict_imu(
        self,
        dt: float,
        ax_lin: float,
        ay_lin: float,
        az_lin: float,
        gz_veh: float,
        gx_veh: float = 0.0,
        gy_veh: float = 0.0,
    ) -> np.ndarray:
        """
        State propagation step using vehicle-frame calibrated IMU inputs.

        Parameters
        ----------
        dt : float
            Timestep duration in seconds.
        ax_lin, ay_lin, az_lin : float
            Vehicle-frame linear acceleration (gravity-subtracted) in m/s².
        gz_veh, gx_veh, gy_veh : float
            Vehicle-frame 3D angular velocities in rad/s.

        Returns
        -------
        Propagated state vector x.
        """
        if dt <= 0.0:
            return self.x

        # Extract current state
        pn, pe, pd = self.x[0], self.x[1], self.x[2]
        vn, ve, vd = self.x[3], self.x[4], self.x[5]
        psi = self.x[6]
        b_ax, b_ay, b_gz = self.x[7], self.x[8], self.x[9]

        # Correct IMU measurements with current bias estimates
        ax_corr = ax_lin - b_ax
        ay_corr = ay_lin - b_ay
        gz_corr = gz_veh - b_gz

        # 1. Heading propagation (planar ground vehicle)
        psi_new = wrap_angle(psi + gz_corr * dt)
        self.q = quaternion_from_euler(0.0, 0.0, psi_new)

        # 2. Body -> NED acceleration rotation
        cos_psi = np.cos(psi_new)
        sin_psi = np.sin(psi_new)
        an = ax_corr * cos_psi - ay_corr * sin_psi
        ae = ax_corr * sin_psi + ay_corr * cos_psi
        ad = az_lin  # Vertical body axis aligned with Down

        # 3. Velocity propagation
        vn_new = vn + an * dt
        ve_new = ve + ae * dt
        vd_new = vd + ad * dt

        # 4. Position propagation (trapezoidal)
        pn_new = pn + 0.5 * (vn + vn_new) * dt
        pe_new = pe + 0.5 * (ve + ve_new) * dt
        pd_new = pd + 0.5 * (vd + vd_new) * dt

        # Updated state vector
        self.x = np.array([
            pn_new, pe_new, pd_new,
            vn_new, ve_new, vd_new,
            psi_new,
            b_ax, b_ay, float(np.clip(b_gz, -0.05, 0.05)),
        ], dtype=np.float64)

        # 5. Jacobian of state transition matrix (F)
        F = np.eye(self.STATE_DIM, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        F[3, 6] = (-ax_corr * sin_psi - ay_corr * cos_psi) * dt
        F[4, 6] = ( ax_corr * cos_psi - ay_corr * sin_psi) * dt
        F[0, 6] = 0.5 * F[3, 6] * dt
        F[1, 6] = 0.5 * F[4, 6] * dt

        F[3, 7] = -cos_psi * dt
        F[3, 8] =  sin_psi * dt
        F[4, 7] = -sin_psi * dt
        F[4, 8] = -cos_psi * dt
        F[6, 9] = -dt

        # 6. Covariance propagation
        Q = np.diag([
            (self._q_pos * dt)**2,
            (self._q_pos * dt)**2,
            (self._q_pos * dt * 2.0)**2,
            (self._q_vel * dt)**2,
            (self._q_vel * dt)**2,
            (self._q_vel * dt)**2,
            (self._q_psi * dt)**2,
            (self._q_acc_bias * dt)**2,
            (self._q_acc_bias * dt)**2,
            (self._q_gyro_bias * dt)**2,
        ])
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

        return self.x

    def predict_ai_dr(
        self,
        dt: float,
        ai_forward_speed: float,
        gz_veh: float = 0.0,
        gx_veh: float = 0.0,
        gy_veh: float = 0.0,
    ) -> np.ndarray:
        """
        Dead-reckoning state propagation using gyro yaw rate + AI forward velocity (MVP.md §12).
        Bypasses noisy accelerometer double-integration to prevent quadratic positional drift during blackouts.
        """
        if dt <= 0.0:
            return self.x

        # 1. Heading propagation (planar ground vehicle)
        gz_corr = gz_veh - self.x[9]
        psi_new = wrap_angle(self.x[6] + gz_corr * dt)
        self.x[6] = psi_new
        self.q = quaternion_from_euler(0.0, 0.0, psi_new)

        # 2. Forward velocity from AI model projected along heading
        v_fwd = max(0.0, ai_forward_speed)
        vn_new = v_fwd * np.cos(psi_new)
        ve_new = v_fwd * np.sin(psi_new)
        vd_new = 0.0

        # 3. Position propagation (trapezoidal)
        pn_new = self.x[0] + 0.5 * (self.x[3] + vn_new) * dt
        pe_new = self.x[1] + 0.5 * (self.x[4] + ve_new) * dt
        pd_new = self.x[2] + 0.5 * (self.x[5] + vd_new) * dt

        self.x[0] = pn_new
        self.x[1] = pe_new
        self.x[2] = pd_new
        self.x[3] = vn_new
        self.x[4] = ve_new
        self.x[5] = vd_new

        # 4. Uncertainty propagation
        self.P[0, 0] += (0.4 * dt)**2
        self.P[1, 1] += (0.4 * dt)**2
        self.P[3, 3] = (self._q_vel * dt)**2
        self.P[4, 4] = (self._q_vel * dt)**2
        self.P[6, 6] += (self._q_psi * dt)**2

        return self.x

    # ------------------------------------------------------------------
    # Measurement Update: GNSS Position & Velocity
    # ------------------------------------------------------------------

    def update_gnss_pv(
        self,
        p_n_meas: float,
        p_e_meas: float,
        p_d_meas: float,
        v_n_meas: float,
        v_e_meas: float,
        pos_std: float = 3.0,
        vel_std: float = 0.3,
    ) -> None:
        """
        Linear measurement update using GNSS NED position and velocity fixes.
        """
        z = np.array([p_n_meas, p_e_meas, p_d_meas, v_n_meas, v_e_meas], dtype=np.float64)

        # Measurement model H (5 x 10)
        H = np.zeros((5, self.STATE_DIM), dtype=np.float64)
        H[0, 0] = 1.0  # p_n
        H[1, 1] = 1.0  # p_e
        H[2, 2] = 1.0  # p_d
        H[3, 3] = 1.0  # v_n
        H[4, 4] = 1.0  # v_e

        # Measurement noise R (5 x 5)
        R = np.diag([
            pos_std**2,
            pos_std**2,
            (pos_std * 2.0)**2,
            vel_std**2,
            vel_std**2,
        ])

        # Innovation
        y = z - H @ self.x

        # Kalman gain & update
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[6] = wrap_angle(self.x[6])

        # Joseph form update for numerical stability: P = (I - KH)P(I - KH)^T + KRK^T
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    # Measurement Update: GNSS Heading / Course
    # ------------------------------------------------------------------

    def update_gnss_heading(
        self,
        heading_rad_meas: float,
        heading_std_rad: float = np.radians(5.0),
    ) -> None:
        """Scalar measurement update for GNSS course angle."""
        z = wrap_angle(heading_rad_meas)
        y = wrap_angle(z - self.x[6])  # Innovation with wrap

        H = np.zeros((1, self.STATE_DIM), dtype=np.float64)
        H[0, 6] = 1.0

        R = np.array([[heading_std_rad**2]], dtype=np.float64)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K * y).flatten()
        self.x[6] = wrap_angle(self.x[6])
        self.q = quaternion_from_euler(0.0, 0.0, self.x[6])

        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    # Measurement Update: AI Forward Velocity (Outages & Tunnel DR)
    # ------------------------------------------------------------------

    def update_ai_velocity(
        self,
        v_forward_meas: float,
        vel_std: float = 0.6,
    ) -> None:
        """
        Nonlinear update using AI model's estimated forward velocity.

        h(x) = v_n * cos(psi) + v_e * sin(psi)
        """
        vn = self.x[3]
        ve = self.x[4]
        psi = self.x[6]

        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        # Expected forward speed
        h_x = vn * cos_psi + ve * sin_psi
        y = v_forward_meas - h_x

        # Jacobian H (1 x 10)
        H = np.zeros((1, self.STATE_DIM), dtype=np.float64)
        H[0, 3] = cos_psi
        H[0, 4] = sin_psi
        H[0, 6] = -vn * sin_psi + ve * cos_psi  # = lateral speed

        R = np.array([[vel_std**2]], dtype=np.float64)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K * y).flatten()
        self.x[6] = wrap_angle(self.x[6])

        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    # Measurement Update: Non-Holonomic Constraints (NHC)
    # ------------------------------------------------------------------

    def update_nhc(
        self,
        lateral_std: float = 0.15,
        vertical_std: float = 0.15,
    ) -> None:
        """
        Enforces vehicle Non-Holonomic Constraints:
            v_lateral  = -v_n * sin(psi) + v_e * cos(psi) ≈ 0
            v_vertical =  v_d ≈ 0
        """
        vn = self.x[3]
        ve = self.x[4]
        vd = self.x[5]
        psi = self.x[6]

        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        # Measurements: [0.0, 0.0]
        z = np.zeros(2, dtype=np.float64)
        h_x = np.array([
            -vn * sin_psi + ve * cos_psi,  # Lateral speed
            vd,                            # Vertical speed
        ])
        y = z - h_x

        # Jacobian H (2 x 10)
        H = np.zeros((2, self.STATE_DIM), dtype=np.float64)
        # Lateral row
        H[0, 3] = -sin_psi
        H[0, 4] =  cos_psi
        H[0, 6] = -vn * cos_psi - ve * sin_psi  # = -forward speed
        # Vertical row
        H[1, 5] = 1.0

        R = np.diag([lateral_std**2, vertical_std**2])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[6] = wrap_angle(self.x[6])

        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    # State Accessors
    # ------------------------------------------------------------------

    @property
    def position_ned(self) -> Tuple[float, float, float]:
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    @property
    def velocity_ned(self) -> Tuple[float, float, float]:
        return float(self.x[3]), float(self.x[4]), float(self.x[5])

    @property
    def speed_ms(self) -> float:
        return float(np.sqrt(self.x[3]**2 + self.x[4]**2))

    @property
    def heading_rad(self) -> float:
        return float(self.x[6])

    @property
    def heading_deg(self) -> float:
        return float(np.degrees(self.x[6]) % 360.0)

    @property
    def accel_biases(self) -> Tuple[float, float]:
        return float(self.x[7]), float(self.x[8])

    @property
    def gyro_bias(self) -> float:
        return float(self.x[9])

    @property
    def state_uncertainty_trace(self) -> float:
        """Trace of position & velocity covariance blocks (confidence indicator)."""
        return float(np.trace(self.P[:6, :6]))
