"""
IDR MVP -- Frame Calibrator
Estimates pitch, roll, and phone-to-vehicle yaw offset to construct the 3D rotation matrix
transforming phone IMU coordinates into the vehicle navigation frame.
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd


class FrameCalibrator:
    """Computes phone-to-vehicle rotation matrix from stationary and moving data."""

    def __init__(self, target_gravity: float = 9.81):
        """
        Args:
            target_gravity: Standard Earth nominal gravity magnitude in m/s^2.
        """
        self.g0 = target_gravity

    @staticmethod
    def rotation_matrix_x(angle_rad: float) -> np.ndarray:
        """3D rotation matrix around X-axis (Roll)."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0,   c,  -s],
            [0.0,   s,   c]
        ])

    @staticmethod
    def rotation_matrix_y(angle_rad: float) -> np.ndarray:
        """3D rotation matrix around Y-axis (Pitch)."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [  c, 0.0,   s],
            [0.0, 1.0, 0.0],
            [ -s, 0.0,   c]
        ])

    @staticmethod
    def rotation_matrix_z(angle_rad: float) -> np.ndarray:
        """3D rotation matrix around Z-axis (Yaw)."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [  c,  -s, 0.0],
            [  s,   c, 0.0],
            [0.0, 0.0, 1.0]
        ])

    def estimate_pitch_and_roll(
        self,
        mean_static_accel: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Estimates Pitch and Roll angles from stationary accelerometer measurements.

        Assumes gravity vector points UPWARDS in phone rest state (e.g. resting flat on table
        yields az approx +9.81 m/s^2).

        Args:
            mean_static_accel: 3-element numpy array [ax_static, ay_static, az_static].

        Returns:
            Tuple of (pitch_rad, roll_rad, gravity_norm).
        """
        ax, ay, az = mean_static_accel
        g_norm = np.linalg.norm(mean_static_accel)

        if g_norm < 1e-3:
            return 0.0, 0.0, self.g0

        # Pitch (around Y axis): rotation bringing gravity to Z-axis
        pitch_rad = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

        # Roll (around X axis): rotation bringing gravity to Z-axis
        roll_rad = np.arctan2(ay, az)

        return pitch_rad, roll_rad, g_norm

    def estimate_yaw_offset(
        self,
        df: pd.DataFrame,
        R_level: np.ndarray,
        moving_mask: pd.Series,
        gnss_course_rad: pd.Series,
        accel_cols: Tuple[str, str, str] = ("ax", "ay", "az"),
    ) -> float:
        """
        Estimates phone-to-vehicle yaw offset using level-frame forward acceleration
        and GNSS course reference while moving.

        Args:
            df: Input DataFrame.
            R_level: 3x3 rotation matrix for Pitch and Roll alignment.
            moving_mask: Boolean mask indicating moving samples.
            gnss_course_rad: GNSS bearing series in radians.
            accel_cols: Column names for phone accelerometer.

        Returns:
            Yaw offset angle in radians.
        """
        moving_idx = df[moving_mask].index

        if len(moving_idx) == 0:
            return 0.0

        # Rotate moving accelerations to level frame
        phone_accel = df.loc[moving_idx, list(accel_cols)].values  # Shape (N, 3)
        level_accel = (R_level @ phone_accel.T).T                # Shape (N, 3)

        # Look for acceleration/deceleration bursts to determine level phone forward axis
        ax_level = level_accel[:, 0]
        ay_level = level_accel[:, 1]
        horiz_norm = np.sqrt(ax_level**2 + ay_level**2)

        # Select samples with clear horizontal acceleration (> 0.5 m/s^2)
        dynamic_mask = horiz_norm > 0.5
        if np.sum(dynamic_mask) > 10:
            # Angle of principal acceleration vector in level phone frame
            phone_forward_angle = np.arctan2(
                np.median(ay_level[dynamic_mask]),
                np.median(ax_level[dynamic_mask])
            )
        else:
            phone_forward_angle = 0.0

        # If GNSS course or reference heading is available, align phone forward with vehicle forward
        if "ref_heading" in df.columns and not df["ref_heading"].isna().all():
            ref_heading_rad = np.radians(df.loc[moving_idx, "ref_heading"].dropna().values)
            if len(ref_heading_rad) > 0:
                yaw_offset_rad = np.median(ref_heading_rad) - phone_forward_angle
                return yaw_offset_rad

        valid_course = gnss_course_rad.loc[moving_idx].dropna()
        if len(valid_course) > 5:
            yaw_offset_rad = np.median(valid_course.values) - phone_forward_angle
            return yaw_offset_rad

        return phone_forward_angle

    def calibrate(
        self,
        df: pd.DataFrame,
        static_mask: pd.Series,
        moving_mask: pd.Series,
        gnss_course_rad: Optional[pd.Series] = None,
        accel_cols: Tuple[str, str, str] = ("ax", "ay", "az"),
        gyro_cols: Tuple[str, str, str] = ("gx", "gy", "gz"),
    ) -> Dict[str, Any]:
        """
        Performs full 3D alignment calibration.

        Args:
            df: Input DataFrame with IMU and GNSS data.
            static_mask: Boolean series indicating stationary periods.
            moving_mask: Boolean series indicating moving periods.
            gnss_course_rad: Optional precomputed GNSS course series.
            accel_cols: Column names for accelerometer.
            gyro_cols: Column names for gyroscope.

        Returns:
            Dict containing calibration parameters and 3x3 rotation matrix R.
        """
        # Step 1: Calculate mean stationary acceleration
        static_idx = df[static_mask].index
        if len(static_idx) > 0:
            mean_static_accel = df.loc[static_idx, list(accel_cols)].mean().values
            mean_static_gyro = df.loc[static_idx, list(gyro_cols)].mean().values
        else:
            print("  [Calibrator] Warning: No stationary samples detected. Using overall mean.")
            mean_static_accel = df[list(accel_cols)].mean().values
            mean_static_gyro = df[list(gyro_cols)].mean().values

        # Step 2: Pitch & Roll calculation
        pitch_rad, roll_rad, g_norm = self.estimate_pitch_and_roll(mean_static_accel)

        # Level rotation matrix (R_level = R_y(pitch) * R_x(roll))
        R_x = self.rotation_matrix_x(roll_rad)
        R_y = self.rotation_matrix_y(pitch_rad)
        R_level = R_y @ R_x

        # Step 3: Phone-to-vehicle Yaw alignment
        if gnss_course_rad is None:
            gnss_course_rad = pd.Series(np.nan, index=df.index)

        yaw_offset_rad = self.estimate_yaw_offset(
            df, R_level, moving_mask, gnss_course_rad, accel_cols
        )

        # Step 4: Combine full 3D Rotation Matrix R = R_z(yaw) * R_y(pitch) * R_x(roll)
        R_z = self.rotation_matrix_z(yaw_offset_rad)
        R_total = R_z @ R_level

        return {
            "pitch_rad": pitch_rad,
            "roll_rad": roll_rad,
            "yaw_offset_rad": yaw_offset_rad,
            "pitch_deg": np.degrees(pitch_rad),
            "roll_deg": np.degrees(roll_rad),
            "yaw_offset_deg": np.degrees(yaw_offset_rad),
            "gravity_norm": g_norm,
            "gyro_bias": mean_static_gyro,
            "rotation_matrix": R_total,
            "R_level": R_level,
        }
