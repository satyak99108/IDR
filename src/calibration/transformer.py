"""
IDR MVP -- Coordinate Transformer
Applies 3D rotation matrices to transform phone-frame IMU data into the vehicle coordinate frame,
and subtracts gravity to produce linear motion accelerations.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


class CoordinateTransformer:
    """Transforms 3D sensor streams between phone frame and vehicle frame."""

    def __init__(self, rotation_matrix: Optional[np.ndarray] = None):
        """
        Args:
            rotation_matrix: 3x3 rotation matrix R transforming Phone -> Vehicle frame.
        """
        self.R = rotation_matrix if rotation_matrix is not None else np.eye(3)

    def set_rotation_matrix(self, rotation_matrix: np.ndarray) -> None:
        """Set or update the active 3x3 rotation matrix."""
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be shape (3, 3).")
        self.R = rotation_matrix

    def transform_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Applies rotation matrix to N 3D vectors.

        Args:
            vectors: (N, 3) numpy array of vectors in Phone frame.

        Returns:
            (N, 3) numpy array of transformed vectors in Vehicle frame.
        """
        return (self.R @ vectors.T).T

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        rotation_matrix: Optional[np.ndarray] = None,
        accel_cols: Tuple[str, str, str] = ("ax", "ay", "az"),
        gyro_cols: Tuple[str, str, str] = ("gx", "gy", "gz"),
        gyro_bias: Optional[np.ndarray] = None,
        gravity_magnitude: float = 9.81,
    ) -> pd.DataFrame:
        """
        Transforms IMU streams in DataFrame to vehicle coordinate frame.

        Columns produced:
            - ax_veh, ay_veh, az_veh: Accelerations in Vehicle frame (including gravity).
            - ax_veh_lin, ay_veh_lin, az_veh_lin: Linear vehicle accelerations (gravity removed).
            - gx_veh, gy_veh, gz_veh: Angular velocities in Vehicle frame.

        Args:
            df: Input DataFrame with phone IMU columns.
            rotation_matrix: 3x3 rotation matrix (overrides constructor value if set).
            accel_cols: Accelerometer column names.
            gyro_cols: Gyroscope column names.
            gyro_bias: Optional (3,) static gyro bias to subtract prior to rotation.
            gravity_magnitude: Expected magnitude of gravity vector in m/s^2.

        Returns:
            DataFrame augmented with vehicle-frame IMU columns.
        """
        R = rotation_matrix if rotation_matrix is not None else self.R
        result = df.copy()

        ax_col, ay_col, az_col = accel_cols
        gx_col, gy_col, gz_col = gyro_cols

        # Extract phone IMU arrays
        phone_accel = result[[ax_col, ay_col, az_col]].values  # (N, 3)
        phone_gyro = result[[gx_col, gy_col, gz_col]].values    # (N, 3)

        # Subtract gyro bias if supplied
        if gyro_bias is not None:
            phone_gyro = phone_gyro - gyro_bias

        # Rotate to vehicle frame
        veh_accel = (R @ phone_accel.T).T  # (N, 3)
        veh_gyro = (R @ phone_gyro.T).T    # (N, 3)

        # Remove gravity along vertical (Z) axis in vehicle frame
        veh_accel_lin = veh_accel.copy()
        veh_accel_lin[:, 2] -= gravity_magnitude

        # Store vehicle-frame columns
        result["ax_veh"] = veh_accel[:, 0]
        result["ay_veh"] = veh_accel[:, 1]
        result["az_veh"] = veh_accel[:, 2]

        result["ax_veh_lin"] = veh_accel_lin[:, 0]
        result["ay_veh_lin"] = veh_accel_lin[:, 1]
        result["az_veh_lin"] = veh_accel_lin[:, 2]

        result["gx_veh"] = veh_gyro[:, 0]
        result["gy_veh"] = veh_gyro[:, 1]
        result["gz_veh"] = veh_gyro[:, 2]

        return result
