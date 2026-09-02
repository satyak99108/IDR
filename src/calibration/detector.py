"""
IDR MVP -- Stationary & Motion Detector
Identifies stationary intervals (for gravity estimation) and moving intervals
with reliable GNSS trajectory (for phone-to-vehicle yaw alignment).
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


class StaticDetector:
    """Detects stationary and linear-motion intervals in IMU and GNSS data."""

    def __init__(
        self,
        sampling_rate_hz: float = 10.0,
        accel_std_threshold: float = 0.25,   # m/s^2
        gyro_norm_threshold: float = 0.08,   # rad/s (~4.5 deg/s)
        moving_speed_threshold: float = 2.0,  # m/s (~7.2 km/h)
        stationary_speed_threshold: float = 0.5, # m/s (~1.8 km/h)
        window_size_sec: float = 1.0,
    ):
        """
        Args:
            sampling_rate_hz: Sampling frequency of IMU stream.
            accel_std_threshold: Max accel std dev within window for static state.
            gyro_norm_threshold: Max gyro norm within window for static state.
            moving_speed_threshold: Min GNSS speed to consider vehicle moving reliably.
            stationary_speed_threshold: Max GNSS speed to consider vehicle stationary.
            window_size_sec: Duration of sliding window in seconds.
        """
        self.fs = sampling_rate_hz
        self.accel_std_thresh = accel_std_threshold
        self.gyro_norm_thresh = gyro_norm_threshold
        self.moving_speed_thresh = moving_speed_threshold
        self.stationary_speed_thresh = stationary_speed_threshold
        self.window_size = int(max(1, round(window_size_sec * sampling_rate_hz)))

    def detect_stationary_samples(
        self,
        df: pd.DataFrame,
        accel_cols: Tuple[str, str, str] = ("ax", "ay", "az"),
        gyro_cols: Tuple[str, str, str] = ("gx", "gy", "gz"),
        speed_col: str = "gnss_speed",
    ) -> pd.Series:
        """
        Computes a boolean mask indicating stationary samples.

        Args:
            df: DataFrame containing IMU and optional GNSS speed data.
            accel_cols: Column names for accelerometer axes.
            gyro_cols: Column names for gyroscope axes.
            speed_col: Column name for GNSS speed.

        Returns:
            Boolean pd.Series (True = stationary, False = moving/dynamic).
        """
        ax_col, ay_col, az_col = accel_cols
        gx_col, gy_col, gz_col = gyro_cols

        # Compute accel magnitude and rolling std dev
        accel_norm = np.sqrt(df[ax_col]**2 + df[ay_col]**2 + df[az_col]**2)
        accel_std = accel_norm.rolling(window=self.window_size, center=True, min_periods=1).std().fillna(0)

        # Compute gyro norm
        gyro_norm = np.sqrt(df[gx_col]**2 + df[gy_col]**2 + df[gz_col]**2)
        gyro_mean = gyro_norm.rolling(window=self.window_size, center=True, min_periods=1).mean().fillna(0)

        # Base IMU static condition
        is_static = (accel_std < self.accel_std_thresh) & (gyro_mean < self.gyro_norm_thresh)

        # Combine with GNSS speed if available
        if speed_col in df.columns and not df[speed_col].isna().all():
            speed = df[speed_col].fillna(0)
            is_static = is_static & (speed < self.stationary_speed_thresh)

        return is_static

    def detect_moving_samples(
        self,
        df: pd.DataFrame,
        speed_col: str = "gnss_speed",
        accel_cols: Tuple[str, str, str] = ("ax", "ay", "az"),
    ) -> pd.Series:
        """
        Computes a boolean mask indicating reliable moving samples for heading estimation.

        Args:
            df: DataFrame containing GNSS speed data.
            speed_col: Column name for GNSS speed.
            accel_cols: Column names for accelerometer axes.

        Returns:
            Boolean pd.Series (True = moving above threshold).
        """
        if speed_col in df.columns and not df[speed_col].isna().all():
            moving = df[speed_col].fillna(0) > self.moving_speed_thresh
        else:
            # Fallback: estimate motion from high variance in horizontal acceleration or gyro
            ax_col, ay_col, _ = accel_cols
            accel_horiz = np.sqrt(df[ax_col]**2 + df[ay_col]**2)
            moving = accel_horiz.rolling(window=self.window_size, center=True, min_periods=1).std().fillna(0) > 0.3

        return moving

    def compute_gnss_course(
        self,
        df: pd.DataFrame,
        lat_col: str = "gnss_lat",
        lon_col: str = "gnss_lon",
    ) -> pd.Series:
        """
        Calculates GNSS course/heading angle (in radians, -pi to +pi) from consecutive lat/lon.

        Args:
            df: DataFrame containing lat and lon coordinates.
            lat_col: Latitude column name.
            lon_col: Longitude column name.

        Returns:
            pd.Series of bearing angles in radians.
        """
        if lat_col not in df.columns or lon_col not in df.columns:
            return pd.Series(np.nan, index=df.index)

        lat = np.radians(df[lat_col])
        lon = np.radians(df[lon_col])

        dlon = lon.diff()
        dlat = lat.diff()

        # Approximate flat-earth bearing calculation for small displacements
        # bearing = atan2(sin(dlon) * cos(lat2), cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon))
        lat_mean = (lat + lat.shift(1)) / 2.0
        y = dlon * np.cos(lat_mean)
        x = dlat

        course_rad = np.arctan2(y, x)
        course_rad.iloc[0] = course_rad.iloc[1] if len(course_rad) > 1 else np.nan

        return course_rad
