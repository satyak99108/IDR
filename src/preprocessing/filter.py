"""
IDR MVP -- Signal Filter
Applies signal processing filters (low-pass, median) to IMU sensor data.
Uses SciPy for all filtering operations per TECH_STACK.md.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as sig


class SignalFilter:
    """Applies configurable filters to sensor data columns."""

    def __init__(
        self,
        sampling_rate_hz: float = 10.0,
    ):
        """
        Args:
            sampling_rate_hz: The data sampling rate in Hz.
        """
        self.fs = sampling_rate_hz

    def butterworth_lowpass(
        self,
        data: np.ndarray,
        cutoff_hz: float = 4.0,
        order: int = 4,
    ) -> np.ndarray:
        """
        Apply a zero-phase Butterworth low-pass filter.

        Args:
            data: 1D signal array.
            cutoff_hz: Cutoff frequency in Hz.
            order: Filter order.

        Returns:
            Filtered signal.
        """
        if len(data) < 3 * order:
            return data

        nyquist = self.fs / 2.0
        if cutoff_hz >= nyquist:
            # Cutoff at or above Nyquist -- no filtering needed
            return data

        normalized_cutoff = cutoff_hz / nyquist
        b, a = sig.butter(order, normalized_cutoff, btype="low")

        # Use filtfilt for zero-phase filtering (no time delay)
        # padlen handles edge effects
        padlen = min(3 * max(len(a), len(b)), len(data) - 1)
        try:
            filtered = sig.filtfilt(b, a, data, padlen=padlen)
        except ValueError:
            # Fallback if filtfilt fails
            filtered = sig.lfilter(b, a, data)

        return filtered

    def median_filter(
        self,
        data: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """
        Apply a median filter for spike removal.

        Args:
            data: 1D signal array.
            kernel_size: Must be odd. Size of the sliding window.

        Returns:
            Median-filtered signal.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1

        if len(data) < kernel_size:
            return data

        return sig.medfilt(data, kernel_size=kernel_size)

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        columns: list[str],
        filter_type: str = "butterworth",
        cutoff_hz: float = 4.0,
        order: int = 4,
        median_kernel: int = 5,
        suffix: str = "",
        inplace: bool = True,
    ) -> pd.DataFrame:
        """
        Apply filtering to specified DataFrame columns.

        Args:
            df: Input DataFrame.
            columns: Column names to filter.
            filter_type: 'butterworth', 'median', or 'both'.
            cutoff_hz: Cutoff frequency for Butterworth filter.
            order: Butterworth filter order.
            median_kernel: Kernel size for median filter.
            suffix: If non-empty, create new columns with this suffix
                    instead of overwriting.
            inplace: If True and suffix is empty, overwrite original columns.

        Returns:
            DataFrame with filtered columns.
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                print(f"  [Filter] Warning: column '{col}' not found, skipping")
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Get data, handle NaNs
            data = df[col].values.astype(float)
            nan_mask = np.isnan(data)

            if nan_mask.all():
                continue

            # Interpolate NaNs temporarily for filtering
            if nan_mask.any():
                valid_indices = np.where(~nan_mask)[0]
                if len(valid_indices) < 4:
                    continue
                data_interp = np.interp(
                    np.arange(len(data)),
                    valid_indices,
                    data[valid_indices],
                )
            else:
                data_interp = data

            # Apply filters
            if filter_type in ("butterworth", "both"):
                data_interp = self.butterworth_lowpass(
                    data_interp, cutoff_hz, order
                )

            if filter_type in ("median", "both"):
                data_interp = self.median_filter(data_interp, median_kernel)

            # Restore NaN positions
            if nan_mask.any():
                data_interp[nan_mask] = np.nan

            # Write result
            output_col = f"{col}{suffix}" if suffix else col
            df[output_col] = data_interp

        return df

    def filter_accelerometer(
        self,
        df: pd.DataFrame,
        accel_columns: list[str],
        cutoff_hz: float = 4.0,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Apply low-pass filter to accelerometer data.
        Default cutoff at 4 Hz removes high-frequency vibration noise
        while preserving vehicle motion dynamics.
        """
        print(f"  [Filter] Filtering accelerometer: {accel_columns} "
              f"(cutoff={cutoff_hz} Hz)")
        return self.filter_dataframe(
            df, accel_columns,
            filter_type="butterworth",
            cutoff_hz=cutoff_hz,
            **kwargs,
        )

    def filter_gyroscope(
        self,
        df: pd.DataFrame,
        gyro_columns: list[str],
        cutoff_hz: float = 4.0,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Apply low-pass filter to gyroscope data.
        Default cutoff at 4 Hz removes high-frequency noise while
        preserving turn dynamics.
        """
        print(f"  [Filter] Filtering gyroscope: {gyro_columns} "
              f"(cutoff={cutoff_hz} Hz)")
        return self.filter_dataframe(
            df, gyro_columns,
            filter_type="butterworth",
            cutoff_hz=cutoff_hz,
            **kwargs,
        )
