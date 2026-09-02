"""
IDR MVP -- Sensor Synchronizer
Handles timestamp alignment between smartphone and vehicle data streams.
Manages the GPS 1 Hz vs IMU 10 Hz rate difference.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate


class SensorSynchronizer:
    """Synchronizes and aligns multi-rate sensor data streams."""

    def __init__(self, target_rate_hz: float = 10.0):
        """
        Args:
            target_rate_hz: Target output sampling rate in Hz.
        """
        self.target_rate = target_rate_hz

    def validate_timestamps(
        self,
        df: pd.DataFrame,
        time_col: str,
    ) -> dict:
        """
        Validate timestamp integrity.

        Returns:
            Dictionary with validation results.
        """
        timestamps = pd.to_numeric(df[time_col], errors="coerce")
        valid = timestamps.dropna()

        result = {
            "total_timestamps": len(timestamps),
            "valid_timestamps": len(valid),
            "invalid_timestamps": int(timestamps.isna().sum()),
            "is_monotonic": bool(valid.is_monotonic_increasing),
            "has_duplicates": bool(valid.duplicated().any()),
            "n_duplicates": int(valid.duplicated().sum()),
        }

        if len(valid) > 1:
            dt = np.diff(valid.values)
            result["dt_stats"] = {
                "min": float(np.min(dt)),
                "max": float(np.max(dt)),
                "mean": float(np.mean(dt)),
                "median": float(np.median(dt)),
            }

        return result

    def forward_fill_gnss(
        self,
        df: pd.DataFrame,
        gnss_columns: list[str],
        limit: int = 15,
    ) -> pd.DataFrame:
        """
        Forward-fill GNSS data to match the higher IMU rate.
        GPS typically updates at 1 Hz while IMU runs at 10 Hz,
        so we hold the last GPS value for up to `limit` samples.

        Args:
            df: Input DataFrame.
            gnss_columns: GNSS column names (lat, lon, speed, etc.).
            limit: Maximum number of samples to forward-fill.

        Returns:
            DataFrame with GNSS columns forward-filled.
        """
        df = df.copy()
        for col in gnss_columns:
            if col in df.columns:
                n_before = df[col].isna().sum()
                df[col] = df[col].ffill(limit=limit)
                n_after = df[col].isna().sum()
                filled = n_before - n_after
                if filled > 0:
                    print(f"  [Sync] Forward-filled {filled:,} samples in '{col}'")
        return df

    def interpolate_gnss(
        self,
        df: pd.DataFrame,
        time_col: str,
        gnss_columns: list[str],
        method: str = "linear",
    ) -> pd.DataFrame:
        """
        Interpolate GNSS data to match IMU timestamps using
        time-based interpolation.

        Args:
            df: Input DataFrame.
            time_col: Timestamp column.
            gnss_columns: GNSS column names.
            method: Interpolation method.

        Returns:
            DataFrame with interpolated GNSS data.
        """
        df = df.copy()
        timestamps = pd.to_numeric(df[time_col], errors="coerce")

        for col in gnss_columns:
            if col not in df.columns:
                continue

            series = df[col].copy()
            valid_mask = series.notna() & timestamps.notna()

            if valid_mask.sum() < 2:
                continue

            # Create interpolation function from valid points
            t_valid = timestamps[valid_mask].values
            v_valid = series[valid_mask].values

            try:
                interp_func = interpolate.interp1d(
                    t_valid, v_valid,
                    kind=method,
                    bounds_error=False,
                    fill_value="extrapolate",
                )

                # Interpolate at all timestamps
                all_t = timestamps.dropna()
                interpolated = interp_func(all_t.values)

                # Only fill where original was NaN and timestamp is valid
                fill_mask = series.isna() & timestamps.notna()
                df.loc[fill_mask, col] = interpolated[
                    timestamps[fill_mask].index.map(
                        lambda i: np.searchsorted(all_t.index, i)
                    )
                ]
            except Exception as e:
                print(f"  [Sync] Warning: interpolation failed for '{col}': {e}")
                # Fallback to forward-fill
                df[col] = df[col].ffill(limit=15)

        return df

    def resample_to_uniform(
        self,
        df: pd.DataFrame,
        time_col: str,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Resample data to a uniform time grid at the target rate.

        Args:
            df: Input DataFrame.
            time_col: Timestamp column.
            columns: Columns to resample. If None, resample all numeric.

        Returns:
            Resampled DataFrame with uniform time spacing.
        """
        timestamps = pd.to_numeric(df[time_col], errors="coerce").dropna()

        if len(timestamps) < 2:
            return df

        t_start = timestamps.iloc[0]
        t_end = timestamps.iloc[-1]

        # Detect time unit
        dt_median = np.median(np.diff(timestamps.values))
        if dt_median > 1000:
            # Milliseconds
            dt_target = 1000.0 / self.target_rate
        elif dt_median > 1_000_000:
            # Microseconds
            dt_target = 1_000_000.0 / self.target_rate
        else:
            # Seconds
            dt_target = 1.0 / self.target_rate

        # Create uniform time grid
        t_uniform = np.arange(t_start, t_end, dt_target)

        if columns is None:
            columns = [c for c in df.columns
                       if c != time_col
                       and pd.api.types.is_numeric_dtype(df[c])]

        result = pd.DataFrame({time_col: t_uniform})

        for col in columns:
            if col not in df.columns:
                continue

            series = df[col]
            valid_mask = series.notna() & timestamps.index.isin(df.index)

            valid_t = timestamps.loc[series.notna()].values
            valid_v = series.dropna().values

            if len(valid_t) < 2:
                result[col] = np.nan
                continue

            try:
                interp_func = interpolate.interp1d(
                    valid_t, valid_v,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                result[col] = interp_func(t_uniform)
            except Exception:
                result[col] = np.nan

        print(f"  [Sync] Resampled to uniform {self.target_rate} Hz grid: "
              f"{len(df)} → {len(result)} samples")

        return result

    def synchronize(
        self,
        df: pd.DataFrame,
        time_col: str,
        gnss_columns: list[str],
        gnss_strategy: str = "forward_fill",
    ) -> pd.DataFrame:
        """
        Run the full synchronization pipeline.

        Args:
            df: Input DataFrame.
            time_col: Timestamp column.
            gnss_columns: GNSS data columns.
            gnss_strategy: 'forward_fill' or 'interpolate' for GNSS data.

        Returns:
            Synchronized DataFrame.
        """
        print(f"  [Sync] Starting synchronization...")

        # Validate timestamps
        validation = self.validate_timestamps(df, time_col)
        print(f"  [Sync] Timestamps: {validation['valid_timestamps']:,} valid, "
              f"{validation['invalid_timestamps']} invalid, "
              f"monotonic={validation['is_monotonic']}")

        # Handle GNSS rate difference
        existing_gnss = [c for c in gnss_columns if c in df.columns]
        if existing_gnss:
            if gnss_strategy == "interpolate":
                df = self.interpolate_gnss(df, time_col, existing_gnss)
            else:
                df = self.forward_fill_gnss(df, existing_gnss)

        print(f"  [Sync] Synchronization complete. Shape: {df.shape}")
        return df
