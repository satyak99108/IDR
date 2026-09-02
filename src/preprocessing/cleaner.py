"""
IDR MVP -- Data Cleaner
Handles missing/invalid samples, outlier detection, and duplicate removal
for IO-VNBD sensor data.
"""

from typing import Optional

import numpy as np
import pandas as pd


class DataCleaner:
    """Cleans IO-VNBD sensor data: NaN handling, outliers, duplicates."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame to clean (will NOT be modified in-place;
                methods return new DataFrames).
        """
        self.original = df
        self.log: list[str] = []

    def _log(self, msg: str):
        self.log.append(msg)
        print(f"  [Cleaner] {msg}")

    def remove_duplicate_timestamps(
        self,
        df: pd.DataFrame,
        time_col: str,
    ) -> pd.DataFrame:
        """Remove rows with duplicate timestamps, keeping the first."""
        n_before = len(df)
        df = df.drop_duplicates(subset=[time_col], keep="first")
        n_removed = n_before - len(df)
        if n_removed > 0:
            self._log(f"Removed {n_removed:,} duplicate timestamps")
        else:
            self._log("No duplicate timestamps found")
        return df.reset_index(drop=True)

    def sort_by_time(
        self,
        df: pd.DataFrame,
        time_col: str,
    ) -> pd.DataFrame:
        """Sort DataFrame by timestamp column."""
        df = df.sort_values(time_col).reset_index(drop=True)
        self._log(f"Sorted by '{time_col}'")
        return df

    def interpolate_small_gaps(
        self,
        df: pd.DataFrame,
        columns: list[str],
        max_gap: int = 5,
        method: str = "linear",
    ) -> pd.DataFrame:
        """
        Interpolate NaN values in specified columns for small gaps.

        Args:
            df: Input DataFrame.
            columns: Columns to interpolate.
            max_gap: Maximum consecutive NaNs to interpolate.
            method: Interpolation method ('linear', 'nearest', etc.).

        Returns:
            DataFrame with small gaps interpolated.
        """
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            n_before = df[col].isna().sum()
            df[col] = df[col].interpolate(
                method=method,
                limit=max_gap,
                limit_direction="both",
            )
            n_after = df[col].isna().sum()
            filled = n_before - n_after
            if filled > 0:
                self._log(
                    f"Interpolated {filled:,} NaNs in '{col}' "
                    f"(remaining: {n_after:,})"
                )
        return df

    def flag_large_gaps(
        self,
        df: pd.DataFrame,
        time_col: str,
        threshold_seconds: float = 1.0,
    ) -> pd.DataFrame:
        """
        Add a boolean column flagging rows after large time gaps.

        Args:
            df: Input DataFrame.
            time_col: Timestamp column.
            threshold_seconds: Gap threshold in seconds.

        Returns:
            DataFrame with '_gap_flag' column added.
        """
        df = df.copy()
        timestamps = pd.to_numeric(df[time_col], errors="coerce")
        dt = np.diff(timestamps.values, prepend=timestamps.values[0])

        # Auto-detect time unit using column name hints + magnitude heuristic
        median_dt = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
        time_col_lower = time_col.lower()
        if "(ms)" in time_col_lower or "_ms" in time_col_lower or "millis" in time_col_lower:
            dt = dt / 1000.0  # milliseconds to seconds
        elif "(us)" in time_col_lower or "_us" in time_col_lower or "micro" in time_col_lower:
            dt = dt / 1_000_000.0  # microseconds to seconds
        elif median_dt > 1_000_000:
            dt = dt / 1_000_000.0
        elif median_dt > 1000:
            dt = dt / 1000.0
        elif median_dt > 10 and (1.0 / median_dt) < 1.0:
            # Unreasonably low rate if treated as seconds -- likely ms
            dt = dt / 1000.0

        df["_gap_flag"] = dt > threshold_seconds
        n_gaps = int(df["_gap_flag"].sum())
        self._log(f"Flagged {n_gaps} large gaps (>{threshold_seconds}s)")
        return df

    def detect_outliers_zscore(
        self,
        df: pd.DataFrame,
        columns: list[str],
        z_threshold: float = 5.0,
    ) -> pd.DataFrame:
        """
        Detect and replace outliers using Z-score method.
        Outliers are replaced with NaN (can be interpolated after).

        Args:
            df: Input DataFrame.
            columns: Columns to check for outliers.
            z_threshold: Z-score threshold (default 5.0 is conservative
                        for IMU data which can have legitimate spikes).

        Returns:
            DataFrame with outliers replaced by NaN.
        """
        df = df.copy()
        total_outliers = 0

        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            series = df[col].dropna()
            if len(series) < 10:
                continue

            mean = series.mean()
            std = series.std()
            if std == 0:
                continue

            z_scores = np.abs((df[col] - mean) / std)
            outlier_mask = z_scores > z_threshold
            n_outliers = int(outlier_mask.sum())

            if n_outliers > 0:
                df.loc[outlier_mask, col] = np.nan
                total_outliers += n_outliers
                self._log(
                    f"Detected {n_outliers:,} outliers in '{col}' "
                    f"(|z|>{z_threshold})"
                )

        if total_outliers == 0:
            self._log("No outliers detected in specified columns")

        return df

    def drop_all_nan_rows(
        self,
        df: pd.DataFrame,
        subset: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Drop rows where all specified columns are NaN.

        Args:
            df: Input DataFrame.
            subset: Columns to check. If None, check all.

        Returns:
            DataFrame with fully-NaN rows removed.
        """
        n_before = len(df)
        df = df.dropna(subset=subset, how="all").reset_index(drop=True)
        n_removed = n_before - len(df)
        if n_removed > 0:
            self._log(f"Dropped {n_removed:,} all-NaN rows")
        return df

    def clean(
        self,
        time_col: str,
        sensor_columns: list[str],
        interpolate_max_gap: int = 5,
        outlier_z_threshold: float = 5.0,
        gap_threshold_seconds: float = 1.0,
    ) -> pd.DataFrame:
        """
        Run the full cleaning pipeline.

        Args:
            time_col: Timestamp column name.
            sensor_columns: List of sensor data columns to clean.
            interpolate_max_gap: Max consecutive NaNs to interpolate.
            outlier_z_threshold: Z-score threshold for outlier detection.
            gap_threshold_seconds: Time gap threshold for flagging.

        Returns:
            Cleaned DataFrame.
        """
        self._log("Starting cleaning pipeline...")
        df = self.original.copy()

        # 1. Sort by time
        df = self.sort_by_time(df, time_col)

        # 2. Remove duplicate timestamps
        df = self.remove_duplicate_timestamps(df, time_col)

        # 3. Flag large gaps
        df = self.flag_large_gaps(df, time_col, gap_threshold_seconds)

        # 4. Detect outliers
        df = self.detect_outliers_zscore(df, sensor_columns, outlier_z_threshold)

        # 5. Interpolate small gaps
        df = self.interpolate_small_gaps(df, sensor_columns, interpolate_max_gap)

        # 6. Drop rows where all sensor columns are NaN
        df = self.drop_all_nan_rows(df, sensor_columns)

        self._log(
            f"Cleaning complete. "
            f"Shape: {self.original.shape} → {df.shape}"
        )
        return df

    def get_log(self) -> list[str]:
        """Return the cleaning operation log."""
        return self.log.copy()
