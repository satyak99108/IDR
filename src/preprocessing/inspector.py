"""
IDR MVP -- Dataset Inspector
Inspects IO-VNBD data: fields, sampling rates, sensor identification,
data quality reports.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class DatasetInspector:
    """Inspects and reports on IO-VNBD dataset characteristics."""

    # Common column name patterns for sensor identification
    # These are flexible patterns to match various naming conventions
    ACCEL_PATTERNS = [
        "accel", "acc_", "ax", "ay", "az",
        "accelerometer", "linear_acc", "user_acc",
    ]
    GYRO_PATTERNS = [
        "gyro", "gyr_", "gx", "gy", "gz",
        "gyroscope", "angular",
    ]
    MAG_PATTERNS = [
        "mag", "magnet", "mx", "my", "mz",
        "magnetometer",
    ]
    GNSS_PATTERNS = [
        "gps", "gnss", "lat", "lon", "long",
        "latitude", "longitude", "altitude",
    ]
    VELOCITY_PATTERNS = [
        "vel", "speed", "velocity", "v_",
    ]
    HEADING_PATTERNS = [
        "head", "yaw", "bearing", "course", "azimuth",
    ]
    TIMESTAMP_PATTERNS = [
        "time", "timestamp", "epoch", "t_", "unix",
    ]

    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        """
        Args:
            df: The DataFrame to inspect.
            name: A label for this dataset (e.g., trip name).
        """
        self.df = df
        self.name = name

    def identify_columns(self) -> dict[str, list[str]]:
        """
        Identify sensor columns by matching against known patterns.

        Returns:
            Dictionary mapping sensor type to matching column names.
        """
        columns_lower = {col: col.lower() for col in self.df.columns}
        result = {
            "accelerometer": [],
            "gyroscope": [],
            "magnetometer": [],
            "gnss": [],
            "velocity": [],
            "heading": [],
            "timestamp": [],
            "reference": [],
            "other": [],
        }

        classified = set()

        # Check for REF_ (ground truth) columns first
        for col, col_low in columns_lower.items():
            if col_low.startswith("ref_") or "reference" in col_low:
                result["reference"].append(col)
                classified.add(col)

        for col, col_low in columns_lower.items():
            if col in classified:
                continue

            matched = False
            for pattern in self.TIMESTAMP_PATTERNS:
                if pattern in col_low:
                    result["timestamp"].append(col)
                    classified.add(col)
                    matched = True
                    break
            if matched:
                continue

            for pattern in self.ACCEL_PATTERNS:
                if pattern in col_low:
                    result["accelerometer"].append(col)
                    classified.add(col)
                    matched = True
                    break
            if matched:
                continue

            for pattern in self.GYRO_PATTERNS:
                if pattern in col_low:
                    result["gyroscope"].append(col)
                    classified.add(col)
                    matched = True
                    break
            if matched:
                continue

            for pattern in self.MAG_PATTERNS:
                if pattern in col_low:
                    result["magnetometer"].append(col)
                    classified.add(col)
                    matched = True
                    break
            if matched:
                continue

            for pattern in self.GNSS_PATTERNS:
                if pattern in col_low:
                    # Don't classify 'speed' columns as GNSS -- they go to velocity
                    if "speed" in col_low or "velocity" in col_low:
                        break
                    result["gnss"].append(col)
                    classified.add(col)
                    matched = True
                    break
            if matched:
                continue

            for pattern in self.VELOCITY_PATTERNS:
                if pattern in col_low:
                    result["velocity"].append(col)
                    classified.add(col)
                    matched = True
                    break
            if matched:
                continue

            for pattern in self.HEADING_PATTERNS:
                if pattern in col_low:
                    result["heading"].append(col)
                    classified.add(col)
                    matched = True
                    break
            if matched:
                continue

        # Everything else
        result["other"] = [c for c in self.df.columns if c not in classified]

        return result

    def compute_sampling_rate(
        self,
        time_col: Optional[str] = None,
    ) -> dict:
        """
        Compute the actual sampling rate from timestamp differences.

        Args:
            time_col: Name of the timestamp column. If None, auto-detect.

        Returns:
            Dictionary with sampling rate statistics.
        """
        if time_col is None:
            # Auto-detect timestamp column
            identified = self.identify_columns()
            time_cols = identified.get("timestamp", [])
            if not time_cols:
                return {"error": "No timestamp column found"}
            time_col = time_cols[0]

        if time_col not in self.df.columns:
            return {"error": f"Column '{time_col}' not found"}

        timestamps = pd.to_numeric(self.df[time_col], errors="coerce").dropna()

        if len(timestamps) < 2:
            return {"error": "Not enough valid timestamps"}

        dt = np.diff(timestamps.values)

        # Filter out obviously wrong dt values (negative or extremely large)
        dt_positive = dt[dt > 0]
        if len(dt_positive) == 0:
            return {"error": "No positive time differences found"}

        # Determine if timestamps are in seconds, milliseconds, or microseconds
        # Strategy: 1) check column name for hints, 2) use dt magnitude heuristic
        median_dt = np.median(dt_positive)
        time_unit = "seconds"
        dt_seconds = dt_positive

        # Check column name for unit hints
        time_col_lower = time_col.lower()
        if "(ms)" in time_col_lower or "_ms" in time_col_lower or "millis" in time_col_lower:
            dt_seconds = dt_positive / 1000.0
            time_unit = "milliseconds (from column name)"
        elif "(us)" in time_col_lower or "_us" in time_col_lower or "micro" in time_col_lower:
            dt_seconds = dt_positive / 1_000_000.0
            time_unit = "microseconds (from column name)"
        elif median_dt > 1_000_000:
            # Very large dt -- likely microseconds
            dt_seconds = dt_positive / 1_000_000.0
            time_unit = "microseconds"
        elif median_dt > 1000:
            # Large dt -- likely milliseconds
            dt_seconds = dt_positive / 1000.0
            time_unit = "milliseconds"
        else:
            # Could be seconds or milliseconds with small dt
            # If treating as seconds gives < 1 Hz, likely milliseconds
            rate_as_seconds = 1.0 / median_dt if median_dt > 0 else 0
            if rate_as_seconds < 1.0 and median_dt > 10:
                # Unreasonably low rate -- probably milliseconds
                dt_seconds = dt_positive / 1000.0
                time_unit = "milliseconds (heuristic)"
            else:
                dt_seconds = dt_positive
                time_unit = "seconds"

        sampling_rate = 1.0 / np.median(dt_seconds)

        return {
            "time_column": time_col,
            "time_unit_detected": time_unit,
            "n_samples": len(timestamps),
            "duration_seconds": float(np.sum(dt_seconds)),
            "dt_mean": float(np.mean(dt_seconds)),
            "dt_median": float(np.median(dt_seconds)),
            "dt_min": float(np.min(dt_seconds)),
            "dt_max": float(np.max(dt_seconds)),
            "dt_std": float(np.std(dt_seconds)),
            "sampling_rate_hz": float(sampling_rate),
            "n_gaps": int(np.sum(dt_seconds > 3 * np.median(dt_seconds))),
        }

    def data_quality_report(self) -> dict:
        """
        Generate a data quality report.

        Returns:
            Dictionary with quality metrics per column and overall.
        """
        total_rows = len(self.df)
        report = {
            "total_rows": total_rows,
            "total_columns": len(self.df.columns),
            "memory_usage_mb": round(
                self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2
            ),
            "columns": {},
            "summary": {},
        }

        total_missing = 0
        total_cells = 0

        for col in self.df.columns:
            series = self.df[col]
            n_missing = int(series.isna().sum())
            n_zeros = 0
            col_stats = {
                "dtype": str(series.dtype),
                "missing_count": n_missing,
                "missing_pct": round(100 * n_missing / total_rows, 2) if total_rows > 0 else 0,
                "n_unique": int(series.nunique()),
            }

            if pd.api.types.is_numeric_dtype(series):
                valid = series.dropna()
                if len(valid) > 0:
                    col_stats.update({
                        "min": float(valid.min()),
                        "max": float(valid.max()),
                        "mean": float(valid.mean()),
                        "std": float(valid.std()),
                        "median": float(valid.median()),
                    })
                    n_zeros = int((valid == 0).sum())
                    col_stats["n_zeros"] = n_zeros

            report["columns"][col] = col_stats
            total_missing += n_missing
            total_cells += total_rows

        report["summary"] = {
            "total_missing_cells": total_missing,
            "total_cells": total_cells,
            "overall_missing_pct": round(
                100 * total_missing / total_cells, 2
            ) if total_cells > 0 else 0,
            "columns_with_missing": sum(
                1 for c in report["columns"].values() if c["missing_count"] > 0
            ),
            "fully_complete_columns": sum(
                1 for c in report["columns"].values() if c["missing_count"] == 0
            ),
        }

        return report

    def basic_statistics(self) -> pd.DataFrame:
        """Return standard describe() output for numeric columns."""
        return self.df.describe()

    def print_summary(self):
        """Print a comprehensive human-readable summary."""
        print(f"\n{'='*70}")
        print(f"  DATASET INSPECTION: {self.name}")
        print(f"{'='*70}")

        # Shape
        print(f"\n  Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"  Memory: {self.df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")

        # Columns
        print(f"\n  ALL COLUMNS ({len(self.df.columns)}):")
        for i, col in enumerate(self.df.columns):
            print(f"    [{i:2d}] {col:<30} dtype={self.df[col].dtype}")

        # Sensor identification
        identified = self.identify_columns()
        print(f"\n  SENSOR IDENTIFICATION:")
        for sensor_type, cols in identified.items():
            if cols:
                print(f"    {sensor_type:15s}: {cols}")

        # Sampling rate
        sr = self.compute_sampling_rate()
        print(f"\n  SAMPLING RATE:")
        if "error" not in sr:
            print(f"    Rate:     {sr['sampling_rate_hz']:.2f} Hz")
            print(f"    Duration: {sr['duration_seconds']:.1f} seconds "
                  f"({sr['duration_seconds']/60:.1f} minutes)")
            print(f"    Δt mean:  {sr['dt_mean']*1000:.2f} ms")
            print(f"    Δt std:   {sr['dt_std']*1000:.2f} ms")
            print(f"    Gaps (>3×median): {sr['n_gaps']}")
        else:
            print(f"    Error: {sr['error']}")

        # Data quality
        quality = self.data_quality_report()
        print(f"\n  DATA QUALITY:")
        print(f"    Total missing cells: {quality['summary']['total_missing_cells']:,} "
              f"({quality['summary']['overall_missing_pct']}%)")
        print(f"    Columns with missing data: "
              f"{quality['summary']['columns_with_missing']}")

        # Show columns with missing data
        missing_cols = [
            (col, info["missing_count"], info["missing_pct"])
            for col, info in quality["columns"].items()
            if info["missing_count"] > 0
        ]
        if missing_cols:
            print(f"\n    Columns with missing values:")
            for col, count, pct in sorted(missing_cols, key=lambda x: -x[1]):
                print(f"      {col:<30} {count:>8,} ({pct}%)")

        # First few rows
        print(f"\n  FIRST 3 ROWS:")
        print(self.df.head(3).to_string(index=False))

        print(f"\n{'='*70}\n")

    def save_report(self, output_path: str):
        """Save the inspection report as JSON."""
        report = {
            "name": self.name,
            "shape": list(self.df.shape),
            "columns": list(self.df.columns),
            "identified_sensors": self.identify_columns(),
            "sampling_rate": self.compute_sampling_rate(),
            "data_quality": self.data_quality_report(),
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"  Report saved to: {output}")
