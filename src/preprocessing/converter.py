"""
IDR MVP -- Format Converter
Converts cleaned, filtered data into the consistent internal format
specified in MVP.md:
    timestamp | ax | ay | az | gx | gy | gz | gnss_lat | gnss_lon | gnss_speed | ref_lat | ref_lon

Saves output as CSV and/or Parquet.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


class FormatConverter:
    """Converts IO-VNBD data into the IDR standard internal format."""

    # Standard output column names
    STANDARD_COLUMNS = [
        "timestamp",
        "ax", "ay", "az",
        "gx", "gy", "gz",
        "gnss_lat", "gnss_lon", "gnss_speed",
        "ref_lat", "ref_lon",
    ]

    # Optional extra columns we preserve if available
    OPTIONAL_COLUMNS = [
        "mx", "my", "mz",          # Magnetometer
        "gnss_accuracy",            # GNSS accuracy
        "heading",                  # Heading/bearing
        "ref_speed",                # Reference speed
        "ref_heading",              # Reference heading
    ]

    def __init__(self, column_mapping: Optional[dict[str, str]] = None):
        """
        Args:
            column_mapping: Dictionary mapping source column names to
                          standard names. E.g. {'Acc_X': 'ax', 'GPS_Lat': 'gnss_lat'}.
                          If None, will attempt auto-mapping.
        """
        self.column_mapping = column_mapping or {}

    def auto_map_columns(
        self,
        source_columns: list[str],
        identified_sensors: dict[str, list[str]],
    ) -> dict[str, str]:
        """
        Automatically map source columns to standard format using
        the inspector's sensor identification.

        Args:
            source_columns: Available column names in the source data.
            identified_sensors: Output from DatasetInspector.identify_columns().

        Returns:
            Dictionary mapping source → standard column names.
        """
        mapping = {}

        # Timestamp
        time_cols = identified_sensors.get("timestamp", [])
        if time_cols:
            mapping[time_cols[0]] = "timestamp"

        # Accelerometer -- expect 3 axes
        accel_cols = identified_sensors.get("accelerometer", [])
        if len(accel_cols) >= 3:
            mapping[accel_cols[0]] = "ax"
            mapping[accel_cols[1]] = "ay"
            mapping[accel_cols[2]] = "az"

        # Gyroscope -- expect 3 axes
        # Handle both X/Y/Z and Roll/Pitch/Yaw naming conventions
        # In vehicle frame: Roll=X, Pitch=Y, Yaw=Z
        gyro_cols = identified_sensors.get("gyroscope", [])
        if len(gyro_cols) >= 3:
            gyro_lower = {c: c.lower() for c in gyro_cols}
            gx_col = gy_col = gz_col = None
            for col, col_low in gyro_lower.items():
                if " x " in col_low or "_x" in col_low or "x(" in col_low or "roll" in col_low:
                    gx_col = col
                elif " y " in col_low or "_y" in col_low or "y(" in col_low or "pitch" in col_low:
                    gy_col = col
                elif " z " in col_low or "_z" in col_low or "z(" in col_low or "yaw" in col_low:
                    gz_col = col
            if gx_col and gy_col and gz_col:
                mapping[gx_col] = "gx"
                mapping[gy_col] = "gy"
                mapping[gz_col] = "gz"
            else:
                # Fallback: use order
                mapping[gyro_cols[0]] = "gx"
                mapping[gyro_cols[1]] = "gy"
                mapping[gyro_cols[2]] = "gz"

        # GNSS
        gnss_cols = identified_sensors.get("gnss", [])
        gnss_lower = {c: c.lower() for c in gnss_cols}
        for col, col_low in gnss_lower.items():
            if "lat" in col_low and "gnss_lat" not in mapping.values():
                mapping[col] = "gnss_lat"
            elif "lon" in col_low and "gnss_lon" not in mapping.values():
                mapping[col] = "gnss_lon"

        # Velocity / Speed
        vel_cols = identified_sensors.get("velocity", [])
        if vel_cols and "gnss_speed" not in mapping.values():
            mapping[vel_cols[0]] = "gnss_speed"

        # Heading
        heading_cols = identified_sensors.get("heading", [])
        if heading_cols and "heading" not in mapping.values():
            mapping[heading_cols[0]] = "heading"

        # Magnetometer
        mag_cols = identified_sensors.get("magnetometer", [])
        if len(mag_cols) >= 3:
            mapping[mag_cols[0]] = "mx"
            mapping[mag_cols[1]] = "my"
            mapping[mag_cols[2]] = "mz"

        # Reference / Ground Truth (from vehicle)
        ref_cols = identified_sensors.get("reference", [])
        ref_lower = {c: c.lower() for c in ref_cols}
        for col, col_low in ref_lower.items():
            if "lat" in col_low and "ref_lat" not in mapping.values():
                mapping[col] = "ref_lat"
            elif "lon" in col_low and "ref_lon" not in mapping.values():
                mapping[col] = "ref_lon"
            elif ("vel" in col_low or "speed" in col_low) and "ref_speed" not in mapping.values():
                mapping[col] = "ref_speed"
            elif ("head" in col_low or "course" in col_low) and "ref_heading" not in mapping.values():
                mapping[col] = "ref_heading"

        return mapping

    def convert(
        self,
        df: pd.DataFrame,
        identified_sensors: Optional[dict[str, list[str]]] = None,
        keep_extra: bool = False,
    ) -> pd.DataFrame:
        """
        Convert a DataFrame to the standard internal format.

        Args:
            df: Source DataFrame.
            identified_sensors: Sensor identification from DatasetInspector.
            keep_extra: If True, keep unmapped columns with original names.

        Returns:
            DataFrame with standard column names.
        """
        # Build column mapping
        if not self.column_mapping and identified_sensors:
            self.column_mapping = self.auto_map_columns(
                list(df.columns), identified_sensors
            )

        if not self.column_mapping:
            print("  [Converter] Warning: No column mapping available. "
                  "Returning original columns.")
            return df.copy()

        # Apply mapping
        result = pd.DataFrame()

        # Map known columns
        for source_col, target_col in self.column_mapping.items():
            if source_col in df.columns:
                result[target_col] = df[source_col].values

        # Add any standard columns that weren't mapped (as NaN)
        for col in self.STANDARD_COLUMNS:
            if col not in result.columns:
                result[col] = pd.NA
                print(f"  [Converter] Warning: '{col}' not found in source data, "
                      f"filled with NaN")

        # Keep extra columns if requested
        if keep_extra:
            mapped_sources = set(self.column_mapping.keys())
            for col in df.columns:
                if col not in mapped_sources and col not in result.columns:
                    result[f"_extra_{col}"] = df[col].values

        # Reorder: standard columns first
        ordered_cols = [c for c in self.STANDARD_COLUMNS if c in result.columns]
        optional_present = [c for c in self.OPTIONAL_COLUMNS if c in result.columns]
        extra_cols = [c for c in result.columns
                      if c not in ordered_cols and c not in optional_present]
        result = result[ordered_cols + optional_present + extra_cols]

        print(f"  [Converter] Converted {len(df.columns)} columns → "
              f"{len(result.columns)} columns")
        print(f"  [Converter] Column mapping used:")
        for src, tgt in sorted(self.column_mapping.items(), key=lambda x: x[1]):
            print(f"    {src:<30} → {tgt}")

        return result

    def save(
        self,
        df: pd.DataFrame,
        output_dir: str,
        trip_name: str,
        formats: list[str] = None,
    ) -> list[str]:
        """
        Save the converted DataFrame.

        Args:
            df: Converted DataFrame.
            output_dir: Output directory path.
            trip_name: Trip identifier for the filename.
            formats: List of formats to save ('csv', 'parquet'). Default: ['csv'].

        Returns:
            List of saved file paths.
        """
        if formats is None:
            formats = ["csv"]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for fmt in formats:
            if fmt == "csv":
                filepath = output_path / f"{trip_name}_processed.csv"
                df.to_csv(filepath, index=False)
                saved_files.append(str(filepath))
                print(f"  [Converter] Saved CSV: {filepath}")

            elif fmt == "parquet":
                try:
                    filepath = output_path / f"{trip_name}_processed.parquet"
                    df.to_parquet(filepath, index=False)
                    saved_files.append(str(filepath))
                    print(f"  [Converter] Saved Parquet: {filepath}")
                except ImportError:
                    print("  [Converter] Warning: pyarrow not installed, "
                          "skipping Parquet format")

        return saved_files

    def get_mapping(self) -> dict[str, str]:
        """Return the current column mapping."""
        return self.column_mapping.copy()
