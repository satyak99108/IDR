"""
IDR MVP -- Phase 1 Runner
=========================
Main orchestration script for Phase 1: Dataset & Preprocessing.

Executes the full pipeline:
1. Load one trip from IO-VNBD
2. Inspect: print all fields, compute sampling rates, identify sensor types
3. Clean: handle missing data, outliers
4. Filter: apply low-pass filters
5. Synchronize: align sensor streams
6. Convert: produce the standard internal format
7. Visualize: generate and save plots
8. Save processed data

Usage:
    python scripts/run_phase1.py
    python scripts/run_phase1.py --trip 0 --data-type smartphone
    python scripts/run_phase1.py --nrows 10000  # Quick test with limited rows
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    DatasetLoader,
    DatasetInspector,
    DataCleaner,
    SignalFilter,
    SensorSynchronizer,
    FormatConverter,
)

# --- Configuration ---
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "IO-VNBD"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase1"

# Plot styling
sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({
    "figure.figsize": (14, 8),
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

# Color palette for IDR plots
IDR_COLORS = {
    "accel_x": "#FF6B6B",
    "accel_y": "#4ECDC4",
    "accel_z": "#45B7D1",
    "gyro_x": "#F7DC6F",
    "gyro_y": "#82E0AA",
    "gyro_z": "#BB8FCE",
    "gnss": "#2ECC71",
    "reference": "#E74C3C",
    "filtered": "#3498DB",
    "raw": "#95A5A6",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="IDR MVP -- Phase 1: Dataset & Preprocessing"
    )
    parser.add_argument(
        "--trip", type=int, default=0,
        help="Trip index to process (default: 0)"
    )
    parser.add_argument(
        "--trip-name", type=str, default=None,
        help="Specific trip name to process (e.g. 's1', 'm', 'vta10')"
    )
    parser.add_argument(
        "--data-type", type=str, default="synced",
        choices=["smartphone", "vehicle", "synced", "pair"],
        help="Data type to load (default: synced / pair with ground truth)"
    )
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="Limit rows to load (for quick testing)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--cutoff-hz", type=float, default=4.0,
        help="Low-pass filter cutoff frequency in Hz (default: 4.0)"
    )
    return parser.parse_args()


def print_banner():
    print("")
    print("+" + "="*62 + "+")
    print("|" + "  IDR MVP -- Phase 1: Dataset & Preprocessing  ".center(62) + "|")
    print("|" + "  Intelligent Dead Reckoning Prototype          ".center(62) + "|")
    print("+" + "="*62 + "+")
    print("")


# --------------------------- PLOTTING ---------------------------

def plot_accelerometer(df, accel_cols, time_col, save_path, title_suffix=""):
    """Plot raw accelerometer signals (ax, ay, az) vs time."""
    if not accel_cols or len(accel_cols) < 3:
        print("  [Plot] Skipping accelerometer plot -- insufficient columns")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Accelerometer Signals {title_suffix}", fontsize=15, fontweight="bold")

    labels = ["X-axis", "Y-axis", "Z-axis"]
    colors = [IDR_COLORS["accel_x"], IDR_COLORS["accel_y"], IDR_COLORS["accel_z"]]

    time_data = df[time_col].values if time_col in df.columns else np.arange(len(df))

    # Normalize time to start from 0 seconds
    if time_col in df.columns:
        time_data = time_data - time_data[0]
        median_dt = np.median(np.diff(time_data[time_data > 0][:100])) if len(time_data) > 1 else 1
        if median_dt > 1000:
            time_data = time_data / 1000.0  # ms to seconds
        elif median_dt > 1_000_000:
            time_data = time_data / 1_000_000.0

    for i, (col, label, color) in enumerate(zip(accel_cols[:3], labels, colors)):
        if col in df.columns:
            axes[i].plot(time_data, df[col].values, color=color, linewidth=0.5, alpha=0.8)
            axes[i].set_ylabel(f"Accel {label}\n(m/s²)", fontsize=10)
            axes[i].set_title(f"Accelerometer {label}: {col}", fontsize=11)
            axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [Plot] Saved: {save_path}")


def plot_gyroscope(df, gyro_cols, time_col, save_path, title_suffix=""):
    """Plot raw gyroscope signals (gx, gy, gz) vs time."""
    if not gyro_cols or len(gyro_cols) < 3:
        print("  [Plot] Skipping gyroscope plot -- insufficient columns")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Gyroscope Signals {title_suffix}", fontsize=15, fontweight="bold")

    labels = ["X-axis (Roll)", "Y-axis (Pitch)", "Z-axis (Yaw)"]
    colors = [IDR_COLORS["gyro_x"], IDR_COLORS["gyro_y"], IDR_COLORS["gyro_z"]]

    time_data = df[time_col].values if time_col in df.columns else np.arange(len(df))
    if time_col in df.columns:
        time_data = time_data - time_data[0]
        median_dt = np.median(np.diff(time_data[time_data > 0][:100])) if len(time_data) > 1 else 1
        if median_dt > 1000:
            time_data = time_data / 1000.0
        elif median_dt > 1_000_000:
            time_data = time_data / 1_000_000.0

    for i, (col, label, color) in enumerate(zip(gyro_cols[:3], labels, colors)):
        if col in df.columns:
            axes[i].plot(time_data, df[col].values, color=color, linewidth=0.5, alpha=0.8)
            axes[i].set_ylabel(f"Gyro {label}\n(rad/s)", fontsize=10)
            axes[i].set_title(f"Gyroscope {label}: {col}", fontsize=11)
            axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [Plot] Saved: {save_path}")


def plot_gnss_trajectory(df, lat_col, lon_col, save_path, ref_lat_col=None, ref_lon_col=None):
    """Plot GNSS/reference ground-truth trajectory."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.suptitle("GNSS / Reference Trajectory", fontsize=15, fontweight="bold")

    has_gnss = lat_col in df.columns and lon_col in df.columns
    has_ref = ref_lat_col and ref_lon_col and ref_lat_col in df.columns and ref_lon_col in df.columns

    if has_gnss:
        gnss_data = df[[lat_col, lon_col]].dropna()
        if len(gnss_data) > 0:
            ax.scatter(
                gnss_data[lon_col], gnss_data[lat_col],
                c=np.arange(len(gnss_data)), cmap="viridis",
                s=2, alpha=0.6, label="GNSS Track"
            )

    if has_ref:
        ref_data = df[[ref_lat_col, ref_lon_col]].dropna()
        if len(ref_data) > 0:
            ax.plot(
                ref_data[ref_lon_col], ref_data[ref_lat_col],
                color=IDR_COLORS["reference"], linewidth=1.5,
                alpha=0.7, label="Reference/Ground Truth"
            )

    if not has_gnss and not has_ref:
        ax.text(0.5, 0.5, "No GNSS/Reference data available",
                transform=ax.transAxes, ha="center", va="center", fontsize=14)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [Plot] Saved: {save_path}")


def plot_sampling_rate(df, time_col, save_path):
    """Plot sampling rate histogram from timestamp differences."""
    if time_col not in df.columns:
        print("  [Plot] Skipping sampling rate plot -- no timestamp column")
        return

    timestamps = df[time_col].dropna().values
    if len(timestamps) < 2:
        return

    dt = np.diff(timestamps)
    dt_positive = dt[dt > 0]

    # Auto-detect time unit and convert to seconds
    median_dt = np.median(dt_positive) if len(dt_positive) > 0 else 1
    if median_dt > 1000:
        dt_positive = dt_positive / 1000.0
        unit_label = "converted from ms"
    elif median_dt > 1_000_000:
        dt_positive = dt_positive / 1_000_000.0
        unit_label = "converted from μs"
    else:
        unit_label = "seconds"

    rates = 1.0 / dt_positive[dt_positive > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sampling Rate Analysis", fontsize=15, fontweight="bold")

    # Histogram of inter-sample intervals
    ax1.hist(dt_positive * 1000, bins=100, color="#3498DB", alpha=0.7, edgecolor="white")
    ax1.set_xlabel(f"Inter-sample interval (ms) [{unit_label}]", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Inter-sample Interval Distribution", fontsize=12)
    ax1.axvline(np.median(dt_positive) * 1000, color="#E74C3C", linestyle="--",
                label=f"Median: {np.median(dt_positive)*1000:.1f} ms")
    ax1.legend()

    # Histogram of instantaneous sampling rates
    rates_clipped = rates[rates < 100]  # Clip unrealistic rates
    if len(rates_clipped) > 0:
        ax2.hist(rates_clipped, bins=100, color="#2ECC71", alpha=0.7, edgecolor="white")
        ax2.set_xlabel("Instantaneous Rate (Hz)", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("Instantaneous Sampling Rate Distribution", fontsize=12)
        ax2.axvline(np.median(rates_clipped), color="#E74C3C", linestyle="--",
                    label=f"Median: {np.median(rates_clipped):.1f} Hz")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [Plot] Saved: {save_path}")


def plot_filter_comparison(df_raw, df_filtered, columns, time_col, save_path):
    """Plot before/after filtering comparison for specified columns."""
    n_cols = min(len(columns), 3)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(n_cols, 1, figsize=(14, 4 * n_cols), sharex=True)
    if n_cols == 1:
        axes = [axes]

    fig.suptitle("Signal Filtering: Before vs After", fontsize=15, fontweight="bold")

    time_data = df_raw[time_col].values if time_col in df_raw.columns else np.arange(len(df_raw))
    if time_col in df_raw.columns:
        time_data = time_data - time_data[0]
        median_dt = np.median(np.diff(time_data[time_data > 0][:100])) if len(time_data) > 1 else 1
        if median_dt > 1000:
            time_data = time_data / 1000.0
        elif median_dt > 1_000_000:
            time_data = time_data / 1_000_000.0

    # Show only first 2000 samples for clarity
    n_show = min(2000, len(df_raw))

    for i, col in enumerate(columns[:n_cols]):
        if col in df_raw.columns and col in df_filtered.columns:
            axes[i].plot(time_data[:n_show], df_raw[col].values[:n_show],
                        color=IDR_COLORS["raw"], linewidth=0.5, alpha=0.5, label="Raw")
            axes[i].plot(time_data[:n_show], df_filtered[col].values[:n_show],
                        color=IDR_COLORS["filtered"], linewidth=1.2, alpha=0.9, label="Filtered")
            axes[i].set_ylabel(col, fontsize=10)
            axes[i].legend(fontsize=9, loc="upper right")
            axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [Plot] Saved: {save_path}")


def plot_data_quality(quality_report, save_path):
    """Plot data quality summary -- missing values bar chart."""
    columns = quality_report["columns"]

    # Get columns with missing data
    missing_data = {
        col: info["missing_pct"]
        for col, info in columns.items()
        if info["missing_pct"] > 0
    }

    fig, ax = plt.subplots(1, 1, figsize=(14, max(6, len(missing_data) * 0.4)))
    fig.suptitle("Data Quality: Missing Values", fontsize=15, fontweight="bold")

    if missing_data:
        sorted_items = sorted(missing_data.items(), key=lambda x: x[1], reverse=True)
        cols_list = [item[0] for item in sorted_items]
        pcts = [item[1] for item in sorted_items]

        colors = ["#E74C3C" if p > 50 else "#F39C12" if p > 10 else "#2ECC71" for p in pcts]
        bars = ax.barh(cols_list, pcts, color=colors, alpha=0.8, edgecolor="white")
        ax.set_xlabel("Missing (%)", fontsize=12)
        ax.set_title(f"Columns with Missing Data ({len(missing_data)} of {len(columns)})",
                     fontsize=12)

        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", ha="left", va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No missing data! [OK]",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=20, color="#2ECC71", fontweight="bold")

    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [Plot] Saved: {save_path}")


# --------------------------- MAIN PIPELINE ---------------------------

def main():
    args = parse_args()
    print_banner()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ======================= STEP 1: LOAD =======================
    print("=" * 60)
    print("  STEP 1: Loading IO-VNBD Dataset")
    print("=" * 60)

    loader = DatasetLoader(str(DATASET_ROOT))
    summary = loader.summary()

    print(f"\n  Dataset Summary:")
    print(f"    Root:            {summary['dataset_root']}")
    print(f"    Total CSV files: {summary['total_csv_files']}")
    print(f"    Smartphone (S-): {summary['smartphone_files']}")
    print(f"    Vehicle (V-):    {summary['vehicle_files']}")
    print(f"    Synced folder:   {summary['synced_folder']}")
    print(f"    Synced files:    {summary['synced_files']}")
    print(f"    Trip names:      {summary['trip_names'][:10]}")

    print(f"\n  All files:")
    for f in summary["all_files"][:20]:
        print(f"    {f}")
    if len(summary["all_files"]) > 20:
        print(f"    ... and {len(summary['all_files']) - 20} more")

    # Save dataset summary
    with open(RESULTS_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Load a trip
    print(f"\n  Loading trip {args.trip_name or args.trip} ({args.data_type})...")
    try:
        if args.data_type in ("synced", "pair") or args.trip_name is not None:
            df_raw, metadata = loader.load_synced_pair(
                trip_index=args.trip,
                trip_name=args.trip_name,
                nrows=args.nrows,
            )
        else:
            df_raw, metadata = loader.load_trip(
                trip_index=args.trip,
                data_type=args.data_type,
                nrows=args.nrows,
            )
    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"\n  Error loading trip: {e}")
        print("  Trying to load any available CSV file...")
        # Fallback: load the first CSV file found
        if loader.all_csv_files:
            df_raw = loader.load_csv(loader.all_csv_files[0], nrows=args.nrows)
            metadata = {
                "filepath": str(loader.all_csv_files[0]),
                "filename": loader.all_csv_files[0].name,
                "shape": df_raw.shape,
                "columns": list(df_raw.columns),
            }
        else:
            print("  FATAL: No CSV files found in dataset!")
            sys.exit(1)

    print(f"\n  Loaded: {metadata['filename']}")
    print(f"  Shape:  {metadata['shape']}")

    # ======================= STEP 2: INSPECT =======================
    print("\n" + "=" * 60)
    print("  STEP 2: Inspecting Dataset")
    print("=" * 60)

    trip_name = Path(metadata["filename"]).stem
    inspector = DatasetInspector(df_raw, name=trip_name)
    inspector.print_summary()

    # Get sensor identification for downstream use
    identified = inspector.identify_columns()
    sampling_info = inspector.compute_sampling_rate()
    quality_report = inspector.data_quality_report()

    # Save inspection report
    inspector.save_report(str(RESULTS_DIR / "inspection_report.json"))

    # Determine time column
    time_col = identified["timestamp"][0] if identified["timestamp"] else df_raw.columns[0]
    print(f"\n  Using timestamp column: '{time_col}'")

    # Determine sensor columns
    accel_cols = identified["accelerometer"]
    gyro_cols = identified["gyroscope"]
    gnss_cols = identified["gnss"]
    sensor_cols = accel_cols + gyro_cols

    print(f"  Accelerometer columns: {accel_cols}")
    print(f"  Gyroscope columns:     {gyro_cols}")
    print(f"  GNSS columns:          {gnss_cols}")

    # Store a copy for before/after comparison
    df_before_filter = df_raw.copy()

    # ======================= STEP 3: CLEAN =======================
    print("\n" + "=" * 60)
    print("  STEP 3: Cleaning Data")
    print("=" * 60)

    cleaner = DataCleaner(df_raw)
    df_cleaned = cleaner.clean(
        time_col=time_col,
        sensor_columns=sensor_cols,
        interpolate_max_gap=5,
        outlier_z_threshold=5.0,
        gap_threshold_seconds=1.0,
    )

    # ======================= STEP 4: FILTER =======================
    print("\n" + "=" * 60)
    print("  STEP 4: Filtering Signals")
    print("=" * 60)

    sampling_rate = sampling_info.get("sampling_rate_hz", 10.0)
    print(f"  Detected sampling rate: {sampling_rate:.2f} Hz")

    sig_filter = SignalFilter(sampling_rate_hz=sampling_rate)

    # Filter accelerometer
    if accel_cols:
        df_filtered = sig_filter.filter_accelerometer(
            df_cleaned, accel_cols, cutoff_hz=args.cutoff_hz
        )
    else:
        df_filtered = df_cleaned

    # Filter gyroscope
    if gyro_cols:
        df_filtered = sig_filter.filter_gyroscope(
            df_filtered, gyro_cols, cutoff_hz=args.cutoff_hz
        )

    # ======================= STEP 5: SYNCHRONIZE =======================
    print("\n" + "=" * 60)
    print("  STEP 5: Synchronizing Sensor Streams")
    print("=" * 60)

    synchronizer = SensorSynchronizer(target_rate_hz=sampling_rate)
    df_synced = synchronizer.synchronize(
        df_filtered,
        time_col=time_col,
        gnss_columns=gnss_cols,
        gnss_strategy="forward_fill",
    )

    # ======================= STEP 6: CONVERT =======================
    print("\n" + "=" * 60)
    print("  STEP 6: Converting to Standard Format")
    print("=" * 60)

    converter = FormatConverter()
    df_standard = converter.convert(
        df_synced,
        identified_sensors=identified,
        keep_extra=True,
    )

    # Save processed data
    saved_files = converter.save(
        df_standard,
        output_dir=str(OUTPUT_DIR),
        trip_name=trip_name,
        formats=["csv"],
    )

    # Print final format
    print(f"\n  Standard format columns: {list(df_standard.columns)}")
    print(f"\n  Final shape: {df_standard.shape}")
    print(f"\n  First 5 rows of standard format:")
    print(df_standard.head().to_string(index=False))

    # ======================= STEP 7: VISUALIZE =======================
    if not args.no_plots:
        print("\n" + "=" * 60)
        print("  STEP 7: Generating Visualizations")
        print("=" * 60)

        # Use standard column names for plots if available
        std_accel = [c for c in ["ax", "ay", "az"] if c in df_standard.columns]
        std_gyro = [c for c in ["gx", "gy", "gz"] if c in df_standard.columns]
        std_time = "timestamp" if "timestamp" in df_standard.columns else time_col

        # Plot 1: Raw accelerometer
        plot_accelerometer(
            df_before_filter, accel_cols, time_col,
            str(RESULTS_DIR / "01_accelerometer_raw.png"),
            title_suffix="(Raw)"
        )

        # Plot 2: Raw gyroscope
        plot_gyroscope(
            df_before_filter, gyro_cols, time_col,
            str(RESULTS_DIR / "02_gyroscope_raw.png"),
            title_suffix="(Raw)"
        )

        # Plot 3: GNSS/Reference trajectory
        # Check standard columns first, then raw columns
        lat_col = "gnss_lat" if "gnss_lat" in df_standard.columns and df_standard["gnss_lat"].notna().any() else None
        lon_col = "gnss_lon" if "gnss_lon" in df_standard.columns and df_standard["gnss_lon"].notna().any() else None
        ref_lat = "ref_lat" if "ref_lat" in df_standard.columns and df_standard["ref_lat"].notna().any() else None
        ref_lon = "ref_lon" if "ref_lon" in df_standard.columns and df_standard["ref_lon"].notna().any() else None

        # Fallback to searching in df_raw if not found
        if lat_col is None or lon_col is None:
            gnss_lower = {c: c.lower() for c in gnss_cols}
            for col, col_low in gnss_lower.items():
                if "lat" in col_low and lat_col is None:
                    lat_col = col
                elif "lon" in col_low and lon_col is None:
                    lon_col = col

        plot_gnss_trajectory(
            df_standard if ("gnss_lat" in df_standard.columns or "ref_lat" in df_standard.columns) else df_raw,
            lat_col, lon_col,
            str(RESULTS_DIR / "03_gnss_trajectory.png"),
            ref_lat_col=ref_lat, ref_lon_col=ref_lon,
        )

        # Plot 4: Sampling rate analysis
        plot_sampling_rate(
            df_raw, time_col,
            str(RESULTS_DIR / "04_sampling_rate.png"),
        )

        # Plot 5: Filter comparison
        plot_filter_comparison(
            df_before_filter, df_filtered,
            accel_cols[:3] if accel_cols else [],
            time_col,
            str(RESULTS_DIR / "05_filter_comparison.png"),
        )

        # Plot 6: Data quality
        plot_data_quality(
            quality_report,
            str(RESULTS_DIR / "06_data_quality.png"),
        )

    # ======================= SUMMARY =======================
    print("\n" + "=" * 60)
    print("  PHASE 1 COMPLETE -- Summary")
    print("=" * 60)

    print(f"""
  Dataset:         {metadata.get('filename', 'N/A')}
  Raw shape:       {df_raw.shape}
  Cleaned shape:   {df_cleaned.shape}
  Final shape:     {df_standard.shape}
  Sampling rate:   {sampling_rate:.2f} Hz
  Duration:        {sampling_info.get('duration_seconds', 0):.1f} seconds
                   ({sampling_info.get('duration_seconds', 0)/60:.1f} minutes)

  Sensor columns found:
    Accelerometer: {accel_cols}
    Gyroscope:     {gyro_cols}
    GNSS:          {gnss_cols}

  Outputs saved:
    Processed data: {saved_files}
    Results dir:    {RESULTS_DIR}

  Cleaning log:
""")
    for entry in cleaner.get_log():
        print(f"    • {entry}")

    # Save column mapping for future phases
    mapping_info = {
        "column_mapping": converter.get_mapping(),
        "identified_sensors": identified,
        "sampling_rate_hz": sampling_rate,
        "time_column": time_col,
        "trip_name": trip_name,
    }
    with open(RESULTS_DIR / "column_mapping.json", "w") as f:
        json.dump(mapping_info, f, indent=2, default=str)
    print(f"  Column mapping saved to: {RESULTS_DIR / 'column_mapping.json'}")

    print(f"\n{'=' * 60}")
    print("  [OK] Phase 1 pipeline executed successfully!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
