"""
IDR MVP -- Phase 2 Runner: Calibration & Alignment
===================================================
Main orchestration script for Phase 2: Frame Calibration & Coordinate Alignment.

Executes the full Phase 2 pipeline:
1. Load preprocessed sensor dataset (Phase 1 output)
2. Detect stationary intervals & compute gravity vector
3. Estimate initial Pitch and Roll orientation angles
4. Detect linear vehicle motion intervals & compute phone-to-vehicle Yaw offset
5. Build 3D rotation matrix (Phone Frame -> Vehicle Frame)
6. Transform IMU acceleration and gyroscope signals to vehicle coordinate frame
7. Generate and save diagnostic visualization plots
8. Export calibrated dataset and calibration parameters JSON

Usage:
    python scripts/run_phase2.py
    python scripts/run_phase2.py --input data/processed/SYNC_s1_processed.csv
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
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import StaticDetector, FrameCalibrator, CoordinateTransformer

# --- Configuration & Palette ---
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "SYNC_s1_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"

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

COLORS = {
    "phone_x": "#FF6B6B",
    "phone_y": "#4ECDC4",
    "phone_z": "#45B7D1",
    "veh_forward": "#2ECC71",
    "veh_lateral": "#F39C12",
    "veh_vertical": "#9B59B6",
    "static": "#E74C3C",
    "moving": "#3498DB",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="IDR MVP -- Phase 2: Calibration & Alignment"
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT),
        help=f"Path to Phase 1 processed CSV (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help=f"Output directory for calibrated data (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR),
        help=f"Directory for plots and results (default: {RESULTS_DIR})"
    )
    return parser.parse_args()


def plot_stationary_detection(df: pd.DataFrame, static_mask: pd.Series, moving_mask: pd.Series, save_path: Path):
    """Plot IMU signal magnitudes with detected static/moving windows highlighted."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8))

    time = df["timestamp"] - df["timestamp"].iloc[0] if "timestamp" in df.columns else df.index

    accel_norm = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    gyro_norm = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)

    # Subplot 1: Accel Norm
    ax1.plot(time, accel_norm, color="#34495E", alpha=0.7, label="Accel Magnitude (m/s²)")
    ax1.fill_between(time, 0, accel_norm.max(), where=static_mask, color=COLORS["static"], alpha=0.3, label="Stationary Window")
    ax1.set_ylabel("Accel Norm (m/s²)")
    ax1.set_title("Phase 2 Calibration: Stationary & Moving Window Detection")
    ax1.legend(loc="upper right")

    # Subplot 2: Gyro Norm / Speed
    ax2.plot(time, gyro_norm, color="#7F8C8D", alpha=0.7, label="Gyro Magnitude (rad/s)")
    if "gnss_speed" in df.columns and not df["gnss_speed"].isna().all():
        ax2_speed = ax2.twinx()
        ax2_speed.plot(time, df["gnss_speed"], color="#27AE60", alpha=0.6, label="GNSS Speed (m/s)")
        ax2_speed.set_ylabel("Speed (m/s)", color="#27AE60")
        ax2_speed.grid(False)

    ax2.fill_between(time, 0, gyro_norm.max(), where=moving_mask, color=COLORS["moving"], alpha=0.25, label="Moving Window (GNSS > 2m/s)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Gyro Norm (rad/s)")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [Plots] Saved: {save_path.name}")


def plot_phone_vs_vehicle_accel(df: pd.DataFrame, save_path: Path):
    """Plot Raw Phone Frame Accelerations vs Calibrated Vehicle Frame Accelerations."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 9))

    time = df["timestamp"] - df["timestamp"].iloc[0] if "timestamp" in df.columns else df.index

    # Raw Phone Frame
    ax1.plot(time, df["ax"], label="ax (Phone X)", color=COLORS["phone_x"], alpha=0.8)
    ax1.plot(time, df["ay"], label="ay (Phone Y)", color=COLORS["phone_y"], alpha=0.8)
    ax1.plot(time, df["az"], label="az (Phone Z)", color=COLORS["phone_z"], alpha=0.8)
    ax1.set_ylabel("Accel (m/s²)")
    ax1.set_title("Phone Frame Accelerations (Raw / Filtered)")
    ax1.legend(loc="upper right")

    # Calibrated Vehicle Frame
    ax2.plot(time, df["ax_veh"], label="ax_veh (Forward)", color=COLORS["veh_forward"], alpha=0.85)
    ax2.plot(time, df["ay_veh"], label="ay_veh (Lateral)", color=COLORS["veh_lateral"], alpha=0.85)
    ax2.plot(time, df["az_veh"], label="az_veh (Vertical)", color=COLORS["veh_vertical"], alpha=0.85)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Accel (m/s²)")
    ax2.set_title("Calibrated Vehicle Frame Accelerations")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [Plots] Saved: {save_path.name}")


def plot_linear_vehicle_accel(df: pd.DataFrame, save_path: Path):
    """Plot Linear Vehicle Accelerations (Gravity Subtracted)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    time = df["timestamp"] - df["timestamp"].iloc[0] if "timestamp" in df.columns else df.index

    ax.plot(time, df["ax_veh_lin"], label="ax_veh_lin (Forward Motion)", color=COLORS["veh_forward"], alpha=0.85)
    ax.plot(time, df["ay_veh_lin"], label="ay_veh_lin (Lateral Sway)", color=COLORS["veh_lateral"], alpha=0.85)
    ax.plot(time, df["az_veh_lin"], label="az_veh_lin (Vertical Bump)", color=COLORS["veh_vertical"], alpha=0.6)

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Linear Acceleration (m/s²)")
    ax.set_title("Vehicle Linear Acceleration (Gravity Subtracted)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [Plots] Saved: {save_path.name}")


def plot_calibration_summary(calib_res: dict, save_path: Path):
    """Display visual summary panel of calibration angles and rotation matrix."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    summary_text = (
        "IDR Phase 2 -- Calibration & Coordinate Alignment Summary\n"
        "=========================================================\n\n"
        f"  • Estimated Pitch (Pitch θ):      {calib_res['pitch_deg']:+.2f}° ({calib_res['pitch_rad']:+.4f} rad)\n"
        f"  • Estimated Roll (Roll φ):       {calib_res['roll_deg']:+.2f}° ({calib_res['roll_rad']:+.4f} rad)\n"
        f"  • Phone-to-Vehicle Yaw Offset (ψ): {calib_res['yaw_offset_deg']:+.2f}° ({calib_res['yaw_offset_rad']:+.4f} rad)\n"
        f"  • Static Gravity Magnitude (g):   {calib_res['gravity_norm']:.4f} m/s²\n\n"
        "  • Static Gyro Biases (rad/s):\n"
        f"      gx_bias: {calib_res['gyro_bias'][0]:+.6f} | gy_bias: {calib_res['gyro_bias'][1]:+.6f} | gz_bias: {calib_res['gyro_bias'][2]:+.6f}\n\n"
        "  • 3D Rotation Matrix R (Phone -> Vehicle Frame):\n"
        f"      [ {calib_res['rotation_matrix'][0,0]:+8.4f}, {calib_res['rotation_matrix'][0,1]:+8.4f}, {calib_res['rotation_matrix'][0,2]:+8.4f} ]\n"
        f"      [ {calib_res['rotation_matrix'][1,0]:+8.4f}, {calib_res['rotation_matrix'][1,1]:+8.4f}, {calib_res['rotation_matrix'][1,2]:+8.4f} ]\n"
        f"      [ {calib_res['rotation_matrix'][2,0]:+8.4f}, {calib_res['rotation_matrix'][2,1]:+8.4f}, {calib_res['rotation_matrix'][2,2]:+8.4f} ]\n"
    )

    ax.text(
        0.05, 0.95, summary_text,
        transform=ax.transAxes,
        fontsize=12,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=1.0", facecolor="#2C3E50", alpha=0.95, edgecolor="#34495E")
    )
    # White text color for contrast
    for text in ax.texts:
        text.set_color("#ECF0F1")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [Plots] Saved: {save_path.name}")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  IDR MVP -- Phase 2: Calibration & Alignment")
    print("=" * 70)

    # 1. Load Data
    if not input_path.exists():
        # Fallback to search any processed CSV in output_dir
        processed_files = list(output_dir.glob("*_processed.csv"))
        if processed_files:
            input_path = processed_files[0]
            print(f"  [Runner] Specified file not found. Fallback to: {input_path}")
        else:
            raise FileNotFoundError(f"No processed CSV dataset found at {input_path}. Please run Phase 1 first.")

    print(f"  [Runner] Loading preprocessed dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  [Runner] Loaded {len(df)} rows, columns: {list(df.columns)}")

    # 2. Detection
    detector = StaticDetector(sampling_rate_hz=10.0)
    static_mask = detector.detect_stationary_samples(df)
    moving_mask = detector.detect_moving_samples(df)
    gnss_course = detector.compute_gnss_course(df)

    n_static = static_mask.sum()
    n_moving = moving_mask.sum()
    print(f"  [Detector] Stationary samples: {n_static} / {len(df)} ({n_static/len(df)*100:.1f}%)")
    print(f"  [Detector] Moving samples:     {n_moving} / {len(df)} ({n_moving/len(df)*100:.1f}%)")

    # 3. Calibration
    calibrator = FrameCalibrator()
    calib_res = calibrator.calibrate(
        df,
        static_mask=static_mask,
        moving_mask=moving_mask,
        gnss_course_rad=gnss_course,
    )

    print("\n  --- Calibration Results ---")
    print(f"  Pitch Angle (θ):        {calib_res['pitch_deg']:+.2f}°")
    print(f"  Roll Angle (φ):         {calib_res['roll_deg']:+.2f}°")
    print(f"  Yaw Offset (ψ):         {calib_res['yaw_offset_deg']:+.2f}°")
    print(f"  Gravity Norm:           {calib_res['gravity_norm']:.4f} m/s²")
    print(f"  Static Gyro Bias:       gx={calib_res['gyro_bias'][0]:.6f}, gy={calib_res['gyro_bias'][1]:.6f}, gz={calib_res['gyro_bias'][2]:.6f}")

    # 4. Transformation
    transformer = CoordinateTransformer(rotation_matrix=calib_res["rotation_matrix"])
    df_calibrated = transformer.transform_dataframe(
        df,
        gyro_bias=calib_res["gyro_bias"],
        gravity_magnitude=calib_res["gravity_norm"],
    )

    # 5. Save Calibrated Dataset & Params JSON
    stem = input_path.stem.replace("_processed", "")
    save_csv_path = output_dir / f"{stem}_calibrated.csv"
    df_calibrated.to_csv(save_csv_path, index=False)
    print(f"\n  [Saver] Saved calibrated dataset to: {save_csv_path}")

    # Export params dict
    json_params = {
        "pitch_deg": float(calib_res["pitch_deg"]),
        "roll_deg": float(calib_res["roll_deg"]),
        "yaw_offset_deg": float(calib_res["yaw_offset_deg"]),
        "gravity_norm": float(calib_res["gravity_norm"]),
        "gyro_bias": calib_res["gyro_bias"].tolist(),
        "rotation_matrix": calib_res["rotation_matrix"].tolist(),
    }
    params_json_path = results_dir / "calibration_params.json"
    with open(params_json_path, "w", encoding="utf-8") as f:
        json.dump(json_params, f, indent=2)
    print(f"  [Saver] Saved calibration parameters to: {params_json_path}")

    # 6. Generate Plots
    print("\n  --- Generating Phase 2 Diagnostic Visualizations ---")
    plot_stationary_detection(df, static_mask, moving_mask, results_dir / "01_stationary_detection.png")
    plot_phone_vs_vehicle_accel(df_calibrated, results_dir / "02_phone_vs_vehicle_accel.png")
    plot_linear_vehicle_accel(df_calibrated, results_dir / "03_vehicle_linear_accel.png")
    plot_calibration_summary(calib_res, results_dir / "05_calibration_summary.png")

    print("\n" + "=" * 70)
    print("  Phase 2 Calibration & Alignment Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
