"""
IDR MVP -- Phase 7 Runner: AI-Assisted Dead Reckoning
=======================================================
Orchestrates Phase 7: Three-way trajectory comparison (MVP.md §11).

Pipelines:
    A. Raw INS:        IMU → StrapdownINS
    B. AI + INS:       IMU → AI velocity → AIAssistedINS
    C. AI + INS + NHC: IMU → AI velocity → AIAssistedINS + NHC

Pipeline:
    1. Load calibrated dataset + calibration params.
    2. Load trained AI model weights + scaler from Phase 5.
    3. Generate AI velocity predictions for the full trip.
    4. Run Pipeline A (Raw INS).
    5. Run Pipeline B (AI + INS).
    6. Run Pipeline C (AI + INS + NHC).
    7. Compute comparative drift metrics.
    8. Save metrics JSON + trajectory CSVs to results/phase7/.
    9. Generate 5 diagnostic plots.

Usage:
    python -u scripts/run_phase7.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ins import StrapdownINS, AIAssistedINS, NHCCorrector
from src.ins.integration import haversine_distance
from src.ai import IMUScaler, IMUSequenceDataset, NumpyCNNInference


# ---------------------------------------------------------------------------
# Configuration & Theme
# ---------------------------------------------------------------------------

DEFAULT_INPUT   = PROJECT_ROOT / "data" / "processed" / "SYNC_s1_calibrated.csv"
DEFAULT_PARAMS  = PROJECT_ROOT / "results" / "phase2" / "calibration_params.json"
MODELS_DIR      = PROJECT_ROOT / "models"
RESULTS_DIR     = PROJECT_ROOT / "results" / "phase7"

sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({
    "figure.figsize":     (14, 7),
    "figure.dpi":         120,
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.2,
})

PALETTE = {
    "gt":       "#2ECC71",   # Ground truth   — emerald green
    "raw_ins":  "#E74C3C",   # Raw INS        — red
    "ai_ins":   "#3498DB",   # AI + INS       — blue
    "ai_nhc":   "#9B59B6",   # AI + INS + NHC — purple
    "gnss":     "#F39C12",   # GNSS track     — amber
    "nhc_corr": "#1ABC9C",   # NHC correction — teal
    "ai_vel":   "#E67E22",   # AI velocity    — orange
}


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="IDR MVP -- Phase 7: AI-Assisted Dead Reckoning"
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT),
        help=f"Path to Phase 2 calibrated CSV (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--params", type=str, default=str(DEFAULT_PARAMS),
        help=f"Path to calibration_params.json (default: {DEFAULT_PARAMS})"
    )
    parser.add_argument(
        "--models-dir", type=str, default=str(MODELS_DIR),
        help=f"Directory containing trained AI model (default: {MODELS_DIR})"
    )
    parser.add_argument(
        "--window-size", type=int, default=50,
        help="AI model sliding window size in samples (default: 50)"
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR),
        help=f"Directory for outputs (default: {RESULTS_DIR})"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_first_gnss_fix(df: pd.DataFrame, min_speed_ms: float = 0.5) -> int:
    """Returns the index of the first row with valid GNSS position."""
    valid_gnss = df["gnss_lat"].notna() & df["gnss_lon"].notna()
    if not valid_gnss.any():
        return 0
    moving = valid_gnss & (df.get("gnss_speed", pd.Series(0.0, index=df.index)) >= min_speed_ms)
    if moving.any():
        return df[moving].index[0]
    return df[valid_gnss].index[0]


def get_initial_heading(df: pd.DataFrame, start_idx: int) -> float:
    """Returns best available heading at start_idx (degrees)."""
    row = df.loc[start_idx]
    for col in ["ref_heading", "heading"]:
        if col in df.columns:
            h = row.get(col, np.nan)
            if pd.notna(h):
                return float(h)
    return 0.0


def compute_position_errors(traj_df: pd.DataFrame, ref_df: pd.DataFrame) -> np.ndarray:
    """Computes per-sample position error in metres between trajectory and reference."""
    n = min(len(traj_df), len(ref_df))
    errors = np.full(n, np.nan)

    ref_lat = ref_df["ref_lat"].to_numpy(dtype=float)[:n] if "ref_lat" in ref_df.columns else np.full(n, np.nan)
    ref_lon = ref_df["ref_lon"].to_numpy(dtype=float)[:n] if "ref_lon" in ref_df.columns else np.full(n, np.nan)

    if "gnss_lat" in ref_df.columns and "gnss_lon" in ref_df.columns:
        gnss_lat = ref_df["gnss_lat"].to_numpy(dtype=float)[:n]
        gnss_lon = ref_df["gnss_lon"].to_numpy(dtype=float)[:n]
        nan_mask = np.isnan(ref_lat) | np.isnan(ref_lon)
        ref_lat = np.where(nan_mask, gnss_lat, ref_lat)
        ref_lon = np.where(nan_mask, gnss_lon, ref_lon)

    valid = ~(np.isnan(ref_lat) | np.isnan(ref_lon))
    if np.any(valid):
        ins_lat = np.radians(traj_df["lat_deg"].to_numpy(dtype=float)[:n])
        ins_lon = np.radians(traj_df["lon_deg"].to_numpy(dtype=float)[:n])
        errors[valid] = haversine_distance(
            ins_lat[valid], ins_lon[valid],
            np.radians(ref_lat[valid]), np.radians(ref_lon[valid])
        )

    return errors


def compute_drift_metrics(errors: np.ndarray, ref_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute drift metrics from position error array."""
    valid = errors[~np.isnan(errors)]
    if len(valid) == 0:
        return {"error": "No valid errors"}

    # Compute total reference distance
    total_dist = 0.0
    for col_lat, col_lon in [("ref_lat", "ref_lon"), ("gnss_lat", "gnss_lon")]:
        if col_lat in ref_df.columns and not ref_df[col_lat].isna().all():
            lats = ref_df[col_lat].to_numpy(dtype=float)
            lons = ref_df[col_lon].to_numpy(dtype=float)
            valid_step = (
                ~np.isnan(lats[:-1]) & ~np.isnan(lons[:-1]) &
                ~np.isnan(lats[1:])  & ~np.isnan(lons[1:])
            )
            if np.any(valid_step):
                lat1 = np.radians(lats[:-1][valid_step])
                lon1 = np.radians(lons[:-1][valid_step])
                lat2 = np.radians(lats[1:][valid_step])
                lon2 = np.radians(lons[1:][valid_step])
                total_dist = float(np.sum(haversine_distance(lat1, lon1, lat2, lon2)))
            break

    return {
        "rmse_m": float(np.sqrt(np.mean(valid**2))),
        "mae_m": float(np.mean(valid)),
        "max_error_m": float(np.max(valid)),
        "final_error_m": float(valid[-1]),
        "total_distance_m": total_dist,
        "drift_percent": float(np.mean(valid) / total_dist * 100) if total_dist > 1.0 else np.nan,
        "n_samples": len(valid),
    }


def generate_ai_velocity(
    df: pd.DataFrame,
    weights_path: Path,
    scaler_path: Path,
    window_size: int = 50,
) -> np.ndarray:
    """
    Generate AI velocity predictions for the full trip.

    Returns an array of length len(df), with NaN for the first
    (window_size - 1) samples that don't have a full window.
    """
    print(f"  Loading AI model weights: {weights_path}")
    engine = NumpyCNNInference.load_weights_npz(weights_path)

    print(f"  Loading feature scaler: {scaler_path}")
    scaler = IMUScaler.load(scaler_path)

    print("  Extracting IMU features...")
    features, _ = IMUSequenceDataset.extract_features(df)
    features_scaled = scaler.transform(features)

    n = len(df)
    ai_vel = np.full(n, np.nan, dtype=np.float64)

    print(f"  Running AI inference ({n - window_size + 1:,} windows)...")
    for i in range(window_size - 1, n):
        window = features_scaled[i - window_size + 1 : i + 1]  # (W, 6)
        pred = engine.predict(window)
        if isinstance(pred, (float, np.floating)):
            ai_vel[i] = max(0.0, float(pred))
        else:
            ai_vel[i] = max(0.0, float(pred.flatten()[0]))

    valid_count = np.sum(~np.isnan(ai_vel))
    print(f"  AI velocity predictions: {valid_count:,} valid / {n:,} total")
    print(f"  Ramp-up: first {window_size - 1} samples use fallback (NaN → 0)")

    return ai_vel


# ---------------------------------------------------------------------------
# Plotting Suite — 5 Diagnostic Plots
# ---------------------------------------------------------------------------

def plot_01_trajectory_comparison(
    traj_a: pd.DataFrame,
    traj_b: pd.DataFrame,
    traj_c: pd.DataFrame,
    ref_df: pd.DataFrame,
    output_path: Path,
):
    """Plot 1: Three-trajectory overlay — the money plot."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Reference / ground truth
    for col_lat, col_lon, label, color in [
        ("ref_lat", "ref_lon", "Ground Truth", PALETTE["gt"]),
        ("gnss_lat", "gnss_lon", "GNSS Track", PALETTE["gnss"]),
    ]:
        if col_lat in ref_df.columns and not ref_df[col_lat].isna().all():
            ax.plot(ref_df[col_lon], ref_df[col_lat],
                    color=color, linewidth=2.5, label=label, zorder=5, alpha=0.85)

    # Pipeline trajectories
    ax.plot(traj_a["lon_deg"], traj_a["lat_deg"],
            color=PALETTE["raw_ins"], linewidth=1.5, linestyle="--",
            label="A: Raw INS", zorder=3, alpha=0.75)

    ax.plot(traj_b["lon_deg"], traj_b["lat_deg"],
            color=PALETTE["ai_ins"], linewidth=1.8, linestyle="-.",
            label="B: AI + INS", zorder=4, alpha=0.85)

    ax.plot(traj_c["lon_deg"], traj_c["lat_deg"],
            color=PALETTE["ai_nhc"], linewidth=2.0,
            label="C: AI + INS + NHC", zorder=4, alpha=0.9)

    # Start marker
    ax.scatter([traj_a["lon_deg"].iloc[0]], [traj_a["lat_deg"].iloc[0]],
               s=120, color="black", marker="o", zorder=10, label="Start")

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Phase 7 — Three-Way Trajectory Comparison\n"
                 "(Ground Truth vs Raw INS vs AI+INS vs AI+INS+NHC)",
                 fontweight="bold", pad=12)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 1] Saved: {output_path.name}")


def plot_02_position_error(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    errors_c: np.ndarray,
    output_path: Path,
):
    """Plot 2: Position error over time for all three pipelines."""
    fig, ax = plt.subplots(figsize=(16, 6))

    time_s = np.arange(len(errors_a)) * 0.1  # 10 Hz

    ax.plot(time_s, errors_a, color=PALETTE["raw_ins"], linewidth=1.2,
            label="A: Raw INS", alpha=0.7)
    ax.plot(time_s, errors_b, color=PALETTE["ai_ins"], linewidth=1.5,
            label="B: AI + INS", alpha=0.85)
    ax.plot(time_s, errors_c, color=PALETTE["ai_nhc"], linewidth=1.8,
            label="C: AI + INS + NHC", alpha=0.9)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Position Error (metres)")
    ax.set_title("Phase 7 — Position Error Over Time (Three Pipelines)",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 2] Saved: {output_path.name}")


def plot_03_drift_comparison(
    metrics_a: Dict,
    metrics_b: Dict,
    metrics_c: Dict,
    output_path: Path,
):
    """Plot 3: Drift % comparison bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = ["A: Raw INS", "B: AI + INS", "C: AI + INS + NHC"]
    colors = [PALETTE["raw_ins"], PALETTE["ai_ins"], PALETTE["ai_nhc"]]

    # Drift %
    drift_vals = [
        metrics_a.get("drift_percent", 0) or 0,
        metrics_b.get("drift_percent", 0) or 0,
        metrics_c.get("drift_percent", 0) or 0,
    ]
    bars = ax1.bar(labels, drift_vals, color=colors, edgecolor="white", linewidth=1.5)
    ax1.axhline(10.0, color="#2C3E50", linestyle="--", linewidth=1.5, label="10% Target")
    for bar, val in zip(bars, drift_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Drift (%)")
    ax1.set_title("Positional Drift %", fontweight="bold")
    ax1.legend(fontsize=9)

    # RMSE
    rmse_vals = [
        metrics_a.get("rmse_m", 0) or 0,
        metrics_b.get("rmse_m", 0) or 0,
        metrics_c.get("rmse_m", 0) or 0,
    ]
    bars2 = ax2.bar(labels, rmse_vals, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars2, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}m", ha="center", fontweight="bold", fontsize=11)
    ax2.set_ylabel("RMSE (metres)")
    ax2.set_title("Position RMSE", fontweight="bold")

    plt.suptitle("Phase 7 — Drift & Error Comparison", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 3] Saved: {output_path.name}")


def plot_04_velocity_comparison(
    ref_df: pd.DataFrame,
    ai_vel: np.ndarray,
    traj_a: pd.DataFrame,
    output_path: Path,
    sample_limit: int = 2000,
):
    """Plot 4: Velocity comparison — GT vs AI vs Raw INS integrated."""
    fig, ax = plt.subplots(figsize=(16, 6))

    n = min(len(ref_df), len(traj_a), len(ai_vel), sample_limit)
    time_s = np.arange(n) * 0.1

    # Reference speed
    for col in ["ref_speed", "gnss_speed"]:
        if col in ref_df.columns and not ref_df[col].isna().all():
            ref_spd = ref_df[col].values[:n]
            # Convert km/h to m/s if needed
            if np.nanmedian(ref_spd) > 10.0:
                ref_spd = ref_spd / 3.6
            ax.plot(time_s, ref_spd, color=PALETTE["gt"], linewidth=2.0,
                    label="Ground Truth Speed (m/s)", alpha=0.85)
            break

    # AI velocity
    ai_plot = ai_vel[:n]
    valid_mask = ~np.isnan(ai_plot)
    if valid_mask.any():
        ax.plot(time_s[valid_mask], ai_plot[valid_mask],
                color=PALETTE["ai_vel"], linewidth=1.5, linestyle="--",
                label="AI Predicted Speed (m/s)", alpha=0.8)

    # Raw INS integrated speed
    ax.plot(time_s, traj_a["speed_ms"].values[:n],
            color=PALETTE["raw_ins"], linewidth=1.0, alpha=0.5,
            label="Raw INS Speed (m/s)")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Phase 7 — Velocity Comparison (GT vs AI vs Raw INS)",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 4] Saved: {output_path.name}")


def plot_05_nhc_correction(
    traj_b: pd.DataFrame,
    traj_c: pd.DataFrame,
    output_path: Path,
    sample_limit: int = 2000,
):
    """Plot 5: NHC correction magnitude over time (difference between B and C)."""
    fig, ax = plt.subplots(figsize=(16, 5))

    n = min(len(traj_b), len(traj_c), sample_limit)
    time_s = np.arange(n) * 0.1

    # Speed difference between B and C as proxy for NHC correction effect
    speed_b = traj_b["speed_ms"].values[:n]
    speed_c = traj_c["speed_ms"].values[:n]
    speed_diff = np.abs(speed_b - speed_c)

    # Position difference
    lat_b = np.radians(traj_b["lat_deg"].to_numpy(dtype=float)[:n])
    lon_b = np.radians(traj_b["lon_deg"].to_numpy(dtype=float)[:n])
    lat_c = np.radians(traj_c["lat_deg"].to_numpy(dtype=float)[:n])
    lon_c = np.radians(traj_c["lon_deg"].to_numpy(dtype=float)[:n])
    pos_diff = haversine_distance(lat_b, lon_b, lat_c, lon_c)

    ax.plot(time_s, pos_diff, color=PALETTE["nhc_corr"], linewidth=1.5,
            label="Position Difference B→C (m)")
    ax.fill_between(time_s, 0, pos_diff, alpha=0.15, color=PALETTE["nhc_corr"])

    ax2 = ax.twinx()
    ax2.plot(time_s, speed_diff, color=PALETTE["ai_vel"], linewidth=1.0,
             linestyle="--", alpha=0.6, label="Speed Difference (m/s)")
    ax2.set_ylabel("Speed Diff (m/s)", color=PALETTE["ai_vel"])

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Position Difference (metres)")
    ax.set_title("Phase 7 — NHC Correction Effect (Pipeline B vs C)",
                 fontweight="bold", pad=10)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 5] Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    input_path  = Path(args.input)
    params_path = Path(args.params)
    models_dir  = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IDR MVP -- Phase 7: AI-Assisted Dead Reckoning")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load Calibrated Dataset
    # ------------------------------------------------------------------
    print(f"\n[Step 1/7] Loading calibrated dataset: {input_path}")
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} samples ({len(df)/10.0:.1f} seconds at 10 Hz)")

    # Load calibration params
    gravity_norm = 9.81
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            calib_params = json.load(f)
        gravity_norm = calib_params.get("gravity_norm", 9.81)
        print(f"  Gravity: {gravity_norm:.4f} m/s²")

    # ------------------------------------------------------------------
    # Step 2: Generate AI Velocity Predictions
    # ------------------------------------------------------------------
    print(f"\n[Step 2/7] Generating AI velocity predictions...")
    weights_path = models_dir / "velocity_cnn_weights.npz"
    scaler_path = models_dir / "scaler_params.json"

    if not weights_path.exists() or not scaler_path.exists():
        print(f"ERROR: AI model not found at {models_dir}")
        print("  Run Phase 5 (run_phase5.py) first to train the velocity model.")
        sys.exit(1)

    ai_vel = generate_ai_velocity(df, weights_path, scaler_path, args.window_size)

    # ------------------------------------------------------------------
    # Step 3: Initialize all three pipelines
    # ------------------------------------------------------------------
    print("\n[Step 3/7] Initializing navigation pipelines...")

    start_idx = find_first_gnss_fix(df)
    init_lat = float(df.loc[start_idx, "gnss_lat"])
    init_lon = float(df.loc[start_idx, "gnss_lon"])
    init_hdg = get_initial_heading(df, start_idx)
    init_alt = float(df.loc[start_idx].get("ALTITUDE (m)", 0.0) or 0.0)
    init_ts = int(df.loc[start_idx, "timestamp"])

    gnss_spd = float(df.loc[start_idx].get("gnss_speed", 0.0) or 0.0)
    hdg_rad = np.radians(init_hdg)
    init_vn = gnss_spd * np.cos(hdg_rad)
    init_ve = gnss_spd * np.sin(hdg_rad)

    print(f"  Start index: {start_idx}")
    print(f"  Lat: {init_lat:.6f}°  Lon: {init_lon:.6f}°")
    print(f"  Heading: {init_hdg:.1f}°  Speed: {gnss_spd:.2f} m/s")

    df_run = df.loc[start_idx:].reset_index(drop=True)
    ai_vel_run = ai_vel[start_idx:]

    init_kwargs = dict(
        lat_deg=init_lat, lon_deg=init_lon, heading_deg=init_hdg,
        alt_m=init_alt, v_north=init_vn, v_east=init_ve,
        v_down=0.0, timestamp_ms=init_ts,
    )

    # ------------------------------------------------------------------
    # Step 4: Run Pipeline A — Raw INS
    # ------------------------------------------------------------------
    print(f"\n[Step 4/7] Running Pipeline A: Raw INS ({len(df_run):,} samples)...")
    raw_ins = StrapdownINS(gravity_magnitude=gravity_norm)
    raw_ins.initialize(**init_kwargs)
    traj_a = raw_ins.run_batch(df_run)
    print(f"  Pipeline A complete: {len(traj_a):,} trajectory points")

    # ------------------------------------------------------------------
    # Step 5: Run Pipeline B — AI + INS
    # ------------------------------------------------------------------
    print(f"\n[Step 5/7] Running Pipeline B: AI + INS ({len(df_run):,} samples)...")
    ai_ins = AIAssistedINS(enable_nhc=False)
    ai_ins.initialize(**init_kwargs)
    traj_b = ai_ins.run_batch_ai(df_run, ai_vel_run)
    print(f"  Pipeline B complete: {len(traj_b):,} trajectory points")

    # ------------------------------------------------------------------
    # Step 6: Run Pipeline C — AI + INS + NHC
    # ------------------------------------------------------------------
    print(f"\n[Step 6/7] Running Pipeline C: AI + INS + NHC ({len(df_run):,} samples)...")
    ai_nhc_ins = AIAssistedINS(enable_nhc=True)
    ai_nhc_ins.initialize(**init_kwargs)
    traj_c = ai_nhc_ins.run_batch_ai(df_run, ai_vel_run)
    print(f"  Pipeline C complete: {len(traj_c):,} trajectory points")

    # ------------------------------------------------------------------
    # Step 7: Compute Metrics & Generate Plots
    # ------------------------------------------------------------------
    print("\n[Step 7/7] Computing drift metrics & generating plots...")

    errors_a = compute_position_errors(traj_a, df_run)
    errors_b = compute_position_errors(traj_b, df_run)
    errors_c = compute_position_errors(traj_c, df_run)

    metrics_a = compute_drift_metrics(errors_a, df_run)
    metrics_b = compute_drift_metrics(errors_b, df_run)
    metrics_c = compute_drift_metrics(errors_c, df_run)

    # Print comparison table
    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║        Phase 7 — Three-Way Drift Comparison                ║")
    print("  ╠══════════════════════╦══════════╦══════════╦═══════════════╣")
    print("  ║ Metric               ║ A: Raw   ║ B: AI    ║ C: AI + NHC  ║")
    print("  ╠══════════════════════╬══════════╬══════════╬═══════════════╣")
    for key, label in [
        ("rmse_m",        "RMSE (m)"),
        ("mae_m",         "MAE (m)"),
        ("max_error_m",   "Max Error (m)"),
        ("final_error_m", "Final Error (m)"),
        ("drift_percent", "Drift (%)"),
    ]:
        va = metrics_a.get(key, np.nan)
        vb = metrics_b.get(key, np.nan)
        vc = metrics_c.get(key, np.nan)
        if va is None: va = np.nan
        if vb is None: vb = np.nan
        if vc is None: vc = np.nan
        print(f"  ║ {label:<20s} ║ {va:>8.1f} ║ {vb:>8.1f} ║ {vc:>13.1f} ║")
    print("  ╚══════════════════════╩══════════╩══════════╩═══════════════╝")

    # Save metrics
    all_metrics = {
        "pipeline_a_raw_ins": metrics_a,
        "pipeline_b_ai_ins": metrics_b,
        "pipeline_c_ai_ins_nhc": metrics_c,
    }
    metrics_path = results_dir / "drift_comparison.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        def _safe(v):
            if isinstance(v, float) and np.isnan(v):
                return None
            return v
        safe_metrics = {}
        for k, v in all_metrics.items():
            safe_metrics[k] = {mk: _safe(mv) for mk, mv in v.items()}
        json.dump(safe_metrics, f, indent=2)
    print(f"\n  Saved metrics JSON: {metrics_path}")

    # Save trajectory CSVs
    traj_a.to_csv(results_dir / "trajectory_a_raw_ins.csv", index=False)
    traj_b.to_csv(results_dir / "trajectory_b_ai_ins.csv", index=False)
    traj_c.to_csv(results_dir / "trajectory_c_ai_ins_nhc.csv", index=False)
    print("  Saved trajectory CSVs")

    # Generate plots
    print("\n  Generating diagnostic plots...")
    plot_01_trajectory_comparison(traj_a, traj_b, traj_c, df_run,
                                  results_dir / "01_trajectory_comparison.png")
    plot_02_position_error(errors_a, errors_b, errors_c,
                           results_dir / "02_position_error_comparison.png")
    plot_03_drift_comparison(metrics_a, metrics_b, metrics_c,
                             results_dir / "03_drift_comparison_bar.png")
    plot_04_velocity_comparison(df_run, ai_vel_run, traj_a,
                                results_dir / "04_velocity_comparison.png")
    plot_05_nhc_correction(traj_b, traj_c,
                           results_dir / "05_nhc_correction_effect.png")

    print("\n" + "=" * 70)
    print("PHASE 7 COMPLETE: AI-Assisted Dead Reckoning finished!")
    print(f"Results saved in: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
