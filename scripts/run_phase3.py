"""
IDR MVP -- Phase 3 Runner: Baseline Dead Reckoning
====================================================
Orchestrates Phase 3: Strapdown INS baseline dead-reckoning evaluation.

Pipeline:
1. Load calibrated dataset (Phase 2 output)
2. Load calibration parameters (gravity, initial heading)
3. Initialize StrapdownINS at first valid GNSS fix
4. Run full strapdown integration
5. Compute drift vs ground truth / GNSS reference
6. Save trajectory CSV and drift metrics JSON
7. Generate 5 diagnostic plots

Deliverable (per MVP.md §7):
    Ground Truth vs Raw INS Trajectory — the baseline for all later improvements.

Usage:
    python scripts/run_phase3.py
    python scripts/run_phase3.py --input data/processed/SYNC_s1_calibrated.csv
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ins import StrapdownINS
from src.ins.integration import haversine_distance

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT   = PROJECT_ROOT / "data" / "processed" / "SYNC_s1_calibrated.csv"
DEFAULT_PARAMS  = PROJECT_ROOT / "results" / "phase2" / "calibration_params.json"
OUTPUT_DIR      = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR     = PROJECT_ROOT / "results" / "phase3"

sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({
    "figure.figsize": (14, 7),
    "figure.dpi":     120,
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "savefig.bbox":   "tight",
    "savefig.pad_inches": 0.2,
})

PALETTE = {
    "gt":     "#2ECC71",   # Ground truth — green
    "ins":    "#E74C3C",   # Raw INS     — red
    "gnss":   "#3498DB",   # GNSS track  — blue
    "error":  "#F39C12",   # Error       — amber
    "drift":  "#9B59B6",   # Drift %     — purple
    "speed":  "#1ABC9C",   # Speed       — teal
    "heading":"#E67E22",   # Heading     — orange
}


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="IDR MVP -- Phase 3: Baseline Dead Reckoning"
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT),
        help=f"Path to Phase 2 calibrated CSV (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--params", type=str, default=str(DEFAULT_PARAMS),
        help=f"Path to calibration_params.json from Phase 2 (default: {DEFAULT_PARAMS})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Directory for trajectory CSV output"
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR),
        help="Directory for plots and metrics"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def find_first_gnss_fix(df: pd.DataFrame, min_speed_ms: float = 0.5) -> int:
    """
    Returns the index of the first row with valid GNSS and moving state.
    Falls back to the first row with non-null GNSS if no moving fix exists.

    Args:
        df:            Calibrated DataFrame.
        min_speed_ms:  Minimum GNSS speed (m/s) to consider 'moving'.

    Returns:
        DataFrame row index integer.
    """
    valid_gnss = df["gnss_lat"].notna() & df["gnss_lon"].notna()
    if not valid_gnss.any():
        return 0  # No GNSS at all — use row 0

    moving = valid_gnss & (df.get("gnss_speed", pd.Series(0.0, index=df.index)) >= min_speed_ms)
    if moving.any():
        return df[moving].index[0]

    return df[valid_gnss].index[0]


def get_initial_heading(df: pd.DataFrame, start_idx: int) -> float:
    """
    Returns the best available heading at the start index (degrees, 0=North, CW).

    Priority:
      1. ref_heading (ground truth from dataset)
      2. GNSS course (heading column)
      3. 0° (North) as fallback
    """
    row = df.loc[start_idx]

    if "ref_heading" in df.columns:
        h = row.get("ref_heading", np.nan)
        if pd.notna(h):
            return float(h)

    if "heading" in df.columns:
        h = row.get("heading", np.nan)
        if pd.notna(h):
            return float(h)

    return 0.0


# ---------------------------------------------------------------------------
# Drift / Error Metrics
# ---------------------------------------------------------------------------

def compute_drift_metrics(
    traj_df: pd.DataFrame,
    ref_df: pd.DataFrame,
) -> tuple:
    """
    Computes position error and drift metrics between INS trajectory and ground truth.

    Args:
        traj_df:  INS trajectory DataFrame with lat_deg, lon_deg columns.
        ref_df:   Reference DataFrame with ref_lat, ref_lon columns, aligned by index.

    Returns:
        Tuple of (metrics_dict, errors_array).
    """
    n = min(len(traj_df), len(ref_df))
    ins_lat = np.radians(traj_df["lat_deg"].to_numpy(dtype=float)[:n])
    ins_lon = np.radians(traj_df["lon_deg"].to_numpy(dtype=float)[:n])
    ref_lat = ref_df["ref_lat"].to_numpy(dtype=float)[:n]
    ref_lon = ref_df["ref_lon"].to_numpy(dtype=float)[:n]

    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)
    valid = ~(np.isnan(ref_lat_rad) | np.isnan(ref_lon_rad))

    errors_m = np.full(n, np.nan)
    if np.any(valid):
        errors_m[valid] = haversine_distance(
            ins_lat[valid], ins_lon[valid],
            ref_lat_rad[valid], ref_lon_rad[valid]
        )

    valid_ref_lat = ref_lat_rad[valid]
    valid_ref_lon = ref_lon_rad[valid]
    if len(valid_ref_lat) > 1:
        step_dists = haversine_distance(
            valid_ref_lat[:-1], valid_ref_lon[:-1],
            valid_ref_lat[1:], valid_ref_lon[1:]
        )
        total_dist = float(np.sum(step_dists))
    else:
        total_dist = 0.0

    errors_arr = errors_m[~np.isnan(errors_m)]

    if len(errors_arr) == 0:
        return {"error": "No valid reference positions found."}

    rmse          = float(np.sqrt(np.mean(errors_arr**2)))
    mae           = float(np.mean(errors_arr))
    max_error     = float(np.max(errors_arr))
    final_error   = float(errors_arr[-1])
    drift_pct     = (mae / total_dist * 100.0) if total_dist > 1.0 else np.nan

    return {
        "rmse_m":          rmse,
        "mae_m":           mae,
        "max_error_m":     max_error,
        "final_error_m":   final_error,
        "total_distance_m":total_dist,
        "drift_percent":   drift_pct,
        "n_samples":       len(errors_arr),
    }, errors_m


def compute_distance_series(ref_df: pd.DataFrame) -> np.ndarray:
    """Computes cumulative ground-truth distance array (metres) from ref_lat/ref_lon."""
    lats = ref_df["ref_lat"].to_numpy(dtype=float)
    lons = ref_df["ref_lon"].to_numpy(dtype=float)
    n = len(lats)
    if n == 0:
        return np.array([0.0])

    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    valid_step = (
        ~np.isnan(lat_rad[:-1]) & ~np.isnan(lon_rad[:-1]) &
        ~np.isnan(lat_rad[1:])  & ~np.isnan(lon_rad[1:])
    )

    step_dists = np.zeros(n - 1)
    if np.any(valid_step):
        step_dists[valid_step] = haversine_distance(
            lat_rad[:-1][valid_step], lon_rad[:-1][valid_step],
            lat_rad[1:][valid_step],  lon_rad[1:][valid_step]
        )

    cumulative = np.zeros(n)
    cumulative[1:] = np.cumsum(step_dists)
    return cumulative


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_trajectory(traj_df, ref_df, gnss_df, save_path: Path):
    """Plot 01: Ground Truth vs INS vs GNSS trajectory (lon/lat space)."""
    fig, ax = plt.subplots(figsize=(12, 9))

    # Ground Truth
    ref_valid = ref_df[ref_df["ref_lat"].notna() & ref_df["ref_lon"].notna()]
    ax.plot(
        ref_valid["ref_lon"], ref_valid["ref_lat"],
        color=PALETTE["gt"], linewidth=2.0, label="Ground Truth (ref)", zorder=3
    )

    # Raw INS
    ax.plot(
        traj_df["lon_deg"], traj_df["lat_deg"],
        color=PALETTE["ins"], linewidth=1.5, alpha=0.85,
        linestyle="--", label="Raw Strapdown INS", zorder=4
    )

    # GNSS track
    gnss_valid = gnss_df[gnss_df["gnss_lat"].notna() & gnss_df["gnss_lon"].notna()]
    ax.scatter(
        gnss_valid["gnss_lon"], gnss_valid["gnss_lat"],
        color=PALETTE["gnss"], s=8, alpha=0.5, label="GNSS Track", zorder=2
    )

    # Start / End markers
    ax.scatter(
        [traj_df["lon_deg"].iloc[0]], [traj_df["lat_deg"].iloc[0]],
        marker="o", color="#27AE60", s=120, zorder=5, label="INS Start"
    )
    ax.scatter(
        [traj_df["lon_deg"].iloc[-1]], [traj_df["lat_deg"].iloc[-1]],
        marker="X", color=PALETTE["ins"], s=120, zorder=5, label="INS End"
    )

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Phase 3 — Ground Truth vs Raw Strapdown INS Trajectory")
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [Plots] Saved: {save_path.name}")


def plot_position_error(errors_m, distance_arr, save_path: Path):
    """Plot 02: Position error over cumulative distance."""
    fig, ax = plt.subplots(figsize=(14, 6))

    valid = [(d, e) for d, e in zip(distance_arr, errors_m) if not np.isnan(e)]
    if not valid:
        plt.close(fig)
        return

    dists, errs = zip(*valid)
    ax.plot(dists, errs, color=PALETTE["error"], linewidth=1.8, label="Position Error (m)")
    ax.fill_between(dists, 0, errs, color=PALETTE["error"], alpha=0.2)
    ax.set_xlabel("Distance Travelled (m)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Phase 3 — Raw INS Position Error vs Distance Travelled")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [Plots] Saved: {save_path.name}")


def plot_velocity_comparison(traj_df, ref_df, save_path: Path):
    """Plot 03: INS speed vs reference speed."""
    fig, ax = plt.subplots(figsize=(14, 6))

    n = min(len(traj_df), len(ref_df))
    time_s = np.arange(n) * 0.1  # 10 Hz

    ins_speed = traj_df["speed_ms"].values[:n]
    ref_speed_valid = ref_df["ref_speed"].values[:n] if "ref_speed" in ref_df.columns else None

    ax.plot(time_s, ins_speed, color=PALETTE["ins"], linewidth=1.5, label="INS Speed (m/s)")

    if ref_speed_valid is not None:
        # ref_speed may be in km/h from dataset — check magnitude
        rs = ref_speed_valid.copy().astype(float)
        if np.nanmedian(rs) > 10.0:   # likely km/h
            rs = rs / 3.6
        ax.plot(time_s, rs, color=PALETTE["gt"], linewidth=1.5,
                linestyle="--", label="Reference Speed (m/s)", alpha=0.85)

    if "gnss_speed" in ref_df.columns:
        gs = ref_df["gnss_speed"].values[:n].astype(float)
        ax.plot(time_s, gs, color=PALETTE["gnss"], linewidth=1.0,
                alpha=0.6, label="GNSS Speed (m/s)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Phase 3 — INS Speed vs Reference Speed")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [Plots] Saved: {save_path.name}")


def plot_drift_percent(errors_m, distance_arr, save_path: Path):
    """Plot 04: Cumulative drift % = cumulative error / distance × 100."""
    fig, ax = plt.subplots(figsize=(14, 6))

    drift_pcts = []
    for d, e in zip(distance_arr, errors_m):
        if np.isnan(e) or d < 1.0:
            drift_pcts.append(np.nan)
        else:
            drift_pcts.append(e / d * 100.0)

    valid = [(d, dp) for d, dp in zip(distance_arr, drift_pcts)
             if not np.isnan(dp) and d > 10.0]
    if not valid:
        plt.close(fig)
        return

    dists, drifts = zip(*valid)
    ax.plot(dists, drifts, color=PALETTE["drift"], linewidth=1.8, label="Drift %")
    ax.axhline(10.0, color="#E74C3C", linestyle="--", linewidth=1.5,
               label="10% Target Threshold")
    ax.fill_between(dists, 0, drifts, color=PALETTE["drift"], alpha=0.15)
    ax.set_xlabel("Distance Travelled (m)")
    ax.set_ylabel("Positional Drift (%)")
    ax.set_title("Phase 3 — Raw INS Drift % vs Distance (Target: <10%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [Plots] Saved: {save_path.name}")


def plot_heading_comparison(traj_df, ref_df, save_path: Path):
    """Plot 05: INS heading vs reference heading over time."""
    fig, ax = plt.subplots(figsize=(14, 6))

    n = min(len(traj_df), len(ref_df))
    time_s = np.arange(n) * 0.1

    ax.plot(time_s, traj_df["heading_deg"].values[:n],
            color=PALETTE["ins"], linewidth=1.5, label="INS Heading (°)")

    if "ref_heading" in ref_df.columns and not ref_df["ref_heading"].isna().all():
        ax.plot(time_s, ref_df["ref_heading"].values[:n],
                color=PALETTE["gt"], linewidth=1.5, linestyle="--",
                label="Reference Heading (°)", alpha=0.85)

    if "heading" in ref_df.columns and not ref_df["heading"].isna().all():
        ax.plot(time_s, ref_df["heading"].values[:n],
                color=PALETTE["gnss"], linewidth=1.0, alpha=0.6,
                label="GNSS Course (°)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading (°)")
    ax.set_title("Phase 3 — INS Heading vs Reference Heading")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [Plots] Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    input_path  = Path(args.input)
    params_path = Path(args.params)
    output_dir  = Path(args.output_dir)
    results_dir = Path(args.results_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  IDR MVP -- Phase 3: Baseline Strapdown Dead Reckoning")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load Calibrated Dataset
    # ------------------------------------------------------------------
    if not input_path.exists():
        processed_files = list(output_dir.glob("*_calibrated.csv"))
        if processed_files:
            input_path = processed_files[0]
            print(f"  [Runner] Fallback to: {input_path}")
        else:
            raise FileNotFoundError(
                f"No calibrated CSV found at {input_path}. "
                "Please run Phase 2 (run_phase2.py) first."
            )

    print(f"  [Runner] Loading calibrated dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  [Runner] Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ------------------------------------------------------------------
    # 2. Load Calibration Parameters (gravity from Phase 2)
    # ------------------------------------------------------------------
    gravity_norm = 9.81
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            calib_params = json.load(f)
        gravity_norm = calib_params.get("gravity_norm", 9.81)
        print(f"  [Runner] Loaded calibration params: gravity = {gravity_norm:.4f} m/s²")
    else:
        print(f"  [Runner] WARNING: calibration_params.json not found at {params_path}. "
              "Using g = 9.81 m/s²")

    # ------------------------------------------------------------------
    # 3. Find Initial GNSS Fix & Heading
    # ------------------------------------------------------------------
    start_idx = find_first_gnss_fix(df)
    init_lat  = df.loc[start_idx, "gnss_lat"]
    init_lon  = df.loc[start_idx, "gnss_lon"]
    init_hdg  = get_initial_heading(df, start_idx)
    init_alt  = float(df.loc[start_idx].get("ALTITUDE (m)", 0.0) or 0.0)
    init_ts   = int(df.loc[start_idx, "timestamp"])

    # Initial velocity from GNSS speed + heading
    gnss_spd  = float(df.loc[start_idx].get("gnss_speed", 0.0) or 0.0)
    hdg_rad   = np.radians(init_hdg)
    init_vn   = gnss_spd * np.cos(hdg_rad)
    init_ve   = gnss_spd * np.sin(hdg_rad)

    print(f"\n  [Init] Start index: {start_idx}")
    print(f"  [Init] Lat: {init_lat:.6f}°  Lon: {init_lon:.6f}°")
    print(f"  [Init] Heading: {init_hdg:.1f}°  |  Speed: {gnss_spd:.2f} m/s")

    # ------------------------------------------------------------------
    # 4. Run Strapdown INS
    # ------------------------------------------------------------------
    ins = StrapdownINS(gravity_magnitude=gravity_norm)
    ins.initialize(
        lat_deg=float(init_lat),
        lon_deg=float(init_lon),
        heading_deg=float(init_hdg),
        alt_m=init_alt,
        v_north=init_vn,
        v_east=init_ve,
        v_down=0.0,
        timestamp_ms=init_ts,
    )

    # Run on the subset starting from start_idx
    df_run = df.loc[start_idx:].reset_index(drop=True)

    print(f"\n  [INS] Running strapdown integration on {len(df_run):,} samples ...")
    traj_df = ins.run_batch(df_run)
    print(f"  [INS] Integration complete. Trajectory rows: {len(traj_df):,}")

    # ------------------------------------------------------------------
    # 5. Compute Drift Metrics
    # ------------------------------------------------------------------
    ref_available = "ref_lat" in df_run.columns and not df_run["ref_lat"].isna().all()
    gnss_available = "gnss_lat" in df_run.columns and not df_run["gnss_lat"].isna().all()

    if ref_available:
        print("\n  [Metrics] Computing drift vs ground truth (ref_lat / ref_lon) ...")
        metrics_result = compute_drift_metrics(traj_df, df_run)
        if isinstance(metrics_result, tuple):
            metrics, errors_m = metrics_result
        else:
            metrics = metrics_result
            errors_m = [np.nan] * len(traj_df)
    elif gnss_available:
        print("\n  [Metrics] No ref_lat found. Computing drift vs GNSS track ...")
        # Use GNSS as pseudo-reference
        gnss_ref = df_run.rename(columns={"gnss_lat": "ref_lat", "gnss_lon": "ref_lon"})
        metrics_result = compute_drift_metrics(traj_df, gnss_ref)
        if isinstance(metrics_result, tuple):
            metrics, errors_m = metrics_result
        else:
            metrics = metrics_result
            errors_m = [np.nan] * len(traj_df)
    else:
        print("\n  [Metrics] WARNING: No reference positions available for drift evaluation.")
        metrics = {}
        errors_m = [np.nan] * len(traj_df)

    if metrics and "error" not in metrics:
        print(f"\n  --- Phase 3 Drift Metrics ---")
        print(f"  RMSE:              {metrics.get('rmse_m', float('nan')):.2f} m")
        print(f"  MAE:               {metrics.get('mae_m', float('nan')):.2f} m")
        print(f"  Max Error:         {metrics.get('max_error_m', float('nan')):.2f} m")
        print(f"  Final Error:       {metrics.get('final_error_m', float('nan')):.2f} m")
        print(f"  Total Distance:    {metrics.get('total_distance_m', float('nan')):.1f} m")
        print(f"  Drift %:           {metrics.get('drift_percent', float('nan')):.2f} %")

    # ------------------------------------------------------------------
    # 6. Save Outputs
    # ------------------------------------------------------------------
    traj_csv_path = output_dir / "ins_trajectory_phase3.csv"
    traj_df.to_csv(traj_csv_path, index=False)
    print(f"\n  [Saver] Saved trajectory: {traj_csv_path}")

    metrics_json_path = results_dir / "drift_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        # Convert any NaN to None for JSON
        def _safe(v):
            if isinstance(v, float) and np.isnan(v):
                return None
            return v
        json.dump({k: _safe(v) for k, v in metrics.items()}, f, indent=2)
    print(f"  [Saver] Saved drift metrics: {metrics_json_path}")

    # ------------------------------------------------------------------
    # 7. Generate Plots
    # ------------------------------------------------------------------
    print("\n  --- Generating Phase 3 Diagnostic Plots ---")

    distance_arr = compute_distance_series(
        df_run.rename(columns={"gnss_lat": "ref_lat", "gnss_lon": "ref_lon"})
        if not ref_available else df_run
    )

    # Use GNSS as fallback for ref on some plots
    ref_for_plots = df_run if ref_available else df_run.rename(
        columns={"gnss_lat": "ref_lat", "gnss_lon": "ref_lon",
                 "gnss_speed": "ref_speed"}
    )

    plot_trajectory(
        traj_df, ref_for_plots, df_run,
        results_dir / "01_trajectory_gt_vs_ins.png"
    )
    plot_position_error(
        errors_m, distance_arr,
        results_dir / "02_position_error.png"
    )
    plot_velocity_comparison(
        traj_df, df_run,
        results_dir / "03_velocity_comparison.png"
    )
    plot_drift_percent(
        errors_m, distance_arr,
        results_dir / "04_drift_percent.png"
    )
    plot_heading_comparison(
        traj_df, df_run,
        results_dir / "05_heading_comparison.png"
    )

    print("\n" + "=" * 70)
    print("  Phase 3 Baseline Dead Reckoning Completed Successfully!")
    print("  Outputs:")
    print(f"    Trajectory CSV : {traj_csv_path}")
    print(f"    Drift Metrics  : {metrics_json_path}")
    print(f"    Plots          : {results_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
