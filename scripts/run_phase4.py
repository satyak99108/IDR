"""
IDR MVP -- Phase 4 Runner: GNSS Outage Simulator & Benchmark
=============================================================
Orchestrates Phase 4: GNSS blackout simulation and dead-reckoning evaluation.

Pipeline:
1. Load calibrated dataset (Phase 2 output) and calibration parameters.
2. Configure 4 benchmark outage scenarios (MVP 30s tunnel, 10s underpass, 60s mountain tunnel, multi-tunnel corridor).
3. Apply blackout masks to sensor and GNSS data streams.
4. Execute dead reckoning through simulated blackouts (GNSS available before/after, pure DR inside).
5. Compute drift % and error metrics using OutageEvaluator.
6. Export masked dataset to data/processed/SYNC_s1_outage_simulated.csv.
7. Save evaluation JSON and summary CSV to results/phase4/.
8. Generate 5 presentation-quality diagnostic plots.

Deliverable (per MVP.md §8 & §16):
    Artificial GNSS blackout windows simulating tunnel entry/exit,
    measuring position drift % and baseline dead-reckoning degradation.

Usage:
    python scripts/run_phase4.py
    python scripts/run_phase4.py --input data/processed/SYNC_s1_calibrated.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ins import StrapdownINS
from src.ins.integration import haversine_distance, haversine_distance_deg
from src.simulation import (
    GNSSOutageSimulator,
    OutageScenario,
    OutageWindow,
    OutageEvaluator,
    ScenarioMetrics,
)

# ---------------------------------------------------------------------------
# Configuration & Theme
# ---------------------------------------------------------------------------

DEFAULT_INPUT   = PROJECT_ROOT / "data" / "processed" / "SYNC_s1_calibrated.csv"
DEFAULT_PARAMS  = PROJECT_ROOT / "results" / "phase2" / "calibration_params.json"
OUTPUT_DIR      = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR     = PROJECT_ROOT / "results" / "phase4"

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
    "gt":         "#2ECC71",   # Ground truth — emerald green
    "gnss":       "#3498DB",   # GNSS active  — vivid blue
    "dr":         "#E74C3C",   # Dead reckoning inside outage — red
    "outage_bg":  "#FADBD8",   # Shaded outage background — soft light red
    "tunnel_box": "#E74C3C",   # Tunnel indicator
    "error":      "#E67E22",   # Error line — orange
    "target":     "#9B59B6",   # Target threshold line — purple
    "bar_10s":    "#3498DB",
    "bar_30s":    "#F39C12",
    "bar_60s":    "#E74C3C",
    "bar_multi":  "#8E44AD",
}


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="IDR MVP -- Phase 4: GNSS Outage Simulator & Benchmark"
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
        help=f"Directory for simulated outage CSV (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR),
        help=f"Directory for plots and metrics (default: {RESULTS_DIR})"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Simulation Execution Helpers
# ---------------------------------------------------------------------------

def run_dead_reckoning_outage_simulation(
    df_masked: pd.DataFrame,
    start_idx: int = 0,
) -> pd.DataFrame:
    """
    Simulates vehicle positioning with GNSS available outside outages and
    baseline dead-reckoning inside outages.

    - Outside Outage: Position is locked to GNSS / Ground Truth reference.
    - Inside Outage: INS integrates IMU accelerations and gyroscope angular rates
      forward from the entrance state, drifting freely until GNSS recovery.
    """
    df_sim = df_masked.copy()
    n_samples = len(df_sim)

    time_col = "timestamp" if "timestamp" in df_sim.columns else "timestamp_ms"
    timestamps = df_sim[time_col].values

    pred_lat = np.full(n_samples, np.nan)
    pred_lon = np.full(n_samples, np.nan)
    pred_speed = np.full(n_samples, np.nan)
    pred_heading = np.full(n_samples, np.nan)
    nav_mode = []

    ins = StrapdownINS()
    in_outage_prev = False
    prev_ts = timestamps[0]

    # Pre-extract numpy arrays for 100x faster execution
    is_outage_arr = df_sim["is_outage"].values.astype(bool)
    gnss_avail_arr = df_sim["gnss_available"].values.astype(int)

    ref_lats = df_sim["ref_lat"].values if "ref_lat" in df_sim.columns else df_sim["gnss_lat"].values
    ref_lons = df_sim["ref_lon"].values if "ref_lon" in df_sim.columns else df_sim["gnss_lon"].values
    raw_ref_spds = df_sim["ref_speed"].values if "ref_speed" in df_sim.columns else (df_sim["gnss_speed"].values if "gnss_speed" in df_sim.columns else np.zeros(n_samples))
    ref_hdgs = df_sim["ref_heading"].values if "ref_heading" in df_sim.columns else (df_sim["heading"].values if "heading" in df_sim.columns else np.zeros(n_samples))

    ax_lins = df_sim["ax_veh_lin"].values if "ax_veh_lin" in df_sim.columns else df_sim["ax_veh"].values
    ay_lins = df_sim["ay_veh_lin"].values if "ay_veh_lin" in df_sim.columns else df_sim["ay_veh"].values
    az_lins = df_sim["az_veh_lin"].values if "az_veh_lin" in df_sim.columns else df_sim["az_veh"].values
    gx_vehs = df_sim["gx_veh"].values if "gx_veh" in df_sim.columns else df_sim["gx"].values
    gy_vehs = df_sim["gy_veh"].values if "gy_veh" in df_sim.columns else df_sim["gy"].values
    gz_vehs = df_sim["gz_veh"].values if "gz_veh" in df_sim.columns else df_sim["gz"].values

    for i in range(n_samples):
        ts = int(timestamps[i])
        dt = (ts - prev_ts) / 1000.0 if i > 0 else 0.1
        if dt <= 0.0 or dt > 5.0:
            dt = 0.1

        is_outage = is_outage_arr[i]
        gnss_avail = gnss_avail_arr[i]

        ref_lat = float(ref_lats[i])
        ref_lon = float(ref_lons[i])
        raw_ref_spd = float(raw_ref_spds[i])
        ref_spd_ms = float(raw_ref_spd / 3.6 if raw_ref_spd > 15.0 else raw_ref_spd)
        ref_hdg = float(ref_hdgs[i])

        ax_lin = float(ax_lins[i])
        ay_lin = float(ay_lins[i])
        az_lin = float(az_lins[i])
        gx_veh = float(gx_vehs[i])
        gy_veh = float(gy_vehs[i])
        gz_veh = float(gz_vehs[i])

        if not is_outage and gnss_avail == 1:
            pred_lat[i] = ref_lat
            pred_lon[i] = ref_lon
            pred_speed[i] = ref_spd_ms
            pred_heading[i] = ref_hdg
            nav_mode.append("GNSS_FIX")

            v_north = float(ref_spd_ms * np.cos(np.radians(ref_hdg)))
            v_east  = float(ref_spd_ms * np.sin(np.radians(ref_hdg)))

            ins.initialize(
                lat_deg=ref_lat,
                lon_deg=ref_lon,
                heading_deg=ref_hdg,
                alt_m=0.0,
                v_north=v_north,
                v_east=v_east,
                v_down=0.0,
                timestamp_ms=ts,
            )
            in_outage_prev = False

        else:
            if not in_outage_prev:
                prev_idx = max(0, i - 1)
                p_lat = pred_lat[prev_idx] if not np.isnan(pred_lat[prev_idx]) else ref_lat
                p_lon = pred_lon[prev_idx] if not np.isnan(pred_lon[prev_idx]) else ref_lon
                p_spd = pred_speed[prev_idx] if not np.isnan(pred_speed[prev_idx]) else ref_spd_ms
                p_hdg = pred_heading[prev_idx] if not np.isnan(pred_heading[prev_idx]) else ref_hdg

                v_north = float(p_spd * np.cos(np.radians(p_hdg)))
                v_east  = float(p_spd * np.sin(np.radians(p_hdg)))

                ins.initialize(
                    lat_deg=float(p_lat),
                    lon_deg=float(p_lon),
                    heading_deg=float(p_hdg),
                    alt_m=0.0,
                    v_north=v_north,
                    v_east=v_east,
                    v_down=0.0,
                    timestamp_ms=int(timestamps[prev_idx]),
                )

            state = ins.update(
                dt=dt,
                ax_lin=ax_lin,
                ay_lin=ay_lin,
                az_lin=az_lin,
                gx_veh=gx_veh,
                gy_veh=gy_veh,
                gz_veh=gz_veh,
                timestamp_ms=ts,
            )

            pred_lat[i] = state.lat_deg
            pred_lon[i] = state.lon_deg
            pred_speed[i] = state.speed_ms
            pred_heading[i] = np.degrees(state.heading_rad) % 360.0
            nav_mode.append("DEAD_RECKONING")
            in_outage_prev = True

        prev_ts = ts

    df_sim["pred_lat"] = pred_lat
    df_sim["pred_lon"] = pred_lon
    df_sim["pred_speed"] = pred_speed
    df_sim["pred_heading"] = pred_heading
    df_sim["nav_mode"] = nav_mode

    # Vectorized step-by-step position error
    pos_err = haversine_distance_deg(pred_lat, pred_lon, ref_lats, ref_lons)
    pos_err = np.nan_to_num(pos_err, nan=0.0)
    df_sim["position_error_m"] = pos_err

    return df_sim


# ---------------------------------------------------------------------------
# Plotting & Visualization Suite
# ---------------------------------------------------------------------------

def plot_01_tunnel_outage_trajectory(
    df_sim: pd.DataFrame,
    scenario: OutageScenario,
    output_path: Path,
):
    """Plot 1: 2D Trajectory with shaded tunnel outage regions."""
    fig, ax = plt.subplots(figsize=(14, 8))

    ref_lat = df_sim["ref_lat"].dropna().values
    ref_lon = df_sim["ref_lon"].dropna().values

    # Plot full Ground Truth
    ax.plot(
        ref_lon, ref_lat,
        color=PALETTE["gt"], linewidth=2.5, linestyle="-",
        label="Ground Truth Reference", zorder=3,
    )

    # Plot predicted trajectory
    pred_mask_gnss = df_sim["nav_mode"] == "GNSS_FIX"
    pred_mask_dr = df_sim["nav_mode"] == "DEAD_RECKONING"

    # GNSS Active Segments
    ax.scatter(
        df_sim.loc[pred_mask_gnss, "pred_lon"],
        df_sim.loc[pred_mask_gnss, "pred_lat"],
        color=PALETTE["gnss"], s=10, alpha=0.6,
        label="GNSS Active (Fixed)", zorder=4,
    )

    # Dead Reckoning Segments (Outages)
    for window in scenario.windows:
        w_df = df_sim[df_sim["outage_id"] == window.outage_id]
        if len(w_df) > 0:
            ax.plot(
                w_df["pred_lon"], w_df["pred_lat"],
                color=PALETTE["dr"], linewidth=3.5, linestyle="--",
                label=f"Dead Reckoning ({window.label})" if window.outage_id == 1 else "",
                zorder=5,
            )
            # Mark tunnel start and end
            ax.scatter(
                w_df["pred_lon"].iloc[0], w_df["pred_lat"].iloc[0],
                color="#E74C3C", marker="X", s=140, edgecolors="black",
                zorder=6, label="Tunnel Entrance" if window.outage_id == 1 else "",
            )
            ax.scatter(
                w_df["pred_lon"].iloc[-1], w_df["pred_lat"].iloc[-1],
                color="#8E44AD", marker="o", s=140, edgecolors="black",
                zorder=6, label="Tunnel Exit (Drift Endpoint)" if window.outage_id == 1 else "",
            )

    # Mark Start & End of overall trip
    ax.scatter(ref_lon[0], ref_lat[0], color="#27AE60", marker="s", s=120, edgecolors="black", zorder=7, label="Trip Start")
    ax.scatter(ref_lon[-1], ref_lat[-1], color="#C0392B", marker="*", s=160, edgecolors="black", zorder=7, label="Trip End")

    ax.set_title(f"Phase 4 — Trajectory in Simulated GNSS Outage ({scenario.description})", pad=12, fontweight="bold")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 1] Saved: {output_path.name}")


def plot_02_position_error_timeline(
    df_sim: pd.DataFrame,
    scenario: OutageScenario,
    output_path: Path,
):
    """Plot 2: Position error timeline with shaded blackout window regions."""
    fig, ax = plt.subplots(figsize=(14, 7))

    time_s = df_sim["time_elapsed_s"].values
    pos_err = df_sim["position_error_m"].values

    # Plot error line
    ax.plot(time_s, pos_err, color=PALETTE["error"], linewidth=2.2, label="Position Error (m)", zorder=4)

    # Highlight blackout spans
    for i, w in enumerate(scenario.windows):
        ax.axvspan(
            w.start_time_s, w.end_time_s,
            color=PALETTE["outage_bg"], alpha=0.65, zorder=2,
            label="GNSS Blackout (Tunnel)" if i == 0 else "",
        )
        # Add annotation text
        mid_t = 0.5 * (w.start_time_s + w.end_time_s)
        w_df = df_sim[df_sim["outage_id"] == w.outage_id]
        if len(w_df) > 0:
            peak_err = w_df["position_error_m"].max()
            ax.text(
                mid_t, peak_err * 0.85 + 2.0,
                f"{w.label}\nDuration: {w.duration_s:.1f}s\nPeak Err: {peak_err:.1f}m",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#E74C3C"),
                zorder=5,
            )

    ax.set_title(f"Phase 4 — Position Error Growth During GNSS Outage Timeline ({scenario.name})", pad=12, fontweight="bold")
    ax.set_xlabel("Time Elapsed (seconds)")
    ax.set_ylabel("Position Error (metres)")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 2] Saved: {output_path.name}")


def plot_03_drift_percentage_by_scenario(
    scenario_metrics_dict: Dict[str, ScenarioMetrics],
    output_path: Path,
):
    """Plot 3: Comparative bar chart of Drift % across scenarios vs MVP <10% Target."""
    fig, ax = plt.subplots(figsize=(12, 7))

    names = []
    drifts = []
    colors = [PALETTE["bar_10s"], PALETTE["bar_30s"], PALETTE["bar_60s"], PALETTE["bar_multi"]]

    for s_name, m in scenario_metrics_dict.items():
        clean_name = m.description.split("(")[0].strip()
        names.append(f"{clean_name}\n({m.total_outage_time_s:.0f}s outage)")
        drifts.append(m.weighted_drift_percent)

    bars = ax.bar(names, drifts, color=colors[:len(names)], width=0.55, edgecolor="black", linewidth=1.2, zorder=3)

    # Add Target 10% line from MVP.md §16
    ax.axhline(
        10.0, color=PALETTE["target"], linestyle="--", linewidth=2.2,
        label="MVP Target Threshold (Drift < 10%)", zorder=4,
    )

    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, h + 1.5,
            f"{h:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title("Phase 4 — Baseline Raw INS Drift % across GNSS Outage Scenarios", pad=12, fontweight="bold")
    ax.set_ylabel("Drift % = (Error at Outage Exit / Distance Travelled) × 100%")
    ax.set_ylim(0, max(drifts) * 1.25 + 5.0)
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 3] Saved: {output_path.name}")


def plot_04_outage_error_growth_analysis(
    df_sim_dict: Dict[str, Tuple[pd.DataFrame, OutageScenario]],
    output_path: Path,
):
    """Plot 4: Error growth curve comparison (E(t) vs elapsed time inside blackout)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    styles = {
        "short_underpass_10s": ("#3498DB", "-", "10s Underpass"),
        "mvp_standard_30s_tunnel": ("#F39C12", "-", "30s Standard Tunnel"),
        "long_tunnel_60s": ("#E74C3C", "-", "60s Mountain Tunnel"),
    }

    for key, (style_color, line_style, lbl) in styles.items():
        if key in df_sim_dict:
            df_s, sc = df_sim_dict[key]
            # Extract first outage window
            w = sc.windows[0]
            slice_df = df_s[df_s["outage_id"] == w.outage_id].copy()
            if len(slice_df) > 0:
                t_in_outage = slice_df["time_elapsed_s"].values - slice_df["time_elapsed_s"].iloc[0]
                e_in_outage = slice_df["position_error_m"].values
                ax.plot(
                    t_in_outage, e_in_outage,
                    color=style_color, linestyle=line_style, linewidth=2.5,
                    label=f"{lbl} ({w.duration_s:.0f}s)", zorder=3,
                )

    ax.set_title("Phase 4 — Inertial Error Divergence vs Elapsed Time in GNSS Blackout", pad=12, fontweight="bold")
    ax.set_xlabel("Elapsed Time Inside Tunnel (seconds)")
    ax.set_ylabel("Accumulated Position Error (metres)")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", framealpha=0.9)

    # Explanatory annotation
    ax.text(
        0.03, 0.65,
        "Note quadratic error growth (E ~ t^2):\nUnassisted double-integration of\naccelerometer bias causes rapid drift,\nmotivating Phase 5 AI velocity estimation.",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#7F8C8D"),
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 4] Saved: {output_path.name}")


def plot_05_multi_scenario_dashboard(
    df_sim_standard: pd.DataFrame,
    standard_scenario: OutageScenario,
    scenario_metrics_dict: Dict[str, ScenarioMetrics],
    output_path: Path,
):
    """Plot 5: Multi-panel Phase 4 summary dashboard for presentations."""
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))

    # Panel 1: Trajectory with 30s tunnel
    ax1 = axs[0, 0]
    ax1.plot(df_sim_standard["ref_lon"], df_sim_standard["ref_lat"], color=PALETTE["gt"], linewidth=2.0, label="Ground Truth")
    w = standard_scenario.windows[0]
    w_df = df_sim_standard[df_sim_standard["outage_id"] == w.outage_id]
    if len(w_df) > 0:
        ax1.plot(w_df["pred_lon"], w_df["pred_lat"], color=PALETTE["dr"], linewidth=2.8, linestyle="--", label="INS Dead Reckoning (Outage)")
        ax1.scatter(w_df["pred_lon"].iloc[0], w_df["pred_lat"].iloc[0], color="#E74C3C", marker="X", s=90, zorder=5)
        ax1.scatter(w_df["pred_lon"].iloc[-1], w_df["pred_lat"].iloc[-1], color="#8E44AD", marker="o", s=90, zorder=5)
    ax1.set_title("A. 30s Simulated Tunnel Trajectory", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Longitude (deg)")
    ax1.set_ylabel("Latitude (deg)")
    ax1.ticklabel_format(useOffset=False, style="plain")
    ax1.legend(loc="best", fontsize=9)

    # Panel 2: Error timeline
    ax2 = axs[0, 1]
    ax2.plot(df_sim_standard["time_elapsed_s"], df_sim_standard["position_error_m"], color=PALETTE["error"], linewidth=2.0, label="Position Error")
    ax2.axvspan(w.start_time_s, w.end_time_s, color=PALETTE["outage_bg"], alpha=0.7, label="30s Tunnel Blackout")
    ax2.set_title("B. Position Error Timeline (30s Outage)", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (m)")
    ax2.legend(loc="upper left", fontsize=9)

    # Panel 3: Drift % comparison
    ax3 = axs[1, 0]
    s_names = [m.description.split("(")[0].strip() for m in scenario_metrics_dict.values()]
    drifts = [m.weighted_drift_percent for m in scenario_metrics_dict.values()]
    bars = ax3.bar(s_names, drifts, color=["#3498DB", "#F39C12", "#E74C3C", "#8E44AD"], width=0.55, edgecolor="black")
    ax3.axhline(10.0, color=PALETTE["target"], linestyle="--", linewidth=1.8, label="MVP Target (<10%)")
    for b in bars:
        ax3.text(b.get_x() + b.get_width()/2.0, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax3.set_title("C. Drift % Across Blackout Durations", fontweight="bold", fontsize=11)
    ax3.set_ylabel("Drift %")
    ax3.tick_params(axis="x", rotation=15)
    ax3.legend(loc="upper left", fontsize=9)

    # Panel 4: Metrics summary table
    ax4 = axs[1, 1]
    ax4.axis("off")

    table_data = [
        ["Scenario", "Duration", "Distance", "Endpoint Error", "Drift %"],
    ]
    for s_name, m in scenario_metrics_dict.items():
        table_data.append([
            m.description.split("(")[0].strip(),
            f"{m.total_outage_time_s:.0f} s",
            f"{m.total_outage_distance_m:.1f} m",
            f"{m.overall_max_error_m:.1f} m",
            f"{m.weighted_drift_percent:.1f} %",
        ])

    table = ax4.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.32, 0.16, 0.16, 0.18, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.8)

    # Style header
    for k in range(5):
        cell = table[(0, k)]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    ax4.set_title("D. Phase 4 Benchmark Summary", fontweight="bold", fontsize=11, pad=20)

    fig.suptitle("IDR MVP — Phase 4: GNSS Outage Simulator Benchmark Report", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 5] Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    input_path   = Path(args.input)
    params_path  = Path(args.params)
    output_dir   = Path(args.output_dir)
    results_dir  = Path(args.results_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IDR MVP -- Phase 4: GNSS Outage Simulator & Benchmark")
    print("=" * 70)

    # 1. Load calibrated dataset
    print(f"\n[Step 1/5] Loading calibrated dataset: {input_path}")
    if not input_path.exists():
        print(f"ERROR: Calibrated file not found at {input_path}")
        sys.exit(1)
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} samples ({len(df)/10.0:.1f} seconds at 10 Hz)")

    # 2. Define Benchmark Scenarios
    print("\n[Step 2/5] Initializing Outage Scenarios...")
    simulator = GNSSOutageSimulator()

    scenarios = {
        "mvp_standard_30s_tunnel": simulator.create_standard_mvp_scenario(start_sec=30.0, duration_sec=30.0),
        "short_underpass_10s":     simulator.create_short_underpass_scenario(start_sec=40.0, duration_sec=10.0),
        "long_tunnel_60s":         simulator.create_long_tunnel_scenario(start_sec=30.0, duration_sec=60.0),
        "multi_tunnel_corridor":   simulator.create_multi_outage_scenario(),
    }

    for name, sc in scenarios.items():
        print(f"  - {sc.name}: {sc.description} ({sc.num_outages} outage(s), total {sc.total_outage_duration_s:.0f}s)")

    # 3. Simulate and evaluate each scenario
    print("\n[Step 3/5] Simulating GNSS Blackout & Running Dead Reckoning...")
    simulated_dfs = {}
    scenario_metrics_dict = {}
    summary_rows = []

    for name, scenario in scenarios.items():
        print(f"\n  Evaluating {name}...")
        df_masked, resolved_sc = simulator.apply_outage_mask(df, scenario)
        df_sim = run_dead_reckoning_outage_simulation(df_masked)

        metrics = OutageEvaluator.evaluate_scenario(
            df_sim,
            scenario_name=resolved_sc.name,
            description=resolved_sc.description,
            pred_lat_col="pred_lat",
            pred_lon_col="pred_lon",
            ref_lat_col="ref_lat",
            ref_lon_col="ref_lon",
            pred_speed_col="pred_speed",
            ref_speed_col="ref_speed",
            pred_heading_col="pred_heading",
            ref_heading_col="ref_heading",
        )

        simulated_dfs[name] = (df_sim, resolved_sc)
        scenario_metrics_dict[name] = metrics

        print(f"    Outages: {metrics.num_outages} | Outage Time: {metrics.total_outage_time_s:.1f}s")
        print(f"    Distance Travelled in Outage: {metrics.total_outage_distance_m:.1f} m")
        print(f"    Max Position Error: {metrics.overall_max_error_m:.2f} m")
        print(f"    Weighted Drift %: {metrics.weighted_drift_percent:.2f}%")

        summary_rows.append({
            "scenario_name": metrics.scenario_name,
            "description": metrics.description,
            "num_outages": metrics.num_outages,
            "total_outage_time_s": metrics.total_outage_time_s,
            "total_distance_m": metrics.total_outage_distance_m,
            "max_error_m": metrics.overall_max_error_m,
            "rmse_m": metrics.overall_rmse_m,
            "weighted_drift_percent": metrics.weighted_drift_percent,
        })

    # 4. Save Datasets and Evaluation JSON
    print("\n[Step 4/5] Exporting Simulated Datasets & Evaluation Reports...")

    # Save standard MVP masked dataset
    standard_df, standard_sc = simulated_dfs["mvp_standard_30s_tunnel"]
    out_csv_path = output_dir / "SYNC_s1_outage_simulated.csv"
    standard_df.to_csv(out_csv_path, index=False)
    print(f"  Saved masked dataset: {out_csv_path}")

    # Save evaluation metrics JSON
    metrics_json_path = results_dir / "outage_evaluation.json"
    full_eval_dict = {
        "benchmark_summary": summary_rows,
        "scenarios": {k: v.to_dict() for k, v in scenario_metrics_dict.items()},
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(full_eval_dict, f, indent=2)
    print(f"  Saved metrics JSON: {metrics_json_path}")

    # Save summary CSV
    summary_csv_path = results_dir / "outage_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    print(f"  Saved summary CSV: {summary_csv_path}")

    # 5. Generate Diagnostic Plots
    print("\n[Step 5/5] Generating Diagnostic Visualizations...")
    plot_01_tunnel_outage_trajectory(standard_df, standard_sc, results_dir / "01_tunnel_outage_trajectory.png")
    plot_02_position_error_timeline(standard_df, standard_sc, results_dir / "02_position_error_blackout_timeline.png")
    plot_03_drift_percentage_by_scenario(scenario_metrics_dict, results_dir / "03_drift_percentage_by_scenario.png")
    plot_04_outage_error_growth_analysis(simulated_dfs, results_dir / "04_outage_error_growth_analysis.png")
    plot_05_multi_scenario_dashboard(standard_df, standard_sc, scenario_metrics_dict, results_dir / "05_multi_scenario_dashboard.png")

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE: GNSS Outage Simulator successfully benchmarked!")
    print(f"Results and plots written to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
