"""
IDR MVP -- Phase 6 Runner: Motion / Vibration Filtering
=========================================================
Orchestrates Phase 6: Extracts motion features from calibrated IMU data,
classifies every sample into navigation-relevant motion states, and generates
diagnostic plots (MVP.md §10, TECH_STACK.md §4/§6).

Pipeline:
    1. Load calibrated dataset (SYNC_s1_calibrated.csv).
    2. Run MotionFeatureExtractor over the full trip.
    3. Run MotionClassifier to label every window & sample.
    4. Print distribution summary (% stationary, normal, shock, etc.).
    5. Save feature CSV and label CSV to results/phase6/.
    6. Generate 5 presentation-ready diagnostic plots.

Usage:
    python -u scripts/run_phase6.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

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

from src.motion import MotionFeatureExtractor, MotionClassifier, MotionLabel


# ---------------------------------------------------------------------------
# Configuration & Theme
# ---------------------------------------------------------------------------

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "SYNC_s1_calibrated.csv"
RESULTS_DIR   = PROJECT_ROOT / "results" / "phase6"

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

# Colour palette for motion labels
LABEL_COLORS: Dict[str, str] = {
    "NORMAL":     "#2ECC71",   # Emerald green
    "STATIONARY": "#3498DB",   # Blue
    "SHOCK":      "#E74C3C",   # Red
    "VIBRATION":  "#E67E22",   # Orange
    "ABNORMAL":   "#8E44AD",   # Purple
}

FEATURE_PALETTE = {
    "accel_rms":          "#2ECC71",
    "accel_var":          "#E74C3C",
    "gyro_rms":           "#3498DB",
    "jerk_rms":           "#E67E22",
    "freq_energy_ratio":  "#8E44AD",
    "spectral_entropy":   "#1ABC9C",
}


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="IDR MVP -- Phase 6: Motion / Vibration Filtering"
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT),
        help=f"Path to Phase 2 calibrated CSV (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--window-sec", type=float, default=1.0,
        help="Feature window duration in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--step-sec", type=float, default=0.5,
        help="Feature window stride in seconds (default: 0.5 = 50%% overlap)"
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR),
        help=f"Directory for plots and outputs (default: {RESULTS_DIR})"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Plotting Suite — 5 Diagnostic Plots
# ---------------------------------------------------------------------------

def plot_01_feature_timeseries(
    feature_df: pd.DataFrame,
    labels: pd.Series,
    output_path: Path,
    sample_limit: int = 500,
):
    """Plot 1: Key features over time, colour-coded by motion label."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    n = min(len(feature_df), sample_limit)
    time_s = feature_df["timestamp"].values[:n]
    # Normalise to start at 0
    time_s = (time_s - time_s[0])

    lbl_vals = labels.values[:n]

    # Panel 1: Acceleration RMS + Variance
    ax = axes[0]
    ax.plot(time_s, feature_df["accel_rms"].values[:n],
            color=FEATURE_PALETTE["accel_rms"], linewidth=1.5, label="Accel RMS (m/s²)")
    ax.plot(time_s, feature_df["accel_var"].values[:n],
            color=FEATURE_PALETTE["accel_var"], linewidth=1.2, linestyle="--", label="Accel Var (m²/s⁴)")
    _shade_labels(ax, time_s, lbl_vals)
    ax.set_ylabel("Acceleration")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title("Phase 6 — Motion Feature Timeline with Event Markers", fontweight="bold", pad=10)

    # Panel 2: Gyro RMS + Jerk RMS
    ax = axes[1]
    ax.plot(time_s, feature_df["gyro_rms"].values[:n],
            color=FEATURE_PALETTE["gyro_rms"], linewidth=1.5, label="Gyro RMS (rad/s)")
    ax2 = ax.twinx()
    ax2.plot(time_s, feature_df["jerk_rms"].values[:n],
             color=FEATURE_PALETTE["jerk_rms"], linewidth=1.2, linestyle="--", label="Jerk RMS (m/s³)")
    ax2.set_ylabel("Jerk RMS", color=FEATURE_PALETTE["jerk_rms"])
    _shade_labels(ax, time_s, lbl_vals)
    ax.set_ylabel("Gyro RMS")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9, framealpha=0.9)

    # Panel 3: Spectral features
    ax = axes[2]
    ax.plot(time_s, feature_df["freq_energy_ratio"].values[:n],
            color=FEATURE_PALETTE["freq_energy_ratio"], linewidth=1.5, label="HF Energy Ratio")
    ax.plot(time_s, feature_df["spectral_entropy"].values[:n],
            color=FEATURE_PALETTE["spectral_entropy"], linewidth=1.2, linestyle="--", label="Spectral Entropy")
    _shade_labels(ax, time_s, lbl_vals)
    ax.set_ylabel("Spectral Features")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 1] Saved: {output_path.name}")


def plot_02_label_distribution(
    sample_labels: pd.Series,
    output_path: Path,
):
    """Plot 2: Bar chart of motion label distribution."""
    dist = MotionClassifier.label_distribution(sample_labels)
    total = sum(dist.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    names = list(dist.keys())
    counts = [dist[n] for n in names]
    colors = [LABEL_COLORS.get(n, "#888888") for n in names]

    bars = ax1.bar(names, counts, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_title("Motion Label Distribution (Sample Count)", fontweight="bold")
    ax1.set_ylabel("Number of Samples")
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                 f"{count:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Pie chart
    non_zero = [(n, c) for n, c in zip(names, counts) if c > 0]
    if non_zero:
        pie_names, pie_counts = zip(*non_zero)
        pie_colors = [LABEL_COLORS.get(n, "#888888") for n in pie_names]
        wedges, texts, autotexts = ax2.pie(
            pie_counts, labels=pie_names, colors=pie_colors,
            autopct="%1.1f%%", startangle=90, pctdistance=0.75,
            textprops={"fontsize": 10}
        )
        for at in autotexts:
            at.set_fontweight("bold")
    ax2.set_title("Motion Label Proportion", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 2] Saved: {output_path.name}")


def plot_03_accel_with_events(
    df: pd.DataFrame,
    sample_labels: pd.Series,
    output_path: Path,
    sample_limit: int = 3000,
):
    """Plot 3: Raw acceleration magnitude with shock/vibration markers."""
    fig, ax = plt.subplots(figsize=(16, 6))

    n = min(len(df), sample_limit)
    time_s = np.arange(n) * 0.1   # 10 Hz

    # Resolve accel columns
    accel_cols = None
    for candidates in [
        ("ax_veh_lin", "ay_veh_lin", "az_veh_lin"),
        ("ax_veh", "ay_veh", "az_veh"),
        ("ax", "ay", "az"),
    ]:
        if all(c in df.columns for c in candidates):
            accel_cols = candidates
            break

    if accel_cols is None:
        print("  [Plot 3] Skipped: Could not find accelerometer columns")
        plt.close()
        return

    accel = df[list(accel_cols)].values[:n].astype(float)
    accel_mag = np.sqrt(np.sum(accel ** 2, axis=1))

    ax.plot(time_s, accel_mag, color="#2C3E50", linewidth=0.8, alpha=0.8, label="Accel Magnitude (m/s²)")

    # Overlay event markers
    lbl = sample_labels.values[:n]

    shock_mask = lbl == int(MotionLabel.SHOCK)
    if shock_mask.any():
        ax.scatter(time_s[shock_mask], accel_mag[shock_mask],
                   color=LABEL_COLORS["SHOCK"], s=40, marker="v", zorder=5, label="SHOCK", edgecolors="black", linewidths=0.5)

    vib_mask = lbl == int(MotionLabel.VIBRATION)
    if vib_mask.any():
        ax.scatter(time_s[vib_mask], accel_mag[vib_mask],
                   color=LABEL_COLORS["VIBRATION"], s=30, marker="D", zorder=5, label="VIBRATION", edgecolors="black", linewidths=0.5)

    abnorm_mask = lbl == int(MotionLabel.ABNORMAL)
    if abnorm_mask.any():
        ax.scatter(time_s[abnorm_mask], accel_mag[abnorm_mask],
                   color=LABEL_COLORS["ABNORMAL"], s=50, marker="x", zorder=5, label="ABNORMAL", linewidths=1.5)

    stat_mask = lbl == int(MotionLabel.STATIONARY)
    if stat_mask.any():
        # Show stationary as background shading
        _shade_regions(ax, time_s, stat_mask, color=LABEL_COLORS["STATIONARY"], alpha=0.15, label="STATIONARY")

    ax.set_title("Phase 6 — Acceleration Signal with Detected Events", fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Acceleration Magnitude (m/s²)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 3] Saved: {output_path.name}")


def plot_04_freq_energy_heatmap(
    feature_df: pd.DataFrame,
    output_path: Path,
    sample_limit: int = 500,
):
    """Plot 4: Frequency-domain energy features over time as a heatmap."""
    fig, ax = plt.subplots(figsize=(16, 4))

    n = min(len(feature_df), sample_limit)
    time_s = feature_df["timestamp"].values[:n]
    time_s = time_s - time_s[0]

    spectral_feats = feature_df[["freq_energy_ratio", "spectral_entropy", "accel_var"]].values[:n].T

    im = ax.imshow(
        spectral_feats,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="bilinear",
        extent=[time_s[0], time_s[-1], 0, 3],
    )
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["Accel Var", "Spectral Entropy", "HF Energy Ratio"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Phase 6 — Spectral & Variance Feature Heatmap", fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, label="Feature Value", shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 4] Saved: {output_path.name}")


def plot_05_jerk_with_annotations(
    feature_df: pd.DataFrame,
    labels: pd.Series,
    output_path: Path,
    sample_limit: int = 500,
):
    """Plot 5: Jerk RMS timeline with pothole/shock spike annotations."""
    fig, ax = plt.subplots(figsize=(16, 5))

    n = min(len(feature_df), sample_limit)
    time_s = feature_df["timestamp"].values[:n]
    time_s = time_s - time_s[0]

    jerk = feature_df["jerk_rms"].values[:n]
    lbl = labels.values[:n]

    ax.plot(time_s, jerk, color="#2C3E50", linewidth=1.2, label="Jerk RMS (m/s³)")
    ax.fill_between(time_s, 0, jerk, alpha=0.15, color="#3498DB")

    # Mark shocks
    shock_mask = lbl == int(MotionLabel.SHOCK)
    if shock_mask.any():
        shock_times = time_s[shock_mask]
        shock_jerk = jerk[shock_mask]
        ax.scatter(shock_times, shock_jerk,
                   color=LABEL_COLORS["SHOCK"], s=80, marker="v", zorder=5,
                   label="SHOCK (pothole/bump)", edgecolors="black", linewidths=0.8)
        # Annotate the top 5 shocks
        top_indices = np.argsort(shock_jerk)[-5:]
        for idx in top_indices:
            ax.annotate(
                f"{shock_jerk[idx]:.1f}",
                (shock_times[idx], shock_jerk[idx]),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8, fontweight="bold",
                color=LABEL_COLORS["SHOCK"],
                arrowprops=dict(arrowstyle="-", color=LABEL_COLORS["SHOCK"], lw=0.8),
            )

    # Threshold line
    ax.axhline(y=30.0, color="#E74C3C", linewidth=1.0, linestyle=":", alpha=0.6, label="Shock Threshold (30 m/s³)")

    ax.set_title("Phase 6 — Jerk RMS with Pothole/Shock Annotations", fontweight="bold", pad=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Jerk RMS (m/s³)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 5] Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Shading Helpers
# ---------------------------------------------------------------------------

def _shade_labels(ax, time_s, labels, alpha=0.12):
    """Add semi-transparent colour bands for non-NORMAL labels."""
    for lbl in MotionLabel:
        if lbl == MotionLabel.NORMAL:
            continue
        mask = labels == int(lbl)
        if mask.any():
            _shade_regions(ax, time_s, mask, color=LABEL_COLORS.get(lbl.name, "#888"), alpha=alpha)


def _shade_regions(ax, time_s, mask, color, alpha=0.15, label=None):
    """Shade contiguous True regions of a boolean mask."""
    labelled = False
    in_region = False
    start_t = 0.0

    for i, val in enumerate(mask):
        if val and not in_region:
            start_t = time_s[i]
            in_region = True
        elif not val and in_region:
            lbl = label if not labelled else None
            ax.axvspan(start_t, time_s[i - 1], color=color, alpha=alpha, label=lbl)
            labelled = True
            in_region = False

    # Close any open region
    if in_region:
        lbl = label if not labelled else None
        ax.axvspan(start_t, time_s[-1], color=color, alpha=alpha, label=lbl)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    input_path  = Path(args.input)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IDR MVP -- Phase 6: Motion / Vibration Filtering")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load Calibrated Dataset
    # ------------------------------------------------------------------
    print(f"\n[Step 1/5] Loading calibrated dataset: {input_path}")
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        print("  Run Phase 2 first to generate the calibrated dataset.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} samples ({len(df)/10.0:.1f} seconds at 10 Hz)")

    # ------------------------------------------------------------------
    # Step 2: Extract Motion Features
    # ------------------------------------------------------------------
    print(f"\n[Step 2/5] Extracting motion features (window={args.window_sec}s, step={args.step_sec}s)...")
    extractor = MotionFeatureExtractor(
        sampling_rate_hz=10.0,
        window_size_sec=args.window_sec,
        step_size_sec=args.step_sec,
    )

    feature_df = extractor.extract_from_dataframe(df)
    print(f"  Extracted {len(feature_df):,} feature windows")
    print(f"  Window size: {extractor.window_samples} samples ({args.window_sec}s)")
    print(f"  Step size:   {extractor.step_samples} samples ({args.step_sec}s)")

    # Print feature statistics
    print("\n  Feature Statistics (mean ± std):")
    for feat in ["accel_mean", "accel_var", "accel_rms", "accel_peak",
                  "gyro_rms", "gyro_peak", "jerk_rms",
                  "freq_energy_ratio", "spectral_entropy"]:
        vals = feature_df[feat]
        print(f"    {feat:>22s}: {vals.mean():8.3f} ± {vals.std():8.3f}  "
              f"[{vals.min():.3f} — {vals.max():.3f}]")

    # Save feature CSV
    feature_csv_path = results_dir / "motion_features.csv"
    feature_df.to_csv(feature_csv_path, index=False)
    print(f"\n  Saved feature CSV: {feature_csv_path}")

    # ------------------------------------------------------------------
    # Step 3: Classify Motion Labels
    # ------------------------------------------------------------------
    print("\n[Step 3/5] Classifying motion windows...")
    classifier = MotionClassifier()

    # Per-window labels
    window_labels = classifier.classify_feature_dataframe(feature_df)

    # Propagate to per-sample labels
    sample_labels = classifier.label_source_dataframe(df, feature_df)

    # Trust weights
    trust_weights = classifier.get_trust_weights_series(sample_labels)

    # Distribution summary
    w_dist = MotionClassifier.label_distribution(window_labels)
    s_dist = MotionClassifier.label_distribution(sample_labels)
    s_pct  = MotionClassifier.label_distribution_pct(sample_labels)

    print("\n  --- Motion Classification Summary ---")
    print(f"  {'Label':<14s} {'Windows':>10s} {'Samples':>10s} {'Percent':>10s}")
    print(f"  {'─' * 46}")
    for lbl in MotionLabel:
        w = w_dist.get(lbl.name, 0)
        s = s_dist.get(lbl.name, 0)
        p = s_pct.get(lbl.name, 0.0)
        trust = classifier.get_trust_weight(lbl)
        print(f"  {lbl.name:<14s} {w:>10,d} {s:>10,d} {p:>9.1f}%  (trust={trust:.1f})")

    # Save label CSV
    label_df = pd.DataFrame({
        "sample_index": range(len(sample_labels)),
        "motion_label": sample_labels.values,
        "motion_label_name": [MotionLabel(v).name for v in sample_labels.values],
        "trust_weight": trust_weights.values,
    })
    label_csv_path = results_dir / "motion_labels.csv"
    label_df.to_csv(label_csv_path, index=False)
    print(f"\n  Saved label CSV: {label_csv_path}")

    # Save summary JSON
    summary = {
        "total_samples": len(df),
        "total_windows": len(feature_df),
        "window_size_sec": args.window_sec,
        "step_size_sec": args.step_sec,
        "window_distribution": w_dist,
        "sample_distribution": s_dist,
        "sample_distribution_pct": s_pct,
        "thresholds": {
            "stationary_accel_var_max": classifier.thresholds.stationary_accel_var_max,
            "stationary_gyro_rms_max": classifier.thresholds.stationary_gyro_rms_max,
            "shock_jerk_rms_min": classifier.thresholds.shock_jerk_rms_min,
            "shock_accel_peak_min": classifier.thresholds.shock_accel_peak_min,
            "vibration_freq_energy_ratio_min": classifier.thresholds.vibration_freq_energy_ratio_min,
            "vibration_spectral_entropy_min": classifier.thresholds.vibration_spectral_entropy_min,
            "vibration_accel_var_min": classifier.thresholds.vibration_accel_var_min,
            "abnormal_gyro_peak_min": classifier.thresholds.abnormal_gyro_peak_min,
            "abnormal_accel_peak_min": classifier.thresholds.abnormal_accel_peak_min,
        },
    }
    summary_json_path = results_dir / "motion_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary JSON: {summary_json_path}")

    # ------------------------------------------------------------------
    # Step 4: Generate Diagnostic Plots
    # ------------------------------------------------------------------
    print("\n[Step 4/5] Generating Phase 6 diagnostic plots...")
    plot_01_feature_timeseries(
        feature_df, window_labels,
        results_dir / "01_motion_feature_timeseries.png",
    )
    plot_02_label_distribution(
        sample_labels,
        results_dir / "02_motion_label_distribution.png",
    )
    plot_03_accel_with_events(
        df, sample_labels,
        results_dir / "03_accel_signal_with_events.png",
    )
    plot_04_freq_energy_heatmap(
        feature_df,
        results_dir / "04_freq_energy_heatmap.png",
    )
    plot_05_jerk_with_annotations(
        feature_df, window_labels,
        results_dir / "05_jerk_rms_with_annotations.png",
    )

    # ------------------------------------------------------------------
    # Step 5: Summary
    # ------------------------------------------------------------------
    print("\n[Step 5/5] Phase 6 Summary")
    print(f"  Non-normal events detected: "
          f"{s_dist.get('SHOCK', 0) + s_dist.get('VIBRATION', 0) + s_dist.get('ABNORMAL', 0):,} samples "
          f"({s_pct.get('SHOCK', 0) + s_pct.get('VIBRATION', 0) + s_pct.get('ABNORMAL', 0):.1f}%)")
    print(f"  Stationary samples:         {s_dist.get('STATIONARY', 0):,} "
          f"({s_pct.get('STATIONARY', 0):.1f}%)")
    print(f"  Normal driving samples:     {s_dist.get('NORMAL', 0):,} "
          f"({s_pct.get('NORMAL', 0):.1f}%)")
    print(f"  Mean trust weight:          {trust_weights.mean():.3f}")

    print("\n" + "=" * 70)
    print("PHASE 6 COMPLETE: Motion / Vibration Filtering finished!")
    print(f"Results saved in: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
