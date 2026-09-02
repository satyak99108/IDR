"""
IDR MVP -- Phase 5 Runner: AI Forward Velocity Estimator
=========================================================
Orchestrates Phase 5: Trains and evaluates the 1D CNN model for forward vehicle
velocity estimation from 6-axis IMU sliding windows (MVP.md §9 and TECH_STACK.md §4).

Pipeline:
1. Load calibrated dataset (SYNC_s1_calibrated.csv).
2. Extract 6-axis IMU features and ground-truth speed (m/s).
3. Partition chronologically into Train (70%), Val (15%), Test (15%) splits.
4. Fit IMUScaler on training features and normalize all splits.
5. Create sliding window sequences (W=50 samples = 5.0s @ 10 Hz).
6. Train 1D CNN model with Huber loss, Adam optimizer, and callbacks.
7. Export models (Keras .keras, TFLite .tflite, NumPy weights .npz, Scaler .json).
8. Evaluate on held-out test split with VelocityEvaluator.
9. Save metrics JSON and CSV summary to results/phase5/.
10. Generate 5 presentation-ready diagnostic plots.

Usage:
    python -u scripts/run_phase5.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

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
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai import (
    IMUScaler,
    IMUSequenceDataset,
    build_1d_cnn_velocity_model,
    NumpyCNNInference,
    VelocityModelTrainer,
    VelocityEvaluator,
    VelocityMetrics,
)

# ---------------------------------------------------------------------------
# Configuration & Theme
# ---------------------------------------------------------------------------

DEFAULT_INPUT  = PROJECT_ROOT / "data" / "processed" / "SYNC_s1_calibrated.csv"
MODELS_DIR     = PROJECT_ROOT / "models"
RESULTS_DIR    = PROJECT_ROOT / "results" / "phase5"

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
    "pred":       "#E74C3C",   # AI Predicted — red
    "train_loss": "#3498DB",   # Train loss   — blue
    "val_loss":   "#E67E22",   # Val loss     — orange
    "scatter":    "#8E44AD",   # Scatter      — purple
    "hist":       "#1ABC9C",   # Histogram    — teal
    "fit_line":   "#2C3E50",   # Ideal fit    — dark slate
}


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="IDR MVP -- Phase 5: AI Forward Velocity Estimator"
    )
    parser.add_argument(
        "--input", type=str, default=str(DEFAULT_INPUT),
        help=f"Path to Phase 2 calibrated CSV (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--window-size", type=int, default=50,
        help="Sliding window length in samples (default: 50 = 5.0s @ 10 Hz)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Training epochs (default: 30)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--models-dir", type=str, default=str(MODELS_DIR),
        help=f"Directory to save trained models (default: {MODELS_DIR})"
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR),
        help=f"Directory for plots and metrics (default: {RESULTS_DIR})"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Plotting Suite
# ---------------------------------------------------------------------------

def plot_01_velocity_timeseries(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    sample_limit: int = 1500,
):
    """Plot 1: Continuous Ground Truth vs AI Predicted Speed Timeline."""
    fig, ax = plt.subplots(figsize=(15, 6))

    n = min(len(y_test), sample_limit)
    time_s = np.arange(n) * 0.1  # 10 Hz

    y_t_plot = y_test[:n].flatten()
    y_p_plot = y_pred[:n].flatten()

    ax.plot(time_s, y_t_plot, color=PALETTE["gt"], linewidth=2.0, label="Ground Truth Speed (m/s)", zorder=3)
    ax.plot(time_s, y_p_plot, color=PALETTE["pred"], linewidth=1.8, linestyle="--", label="1D CNN Predicted Speed (m/s)", zorder=4)

    ax.set_title("Phase 5 — Ground Truth vs AI 1D CNN Forward Velocity (Test Split)", pad=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Forward Speed (m/s)")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 1] Saved: {output_path.name}")


def plot_02_velocity_scatter_r2(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    r2_score: float,
    mae_ms: float,
    output_path: Path,
):
    """Plot 2: Scatter plot (y_true vs y_pred) with ideal y=x reference line."""
    fig, ax = plt.subplots(figsize=(10, 8))

    y_t = y_test.flatten()
    y_p = y_pred.flatten()

    # Subsample points if large for clean rendering
    if len(y_t) > 3000:
        idx = np.random.choice(len(y_t), 3000, replace=False)
        y_t_sub, y_p_sub = y_t[idx], y_p[idx]
    else:
        y_t_sub, y_p_sub = y_t, y_p

    ax.scatter(y_t_sub, y_p_sub, color=PALETTE["scatter"], alpha=0.35, s=18, edgecolors="none", label="Predictions")

    # Ideal line
    max_val = max(np.max(y_t), np.max(y_p)) * 1.05
    ax.plot([0, max_val], [0, max_val], color="#E74C3C", linestyle="--", linewidth=2.2, label="Ideal Fit (y = x)")

    # Annotation box
    ax.text(
        0.05, 0.88,
        f"R² Score: {r2_score:.4f}\nMAE: {mae_ms:.3f} m/s ({mae_ms*3.6:.2f} km/h)",
        transform=ax.transAxes, fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#7F8C8D"),
    )

    ax.set_title("Phase 5 — True Speed vs 1D CNN Predicted Speed Scatter", pad=12, fontweight="bold")
    ax.set_xlabel("Reference Velocity (m/s)")
    ax.set_ylabel("Predicted Velocity (m/s)")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 2] Saved: {output_path.name}")


def plot_03_loss_curves(
    history: Dict[str, Any],
    output_path: Path,
):
    """Plot 3: Training and Validation Loss / MAE curves over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    epochs = range(1, len(history["loss"]) + 1)

    # Loss subplot
    ax1.plot(epochs, history["loss"], color=PALETTE["train_loss"], linewidth=2.0, label="Train Huber Loss")
    ax1.plot(epochs, history["val_loss"], color=PALETTE["val_loss"], linewidth=2.0, linestyle="--", label="Val Huber Loss")
    ax1.set_title("Training vs Validation Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Huber Loss")
    ax1.legend(loc="upper right")

    # MAE subplot
    ax2.plot(epochs, history["mae"], color=PALETTE["train_loss"], linewidth=2.0, label="Train MAE (m/s)")
    ax2.plot(epochs, history["val_mae"], color=PALETTE["val_loss"], linewidth=2.0, linestyle="--", label="Val MAE (m/s)")
    ax2.set_title("Training vs Validation MAE", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Absolute Error (m/s)")
    ax2.legend(loc="upper right")

    fig.suptitle("Phase 5 — 1D CNN Velocity Model Learning Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 3] Saved: {output_path.name}")


def plot_04_residuals_distribution(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
):
    """Plot 4: Residual prediction errors histogram with fitted Gaussian normal curve."""
    fig, ax = plt.subplots(figsize=(12, 6))

    residuals = (y_pred.flatten() - y_test.flatten()).astype(float)
    mu, std = float(np.mean(residuals)), float(np.std(residuals))

    # Histogram & KDE
    sns.histplot(residuals, bins=50, kde=True, color=PALETTE["hist"], stat="density", ax=ax, alpha=0.55, edgecolor="black")

    # Gaussian curve
    x_grid = np.linspace(residuals.min(), residuals.max(), 200)
    p = stats.norm.pdf(x_grid, mu, std)
    ax.plot(x_grid, p, color="#C0392B", linewidth=2.2, label=f"Normal Fit (μ={mu:.3f}, σ={std:.3f})")

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Phase 5 — Velocity Prediction Residual Error Distribution", pad=12, fontweight="bold")
    ax.set_xlabel("Residual Error = Predicted - True (m/s)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 4] Saved: {output_path.name}")


def plot_05_error_by_speed_bin(
    speed_bin_metrics: Dict[str, Dict[str, float]],
    output_path: Path,
):
    """Plot 5: Bar chart showing MAE across different operational speed regimes."""
    fig, ax = plt.subplots(figsize=(12, 6))

    bin_names = list(speed_bin_metrics.keys())
    maes = [speed_bin_metrics[b]["mae_ms"] for b in bin_names]
    counts = [speed_bin_metrics[b]["count"] for b in bin_names]
    colors = ["#3498DB", "#2ECC71", "#F39C12", "#E74C3C"]

    bars = ax.bar(bin_names, maes, color=colors[:len(bin_names)], width=0.55, edgecolor="black", linewidth=1.2, zorder=3)

    for bar, count in zip(bars, counts):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, h + 0.02,
            f"{h:.3f} m/s\n(n={count})",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title("Phase 5 — Velocity Estimation MAE by Vehicle Speed Regime", pad=12, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (m/s)")
    ax.set_ylim(0, max(maes) * 1.35 + 0.1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  [Plot 5] Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    input_path  = Path(args.input)
    models_dir  = Path(args.models_dir)
    results_dir = Path(args.results_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IDR MVP -- Phase 5: AI Forward Velocity Estimator (1D CNN)")
    print("=" * 70)

    # 1. Load Calibrated Dataset
    print(f"\n[Step 1/6] Loading calibrated dataset: {input_path}")
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} samples ({len(df)/10.0:.1f} seconds at 10 Hz)")

    # 2. Partition & Extract Sliding Windows
    print("\n[Step 2/6] Partitioning dataset & building sliding windows (W=50 samples = 5.0s)...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = (
        IMUSequenceDataset.partition_trip_dataset(
            df,
            window_size=args.window_size,
            step_size=1,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )
    )

    print(f"  Train sequences: {X_train.shape[0]:,} | Shape: {X_train.shape}")
    print(f"  Val sequences:   {X_val.shape[0]:,}   | Shape: {X_val.shape}")
    print(f"  Test sequences:  {X_test.shape[0]:,}  | Shape: {X_test.shape}")

    # Save fitted scaler
    scaler_path = models_dir / "scaler_params.json"
    scaler.save(scaler_path)
    print(f"  Saved fitted feature scaler: {scaler_path}")

    # 3. Train 1D CNN Velocity Model
    print(f"\n[Step 3/6] Training 1D CNN Model (epochs={args.epochs}, batch_size={args.batch_size})...")
    trainer = VelocityModelTrainer(
        input_shape=(args.window_size, 6),
        learning_rate=1e-3,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    history = trainer.train(X_train, y_train, X_val, y_val, verbose=1)

    # 4. Export Models
    print("\n[Step 4/6] Exporting Model Artifacts (Keras, TFLite, NumPy)...")
    exported_paths = trainer.export_models(models_dir, model_name="velocity_cnn")
    for fmt, p in exported_paths.items():
        print(f"  - {fmt.upper()}: {p}")

    # 5. Evaluate on Test Split
    print("\n[Step 5/6] Evaluating Velocity Estimator on Held-out Test Split...")
    y_pred_test = trainer.model.predict(X_test, batch_size=args.batch_size, verbose=0)
    metrics = VelocityEvaluator.evaluate(y_test, y_pred_test)

    print("\n  --- Phase 5 Evaluation Metrics (Test Split) ---")
    print(f"  MAE:                {metrics.mae_ms:.3f} m/s ({metrics.mae_kmh:.2f} km/h)")
    print(f"  RMSE:               {metrics.rmse_ms:.3f} m/s ({metrics.rmse_kmh:.2f} km/h)")
    print(f"  R² Score:           {metrics.r2_score:.4f}")
    print(f"  Pearson Corr:       {metrics.pearson_corr:.4f}")
    print(f"  Max Error:          {metrics.max_error_ms:.3f} m/s")
    print(f"  Median Error:       {metrics.median_error_ms:.3f} m/s")
    print(f"  Test Samples:       {metrics.samples_count:,}")

    # Save metrics JSON
    metrics_json_path = results_dir / "velocity_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"  Saved evaluation JSON: {metrics_json_path}")

    # Save summary CSV
    summary_csv_path = results_dir / "velocity_summary.csv"
    summary_df = pd.DataFrame([{
        "model": "1D_CNN_Velocity",
        "window_size": args.window_size,
        "mae_ms": metrics.mae_ms,
        "rmse_ms": metrics.rmse_ms,
        "r2_score": metrics.r2_score,
        "pearson_corr": metrics.pearson_corr,
        "mae_kmh": metrics.mae_kmh,
        "rmse_kmh": metrics.rmse_kmh,
        "max_error_ms": metrics.max_error_ms,
    }])
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  Saved summary CSV: {summary_csv_path}")

    # 6. Generate Diagnostic Plots
    print("\n[Step 6/6] Generating Phase 5 Diagnostic Plots...")
    plot_01_velocity_timeseries(y_test, y_pred_test, results_dir / "01_velocity_gt_vs_predicted_timeseries.png")
    plot_02_velocity_scatter_r2(y_test, y_pred_test, metrics.r2_score, metrics.mae_ms, results_dir / "02_velocity_scatter_r2_fit.png")
    plot_03_loss_curves(history, results_dir / "03_training_validation_loss_curves.png")
    plot_04_residuals_distribution(y_test, y_pred_test, results_dir / "04_velocity_residuals_distribution.png")
    plot_05_error_by_speed_bin(metrics.speed_bin_metrics, results_dir / "05_velocity_error_by_speed_bin.png")

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE: AI Velocity Estimator successfully trained & validated!")
    print(f"Models saved in: {models_dir}")
    print(f"Results saved in: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
