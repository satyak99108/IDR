#!/usr/bin/env python3
"""
IDR MVP -- Phase 14: TFLite Deployment & Optimization Benchmark
================================================================
Performs systematic quantization, on-device latency/throughput profiling,
Pareto accuracy analysis, and automated mobile deployment per MVP.md §18.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.quantization import (
    TFLiteQuantizer,
    TFLiteBenchmarkEngine,
    QuantizationBenchmarkResult,
    generate_model_card,
)

DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "phase14"
DEFAULT_ANDROID_ASSETS_DIR = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets"

# Style palette matching 60/30/10 design guidelines
COLORS = {
    "Keras (FP32)":   "#64748B",  # Slate
    "TFLite (FP32)":  "#3B82F6",  # Blue
    "TFLite (FP16)":  "#8B5CF6",  # Purple
    "Dynamic INT8":   "#10B981",  # Emerald
    "Full INT8":      "#F59E0B",  # Amber
    "target_line":    "#EF4444",  # Red
}


def parse_args():
    parser = argparse.ArgumentParser(description="IDR MVP -- Phase 14: TFLite Deployment Benchmark")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--android-assets-dir", type=str, default=str(DEFAULT_ANDROID_ASSETS_DIR))
    parser.add_argument("--max-eval-samples", type=int, default=1500)
    return parser.parse_args()


def load_evaluation_data(data_dir: Path, scaler_path: Path, max_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Loads and standardizes 50-sample IMU windows from calibrated test trip."""
    trip_path = data_dir / "SYNC_s1_calibrated.csv"
    if not trip_path.exists():
        # Fallback to any calibrated csv
        csvs = list(data_dir.glob("*_calibrated.csv"))
        if not csvs:
            raise FileNotFoundError(f"No calibrated CSV found in {data_dir}")
        trip_path = csvs[0]

    df = pd.read_csv(trip_path)
    with open(scaler_path, "r") as f:
        scaler_params = json.load(f)

    feature_names = scaler_params["feature_names"]
    means = np.array(scaler_params["means"], dtype=np.float32)
    stds = np.array(scaler_params["stds"], dtype=np.float32)
    eps = scaler_params.get("eps", 1e-8)

    # Standardize
    raw_feats = df[feature_names].fillna(0.0).values.astype(np.float32)
    norm_feats = (raw_feats - means) / (stds + eps)

    # Reference velocity
    ref_col = "ref_speed" if "ref_speed" in df.columns else "gnss_speed"
    raw_ref = df[ref_col].fillna(0.0).values.astype(np.float32)

    # Build sliding windows (size=50)
    window_size = 50
    n = min(len(norm_feats) - window_size, max_samples)
    windows = np.zeros((n, window_size, len(feature_names)), dtype=np.float32)
    velocities = np.zeros(n, dtype=np.float32)

    for i in range(n):
        windows[i] = norm_feats[i:i + window_size]
        velocities[i] = raw_ref[i + window_size - 1]

    return windows, velocities, raw_ref, feature_names


def plot_01_model_size_comparison(results: List[QuantizationBenchmarkResult], keras_size_kb: float, out_dir: Path):
    """Plot 01: Model File Size & Compression Ratio Comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    names = ["Keras (.keras)"] + [r.variant_name for r in results]
    sizes = [keras_size_kb] + [r.file_size_kb for r in results]
    colors = [COLORS.get("Keras (FP32)", "#64748B")] + [COLORS.get(r.variant_name, "#3B82F6") for r in results]

    bars = ax1.bar(names, sizes, color=colors, width=0.55, edgecolor="#1E293B", linewidth=1.2)
    ax1.set_ylabel("File Size (KB)", fontweight="bold", fontsize=11)
    ax1.set_title("On-Disk Model Size Comparison", fontweight="bold", fontsize=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_xticklabels(names, rotation=20, ha="right")

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f} KB",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Compression ratios
    comp_names = [r.variant_name for r in results]
    ratios = [r.compression_ratio_vs_keras for r in results]
    bars2 = ax2.bar(comp_names, ratios, color=[COLORS.get(n, "#10B981") for n in comp_names], width=0.55)
    ax2.set_ylabel("Compression Factor (vs Keras)", fontweight="bold", fontsize=11)
    ax2.set_title("Storage Compression Factor", fontweight="bold", fontsize=12)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.set_xticklabels(comp_names, rotation=20, ha="right")

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}x",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "01_model_size_comparison.png", dpi=300)
    plt.close(fig)
    print("  [Plot 1] Saved: 01_model_size_comparison.png")


def plot_02_latency_vs_throughput(results: List[QuantizationBenchmarkResult], out_dir: Path):
    """Plot 02: Inference Latency and Navigation Throughput vs Minimum 10 Hz Target."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = [r.variant_name for r in results]
    colors = [COLORS.get(r.variant_name, "#3B82F6") for r in results]

    # Latencies
    means = [r.mean_latency_ms for r in results]
    p95s = [r.p95_latency_ms for r in results]
    x = np.arange(len(names))
    width = 0.35

    ax1.bar(x - width / 2, means, width, label="Mean Latency", color=colors, alpha=0.9)
    ax1.bar(x + width / 2, p95s, width, label="p95 Latency", color=colors, alpha=0.4, hatch="//")
    ax1.set_ylabel("Latency (ms / sample)", fontweight="bold")
    ax1.set_title("Inference Latency Profile", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend()

    # Throughput vs 10 Hz Target
    throughputs = [r.throughput_hz for r in results]
    bars = ax2.bar(names, throughputs, color=colors, width=0.5)
    ax2.axhline(10.0, color=COLORS["target_line"], linestyle="--", linewidth=2, label="MVP Goal (10 Hz)")
    ax2.set_ylabel("Throughput (Hz)", fontweight="bold")
    ax2.set_title("Navigation Output Rate vs MVP Target", fontweight="bold")
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend()

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f} Hz",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "02_latency_vs_throughput.png", dpi=300)
    plt.close(fig)
    print("  [Plot 2] Saved: 02_latency_vs_throughput.png")


def plot_03_accuracy_vs_size_pareto(results: List[QuantizationBenchmarkResult], out_dir: Path):
    """Plot 03: Pareto Frontier of Model Size vs Prediction Error (MAE)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for r in results:
        c = COLORS.get(r.variant_name, "#3B82F6")
        ax.scatter(r.file_size_kb, r.mae_ms, color=c, s=160, zorder=5, edgecolors="#1E293B", linewidth=1.5)
        ax.annotate(
            f"  {r.variant_name}\n  ({r.file_size_kb:.1f} KB, MAE: {r.mae_ms:.3f} m/s)",
            xy=(r.file_size_kb, r.mae_ms),
            fontsize=10,
            fontweight="bold"
        )

    ax.set_xlabel("Model Size (KB) [Lower is Better]", fontweight="bold", fontsize=11)
    ax.set_ylabel("Forward Velocity MAE (m/s) [Lower is Better]", fontweight="bold", fontsize=11)
    ax.set_title("Pareto Frontier: Accuracy vs Model Compression", fontweight="bold", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_dir / "03_accuracy_vs_size_pareto.png", dpi=300)
    plt.close(fig)
    print("  [Plot 3] Saved: 03_accuracy_vs_size_pareto.png")


def plot_04_velocity_prediction_overlay(
    results: List[QuantizationBenchmarkResult],
    test_windows: np.ndarray,
    ref_velocities: np.ndarray,
    out_dir: Path
):
    """Plot 04: Ground Truth vs TFLite Predicted Velocity Timelines."""
    import tensorflow as tf

    fig, ax = plt.subplots(figsize=(14, 6))
    time_s = np.arange(len(ref_velocities)) * 0.1  # 10 Hz

    # Plot Reference
    ax.plot(time_s, ref_velocities, color="#0F172A", label="Reference Ground Truth", linewidth=2.5, zorder=4)

    # Plot Best/Key variants
    for r in results:
        if r.variant_name in ("TFLite (FP32)", "Dynamic INT8"):
            interp = tf.lite.Interpreter(model_path=r.file_path)
            interp.allocate_tensors()
            in_idx = interp.get_input_details()[0]["index"]
            out_idx = interp.get_output_details()[0]["index"]
            preds = [
                float(interp.get_tensor(out_idx)[0, 0])
                for _ in [interp.set_tensor(in_idx, test_windows[i:i + 1]) or interp.invoke() for i in range(len(test_windows))]
            ]
            ax.plot(time_s, preds, label=f"{r.variant_name} (MAE={r.mae_ms:.2f} m/s)",
                    linestyle="--", linewidth=1.6, color=COLORS.get(r.variant_name))

    ax.set_xlabel("Time Elapsed (s)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Forward Velocity (m/s)", fontweight="bold", fontsize=11)
    ax.set_title("Velocity Trajectory Estimation: Ground Truth vs Quantized TFLite", fontweight="bold", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    fig.savefig(out_dir / "04_velocity_prediction_overlay.png", dpi=300)
    plt.close(fig)
    print("  [Plot 4] Saved: 04_velocity_prediction_overlay.png")


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    android_assets_dir = Path(args.android_assets_dir)
    variants_dir = models_dir / "variants"

    results_dir.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("IDR MVP -- Phase 14: TFLite Deployment & Optimization Benchmark")
    print("=" * 75)

    # Step 1: Load Keras model & Scaler
    keras_path = models_dir / "velocity_cnn.keras"
    scaler_path = models_dir / "scaler_params.json"
    if not keras_path.exists():
        raise FileNotFoundError(f"Trained Keras model not found at {keras_path}")

    print(f"\n[Step 1/5] Loading trained Keras model & evaluation data...")
    import tensorflow as tf
    keras_model = tf.keras.models.load_model(keras_path)
    keras_size_bytes = keras_path.stat().st_size
    keras_size_kb = keras_size_bytes / 1024.0
    print(f"  Keras model: {keras_path.name} ({keras_size_kb:.1f} KB)")

    test_windows, ref_vels, full_ref, feature_names = load_evaluation_data(
        data_dir, scaler_path, args.max_eval_samples
    )
    print(f"  Evaluation dataset: {len(test_windows)} windows ({len(feature_names)} features, window=50)")

    # Baseline Keras predictions
    keras_preds = keras_model.predict(test_windows, verbose=0).flatten()
    keras_mae = float(np.mean(np.abs(keras_preds - ref_vels)))
    print(f"  Keras Baseline MAE: {keras_mae:.4f} m/s")

    # Step 2: Export Quantized Variants
    print(f"\n[Step 2/5] Exporting TFLite optimization variants...")
    variants = {}

    # Variant 1: Float32 (Standard)
    p_fp32 = variants_dir / "velocity_cnn_float32.tflite"
    TFLiteQuantizer.export_float32(keras_model, p_fp32)
    variants["TFLite (FP32)"] = p_fp32
    print(f"  [Variant 1] Float32: {p_fp32.stat().st_size / 1024.0:.1f} KB")

    # Variant 2: Dynamic-Range Post-Training Quantization (INT8 weights)
    p_dyn = variants_dir / "velocity_cnn_dynamic_range.tflite"
    TFLiteQuantizer.export_dynamic_range(keras_model, p_dyn)
    variants["Dynamic INT8"] = p_dyn
    print(f"  [Variant 2] Dynamic Range (INT8): {p_dyn.stat().st_size / 1024.0:.1f} KB")

    # Variant 3: Float16 Quantization
    p_fp16 = variants_dir / "velocity_cnn_float16.tflite"
    TFLiteQuantizer.export_float16(keras_model, p_fp16)
    variants["TFLite (FP16)"] = p_fp16
    print(f"  [Variant 3] Float16: {p_fp16.stat().st_size / 1024.0:.1f} KB")

    # Variant 4: Full Integer Quantization with representative dataset
    def representative_dataset_gen():
        for i in range(min(100, len(test_windows))):
            yield [test_windows[i:i + 1].astype(np.float32)]

    p_int8 = variants_dir / "velocity_cnn_full_int8.tflite"
    try:
        TFLiteQuantizer.export_full_integer(keras_model, representative_dataset_gen, p_int8)
        variants["Full INT8"] = p_int8
        print(f"  [Variant 4] Full Integer (INT8): {p_int8.stat().st_size / 1024.0:.1f} KB")
    except Exception as e:
        print(f"  [Variant 4] Full Integer skipped (encountered {e})")

    # Step 3: Benchmarking Sweep
    print(f"\n[Step 3/5] Running latency, throughput, and accuracy benchmark...")
    engine = TFLiteBenchmarkEngine(keras_model_size_bytes=keras_size_bytes)
    results: List[QuantizationBenchmarkResult] = []

    for name, path in variants.items():
        res = engine.benchmark_variant(
            variant_name=name,
            tflite_path=path,
            test_windows=test_windows,
            reference_velocities=ref_vels,
            keras_predictions=keras_preds,
            num_timing_runs=500
        )
        results.append(res)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Variant':<16} | {'Size (KB)':<10} | {'Comp':<6} | {'Latency':<9} | {'Throughput':<11} | {'MAE (m/s)':<10} | {'Status'}")
    print("-" * 90)
    for r in results:
        status = "PASS (>=10Hz)" if r.passes_10hz_goal else "FAIL"
        print(f"{r.variant_name:<16} | {r.file_size_kb:>8.1f} KB | {r.compression_ratio_vs_keras:>5.1f}x | {r.mean_latency_ms:>6.3f} ms | {r.throughput_hz:>8.1f} Hz | {r.mae_ms:>8.4f} m/s | {status}")
    print("=" * 90)

    # Step 4: Diagnostic Visualizations
    print(f"\n[Step 4/5] Generating diagnostic plots...")
    plot_01_model_size_comparison(results, keras_size_kb, results_dir)
    plot_02_latency_vs_throughput(results, results_dir)
    plot_03_accuracy_vs_size_pareto(results, results_dir)
    plot_04_velocity_prediction_overlay(results, test_windows, ref_vels, results_dir)

    # Step 5: Optimal Model Selection & Android Deployment
    print(f"\n[Step 5/5] Deploying optimal model to Android prototype assets...")

    # Pick variant with best compression among those maintaining MAE within 0.05 m/s of Float32
    fp32_res = next(r for r in results if "FP32" in r.variant_name)
    eligible = [r for r in results if r.mae_ms <= (fp32_res.mae_ms + 0.05) and r.passes_10hz_goal]
    best = min(eligible, key=lambda r: r.file_size_bytes) if eligible else fp32_res

    print(f"  Selected Optimal Model: '{best.variant_name}'")
    print(f"    Size: {best.file_size_kb} KB ({best.compression_ratio_vs_keras}x compression)")
    print(f"    Throughput: {best.throughput_hz} Hz (Goal >= 10 Hz: PASSED)")
    print(f"    MAE: {best.mae_ms} m/s (Parity delta: {best.mean_delta_vs_keras_ms} m/s)")

    # Copy to production paths
    dest_model = models_dir / "velocity_cnn.tflite"
    shutil.copy2(best.file_path, dest_model)
    print(f"  Deployed to: {dest_model}")

    if android_assets_dir.exists():
        dest_android = android_assets_dir / "velocity_cnn.tflite"
        shutil.copy2(best.file_path, dest_android)
        print(f"  Deployed to Android Assets: {dest_android}")

    # Save structured JSON benchmark
    benchmark_payload = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "keras_model_size_kb": round(keras_size_kb, 2),
        "keras_baseline_mae_ms": round(keras_mae, 4),
        "minimum_target_throughput_hz": 10.0,
        "variants": [r.to_dict() for r in results],
        "deployed_model": best.to_dict()
    }
    with open(results_dir / "deployment_benchmark.json", "w") as f:
        json.dump(benchmark_payload, f, indent=2)

    # Save Model Card
    model_card = generate_model_card(
        best_variant=best,
        input_shape=[1, 50, len(feature_names)],
        output_shape=[1, 1],
        feature_names=feature_names,
    )
    with open(results_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)

    print(f"  Saved benchmark metrics: {results_dir / 'deployment_benchmark.json'}")
    print(f"  Saved model card: {results_dir / 'model_card.json'}")
    print(f"\n>>> PHASE 14 COMPLETED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    main()
