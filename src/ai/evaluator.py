"""
IDR MVP -- AI Velocity Evaluator & Regression Metrics
======================================================
Computes metrics, residual distributions, and speed-regime error breakdowns
for the forward velocity regression model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


@dataclass
class VelocityMetrics:
    """
    Evaluation metrics for forward velocity estimation.
    """
    mae_ms: float
    rmse_ms: float
    max_error_ms: float
    median_error_ms: float
    r2_score: float
    pearson_corr: float
    mae_kmh: float
    rmse_kmh: float
    samples_count: int
    speed_bin_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mae_ms": round(self.mae_ms, 3),
            "rmse_ms": round(self.rmse_ms, 3),
            "max_error_ms": round(self.max_error_ms, 3),
            "median_error_ms": round(self.median_error_ms, 3),
            "r2_score": round(self.r2_score, 4),
            "pearson_corr": round(self.pearson_corr, 4),
            "mae_kmh": round(self.mae_kmh, 3),
            "rmse_kmh": round(self.rmse_kmh, 3),
            "samples_count": self.samples_count,
            "speed_bins": self.speed_bin_metrics,
        }


class VelocityEvaluator:
    """
    Computes velocity regression metrics and performance analysis across operational regimes.
    """

    @classmethod
    def evaluate(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> VelocityMetrics:
        """
        Evaluates predictions against true velocity.
        """
        y_true_flat = y_true.flatten().astype(np.float64)
        y_pred_flat = y_pred.flatten().astype(np.float64)

        errors = y_pred_flat - y_true_flat
        abs_errors = np.abs(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors**2)))
        max_err = float(np.max(abs_errors))
        med_err = float(np.median(abs_errors))

        # R^2 Score
        ss_res = float(np.sum(errors**2))
        ss_tot = float(np.sum((y_true_flat - np.mean(y_true_flat))**2))
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-8 else 0.0

        # Pearson Correlation
        if np.std(y_true_flat) > 1e-8 and np.std(y_pred_flat) > 1e-8:
            corr = float(np.corrcoef(y_true_flat, y_pred_flat)[0, 1])
        else:
            corr = 0.0

        # Speed Regime Bins:
        # Stationary (< 1.0 m/s), Low (1-5 m/s), Medium (5-12 m/s), High (> 12 m/s)
        bins = {
            "Stationary (< 1 m/s)": (y_true_flat < 1.0),
            "Low Speed (1-5 m/s)": (y_true_flat >= 1.0) & (y_true_flat < 5.0),
            "Medium Speed (5-12 m/s)": (y_true_flat >= 5.0) & (y_true_flat < 12.0),
            "High Speed (> 12 m/s)": (y_true_flat >= 12.0),
        }

        speed_bin_dict = {}
        for bin_name, mask in bins.items():
            if np.sum(mask) > 0:
                bin_errors = abs_errors[mask]
                speed_bin_dict[bin_name] = {
                    "count": int(np.sum(mask)),
                    "mae_ms": round(float(np.mean(bin_errors)), 3),
                    "rmse_ms": round(float(np.sqrt(np.mean((errors[mask])**2))), 3),
                }

        return VelocityMetrics(
            mae_ms=mae,
            rmse_ms=rmse,
            max_error_ms=max_err,
            median_error_ms=med_err,
            r2_score=r2,
            pearson_corr=corr,
            mae_kmh=mae * 3.6,
            rmse_kmh=rmse * 3.6,
            samples_count=len(y_true_flat),
            speed_bin_metrics=speed_bin_dict,
        )
