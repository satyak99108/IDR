"""
IDR MVP -- AI Dataset & Sequence Utilities
===========================================
Prepares sliding window IMU sequence datasets for forward velocity estimation.

Pipeline:
    6-Axis IMU Signals [ax_veh, ay_veh, az_veh, gx_veh, gy_veh, gz_veh]
        ↓
    Standardization (IMUScaler fitted on training set)
        ↓
    Sliding Window Extraction (W samples, e.g. W=50 at 10 Hz = 5.0s)
        ↓
    (X, y) Arrays where X in R^(N x W x 6), y in R^(N x 1) in m/s

Classes:
    IMUScaler           -- Z-score standardizer with JSON serialization.
    IMUSequenceDataset  -- Sliding window generator and sequence partitioner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature Scaler
# ---------------------------------------------------------------------------

@dataclass
class IMUScaler:
    """
    Standardizes 6-axis IMU features using mean and standard deviation:
        z = (x - mean) / (std + eps)
    """
    feature_names: List[str] = field(default_factory=lambda: [
        "ax_veh_lin", "ay_veh_lin", "az_veh_lin",
        "gx_veh", "gy_veh", "gz_veh",
    ])
    means: Optional[np.ndarray] = None
    stds: Optional[np.ndarray] = None
    eps: float = 1e-8
    is_fitted: bool = False

    def fit(self, X: np.ndarray) -> IMUScaler:
        """
        Fits mean and standard deviation from 2D (N, C) or 3D (N, W, C) arrays.
        """
        if X.ndim == 3:
            # Reshape to (N*W, C)
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X

        self.means = np.nanmean(X_flat, axis=0)
        self.stds = np.nanstd(X_flat, axis=0)
        # Prevent division by zero
        self.stds = np.where(self.stds < self.eps, 1.0, self.stds)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardizes array X using fitted statistics."""
        if not self.is_fitted or self.means is None or self.stds is None:
            raise RuntimeError("IMUScaler is not fitted yet. Call fit() first.")

        if X.ndim == 3:
            # Shape (N, W, C)
            return (X - self.means[None, None, :]) / self.stds[None, None, :]
        elif X.ndim == 2:
            # Shape (N, C)
            return (X - self.means[None, :]) / self.stds[None, :]
        else:
            return (X - self.means) / self.stds

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits and transforms array X in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Restores original scale from standardized features."""
        if not self.is_fitted or self.means is None or self.stds is None:
            raise RuntimeError("IMUScaler is not fitted yet.")

        if X_scaled.ndim == 3:
            return X_scaled * self.stds[None, None, :] + self.means[None, None, :]
        elif X_scaled.ndim == 2:
            return X_scaled * self.stds[None, :] + self.means[None, :]
        else:
            return X_scaled * self.stds + self.means

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves scaler statistics to JSON."""
        data = {
            "feature_names": self.feature_names,
            "means": self.means.tolist() if self.means is not None else [],
            "stds": self.stds.tolist() if self.stds is not None else [],
            "eps": self.eps,
            "is_fitted": self.is_fitted,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> IMUScaler:
        """Loads scaler statistics from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        scaler = cls(
            feature_names=data.get("feature_names", []),
            eps=data.get("eps", 1e-8),
        )
        if data.get("is_fitted", False):
            scaler.means = np.array(data["means"], dtype=float)
            scaler.stds = np.array(data["stds"], dtype=float)
            scaler.is_fitted = True
        return scaler


# ---------------------------------------------------------------------------
# Dataset Sequence Builder
# ---------------------------------------------------------------------------

class IMUSequenceDataset:
    """
    Constructs sliding window datasets from calibrated IMU and reference velocity logs.
    """

    DEFAULT_FEATURES = [
        "ax_veh_lin", "ay_veh_lin", "az_veh_lin",
        "gx_veh", "gy_veh", "gz_veh",
    ]

    FALLBACK_FEATURES = [
        "ax_veh", "ay_veh", "az_veh",
        "gx", "gy", "gz",
    ]

    @classmethod
    def extract_features(cls, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts the 6-axis IMU feature matrix from a DataFrame.
        """
        if all(col in df.columns for col in cls.DEFAULT_FEATURES):
            cols = cls.DEFAULT_FEATURES
        elif all(col in df.columns for col in cls.FALLBACK_FEATURES):
            cols = cls.FALLBACK_FEATURES
        else:
            # Match any ax, ay, az, gx, gy, gz variations
            matched = []
            for prefix in ["ax", "ay", "az", "gx", "gy", "gz"]:
                candidates = [c for c in df.columns if c.startswith(prefix)]
                if candidates:
                    matched.append(candidates[0])
            if len(matched) == 6:
                cols = matched
            else:
                raise ValueError(f"Could not find 6 IMU columns in DataFrame. Columns: {df.columns.tolist()[:10]}")

        X = df[cols].values.astype(np.float32)
        return X, cols

    @classmethod
    def extract_target_velocity(cls, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts ground-truth forward speed in m/s from reference columns.
        """
        if "ref_speed" in df.columns and not df["ref_speed"].isna().all():
            spd = df["ref_speed"].values.astype(np.float32)
            # Check if speed is in km/h (median > 10)
            if np.nanmedian(spd) > 10.0:
                spd = spd / 3.6
        elif "gnss_speed" in df.columns and not df["gnss_speed"].isna().all():
            spd = df["gnss_speed"].values.astype(np.float32)
        else:
            raise ValueError("No valid reference speed column ('ref_speed' or 'gnss_speed') found.")

        # Interpolate any isolated NaN values
        if np.isnan(spd).any():
            s = pd.Series(spd)
            spd = s.interpolate(method="linear").bfill().ffill().values

        return spd

    @classmethod
    def create_sliding_windows(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        window_size: int = 50,
        step_size: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates sliding window sequences (X, y).

        Args:
            features: 2D array of shape (N, C), e.g. (51746, 6)
            targets: 1D array of shape (N,), forward speed in m/s
            window_size: Number of timesteps per sequence (e.g. 50 samples = 5.0s @ 10 Hz)
            step_size: Stride between successive windows (1 = dense overlap)

        Returns:
            X: Array of shape (num_windows, window_size, C)
            y: Array of shape (num_windows, 1) target speed at the window endpoint
        """
        n_samples, n_channels = features.shape
        if n_samples < window_size:
            raise ValueError(f"Dataset length ({n_samples}) is shorter than window_size ({window_size})")

        num_windows = (n_samples - window_size) // step_size + 1

        X = np.empty((num_windows, window_size, n_channels), dtype=np.float32)
        y = np.empty((num_windows, 1), dtype=np.float32)

        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            X[i] = features[start:end]
            y[i] = targets[end - 1]  # Target is the speed at the current timestep (window end)

        return X, y

    @classmethod
    def partition_trip_dataset(
        cls,
        df: pd.DataFrame,
        window_size: int = 50,
        step_size: int = 1,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, IMUScaler]:
        """
        Splits a trip DataFrame chronologically into train, validation, and test splits
        per MVP.md §4 without data leakage, fits the scaler on the train split only,
        and returns scaled sliding windows.

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
        """
        features, _ = cls.extract_features(df)
        targets = cls.extract_target_velocity(df)

        n_samples = len(df)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * (train_ratio + val_ratio))

        # Split raw sequences
        feat_train = features[:n_train]
        targ_train = targets[:n_train]

        feat_val = features[n_train:n_val]
        targ_val = targets[n_train:n_val]

        feat_test = features[n_val:]
        targ_test = targets[n_val:]

        # Fit scaler ONLY on train split
        scaler = IMUScaler()
        scaler.fit(feat_train)

        # Transform raw features
        feat_train_scaled = scaler.transform(feat_train)
        feat_val_scaled = scaler.transform(feat_val)
        feat_test_scaled = scaler.transform(feat_test)

        # Build sliding windows for each partition
        X_train, y_train = cls.create_sliding_windows(feat_train_scaled, targ_train, window_size, step_size)
        X_val, y_val = cls.create_sliding_windows(feat_val_scaled, targ_val, window_size, step_size)
        X_test, y_test = cls.create_sliding_windows(feat_test_scaled, targ_test, window_size, step_size)

        return X_train, y_train, X_val, y_val, X_test, y_test, scaler
