"""
IDR MVP -- Motion Feature Extractor
=====================================
Computes sliding-window signal-processing features from 6-axis IMU data for
motion/vibration classification (MVP.md §10, TECH_STACK.md §6).

Features per window:
    ┌──────────────────────┬──────────────────────────────────────────────────┐
    │ Feature              │ Description                                      │
    ├──────────────────────┼──────────────────────────────────────────────────┤
    │ accel_mean           │ Mean accelerometer magnitude                     │
    │ accel_var            │ Variance of accelerometer magnitude              │
    │ accel_rms            │ RMS of accelerometer magnitude                   │
    │ accel_peak           │ Peak (max absolute) acceleration in window       │
    │ gyro_rms             │ RMS of gyroscope magnitude                       │
    │ gyro_peak            │ Peak gyroscope magnitude in window               │
    │ jerk_rms             │ RMS of acceleration derivative (jerk)            │
    │ freq_energy_ratio    │ High-frequency energy ratio (>2 Hz / total)      │
    │ spectral_entropy     │ Normalized Shannon entropy of the PSD            │
    └──────────────────────┴──────────────────────────────────────────────────┘

All computations use NumPy + SciPy only (no ML dependencies).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sig


# ---------------------------------------------------------------------------
# Feature name constants
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "accel_mean",
    "accel_var",
    "accel_rms",
    "accel_peak",
    "gyro_rms",
    "gyro_peak",
    "jerk_rms",
    "freq_energy_ratio",
    "spectral_entropy",
]


class MotionFeatureExtractor:
    """
    Extracts motion/vibration features from 6-axis IMU signals using a sliding
    window.  All features are computed with NumPy + SciPy per TECH_STACK.md §6.

    Parameters
    ----------
    sampling_rate_hz : float
        IMU sampling frequency (default 10 Hz for IO-VNBD smartphone data).
    window_size_sec : float
        Duration of the sliding feature window in seconds (default 1.0 s).
    step_size_sec : float
        Stride between successive windows in seconds (default 0.5 s for 50 %
        overlap).  Set equal to ``window_size_sec`` for non-overlapping windows.
    high_freq_cutoff_hz : float
        Boundary between "low" and "high" frequency bands for
        ``freq_energy_ratio`` (default 2.0 Hz).
    """

    def __init__(
        self,
        sampling_rate_hz: float = 10.0,
        window_size_sec: float = 1.0,
        step_size_sec: float = 0.5,
        high_freq_cutoff_hz: float = 2.0,
    ):
        self.fs = sampling_rate_hz
        self.window_samples = max(2, int(round(window_size_sec * sampling_rate_hz)))
        self.step_samples = max(1, int(round(step_size_sec * sampling_rate_hz)))
        self.high_freq_cutoff = high_freq_cutoff_hz

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        accel_cols: Tuple[str, ...] = ("ax_veh_lin", "ay_veh_lin", "az_veh_lin"),
        gyro_cols: Tuple[str, ...] = ("gx_veh", "gy_veh", "gz_veh"),
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Compute motion features over the full DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Calibrated IMU data with at least 3 accel and 3 gyro columns.
        accel_cols : tuple of str
            Column names for accelerometer axes (x, y, z).
        gyro_cols : tuple of str
            Column names for gyroscope axes (x, y, z).
        timestamp_col : str
            Column used for timestamp alignment.

        Returns
        -------
        pd.DataFrame
            One row per window with columns from ``FEATURE_NAMES`` plus
            ``window_start_idx``, ``window_end_idx``, and ``timestamp`` (centre
            of window).
        """
        # Resolve column names with fallback
        accel_cols = self._resolve_columns(df, accel_cols, ("ax_veh", "ay_veh", "az_veh"), ("ax", "ay", "az"))
        gyro_cols = self._resolve_columns(df, gyro_cols, ("gx_veh", "gy_veh", "gz_veh"), ("gx", "gy", "gz"))

        accel = df[list(accel_cols)].values.astype(np.float64)
        gyro = df[list(gyro_cols)].values.astype(np.float64)

        timestamps = df[timestamp_col].values if timestamp_col in df.columns else np.arange(len(df))

        return self.extract(accel, gyro, timestamps)

    def extract(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compute motion features from raw NumPy arrays.

        Parameters
        ----------
        accel : ndarray of shape (N, 3)
            Accelerometer readings [ax, ay, az] in m/s².
        gyro : ndarray of shape (N, 3)
            Gyroscope readings [gx, gy, gz] in rad/s.
        timestamps : ndarray of shape (N,), optional
            Timestamps for each sample.

        Returns
        -------
        pd.DataFrame
            Feature DataFrame with ``FEATURE_NAMES`` columns.
        """
        n_samples = len(accel)
        if n_samples < self.window_samples:
            raise ValueError(
                f"Data length ({n_samples}) is shorter than window size "
                f"({self.window_samples} samples)"
            )

        # Pre-compute magnitudes
        accel_mag = np.sqrt(np.sum(accel ** 2, axis=1))
        gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=1))

        if timestamps is None:
            timestamps = np.arange(n_samples)

        # Compute windows
        window_starts = np.arange(0, n_samples - self.window_samples + 1, self.step_samples)
        n_windows = len(window_starts)

        # Pre-allocate feature matrix
        features = np.empty((n_windows, len(FEATURE_NAMES)), dtype=np.float64)
        meta_start = np.empty(n_windows, dtype=np.int64)
        meta_end = np.empty(n_windows, dtype=np.int64)
        meta_ts = np.empty(n_windows, dtype=np.float64)

        for i, start in enumerate(window_starts):
            end = start + self.window_samples
            a_win = accel_mag[start:end]
            g_win = gyro_mag[start:end]
            a_raw = accel[start:end]  # (W, 3) for jerk

            features[i] = self._compute_window_features(a_win, g_win, a_raw)
            meta_start[i] = start
            meta_end[i] = end - 1
            # Centre timestamp
            mid = start + self.window_samples // 2
            meta_ts[i] = float(timestamps[mid])

        result = pd.DataFrame(features, columns=FEATURE_NAMES)
        result["window_start_idx"] = meta_start
        result["window_end_idx"] = meta_end
        result["timestamp"] = meta_ts

        return result

    # ------------------------------------------------------------------
    # Per-window feature computation
    # ------------------------------------------------------------------

    def _compute_window_features(
        self,
        accel_mag: np.ndarray,
        gyro_mag: np.ndarray,
        accel_3d: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the full feature vector for a single window.

        Returns
        -------
        ndarray of shape (9,) — one value per feature in ``FEATURE_NAMES``.
        """
        w = len(accel_mag)

        # --- Acceleration features ---
        accel_mean = np.mean(accel_mag)
        accel_var = np.var(accel_mag)
        accel_rms = np.sqrt(np.mean(accel_mag ** 2))
        accel_peak = np.max(np.abs(accel_mag))

        # --- Gyroscope features ---
        gyro_rms = np.sqrt(np.mean(gyro_mag ** 2))
        gyro_peak = np.max(np.abs(gyro_mag))

        # --- Jerk (derivative of acceleration magnitude) ---
        if w >= 2:
            dt = 1.0 / self.fs
            jerk = np.diff(accel_mag) / dt
            jerk_rms = np.sqrt(np.mean(jerk ** 2))
        else:
            jerk_rms = 0.0

        # --- Frequency-domain features ---
        freq_energy_ratio, sp_entropy = self._spectral_features(accel_mag)

        return np.array([
            accel_mean,
            accel_var,
            accel_rms,
            accel_peak,
            gyro_rms,
            gyro_peak,
            jerk_rms,
            freq_energy_ratio,
            sp_entropy,
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Spectral helpers (SciPy)
    # ------------------------------------------------------------------

    def _spectral_features(self, signal_1d: np.ndarray) -> Tuple[float, float]:
        """
        Compute frequency energy ratio and spectral entropy in a single Welch PSD pass.

        Returns
        -------
        (freq_energy_ratio, spectral_entropy)
        """
        w = len(signal_1d)
        nperseg = min(w, max(4, w))  # At least 4 points

        try:
            freqs, psd = sig.welch(
                signal_1d,
                fs=self.fs,
                nperseg=nperseg,
                noverlap=nperseg // 2,
                detrend="constant",
            )
        except ValueError:
            return 0.0, 0.0

        total_energy = np.sum(psd)
        if total_energy < 1e-12:
            return 0.0, 0.0

        # Frequency energy ratio
        high_mask = freqs >= self.high_freq_cutoff
        high_energy = np.sum(psd[high_mask])
        ratio = float(high_energy / total_energy)

        # Spectral entropy
        p = psd / total_energy
        p = p[p > 0]
        entropy = -np.sum(p * np.log2(p))
        max_entropy = np.log2(len(psd)) if len(psd) > 1 else 1.0
        sp_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        return ratio, sp_entropy

    def _freq_energy_ratio(self, signal_1d: np.ndarray) -> float:
        """Ratio of energy above ``high_freq_cutoff`` to total energy."""
        return self._spectral_features(signal_1d)[0]

    def _spectral_entropy(self, signal_1d: np.ndarray) -> float:
        """Normalized Shannon entropy of the power spectral density."""
        return self._spectral_features(signal_1d)[1]

    # ------------------------------------------------------------------
    # Column resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_columns(
        df: pd.DataFrame,
        primary: Tuple[str, ...],
        *fallbacks: Tuple[str, ...],
    ) -> Tuple[str, ...]:
        """Try primary column names first, then each fallback set in order."""
        if all(c in df.columns for c in primary):
            return primary
        for fb in fallbacks:
            if all(c in df.columns for c in fb):
                return fb
        raise ValueError(
            f"Cannot find IMU columns. Tried: {primary} and fallbacks. "
            f"Available: {df.columns.tolist()[:15]}"
        )
