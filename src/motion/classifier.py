"""
IDR MVP -- Motion / Vibration Classifier
==========================================
Classifies IMU windows into navigation-relevant motion categories and provides
trust weights for downstream INS / fusion stages (MVP.md §10).

Classification flow:
    Feature Vector (from MotionFeatureExtractor)
        ↓
    Threshold-based rules  (default — no training required)
        ↓
    MotionLabel
        ↓
    Trust Weight (0.0 – 1.0)
        ↓
    Navigation Pipeline

Labels:
    NORMAL      (0) — Normal driving, trust IMU fully.
    STATIONARY  (1) — Vehicle idle/stopped, apply zero-velocity update.
    SHOCK       (2) — Sudden impact / pothole, suppress IMU contribution.
    VIBRATION   (3) — Sustained high-frequency vibration, reduce trust.
    ABNORMAL    (4) — Unclassified abnormal phone motion.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Motion Label Enum
# ---------------------------------------------------------------------------

class MotionLabel(enum.IntEnum):
    """Navigation-relevant motion categories."""
    NORMAL     = 0   # Normal driving — trust IMU fully
    STATIONARY = 1   # Vehicle idle/stopped — zero-velocity update
    SHOCK      = 2   # Sudden impact / pothole
    VIBRATION  = 3   # Sustained high-frequency vibration
    ABNORMAL   = 4   # Unclassified abnormal phone motion


# ---------------------------------------------------------------------------
# Configurable Thresholds
# ---------------------------------------------------------------------------

@dataclass
class MotionThresholds:
    """
    Configurable thresholds for the rule-based classifier.

    Defaults are tuned for consumer-grade smartphone IMU at 10 Hz (IO-VNBD).
    Adjust after inspecting real dataset distributions.
    """
    # --- Stationary detection ---
    stationary_accel_var_max: float = 0.05          # m²/s⁴  — very low accel variance
    stationary_gyro_rms_max: float = 0.05           # rad/s  — near-zero rotation
    stationary_accel_mean_min: float = 9.0           # m/s²   — should be ~9.81 (gravity)
    stationary_accel_mean_max: float = 10.5          # m/s²   — upper bound for gravity

    # --- Shock / pothole detection ---
    shock_jerk_rms_min: float = 30.0                 # m/s³   — sharp acceleration derivative
    shock_accel_peak_min: float = 18.0               # m/s²   — ~1.8 g spike

    # --- Vibration detection ---
    vibration_freq_energy_ratio_min: float = 0.55    # >55% energy above cutoff
    vibration_spectral_entropy_min: float = 0.80     # broadband noise signature
    vibration_accel_var_min: float = 2.0              # m²/s⁴  — sustained high variance

    # --- Abnormal phone motion ---
    abnormal_gyro_peak_min: float = 3.0              # rad/s   — ~172 deg/s
    abnormal_accel_peak_min: float = 25.0            # m/s²   — ~2.5 g

    # --- Trust weights per label ---
    trust_weights: Dict[int, float] = field(default_factory=lambda: {
        MotionLabel.NORMAL:     1.0,
        MotionLabel.STATIONARY: 1.0,   # Stationary is trusted (ZUPT)
        MotionLabel.SHOCK:      0.1,   # Strongly suppress during shocks
        MotionLabel.VIBRATION:  0.4,   # Partially trust during vibration
        MotionLabel.ABNORMAL:   0.0,   # Do not trust at all
    })


# ---------------------------------------------------------------------------
# Motion Classifier
# ---------------------------------------------------------------------------

class MotionClassifier:
    """
    Rule-based motion classifier for IMU feature windows.

    Uses configurable thresholds on signal-processing features to label each
    window.  Priority order (first match wins):

        1. ABNORMAL   — extreme gyro/accel peaks (phone being handled)
        2. SHOCK      — high jerk + accel spike (pothole / bump)
        3. VIBRATION  — high-frequency energy + broadband entropy
        4. STATIONARY — low variance + gravity-only accel
        5. NORMAL     — default
    """

    def __init__(self, thresholds: Optional[MotionThresholds] = None):
        self.thresholds = thresholds or MotionThresholds()

    # ------------------------------------------------------------------
    # Single-window classification
    # ------------------------------------------------------------------

    def classify_window(self, features: Dict[str, float]) -> MotionLabel:
        """
        Classify a single feature window.

        Parameters
        ----------
        features : dict
            Feature dictionary with keys from ``FEATURE_NAMES``
            (accel_mean, accel_var, accel_rms, accel_peak, gyro_rms,
             gyro_peak, jerk_rms, freq_energy_ratio, spectral_entropy).

        Returns
        -------
        MotionLabel
        """
        t = self.thresholds

        accel_var = features.get("accel_var", 0.0)
        accel_mean = features.get("accel_mean", 9.81)
        accel_peak = features.get("accel_peak", 0.0)
        gyro_rms = features.get("gyro_rms", 0.0)
        gyro_peak = features.get("gyro_peak", 0.0)
        jerk_rms = features.get("jerk_rms", 0.0)
        freq_energy = features.get("freq_energy_ratio", 0.0)
        sp_entropy = features.get("spectral_entropy", 0.0)

        # Priority 1: ABNORMAL — extreme phone motion
        if gyro_peak >= t.abnormal_gyro_peak_min and accel_peak >= t.abnormal_accel_peak_min:
            return MotionLabel.ABNORMAL

        # Priority 2: SHOCK — sudden impact / pothole
        if jerk_rms >= t.shock_jerk_rms_min and accel_peak >= t.shock_accel_peak_min:
            return MotionLabel.SHOCK

        # Priority 3: VIBRATION — sustained high-frequency noise
        if (freq_energy >= t.vibration_freq_energy_ratio_min
                and sp_entropy >= t.vibration_spectral_entropy_min
                and accel_var >= t.vibration_accel_var_min):
            return MotionLabel.VIBRATION

        # Priority 4: STATIONARY — vehicle idle
        if (accel_var <= t.stationary_accel_var_max
                and gyro_rms <= t.stationary_gyro_rms_max
                and t.stationary_accel_mean_min <= accel_mean <= t.stationary_accel_mean_max):
            return MotionLabel.STATIONARY

        # Default: NORMAL driving
        return MotionLabel.NORMAL

    # ------------------------------------------------------------------
    # Bulk classification
    # ------------------------------------------------------------------

    def classify_feature_dataframe(self, feature_df: pd.DataFrame) -> pd.Series:
        """
        Classify every row of a feature DataFrame (as produced by
        ``MotionFeatureExtractor.extract``).

        Returns
        -------
        pd.Series of MotionLabel (int-valued), aligned to ``feature_df.index``.
        """
        labels = np.empty(len(feature_df), dtype=np.int32)

        for i, (_, row) in enumerate(feature_df.iterrows()):
            labels[i] = int(self.classify_window(row.to_dict()))

        return pd.Series(labels, index=feature_df.index, name="motion_label")

    def label_source_dataframe(
        self,
        source_df: pd.DataFrame,
        feature_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Propagate per-window labels back to every sample in the source
        DataFrame.

        Each source sample receives the label of the window whose centre is
        closest.  Samples outside all windows receive ``MotionLabel.NORMAL``.

        Parameters
        ----------
        source_df : pd.DataFrame
            Original IMU DataFrame (N rows).
        feature_df : pd.DataFrame
            Feature DataFrame with ``window_start_idx`` and ``window_end_idx``.

        Returns
        -------
        pd.Series of MotionLabel, same length as ``source_df``.
        """
        n = len(source_df)
        sample_labels = np.full(n, int(MotionLabel.NORMAL), dtype=np.int32)

        window_labels = self.classify_feature_dataframe(feature_df)

        for i, (_, row) in enumerate(feature_df.iterrows()):
            start = int(row["window_start_idx"])
            end = int(row["window_end_idx"]) + 1  # inclusive → exclusive
            end = min(end, n)
            label = int(window_labels.iloc[i])
            # Higher-priority (larger enum value for non-NORMAL) overwrites
            # lower-priority labels.  NORMAL (0) never overwrites anything.
            if label != int(MotionLabel.NORMAL):
                mask = sample_labels[start:end] < label
                sample_labels[start:end] = np.where(mask, label, sample_labels[start:end])

        return pd.Series(sample_labels, index=source_df.index, name="motion_label")

    # ------------------------------------------------------------------
    # Trust weight API
    # ------------------------------------------------------------------

    def get_trust_weight(self, label: MotionLabel) -> float:
        """
        Return a 0.0–1.0 multiplier indicating how much the navigation
        pipeline should trust IMU measurements during this motion state.

        NORMAL / STATIONARY → 1.0 (full trust)
        VIBRATION           → 0.4 (partial)
        SHOCK               → 0.1 (strongly suppressed)
        ABNORMAL            → 0.0 (ignore)
        """
        return self.thresholds.trust_weights.get(int(label), 0.5)

    def get_trust_weights_series(self, labels: pd.Series) -> pd.Series:
        """Map a Series of MotionLabel ints to trust weights."""
        weights = labels.map(
            lambda lbl: self.thresholds.trust_weights.get(int(lbl), 0.5)
        )
        return weights.rename("trust_weight")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def label_distribution(labels: pd.Series) -> Dict[str, int]:
        """
        Return a dict mapping label name → count.
        """
        dist = {}
        for lbl in MotionLabel:
            count = int((labels == int(lbl)).sum())
            dist[lbl.name] = count
        return dist

    @staticmethod
    def label_distribution_pct(labels: pd.Series) -> Dict[str, float]:
        """
        Return a dict mapping label name → percentage (0–100).
        """
        total = len(labels)
        if total == 0:
            return {lbl.name: 0.0 for lbl in MotionLabel}
        dist = {}
        for lbl in MotionLabel:
            count = int((labels == int(lbl)).sum())
            dist[lbl.name] = round(100.0 * count / total, 2)
        return dist
