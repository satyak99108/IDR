"""
IDR MVP -- Motion / Vibration Filtering Package
=================================================
Detects non-navigation motion events (stationary, shocks, vibration, abnormal
phone motion) to prevent corrupted IMU samples from degrading the position
estimate (MVP.md §10, TECH_STACK.md §4/§6).

Pipeline:
    Raw IMU
       ↓
    MotionFeatureExtractor  (sliding-window signal features via NumPy + SciPy)
       ↓
    MotionClassifier        (threshold-based labelling + trust weights)
       ↓
    Navigation Pipeline     (downstream INS / fusion uses trust weights)

Exports:
    MotionFeatureExtractor  -- Computes signal-processing features per window.
    MotionClassifier        -- Labels windows and provides trust weights.
    MotionLabel             -- Enum of motion categories.
"""

from .feature_extractor import MotionFeatureExtractor
from .classifier import MotionClassifier, MotionLabel

__all__ = [
    "MotionFeatureExtractor",
    "MotionClassifier",
    "MotionLabel",
]
