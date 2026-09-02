"""
IDR MVP -- Calibration & Alignment Package
Provides modules for stationary window detection, pitch/roll/yaw alignment,
and coordinate frame transformation (Phone -> Vehicle frame).
"""

from .detector import StaticDetector
from .calibrator import FrameCalibrator
from .transformer import CoordinateTransformer

__all__ = [
    "StaticDetector",
    "FrameCalibrator",
    "CoordinateTransformer",
]
