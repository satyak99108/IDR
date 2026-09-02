"""
IDR MVP -- GNSS + INS Fusion Package
======================================
Implements EKF-based GNSS + INS state fusion using FilterPy per MVP.md §12
and TECH_STACK.md §9.

Couples high-rate IMU dead-reckoning with GNSS position/velocity measurements
during normal signal availability, applies AI forward velocity and Non-Holonomic
Constraints (NHC) during outages/tunnels, and provides smooth state recovery
when GNSS re-emerges.

Exports:
    NavigationEKF         -- 10-state Extended Kalman Filter
    GNSSINSFusionEngine   -- High-level coordinator managing frames, rates, and modes
    FusionState           -- Navigation state container
    NavigationMode        -- Enum representing GNSS_INS, DEAD_RECKONING, and RECOVERY
"""

from .ekf import NavigationEKF
from .fusion_engine import GNSSINSFusionEngine, FusionState, NavigationMode

__all__ = [
    "NavigationEKF",
    "GNSSINSFusionEngine",
    "FusionState",
    "NavigationMode",
]
