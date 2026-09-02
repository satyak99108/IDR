"""
IDR MVP -- INS / Dead Reckoning Module
======================================
Provides strapdown INS components for Phase 3 baseline dead reckoning
and AI-assisted dead reckoning with NHC for Phase 7.

Exports:
    StrapdownINS    -- Full strapdown INS integration pipeline (Phase 3)
    AIAssistedINS   -- AI velocity + gyro heading INS (Phase 7)
    NHCCorrector    -- Non-Holonomic Constraint velocity corrector (Phase 7)
    INSState        -- Immutable navigation state dataclass
"""

from .strapdown import StrapdownINS, INSState
from .ai_assisted import AIAssistedINS
from .nhc import NHCCorrector
from .integration import (
    quaternion_multiply,
    quaternion_from_euler,
    quaternion_to_rotation_matrix,
    normalize_quaternion,
    angular_velocity_to_quaternion_rate,
    latlon_update_ned,
)

__all__ = [
    "StrapdownINS",
    "AIAssistedINS",
    "NHCCorrector",
    "INSState",
    "quaternion_multiply",
    "quaternion_from_euler",
    "quaternion_to_rotation_matrix",
    "normalize_quaternion",
    "angular_velocity_to_quaternion_rate",
    "latlon_update_ned",
]
