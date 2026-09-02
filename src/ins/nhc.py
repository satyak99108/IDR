"""
IDR MVP -- Non-Holonomic Constraints (NHC)
============================================
Enforces vehicle motion constraints on the NED velocity vector
(MVP.md §11, TECH_STACK.md §8).

Vehicle constraints:
    Lateral velocity  ≈ 0   (car cannot slide sideways)
    Vertical velocity ≈ 0   (car stays on the road surface)

The NHC operates by:
    1. Rotating NED velocity into the body frame using the current heading.
    2. Zeroing the lateral (v_y_body) and vertical (v_z_body) components.
    3. Rotating back to NED.

This preserves only the forward velocity component, which is the dominant
source of position change for ground vehicles.

Classes:
    NHCCorrector -- Applies NHC to NED velocity given heading.
"""

from __future__ import annotations

import numpy as np


class NHCCorrector:
    """
    Applies Non-Holonomic Constraints to the navigation velocity vector.

    For a ground vehicle:
        - Lateral (sideways) velocity ≈ 0
        - Vertical (up/down) velocity ≈ 0
        - Only forward velocity is retained

    Parameters
    ----------
    lateral_weight : float
        Fraction of lateral velocity to suppress (1.0 = fully zeroed,
        0.0 = no constraint). Default 1.0 for hard NHC.
    vertical_weight : float
        Fraction of vertical velocity to suppress. Default 1.0.
    """

    def __init__(
        self,
        lateral_weight: float = 1.0,
        vertical_weight: float = 1.0,
    ):
        self.lateral_weight = np.clip(lateral_weight, 0.0, 1.0)
        self.vertical_weight = np.clip(vertical_weight, 0.0, 1.0)

    def correct_ned_velocity(
        self,
        v_north: float,
        v_east: float,
        v_down: float,
        heading_rad: float,
    ) -> tuple:
        """
        Apply NHC to a NED velocity vector.

        Parameters
        ----------
        v_north : float
            North velocity (m/s).
        v_east : float
            East velocity (m/s).
        v_down : float
            Down velocity (m/s).
        heading_rad : float
            Vehicle heading in radians (0 = North, CW positive).

        Returns
        -------
        tuple of (v_north_corrected, v_east_corrected, v_down_corrected)
        """
        # --- Step 1: NED → Body rotation ---
        # Body frame: x = forward, y = right, z = down
        # For a level vehicle (roll=0, pitch=0), only yaw matters:
        #   v_forward = v_north * cos(ψ) + v_east * sin(ψ)
        #   v_right   = -v_north * sin(ψ) + v_east * cos(ψ)
        #   v_body_z  = v_down
        cos_h = np.cos(heading_rad)
        sin_h = np.sin(heading_rad)

        v_forward = v_north * cos_h + v_east * sin_h
        v_right = -v_north * sin_h + v_east * cos_h
        v_body_down = v_down

        # --- Step 2: Apply constraints ---
        # Suppress lateral velocity
        v_right_corrected = v_right * (1.0 - self.lateral_weight)
        # Suppress vertical velocity
        v_body_down_corrected = v_body_down * (1.0 - self.vertical_weight)

        # --- Step 3: Body → NED rotation ---
        v_north_new = v_forward * cos_h - v_right_corrected * sin_h
        v_east_new = v_forward * sin_h + v_right_corrected * cos_h
        v_down_new = v_body_down_corrected

        return float(v_north_new), float(v_east_new), float(v_down_new)

    def correction_magnitude(
        self,
        v_north: float,
        v_east: float,
        v_down: float,
        heading_rad: float,
    ) -> float:
        """
        Returns the magnitude of the velocity correction applied by NHC (m/s).
        Useful for diagnostics and plots.
        """
        vn_c, ve_c, vd_c = self.correct_ned_velocity(
            v_north, v_east, v_down, heading_rad
        )
        dv_n = v_north - vn_c
        dv_e = v_east - ve_c
        dv_d = v_down - vd_c
        return float(np.sqrt(dv_n**2 + dv_e**2 + dv_d**2))
