"""
IDR MVP -- Multi-Rate GNSS + INS Fusion Engine
================================================
Coordinates the NavigationEKF across varying sensor sample rates, manages
WGS-84 geodetic transformations, handles outage / tunnel transitions, and
produces unified navigation state estimates (MVP.md §12).

Modes:
    GNSS_INS       -- Normal operating state with full GNSS + IMU fusion.
    DEAD_RECKONING -- GNSS outage / tunnel blackout; driven by IMU + AI + NHC.
    RECOVERY       -- Smooth transition phase upon GNSS signal re-acquisition.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd

from .ekf import NavigationEKF
from .mode_manager import NavigationMode, NavigationModeManager
from src.ins.integration import earth_radii, haversine_distance, latlon_update_ned


@dataclass
class FusionState:
    """Represents a unified navigation state snapshot."""
    timestamp_ms: int
    lat_deg: float
    lon_deg: float
    alt_m: float
    v_north: float
    v_east: float
    v_down: float
    speed_ms: float
    heading_deg: float
    mode: str
    confidence: float
    p_n: float
    p_e: float
    p_d: float
    b_ax: float
    b_ay: float
    b_gz: float
    cov_trace: float
    dn_smooth: float = 0.0
    de_smooth: float = 0.0
    blend_weight: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "lat_deg": self.lat_deg,
            "lon_deg": self.lon_deg,
            "alt_m": self.alt_m,
            "v_north": self.v_north,
            "v_east": self.v_east,
            "v_down": self.v_down,
            "speed_ms": self.speed_ms,
            "heading_deg": self.heading_deg,
            "mode": self.mode,
            "confidence": self.confidence,
            "p_n": self.p_n,
            "p_e": self.p_e,
            "p_d": self.p_d,
            "b_ax": self.b_ax,
            "b_ay": self.b_ay,
            "b_gz": self.b_gz,
            "cov_trace": self.cov_trace,
            "dn_smooth": self.dn_smooth,
            "de_smooth": self.de_smooth,
            "blend_weight": self.blend_weight,
        }


class GNSSINSFusionEngine:
    """
    High-level GNSS + INS fusion engine.

    Manages coordinate transformations, multi-rate updates, and state
    machine transitions between normal GNSS, simulated blackout tunnels,
    and post-blackout recovery via NavigationModeManager.
    """

    def __init__(
        self,
        ekf: Optional[NavigationEKF] = None,
        mode_manager: Optional[NavigationModeManager] = None,
        recovery_window_s: float = 3.0,
        dropout_debounce_samples: int = 5,
        recovery_confirm_samples: int = 5,
    ):
        self.ekf = ekf or NavigationEKF()
        self.recovery_window_s = recovery_window_s
        self.mode_manager = mode_manager or NavigationModeManager(
            dropout_debounce_samples=dropout_debounce_samples,
            recovery_confirm_samples=recovery_confirm_samples,
            recovery_duration_s=recovery_window_s,
        )

        # Geodetic reference origin
        self.origin_lat_rad: Optional[float] = None
        self.origin_lon_rad: Optional[float] = None
        self.origin_alt_m: float = 0.0

        # Current navigation state
        self.mode = NavigationMode.GNSS_INS
        self._curr_lat_deg: Optional[float] = None
        self._curr_lon_deg: Optional[float] = None

    # ------------------------------------------------------------------
    # Geodetic <-> Local NED Coordinate Conversion
    # ------------------------------------------------------------------

    def set_origin(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> None:
        """Sets the tangent-plane local NED origin."""
        self.origin_lat_rad = np.radians(lat_deg)
        self.origin_lon_rad = np.radians(lon_deg)
        self.origin_alt_m = alt_m
        self._curr_lat_deg = lat_deg
        self._curr_lon_deg = lon_deg

    def geodetic_to_ned(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Converts geodetic lat/lon/alt to local NED (m)."""
        if self.origin_lat_rad is None or self.origin_lon_rad is None:
            self.set_origin(lat_deg, lon_deg, alt_m)
            return 0.0, 0.0, 0.0

        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)

        N, M = earth_radii(self.origin_lat_rad)

        dlat = lat_rad - self.origin_lat_rad
        dlon = lon_rad - self.origin_lon_rad

        p_n = dlat * M
        p_e = dlon * (N * np.cos(self.origin_lat_rad))
        p_d = -(alt_m - self.origin_alt_m)

        return float(p_n), float(p_e), float(p_d)

    def ned_to_geodetic(
        self,
        p_n: float,
        p_e: float,
        p_d: float,
    ) -> Tuple[float, float, float]:
        """Converts local NED (m) back to geodetic lat/lon/alt."""
        if self.origin_lat_rad is None or self.origin_lon_rad is None:
            raise RuntimeError("Origin not set. Call set_origin() first.")

        N, M = earth_radii(self.origin_lat_rad)

        dlat = p_n / M
        dlon = p_e / (N * np.cos(self.origin_lat_rad))

        lat_rad = self.origin_lat_rad + dlat
        lon_rad = self.origin_lon_rad + dlon
        alt_m = self.origin_alt_m - p_d

        return float(np.degrees(lat_rad)), float(np.degrees(lon_rad)), float(alt_m)

    # ------------------------------------------------------------------
    # Single-step update
    # ------------------------------------------------------------------

    def step(
        self,
        dt: float,
        ax_lin: float,
        ay_lin: float,
        az_lin: float,
        gz_veh: float,
        gx_veh: float = 0.0,
        gy_veh: float = 0.0,
        gnss_lat: Optional[float] = None,
        gnss_lon: Optional[float] = None,
        gnss_alt: Optional[float] = None,
        gnss_speed: Optional[float] = None,
        gnss_course_deg: Optional[float] = None,
        gnss_accuracy: Optional[float] = None,
        satellites: Optional[int] = None,
        ai_speed: Optional[float] = None,
        is_outage: bool = False,
        timestamp_ms: int = 0,
    ) -> FusionState:
        """
        Executes one prediction and optional measurement updates.
        """
        timestamp_s = timestamp_ms / 1000.0

        # 1. Mode State Machine Update
        has_gnss_sample = (
            gnss_lat is not None
            and gnss_lon is not None
            and not (np.isnan(gnss_lat) or np.isnan(gnss_lon))
        )
        curr_ekf_pos = self.ekf.position_ned
        self.mode, telemetry = self.mode_manager.update(
            timestamp_s=timestamp_s,
            has_gnss_sample=has_gnss_sample,
            is_simulated_outage=is_outage,
            gnss_accuracy=gnss_accuracy,
            satellites=satellites,
            current_ekf_pos_ned=curr_ekf_pos,
        )

        # 2. Prediction Step (IMU-driven in all modes for proper Jacobian/covariance propagation)
        self.ekf.predict_imu(
            dt=dt,
            ax_lin=ax_lin,
            ay_lin=ay_lin,
            az_lin=az_lin,
            gz_veh=gz_veh,
            gx_veh=gx_veh,
            gy_veh=gy_veh,
        )

        # 3. Measurement Updates
        has_gnss_fix = has_gnss_sample and not is_outage

        if self.mode in (NavigationMode.GNSS_INS, NavigationMode.DEGRADED_GNSS, NavigationMode.RECOVERY) and has_gnss_fix:
            # Convert GNSS to NED
            alt = gnss_alt if (gnss_alt is not None and not np.isnan(gnss_alt)) else 0.0
            p_n, p_e, p_d = self.geodetic_to_ned(gnss_lat, gnss_lon, alt)

            # Compute GNSS velocity components from speed and course if available
            spd = gnss_speed if (gnss_speed is not None and not np.isnan(gnss_speed)) else self.ekf.speed_ms
            crs_rad = (
                np.radians(gnss_course_deg)
                if (gnss_course_deg is not None and not np.isnan(gnss_course_deg))
                else self.ekf.heading_rad
            )
            v_n = spd * np.cos(crs_rad)
            v_e = spd * np.sin(crs_rad)

            # Adaptive pos_std from ModeManager (smoothly scales during recovery)
            pos_std = float(telemetry.get("ekf_pos_std", 3.0))
            vel_std = 0.3 if self.mode == NavigationMode.GNSS_INS else 0.5

            # GNSS position + velocity update
            self.ekf.update_gnss_pv(
                p_n_meas=p_n,
                p_e_meas=p_e,
                p_d_meas=p_d,
                v_n_meas=v_n,
                v_e_meas=v_e,
                pos_std=pos_std,
                vel_std=vel_std,
            )

            # GNSS course update if vehicle is moving
            if spd > 1.5 and gnss_course_deg is not None and not np.isnan(gnss_course_deg):
                self.ekf.update_gnss_heading(crs_rad, heading_std_rad=np.radians(4.0))

            # Update tracked geodetic coordinates
            p_n_curr, p_e_curr, p_d_curr = self.ekf.position_ned
            lat_curr, lon_curr, alt_curr = self.ned_to_geodetic(p_n_curr, p_e_curr, p_d_curr)
            self._curr_lat_deg = lat_curr
            self._curr_lon_deg = lon_curr
        else:
            # During blackout (DEAD_RECKONING):
            # Apply AI velocity as a proper EKF measurement update (not direct override)
            # This lets the Kalman gain weight AI prediction vs propagated uncertainty
            if ai_speed is not None and not np.isnan(ai_speed):
                self.ekf.update_ai_velocity(
                    v_forward_meas=max(0.0, ai_speed),
                    vel_std=0.15,  # Strong weight on AI speed to prevent IMU acceleration bias drag
                )

            # Propagate geodetic position directly to prevent tangent-plane distortion
            vn_curr, ve_curr, _ = self.ekf.velocity_ned
            if self._curr_lat_deg is not None and self._curr_lon_deg is not None:
                lat_rad, lon_rad = latlon_update_ned(
                    np.radians(self._curr_lat_deg),
                    np.radians(self._curr_lon_deg),
                    vn_curr, ve_curr, dt,
                )
                lat_curr = float(np.degrees(lat_rad))
                lon_curr = float(np.degrees(lon_rad))
                self._curr_lat_deg = lat_curr
                self._curr_lon_deg = lon_curr
                alt_curr = self.origin_alt_m
                p_n_curr, p_e_curr, p_d_curr = self.geodetic_to_ned(lat_curr, lon_curr, alt_curr)
                self.ekf.x[0] = p_n_curr
                self.ekf.x[1] = p_e_curr
            else:
                p_n_curr, p_e_curr, p_d_curr = self.ekf.position_ned
                lat_curr, lon_curr, alt_curr = self.ned_to_geodetic(p_n_curr, p_e_curr, p_d_curr)
                self._curr_lat_deg = lat_curr
                self._curr_lon_deg = lon_curr

        # 4. Non-Holonomic Constraints (ALL modes — critical during DR to prevent lateral drift)
        # Rigid lateral constraint during DR: vehicle cannot slide sideways on road
        nhc_lat_std = 0.03 if self.mode == NavigationMode.DEAD_RECKONING else 0.15
        self.ekf.update_nhc(lateral_std=nhc_lat_std, vertical_std=0.20)

        # 5. Extract current fused estimate with smooth jump mitigation applied
        p_n_curr, p_e_curr, p_d_curr = self.ekf.position_ned

        if getattr(self.mode_manager, "just_entered_recovery", False):
            self.mode_manager.init_jump_mitigation(
                current_time_s=timestamp_s,
                new_ekf_pos_ned=(p_n_curr, p_e_curr, p_d_curr),
            )

        dn_smooth, de_smooth, blend_weight = self.mode_manager.get_jump_offsets(timestamp_s)

        # Position output incorporates smooth jump mitigation offset
        p_n_out = p_n_curr + dn_smooth
        p_e_out = p_e_curr + de_smooth
        lat_curr, lon_curr, alt_curr = self.ned_to_geodetic(p_n_out, p_e_out, p_d_curr)

        vn_curr, ve_curr, vd_curr = self.ekf.velocity_ned
        speed_curr = self.ekf.speed_ms
        hdg_curr = self.ekf.heading_deg
        b_ax_curr, b_ay_curr = self.ekf.accel_biases
        b_gz_curr = self.ekf.gyro_bias
        trace = self.ekf.state_uncertainty_trace
        conf = float(telemetry.get("confidence", 90.0))

        return FusionState(
            timestamp_ms=timestamp_ms,
            lat_deg=lat_curr,
            lon_deg=lon_curr,
            alt_m=alt_curr,
            v_north=vn_curr,
            v_east=ve_curr,
            v_down=vd_curr,
            speed_ms=speed_curr,
            heading_deg=hdg_curr,
            mode=self.mode.value,
            confidence=round(conf, 1),
            p_n=p_n_out,
            p_e=p_e_out,
            p_d=p_d_curr,
            b_ax=b_ax_curr,
            b_ay=b_ay_curr,
            b_gz=b_gz_curr,
            cov_trace=trace,
            dn_smooth=dn_smooth,
            de_smooth=de_smooth,
            blend_weight=blend_weight,
        )

    # ------------------------------------------------------------------
    # Batch runner
    # ------------------------------------------------------------------

    def run_batch(
        self,
        df: pd.DataFrame,
        ai_velocity: Optional[np.ndarray] = None,
        outage_mask: Optional[np.ndarray] = None,
        init_lat: Optional[float] = None,
        init_lon: Optional[float] = None,
        init_heading_deg: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Processes a full trip DataFrame through the multi-rate fusion engine.
        """
        n = len(df)
        if n == 0:
            return pd.DataFrame()

        # Resolve initial geodetic fix
        first_lat = init_lat if init_lat is not None else float(df["gnss_lat"].dropna().iloc[0])
        first_lon = init_lon if init_lon is not None else float(df["gnss_lon"].dropna().iloc[0])
        first_alt = float(df.get("ALTITUDE (m)", pd.Series(0.0, index=df.index)).dropna().iloc[0] or 0.0)

        first_hdg = (
            init_heading_deg
            if init_heading_deg is not None
            else float(df.get("ref_heading", df.get("heading", pd.Series(0.0))).dropna().iloc[0] or 0.0)
        )

        first_ts = int(df["timestamp"].iloc[0])

        self.set_origin(first_lat, first_lon, first_alt)
        self.ekf.initialize_state(
            p_n=0.0, p_e=0.0, p_d=0.0,
            v_n=0.0, v_e=0.0, v_d=0.0,
            heading_rad=np.radians(first_hdg),
            timestamp_s=first_ts / 1000.0,
        )

        # Arrays
        timestamps = df["timestamp"].values
        ax_arr = df["ax_veh_lin"].values if "ax_veh_lin" in df.columns else df["ax_veh"].values
        ay_arr = df["ay_veh_lin"].values if "ay_veh_lin" in df.columns else df["ay_veh"].values
        az_arr = df["az_veh_lin"].values if "az_veh_lin" in df.columns else df["az_veh"].values
        gx_arr = df["gx_veh"].values if "gx_veh" in df.columns else np.zeros(n)
        gy_arr = df["gy_veh"].values if "gy_veh" in df.columns else np.zeros(n)
        gz_arr = df["gz_veh"].values if "gz_veh" in df.columns else df["gz"].values

        gnss_lats = df["gnss_lat"].values if "gnss_lat" in df.columns else np.full(n, np.nan)
        gnss_lons = df["gnss_lon"].values if "gnss_lon" in df.columns else np.full(n, np.nan)
        gnss_alts = df["ALTITUDE (m)"].values if "ALTITUDE (m)" in df.columns else np.full(n, np.nan)
        gnss_spds = df["gnss_speed"].values if "gnss_speed" in df.columns else np.full(n, np.nan)
        gnss_courses = (
            df["ref_heading"].values
            if "ref_heading" in df.columns and not df["ref_heading"].isna().all()
            else df["heading"].values
            if "heading" in df.columns
            else np.full(n, np.nan)
        )

        gnss_accuracies = (
            pd.to_numeric(df["_extra_GPS ACCURACY (m)"], errors="coerce").values.astype(np.float64)
            if "_extra_GPS ACCURACY (m)" in df.columns
            else np.full(n, np.nan, dtype=np.float64)
        )
        gnss_sats = (
            pd.to_numeric(df["_extra_GPS SATELLITES IN RANGE"], errors="coerce").values.astype(np.float64)
            if "_extra_GPS SATELLITES IN RANGE" in df.columns
            else np.full(n, np.nan, dtype=np.float64)
        )

        ai_spds = ai_velocity if ai_velocity is not None else np.full(n, np.nan)
        outages = outage_mask if outage_mask is not None else np.zeros(n, dtype=bool)

        records: List[dict] = []
        prev_ts = timestamps[0]

        for i in range(n):
            ts = timestamps[i]
            dt = (ts - prev_ts) / 1000.0 if i > 0 else 0.1
            if dt <= 0.0 or dt > 2.0:
                dt = 0.1

            fstate = self.step(
                dt=dt,
                ax_lin=float(ax_arr[i]),
                ay_lin=float(ay_arr[i]),
                az_lin=float(az_arr[i]),
                gz_veh=float(gz_arr[i]),
                gx_veh=float(gx_arr[i]),
                gy_veh=float(gy_arr[i]),
                gnss_lat=None if outages[i] else float(gnss_lats[i]),
                gnss_lon=None if outages[i] else float(gnss_lons[i]),
                gnss_alt=None if outages[i] else float(gnss_alts[i]),
                gnss_speed=None if outages[i] else float(gnss_spds[i]),
                gnss_course_deg=None if outages[i] else float(gnss_courses[i]),
                gnss_accuracy=float(gnss_accuracies[i]) if not np.isnan(gnss_accuracies[i]) else None,
                satellites=int(gnss_sats[i]) if not np.isnan(gnss_sats[i]) else None,
                ai_speed=float(ai_spds[i]) if not np.isnan(ai_spds[i]) else None,
                is_outage=bool(outages[i]),
                timestamp_ms=int(ts),
            )
            records.append(fstate.to_dict())
            prev_ts = ts

        return pd.DataFrame(records)
