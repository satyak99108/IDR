"""
IDR MVP -- End-to-End Navigation Pipeline Engine (Phase 11)
============================================================
Integrates all subsystems developed in Phases 1-10 into a cohesive, production-
ready prototype navigation pipeline (MVP.md §15 and TECH_STACK.md §2).

Architecture:
    Raw / Preprocessed IMU & GNSS
                ↓
    [1. Calibration & Alignment]   (Frame transformation & gravity removal)
                ↓
    [2. AI Velocity Model]         (1D CNN forward velocity inference)
                ↓
    [3. Motion Classifier]         (Vibration/shock/stationary trust weighting)
                ↓
    [4. INS + NHC Integration]     (Strapdown + lateral/vertical constraints)
                ↓
    [5. Multi-Rate GNSS/INS EKF]   (Navigation Mode Manager & Smooth Jump Mitigator)
                ↓
    [6. Offline OSM Map Matcher]   (HMM road network snapping)
                ↓
    [7. Unified Navigation Output] (Lat, Lon, Speed, Heading, Mode, Confidence)

Supports:
    - Batch Processing:      ``process_trip(df) -> pd.DataFrame``
    - Streaming Step-by-Step: ``step(sensor_input) -> NavigationOutput``
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from src.ai import IMUScaler, NumpyCNNInference
from src.fusion import (
    GNSSINSFusionEngine,
    NavigationMode,
    NavigationModeManager,
    NavigationEKF,
)
from src.map_matching import (
    RoadGraph,
    HMMMapMatcher,
    MapMatchResult,
    MapMatcherEngine,
    apply_map_matching,
)
from src.motion import MotionClassifier, MotionFeatureExtractor, MotionLabel

logger = logging.getLogger(__name__)

_DEFAULT_MODELS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "models"
)


# ---------------------------------------------------------------------------
# Pipeline Configuration & Data Containers
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration options for the End-to-End Navigation Pipeline."""
    models_dir: Optional[Union[str, Path]] = None
    use_ai_velocity: bool = True
    use_map_matching: bool = True
    use_motion_classifier: bool = True
    window_size: int = 50               # 5.0 seconds at 10 Hz
    sampling_rate_hz: float = 10.0
    recovery_window_s: float = 3.0
    dropout_debounce_samples: int = 5
    recovery_confirm_samples: int = 5
    road_search_radius_m: float = 60.0
    map_matching_sigma_pos_m: float = 15.0
    map_matching_sigma_heading_deg: float = 25.0
    map_matching_confidence_threshold: float = 0.25
    cache_dir: Optional[str] = None


@dataclass
class SensorInput:
    """Standardized single-timestep sensor telemetry input (MVP.md §19)."""
    timestamp_ms: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    ax_lin: Optional[float] = None
    ay_lin: Optional[float] = None
    az_lin: Optional[float] = None
    gnss_lat: Optional[float] = None
    gnss_lon: Optional[float] = None
    gnss_alt: Optional[float] = None
    gnss_speed: Optional[float] = None
    gnss_course: Optional[float] = None
    gnss_accuracy: Optional[float] = None
    gnss_sats: Optional[int] = None


@dataclass
class NavigationOutput:
    """Standardized single-timestep navigation state output (MVP.md §15, §19)."""
    timestamp_ms: int
    lat_deg: float
    lon_deg: float
    alt_m: float
    speed_ms: float
    heading_deg: float
    mode: str
    confidence: float
    # Map matching fields
    mm_lat: Optional[float] = None
    mm_lon: Optional[float] = None
    mm_road_id: Optional[str] = None
    mm_road_name: Optional[str] = None
    mm_snap_dist_m: Optional[float] = None
    mm_confidence: Optional[float] = None
    is_map_matched: bool = False
    # Final consolidated position (best estimate: map-snapped if high conf, else fusion)
    final_lat: float = 0.0
    final_lon: float = 0.0
    # Diagnostics & Motion state
    motion_state: str = "NORMAL"
    trust_weight: float = 1.0
    ai_speed_ms: Optional[float] = None
    p_n: float = 0.0
    p_e: float = 0.0
    cov_trace: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "lat_deg": self.lat_deg,
            "lon_deg": self.lon_deg,
            "alt_m": self.alt_m,
            "speed_ms": self.speed_ms,
            "heading_deg": self.heading_deg,
            "mode": self.mode,
            "confidence": self.confidence,
            "mm_lat": self.mm_lat,
            "mm_lon": self.mm_lon,
            "mm_road_id": self.mm_road_id,
            "mm_road_name": self.mm_road_name,
            "mm_snap_dist_m": self.mm_snap_dist_m,
            "mm_confidence": self.mm_confidence,
            "is_map_matched": self.is_map_matched,
            "final_lat": self.final_lat,
            "final_lon": self.final_lon,
            "motion_state": self.motion_state,
            "trust_weight": self.trust_weight,
            "ai_speed_ms": self.ai_speed_ms,
            "p_n": self.p_n,
            "p_e": self.p_e,
            "cov_trace": self.cov_trace,
        }


# ---------------------------------------------------------------------------
# Main End-to-End Pipeline Engine
# ---------------------------------------------------------------------------

class IDRNavigationEngine:
    """
    Intelligent Dead Reckoning (IDR) unified navigation engine.

    Connects preprocessing, calibration, AI velocity prediction, motion
    classification, strapdown INS + NHC, multi-rate GNSS/INS EKF fusion, and
    offline OSM map matching into a unified navigation solution.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        road_graph: Optional[RoadGraph] = None,
    ):
        self.config = config or PipelineConfig()
        self.models_dir = Path(self.config.models_dir or _DEFAULT_MODELS_DIR)

        # 1. AI Velocity Subsystem
        self.ai_scaler: Optional[IMUScaler] = None
        self.ai_model: Optional[NumpyCNNInference] = None
        self._init_ai_model()

        # 2. Motion Classification Subsystem
        window_sec = self.config.window_size / max(1.0, self.config.sampling_rate_hz)
        step_sec = 1.0 / max(1.0, self.config.sampling_rate_hz)
        self.feature_extractor = MotionFeatureExtractor(
            sampling_rate_hz=self.config.sampling_rate_hz,
            window_size_sec=window_sec,
            step_size_sec=step_sec,
        )
        self.motion_classifier = MotionClassifier()

        # 3. Fusion Subsystem
        self.mode_manager = NavigationModeManager(
            dropout_debounce_samples=self.config.dropout_debounce_samples,
            recovery_confirm_samples=self.config.recovery_confirm_samples,
            recovery_duration_s=self.config.recovery_window_s,
        )
        self.fusion_engine = GNSSINSFusionEngine(
            mode_manager=self.mode_manager,
            recovery_window_s=self.config.recovery_window_s,
        )

        # 4. Map Matching Subsystem
        self.road_graph = road_graph
        self.map_matcher: Optional[HMMMapMatcher] = None
        if self.road_graph is not None:
            self._init_map_matcher(self.road_graph)

        # 5. Streaming Buffers
        self._imu_window_buffer = collections.deque(maxlen=self.config.window_size)
        self._prev_timestamp_ms: Optional[int] = None
        self._origin_initialized = False

    def _init_ai_model(self) -> None:
        """Loads the trained 1D CNN forward velocity model and scaler."""
        if not self.config.use_ai_velocity:
            return

        weights_path = self.models_dir / "velocity_cnn_weights.npz"
        scaler_path = self.models_dir / "scaler_params.json"

        if weights_path.exists() and scaler_path.exists():
            try:
                self.ai_scaler = IMUScaler.load(scaler_path)
                self.ai_model = NumpyCNNInference.load_weights_npz(weights_path)
                logger.info("AI Velocity Model loaded successfully from %s", self.models_dir)
            except Exception as e:
                logger.warning("Failed loading AI model from %s: %s. Falling back.", self.models_dir, e)
                self.ai_scaler = None
                self.ai_model = None
        else:
            logger.info("AI model weights/scaler not found in %s; using kinematic estimation.", self.models_dir)

    def _init_map_matcher(self, road_graph: RoadGraph) -> None:
        """Initializes the HMM Map Matcher with the given RoadGraph."""
        self.road_graph = road_graph
        self.map_matcher = HMMMapMatcher(
            road_graph=self.road_graph,
            sigma_pos_m=self.config.map_matching_sigma_pos_m,
            sigma_heading_deg=self.config.map_matching_sigma_heading_deg,
            search_radius_m=self.config.road_search_radius_m,
        )

    def reset(self) -> None:
        """Resets all internal states, filters, and streaming buffers."""
        self._imu_window_buffer.clear()
        self._prev_timestamp_ms = None
        self._origin_initialized = False

        self.mode_manager = NavigationModeManager(
            dropout_debounce_samples=self.config.dropout_debounce_samples,
            recovery_confirm_samples=self.config.recovery_confirm_samples,
            recovery_duration_s=self.config.recovery_window_s,
        )
        self.fusion_engine = GNSSINSFusionEngine(
            mode_manager=self.mode_manager,
            recovery_window_s=self.config.recovery_window_s,
        )
        if self.road_graph is not None:
            self._init_map_matcher(self.road_graph)

    # ------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------

    def process_trip(
        self,
        df: pd.DataFrame,
        road_graph: Optional[RoadGraph] = None,
        outage_mask: Optional[np.ndarray] = None,
        init_lat: Optional[float] = None,
        init_lon: Optional[float] = None,
        init_heading_deg: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Executes the full end-to-end navigation pipeline on a trip DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Sensor telemetry log containing IMU and GNSS columns.
        road_graph : RoadGraph, optional
            OSM road network graph. If None, self.road_graph or auto-loaded graph is used.
        outage_mask : np.ndarray, optional
            Boolean mask where True indicates simulated GNSS outage.
        init_lat, init_lon, init_heading_deg : float, optional
            Initial state overrides.

        Returns
        -------
        pd.DataFrame
            Augmented DataFrame containing:
            ``timestamp_ms``, ``lat_deg``, ``lon_deg``, ``speed_ms``, ``heading_deg``,
            ``mode``, ``confidence``, ``mm_lat``, ``mm_lon``, ``final_lat``, ``final_lon``, etc.
        """
        if df.empty:
            return pd.DataFrame()

        n = len(df)
        working_rg = road_graph or self.road_graph

        # 1. Coordinate & Feature Alignment
        df_clean = df.copy()
        feat_cols = self._resolve_feature_columns(df_clean)

        # 2. AI Forward Velocity Estimation
        ai_velocity = self._predict_ai_velocity_batch(df_clean, feat_cols)

        # 3. Motion Classification & Trust Weighting
        motion_labels, trust_weights = self._classify_motion_batch(df_clean, feat_cols)

        # 4. Multi-Rate GNSS/INS EKF Fusion with Mode Management
        fused_df = self.fusion_engine.run_batch(
            df=df_clean,
            ai_velocity=ai_velocity,
            outage_mask=outage_mask,
            init_lat=init_lat,
            init_lon=init_lon,
            init_heading_deg=init_heading_deg,
        )

        if fused_df.empty:
            logger.warning("Fusion engine returned empty DataFrame.")
            return df_clean

        # Add motion state diagnostic columns
        fused_df["motion_state"] = motion_labels
        fused_df["trust_weight"] = trust_weights
        if ai_velocity is not None:
            fused_df["ai_speed_ms"] = ai_velocity

        # 5. Offline OSM Map Matching
        if self.config.use_map_matching and working_rg is not None:
            result_df = apply_map_matching(
                fusion_df=fused_df,
                road_graph=working_rg,
                sigma_pos_m=self.config.map_matching_sigma_pos_m,
                sigma_heading_deg=self.config.map_matching_sigma_heading_deg,
                search_radius_m=self.config.road_search_radius_m,
                cache_dir=self.config.cache_dir,
            )
        else:
            result_df = fused_df.copy()
            # Create default empty map matching columns
            result_df["mm_lat"] = result_df["lat_deg"]
            result_df["mm_lon"] = result_df["lon_deg"]
            result_df["mm_road_id"] = ""
            result_df["mm_road_name"] = ""
            result_df["mm_snap_dist_m"] = 0.0
            result_df["mm_confidence"] = 0.0
            result_df["mm_is_matched"] = False

        # 6. Final Consolidated Position Selection
        # If map-matched with sufficient confidence, snap to road; otherwise retain EKF fusion
        use_mm = (
            result_df["mm_is_matched"]
            & (result_df["mm_confidence"] >= self.config.map_matching_confidence_threshold)
            & (result_df["mm_snap_dist_m"] <= self.config.road_search_radius_m)
        )
        result_df["final_lat"] = np.where(use_mm, result_df["mm_lat"], result_df["lat_deg"])
        result_df["final_lon"] = np.where(use_mm, result_df["mm_lon"], result_df["lon_deg"])
        result_df["is_map_matched"] = use_mm

        return result_df

    # ------------------------------------------------------------------
    # Streaming Step-by-Step Execution
    # ------------------------------------------------------------------

    def step(self, sensor_input: SensorInput) -> NavigationOutput:
        """
        Processes a single-timestep sensor observation for real-time edge streaming.

        Parameters
        ----------
        sensor_input : SensorInput
            Instantaneous sensor reading.

        Returns
        -------
        NavigationOutput
            Estimated vehicle navigation state.
        """
        ts = sensor_input.timestamp_ms
        if self._prev_timestamp_ms is None:
            dt = 0.1
        else:
            dt = (ts - self._prev_timestamp_ms) / 1000.0
            if dt <= 0.0 or dt > 2.0:
                dt = 0.1
        self._prev_timestamp_ms = ts

        # 1. Feature buffer update
        ax_lin = sensor_input.ax_lin if sensor_input.ax_lin is not None else sensor_input.ax
        ay_lin = sensor_input.ay_lin if sensor_input.ay_lin is not None else sensor_input.ay
        az_lin = sensor_input.az_lin if sensor_input.az_lin is not None else sensor_input.az
        imu_feat = np.array([ax_lin, ay_lin, az_lin, sensor_input.gx, sensor_input.gy, sensor_input.gz], dtype=np.float32)
        self._imu_window_buffer.append(imu_feat)

        # 2. AI Velocity Inference (if window is ready)
        ai_speed: Optional[float] = None
        if self.ai_model is not None and self.ai_scaler is not None and len(self._imu_window_buffer) >= self.config.window_size:
            window_arr = np.array(self._imu_window_buffer, dtype=np.float32)  # (W, 6)
            scaled_window = self.ai_scaler.transform(window_arr)
            ai_speed = float(max(0.0, self.ai_model.predict(scaled_window)))
        elif sensor_input.gnss_speed is not None and not np.isnan(sensor_input.gnss_speed):
            ai_speed = float(sensor_input.gnss_speed)

        # 3. Motion state
        motion_state = "NORMAL"
        trust_weight = 1.0
        if self.config.use_motion_classifier and len(self._imu_window_buffer) >= self.config.window_size:
            w_arr = np.array(self._imu_window_buffer, dtype=np.float64)
            # Check for stationary state via variance
            accel_mag = np.linalg.norm(w_arr[:, :3], axis=1)
            gyro_mag = np.linalg.norm(w_arr[:, 3:6], axis=1)
            if np.var(accel_mag) < 0.05 and np.mean(gyro_mag) < 0.08:
                motion_state = "STATIONARY"
                trust_weight = 0.0
                if ai_speed is not None:
                    ai_speed = 0.0

        # 4. Initialize origin if GNSS available and uninitialized
        if not self._origin_initialized:
            if sensor_input.gnss_lat is not None and not np.isnan(sensor_input.gnss_lat):
                lat0 = float(sensor_input.gnss_lat)
                lon0 = float(sensor_input.gnss_lon)
                alt0 = float(sensor_input.gnss_alt or 0.0)
                self.fusion_engine.set_origin(lat0, lon0, alt0)
                init_hdg = float(sensor_input.gnss_course or 0.0)
                self.fusion_engine.ekf.initialize_state(
                    p_n=0.0, p_e=0.0, p_d=0.0,
                    v_n=0.0, v_e=0.0, v_d=0.0,
                    heading_rad=np.radians(init_hdg),
                    timestamp_s=ts / 1000.0,
                )
                self._origin_initialized = True
            else:
                # Return pre-initialization state
                return NavigationOutput(
                    timestamp_ms=ts,
                    lat_deg=0.0,
                    lon_deg=0.0,
                    alt_m=0.0,
                    speed_ms=0.0,
                    heading_deg=0.0,
                    mode=NavigationMode.GNSS_INS.value,
                    confidence=0.0,
                    final_lat=0.0,
                    final_lon=0.0,
                    motion_state=motion_state,
                    trust_weight=trust_weight,
                    ai_speed_ms=ai_speed,
                )

        # 5. Fusion Step
        fusion_state = self.fusion_engine.step(
            dt=dt,
            ax_lin=float(ax_lin),
            ay_lin=float(ay_lin),
            az_lin=float(az_lin),
            gz_veh=float(sensor_input.gz),
            gx_veh=float(sensor_input.gx),
            gy_veh=float(sensor_input.gy),
            gnss_lat=sensor_input.gnss_lat,
            gnss_lon=sensor_input.gnss_lon,
            gnss_alt=sensor_input.gnss_alt,
            gnss_speed=sensor_input.gnss_speed,
            gnss_course_deg=sensor_input.gnss_course,
            gnss_accuracy=sensor_input.gnss_accuracy,
            satellites=sensor_input.gnss_sats,
            ai_speed=ai_speed,
            timestamp_ms=ts,
        )

        # 6. Map Matching (single step)
        mm_lat = fusion_state.lat_deg
        mm_lon = fusion_state.lon_deg
        mm_road_id = None
        mm_road_name = None
        mm_snap_dist = 0.0
        mm_conf = 0.0
        is_matched = False

        if self.config.use_map_matching and self.map_matcher is not None:
            mm_res = self.map_matcher.match_single(
                lat=fusion_state.lat_deg,
                lon=fusion_state.lon_deg,
                heading_deg=fusion_state.heading_deg,
                speed_ms=fusion_state.speed_ms,
                timestamp_ms=ts,
            )
            if mm_res.is_matched:
                mm_lat = mm_res.matched_lat
                mm_lon = mm_res.matched_lon
                mm_road_id = str(mm_res.road_id) if mm_res.road_id else None
                mm_road_name = mm_res.road_name
                mm_snap_dist = mm_res.snap_distance_m
                mm_conf = mm_res.confidence
                is_matched = True

        use_mm = is_matched and mm_conf >= self.config.map_matching_confidence_threshold
        final_lat = mm_lat if use_mm else fusion_state.lat_deg
        final_lon = mm_lon if use_mm else fusion_state.lon_deg

        return NavigationOutput(
            timestamp_ms=ts,
            lat_deg=fusion_state.lat_deg,
            lon_deg=fusion_state.lon_deg,
            alt_m=fusion_state.alt_m,
            speed_ms=fusion_state.speed_ms,
            heading_deg=fusion_state.heading_deg,
            mode=fusion_state.mode,
            confidence=fusion_state.confidence,
            mm_lat=mm_lat if is_matched else None,
            mm_lon=mm_lon if is_matched else None,
            mm_road_id=mm_road_id,
            mm_road_name=mm_road_name,
            mm_snap_dist_m=mm_snap_dist if is_matched else None,
            mm_confidence=mm_conf if is_matched else None,
            is_map_matched=use_mm,
            final_lat=final_lat,
            final_lon=final_lon,
            motion_state=motion_state,
            trust_weight=trust_weight,
            ai_speed_ms=ai_speed,
            p_n=fusion_state.p_n,
            p_e=fusion_state.p_e,
            cov_trace=fusion_state.cov_trace,
        )

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_feature_columns(df: pd.DataFrame) -> List[str]:
        """Resolves available IMU feature columns."""
        if all(c in df.columns for c in ["ax_veh_lin", "ay_veh_lin", "az_veh_lin", "gx_veh", "gy_veh", "gz_veh"]):
            return ["ax_veh_lin", "ay_veh_lin", "az_veh_lin", "gx_veh", "gy_veh", "gz_veh"]
        elif all(c in df.columns for c in ["ax_veh", "ay_veh", "az_veh", "gx_veh", "gy_veh", "gz_veh"]):
            return ["ax_veh", "ay_veh", "az_veh", "gx_veh", "gy_veh", "gz_veh"]
        elif all(c in df.columns for c in ["ax", "ay", "az", "gx", "gy", "gz"]):
            return ["ax", "ay", "az", "gx", "gy", "gz"]
        else:
            raise ValueError("DataFrame lacks standard 6-axis IMU columns.")

    def _predict_ai_velocity_batch(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
    ) -> Optional[np.ndarray]:
        """Runs fast batch inference for forward velocity using 1D CNN."""
        if not self.config.use_ai_velocity:
            return None

        if self.ai_model is not None and self.ai_scaler is not None:
            raw_feats = df[feat_cols].values.astype(np.float32)
            norm_feats = self.ai_scaler.transform(raw_feats)
            N = len(df)
            W = self.config.window_size
            windows = np.zeros((N, W, 6), dtype=np.float32)

            for i in range(N):
                st = max(0, i - W + 1)
                seg = norm_feats[st : i + 1]
                if len(seg) < W:
                    pad = np.repeat(seg[:1], W - len(seg), axis=0)
                    windows[i] = np.vstack([pad, seg])
                else:
                    windows[i] = seg

            preds = self.ai_model.predict(windows).flatten().astype(np.float64)
            preds = np.maximum(0.0, preds)
            return preds

        # Fallback to GNSS speed if AI model is not present
        if "gnss_speed" in df.columns:
            return df["gnss_speed"].bfill().ffill().values.astype(np.float64)

        return None

    def _classify_motion_batch(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
    ) -> Tuple[List[str], np.ndarray]:
        """Extracts motion state labels and trust weights across the trip."""
        n = len(df)
        if not self.config.use_motion_classifier or n < self.config.window_size:
            return ["NORMAL"] * n, np.ones(n, dtype=np.float64)

        try:
            ts_col = "timestamp" if "timestamp" in df.columns else ("timestamp_ms" if "timestamp_ms" in df.columns else None)
            features_df = self.feature_extractor.extract_from_dataframe(
                df,
                accel_cols=(feat_cols[0], feat_cols[1], feat_cols[2]),
                gyro_cols=(feat_cols[3], feat_cols[4], feat_cols[5]),
                timestamp_col=ts_col if ts_col else "timestamp",
            )
            # Propagate window labels back to every sample in source df
            sample_labels = self.motion_classifier.label_source_dataframe(df, features_df)
            weights_series = self.motion_classifier.get_trust_weights_series(sample_labels)

            labels = [MotionLabel(val).name for val in sample_labels]
            weights = weights_series.to_numpy(dtype=np.float64)
            return labels, weights
        except Exception as e:
            logger.warning("Motion classification failed (%s), defaulting to NORMAL.", e)
            return ["NORMAL"] * n, np.ones(n, dtype=np.float64)
