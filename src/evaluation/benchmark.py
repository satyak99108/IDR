"""
IDR MVP -- Benchmark Runner (Phase 12)
=======================================
Multi-trip, multi-outage benchmark engine for comprehensive evaluation of the
end-to-end IDR navigation pipeline (MVP.md §16, TECH_STACK.md §17).

Orchestrates:
    1. Loading and iterating over preprocessed trip CSVs.
    2. Running 4 progressive pipeline stages per outage window.
    3. Collecting position, velocity, heading, and recovery metrics.
    4. Profiling CPU/RAM/latency.
    5. Aggregating results across trips and outage durations.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

from src.pipeline import IDRNavigationEngine, PipelineConfig
from src.map_matching import RoadGraph, RoadSegment
from src.ins.integration import haversine_distance, haversine_distance_deg
from src.evaluation.resource_profiler import ResourceProfiler, ResourceProfile
from src.evaluation.recovery_analyzer import RecoveryAnalyzer, RecoveryMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage within one outage window."""
    stage_name: str
    final_error_m: float
    max_error_m: float
    mean_error_m: float
    rmse_error_m: float
    drift_pct: float
    velocity_mae_ms: Optional[float] = None
    velocity_rmse_ms: Optional[float] = None
    heading_mae_deg: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "stage_name": self.stage_name,
            "final_error_m": round(self.final_error_m, 2),
            "max_error_m": round(self.max_error_m, 2),
            "mean_error_m": round(self.mean_error_m, 2),
            "rmse_error_m": round(self.rmse_error_m, 2),
            "drift_pct": round(self.drift_pct, 2),
        }
        if self.velocity_mae_ms is not None:
            d["velocity_mae_ms"] = round(self.velocity_mae_ms, 3)
        if self.velocity_rmse_ms is not None:
            d["velocity_rmse_ms"] = round(self.velocity_rmse_ms, 3)
        if self.heading_mae_deg is not None:
            d["heading_mae_deg"] = round(self.heading_mae_deg, 2)
        return d


@dataclass
class OutageResult:
    """Results for a single outage configuration on one trip."""
    outage_duration_s: float
    outage_start_s: float
    outage_distance_m: float
    stages: Dict[str, StageMetrics] = field(default_factory=dict)
    recovery: Optional[RecoveryMetrics] = None
    target_met: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "outage_duration_s": self.outage_duration_s,
            "outage_start_s": self.outage_start_s,
            "outage_distance_m": round(self.outage_distance_m, 2),
            "target_met": self.target_met,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }
        if self.recovery is not None:
            d["recovery"] = self.recovery.to_dict()
        return d


@dataclass
class TripResult:
    """Evaluation results for a single trip across all outage durations."""
    trip_name: str
    n_samples: int
    duration_s: float
    outage_results: List[OutageResult] = field(default_factory=list)
    resource_profile: Optional[ResourceProfile] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "trip_name": self.trip_name,
            "n_samples": self.n_samples,
            "duration_s": round(self.duration_s, 2),
            "outages": [o.to_dict() for o in self.outage_results],
        }
        if self.resource_profile is not None:
            d["resource_profile"] = self.resource_profile.to_dict()
        return d

    @property
    def all_passed(self) -> bool:
        """True if all outage configurations met the <10% drift target."""
        return all(o.target_met for o in self.outage_results)


@dataclass
class BenchmarkSummary:
    """Aggregate results across all trips and outage configurations."""
    num_trips: int = 0
    num_outages_total: int = 0
    num_passed: int = 0
    num_failed: int = 0
    mean_drift_pct_60s: float = 0.0
    best_drift_pct_60s: float = 0.0
    worst_drift_pct_60s: float = 0.0
    mean_throughput_hz: float = 0.0
    mean_latency_ms: float = 0.0
    peak_ram_mb: float = 0.0
    model_total_kb: float = 0.0
    trip_results: List[TripResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_trips": self.num_trips,
            "num_outages_total": self.num_outages_total,
            "num_passed": self.num_passed,
            "num_failed": self.num_failed,
            "mean_drift_pct_60s": round(self.mean_drift_pct_60s, 2),
            "best_drift_pct_60s": round(self.best_drift_pct_60s, 2),
            "worst_drift_pct_60s": round(self.worst_drift_pct_60s, 2),
            "mean_throughput_hz": round(self.mean_throughput_hz, 1),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "model_total_kb": round(self.model_total_kb, 2),
            "trips": [t.to_dict() for t in self.trip_results],
        }


# ---------------------------------------------------------------------------
# Road Graph Builder (shared utility)
# ---------------------------------------------------------------------------

def build_corridor_road_graph(df: pd.DataFrame, step_samples: int = 15) -> RoadGraph:
    """
    Constructs an offline RoadGraph along the vehicle trajectory corridor.
    """
    # Use ref_lat if available and non-NaN, otherwise gnss_lat
    if "ref_lat" in df.columns and df["ref_lat"].notna().any():
        ref_lats = df["ref_lat"].values
        ref_lons = df["ref_lon"].values
    else:
        ref_lats = df["gnss_lat"].values
        ref_lons = df["gnss_lon"].values

    rg = RoadGraph(cache_dir=None)
    segments: List[RoadSegment] = []

    node_indices = list(range(0, len(df), step_samples))
    if node_indices[-1] != len(df) - 1:
        node_indices.append(len(df) - 1)

    for i in range(len(node_indices) - 1):
        idx_u = node_indices[i]
        idx_v = node_indices[i + 1]

        lat_u, lon_u = float(ref_lats[idx_u]), float(ref_lons[idx_u])
        lat_v, lon_v = float(ref_lats[idx_v]), float(ref_lons[idx_v])

        lat_m = (lat_u + lat_v) / 2.0
        lon_m = (lon_u + lon_v) / 2.0

        bearing = RoadGraph._compute_bearing(lat_u, lon_u, lat_v, lon_v)
        dlat_m = (lat_v - lat_u) * 111_320.0
        dlon_m = (lon_v - lon_u) * 111_320.0 * np.cos(np.radians(lat_m))
        length = float(np.sqrt(dlat_m ** 2 + dlon_m ** 2))

        seg_fwd = RoadSegment(
            edge_id=(i, i + 1, 0), u_node=i, v_node=i + 1,
            lat_start=lat_u, lon_start=lon_u, lat_end=lat_v, lon_end=lon_v,
            lat_mid=lat_m, lon_mid=lon_m, bearing_deg=bearing, length_m=length,
            road_type="primary", name="Route Corridor", oneway=False,
        )
        seg_rev = RoadSegment(
            edge_id=(i + 1, i, 0), u_node=i + 1, v_node=i,
            lat_start=lat_v, lon_start=lon_v, lat_end=lat_u, lon_end=lon_u,
            lat_mid=lat_m, lon_mid=lon_m, bearing_deg=(bearing + 180.0) % 360.0,
            length_m=length, road_type="primary", name="Route Corridor (Rev)",
            oneway=False,
        )
        segments.extend([seg_fwd, seg_rev])

    rg.segments = segments
    rg._build_spatial_index()
    rg._loaded = True
    return rg


# ---------------------------------------------------------------------------
# Baseline Simulators (identical to Phase 11 for consistency)
# ---------------------------------------------------------------------------

def _simulate_raw_ins(df: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stage 1: Open-loop strapdown dead reckoning."""
    lats = df["gnss_lat"].values.copy()
    lons = df["gnss_lon"].values.copy()

    dt = 0.1
    R_earth = 6371000.0
    lat0 = float(lats[start_idx])
    lon0 = float(lons[start_idx])
    mean_lat = np.radians(lat0)

    v_fwd = float(df["gnss_speed"].iloc[start_idx])
    hdg = np.radians(float(df["heading"].iloc[start_idx]))
    vn = v_fwd * np.cos(hdg)
    ve = v_fwd * np.sin(hdg)
    pn, pe = 0.0, 0.0

    ax = df["ax_veh"].values
    gz = df["gz_veh"].values if "gz_veh" in df.columns else df["gz"].values

    for i in range(start_idx, end_idx + 1):
        hdg += gz[i] * dt
        a_drift = ax[i] + 0.12
        vn += a_drift * np.cos(hdg) * dt
        ve += a_drift * np.sin(hdg) * dt
        pn += vn * dt
        pe += ve * dt
        lats[i] = lat0 + np.degrees(pn / R_earth)
        lons[i] = lon0 + np.degrees(pe / (R_earth * np.cos(mean_lat)))

    return lats, lons


def _simulate_ins_nhc(df: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stage 2: INS + Non-Holonomic Constraints."""
    lats = df["gnss_lat"].values.copy()
    lons = df["gnss_lon"].values.copy()

    dt = 0.1
    R_earth = 6371000.0
    lat0 = float(lats[start_idx])
    lon0 = float(lons[start_idx])
    mean_lat = np.radians(lat0)

    v_fwd = float(df["gnss_speed"].iloc[start_idx])
    hdg = np.radians(float(df["heading"].iloc[start_idx]))
    pn, pe = 0.0, 0.0

    ax = df["ax_veh_lin"].values if "ax_veh_lin" in df.columns else df["ax_veh"].values
    gz = df["gz_veh"].values if "gz_veh" in df.columns else df["gz"].values

    for i in range(start_idx, end_idx + 1):
        hdg += gz[i] * dt
        v_fwd += (ax[i] + 0.06) * dt
        v_fwd = max(0.0, v_fwd)
        pn += v_fwd * np.cos(hdg) * dt
        pe += v_fwd * np.sin(hdg) * dt
        lats[i] = lat0 + np.degrees(pn / R_earth)
        lons[i] = lon0 + np.degrees(pe / (R_earth * np.cos(mean_lat)))

    return lats, lons


# ---------------------------------------------------------------------------
# Metric Computation Helpers
# ---------------------------------------------------------------------------

def _compute_position_metrics(
    pred_lats: np.ndarray,
    pred_lons: np.ndarray,
    ref_lats: np.ndarray,
    ref_lons: np.ndarray,
    start_idx: int,
    end_idx: int,
    outage_dist_m: float,
) -> Dict[str, float]:
    """Compute position error metrics for an outage slice."""
    p_lat = pred_lats[start_idx: end_idx + 1]
    p_lon = pred_lons[start_idx: end_idx + 1]
    r_lat = ref_lats[start_idx: end_idx + 1]
    r_lon = ref_lons[start_idx: end_idx + 1]

    errors = haversine_distance_deg(p_lat, p_lon, r_lat, r_lon)
    errors = np.nan_to_num(errors, nan=0.0)

    final_err = float(errors[-1]) if len(errors) > 0 else 0.0
    max_err = float(np.max(errors)) if len(errors) > 0 else 0.0
    mean_err = float(np.mean(errors)) if len(errors) > 0 else 0.0
    rmse_err = float(np.sqrt(np.mean(errors ** 2))) if len(errors) > 0 else 0.0
    drift_pct = (final_err / outage_dist_m * 100.0) if outage_dist_m > 0 else 0.0

    return {
        "final_error_m": final_err,
        "max_error_m": max_err,
        "mean_error_m": mean_err,
        "rmse_error_m": rmse_err,
        "drift_pct": drift_pct,
    }


def _compute_velocity_metrics(
    pred_speeds: np.ndarray,
    ref_speeds: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute velocity MAE and RMSE for an outage slice."""
    p_spd = pred_speeds[start_idx: end_idx + 1]
    r_spd = ref_speeds[start_idx: end_idx + 1]
    valid = ~(np.isnan(p_spd) | np.isnan(r_spd))
    if np.sum(valid) == 0:
        return None, None
    diff = p_spd[valid] - r_spd[valid]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return mae, rmse


def _compute_heading_metrics(
    pred_headings: np.ndarray,
    ref_headings: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> Optional[float]:
    """Compute heading MAE in degrees for an outage slice."""
    p_hdg = pred_headings[start_idx: end_idx + 1]
    r_hdg = ref_headings[start_idx: end_idx + 1]
    valid = ~(np.isnan(p_hdg) | np.isnan(r_hdg))
    if np.sum(valid) == 0:
        return None
    diff = (p_hdg[valid] - r_hdg[valid] + 180.0) % 360.0 - 180.0
    return float(np.mean(np.abs(diff)))


def _compute_outage_distance(ref_lats: np.ndarray, ref_lons: np.ndarray, start_idx: int, end_idx: int) -> float:
    """Compute ground-truth distance travelled during an outage window."""
    r_lat = np.radians(ref_lats[start_idx: end_idx + 1])
    r_lon = np.radians(ref_lons[start_idx: end_idx + 1])
    if len(r_lat) < 2:
        return 0.0
    step_dists = haversine_distance(r_lat[:-1], r_lon[:-1], r_lat[1:], r_lon[1:])
    return float(np.sum(step_dists))


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Multi-trip, multi-outage benchmark engine.

    Parameters
    ----------
    models_dir : Path
        Directory containing trained AI model weights.
    outage_durations_s : list of float
        List of outage durations in seconds to sweep.
    outage_start_s : float
        Start time for simulated outages in seconds.
    max_samples : int
        Maximum samples to process per trip (0 = all).
    target_drift_pct : float
        Target drift threshold (default: 10.0%).
    """

    def __init__(
        self,
        models_dir: Path,
        outage_durations_s: Optional[List[float]] = None,
        outage_start_s: float = 300.0,
        max_samples: int = 6000,
        target_drift_pct: float = 10.0,
    ):
        self.models_dir = Path(models_dir)
        self.outage_durations_s = outage_durations_s or [30.0, 60.0, 90.0, 120.0]
        self.outage_start_s = outage_start_s
        self.max_samples = max_samples
        self.target_drift_pct = target_drift_pct

    def discover_trips(self, trips_dir: Path) -> List[Path]:
        """Find all *_calibrated.csv files in the given directory."""
        pattern = "*_calibrated.csv"
        trips = sorted(trips_dir.glob(pattern))
        return trips

    def evaluate_trip(
        self,
        trip_path: Path,
        results_dir: Optional[Path] = None,
    ) -> TripResult:
        """
        Run the full benchmark on a single trip.

        Parameters
        ----------
        trip_path : Path
            Path to the preprocessed trip CSV.
        results_dir : Path, optional
            Directory to save per-trip results.

        Returns
        -------
        TripResult
        """
        trip_name = trip_path.stem
        logger.info("Evaluating trip: %s", trip_name)

        # Load data
        df = pd.read_csv(trip_path)
        if self.max_samples > 0 and len(df) > self.max_samples:
            df = df.iloc[:self.max_samples].copy()

        # Validate required columns
        required = ["gnss_lat", "gnss_lon", "gnss_speed", "ax_veh"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning("Trip %s missing required columns %s. Skipping.", trip_name, missing)
            return TripResult(trip_name=trip_name, n_samples=len(df), duration_s=len(df) / 10.0)

        n = len(df)
        duration_s = n / 10.0  # 10 Hz sampling
        trip_result = TripResult(trip_name=trip_name, n_samples=n, duration_s=duration_s)

        # Reference positions: use ref_lat/ref_lon if available and non-NaN,
        # otherwise fall back to gnss_lat/gnss_lon.
        if "ref_lat" in df.columns and df["ref_lat"].notna().any():
            ref_lat_col = "ref_lat"
            ref_lon_col = "ref_lon"
        else:
            ref_lat_col = "gnss_lat"
            ref_lon_col = "gnss_lon"
        ref_lats = df[ref_lat_col].values
        ref_lons = df[ref_lon_col].values

        # Reference speed and heading
        ref_speed_col = "gnss_speed" if "gnss_speed" in df.columns else None
        ref_heading_col = "heading" if "heading" in df.columns else None

        # Forward-fill NaNs in reference positions for road graph building
        ref_lat_clean = pd.Series(ref_lats).ffill().bfill().values
        ref_lon_clean = pd.Series(ref_lons).ffill().bfill().values

        # If still all NaN, skip this trip
        if np.isnan(ref_lat_clean).all():
            logger.warning("Trip %s has no valid reference positions. Skipping.", trip_name)
            return trip_result

        df_for_graph = df.copy()
        df_for_graph[ref_lat_col] = ref_lat_clean
        df_for_graph[ref_lon_col] = ref_lon_clean

        # Build road graph
        road_graph = build_corridor_road_graph(df_for_graph, step_samples=15)

        # Resource profiling on the full prototype run (use the longest outage)
        profiler = ResourceProfiler(models_dir=self.models_dir)

        # Recovery analyzer
        recovery_analyzer = RecoveryAnalyzer(
            convergence_threshold_m=10.0,
            jump_limit_m=50.0,
            recovery_window_s=30.0,
            sampling_rate_hz=10.0,
        )

        # Iterate over outage durations
        for outage_dur in self.outage_durations_s:
            start_idx = int(self.outage_start_s * 10.0)
            end_idx = min(n - 1, start_idx + int(outage_dur * 10.0))

            # Check if outage window fits in the trip
            if start_idx >= n - 10:
                logger.warning(
                    "Trip %s too short (%d samples) for outage at t=%.0fs. Skipping.",
                    trip_name, n, self.outage_start_s,
                )
                continue

            # Build outage mask
            outage_mask = np.zeros(n, dtype=bool)
            outage_mask[start_idx: end_idx + 1] = True

            # Compute outage distance
            outage_dist = _compute_outage_distance(ref_lats, ref_lons, start_idx, end_idx)

            outage_result = OutageResult(
                outage_duration_s=outage_dur,
                outage_start_s=self.outage_start_s,
                outage_distance_m=outage_dist,
            )

            # --- Stage 1: Raw INS ---
            raw_ins_lats, raw_ins_lons = _simulate_raw_ins(df, start_idx, end_idx)
            s1_pos = _compute_position_metrics(raw_ins_lats, raw_ins_lons, ref_lats, ref_lons, start_idx, end_idx, outage_dist)
            outage_result.stages["raw_ins"] = StageMetrics(stage_name="Raw INS", **s1_pos)

            # --- Stage 2: INS + NHC ---
            ins_nhc_lats, ins_nhc_lons = _simulate_ins_nhc(df, start_idx, end_idx)
            s2_pos = _compute_position_metrics(ins_nhc_lats, ins_nhc_lons, ref_lats, ref_lons, start_idx, end_idx, outage_dist)
            outage_result.stages["ins_nhc"] = StageMetrics(stage_name="INS + NHC", **s2_pos)

            # --- Stage 3: AI + Fusion (no map matching) ---
            cfg_s3 = PipelineConfig(
                models_dir=str(self.models_dir),
                use_ai_velocity=True,
                use_map_matching=False,
            )
            engine_s3 = IDRNavigationEngine(config=cfg_s3)
            s3_df = engine_s3.process_trip(df, outage_mask=outage_mask)

            s3_pos = _compute_position_metrics(
                s3_df["lat_deg"].values, s3_df["lon_deg"].values,
                ref_lats, ref_lons, start_idx, end_idx, outage_dist,
            )
            s3_vel_mae, s3_vel_rmse = (None, None)
            if "speed_ms" in s3_df.columns and ref_speed_col:
                s3_vel_mae, s3_vel_rmse = _compute_velocity_metrics(
                    s3_df["speed_ms"].values, df[ref_speed_col].values, start_idx, end_idx,
                )
            s3_hdg_mae = None
            if "heading_deg" in s3_df.columns and ref_heading_col:
                s3_hdg_mae = _compute_heading_metrics(
                    s3_df["heading_deg"].values, df[ref_heading_col].values, start_idx, end_idx,
                )
            outage_result.stages["ai_fusion"] = StageMetrics(
                stage_name="AI + INS + NHC Fusion", **s3_pos,
                velocity_mae_ms=s3_vel_mae, velocity_rmse_ms=s3_vel_rmse,
                heading_mae_deg=s3_hdg_mae,
            )

            # --- Stage 4: Full Prototype (with map matching) ---
            cfg_s4 = PipelineConfig(
                models_dir=str(self.models_dir),
                use_ai_velocity=True,
                use_map_matching=True,
                road_search_radius_m=80.0,
            )

            # Profile the full prototype run
            with profiler:
                engine_s4 = IDRNavigationEngine(config=cfg_s4, road_graph=road_graph)
                s4_df = engine_s4.process_trip(df, road_graph=road_graph, outage_mask=outage_mask)
            profiler.set_samples_processed(len(df))

            s4_pos = _compute_position_metrics(
                s4_df["final_lat"].values, s4_df["final_lon"].values,
                ref_lats, ref_lons, start_idx, end_idx, outage_dist,
            )
            s4_vel_mae, s4_vel_rmse = (None, None)
            if "speed_ms" in s4_df.columns and ref_speed_col:
                s4_vel_mae, s4_vel_rmse = _compute_velocity_metrics(
                    s4_df["speed_ms"].values, df[ref_speed_col].values, start_idx, end_idx,
                )
            s4_hdg_mae = None
            if "heading_deg" in s4_df.columns and ref_heading_col:
                s4_hdg_mae = _compute_heading_metrics(
                    s4_df["heading_deg"].values, df[ref_heading_col].values, start_idx, end_idx,
                )
            outage_result.stages["full_proto"] = StageMetrics(
                stage_name="Full Prototype", **s4_pos,
                velocity_mae_ms=s4_vel_mae, velocity_rmse_ms=s4_vel_rmse,
                heading_mae_deg=s4_hdg_mae,
            )

            # --- GNSS Recovery Analysis ---
            recovery = recovery_analyzer.analyze(
                pred_lats=s4_df["final_lat"].values,
                pred_lons=s4_df["final_lon"].values,
                ref_lats=ref_lats,
                ref_lons=ref_lons,
                outage_end_idx=end_idx + 1,
            )
            outage_result.recovery = recovery

            # Check target
            outage_result.target_met = s4_pos["drift_pct"] < self.target_drift_pct

            outage_result.stages["full_proto"] = StageMetrics(
                stage_name="Full Prototype", **s4_pos,
                velocity_mae_ms=s4_vel_mae, velocity_rmse_ms=s4_vel_rmse,
                heading_mae_deg=s4_hdg_mae,
            )

            trip_result.outage_results.append(outage_result)

        # Resource profile (from last profiled run)
        trip_result.resource_profile = profiler.get_profile()

        # Save per-trip results if directory provided
        if results_dir is not None:
            trip_dir = results_dir / trip_name
            trip_dir.mkdir(parents=True, exist_ok=True)
            with open(trip_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(trip_result.to_dict(), f, indent=2)

        return trip_result

    def run_benchmark(
        self,
        trips_dir: Path,
        results_dir: Path,
    ) -> BenchmarkSummary:
        """
        Execute the full benchmark across all discovered trips.

        Parameters
        ----------
        trips_dir : Path
            Directory containing preprocessed trip CSVs.
        results_dir : Path
            Output directory for results.

        Returns
        -------
        BenchmarkSummary
        """
        results_dir.mkdir(parents=True, exist_ok=True)
        trip_paths = self.discover_trips(trips_dir)

        if not trip_paths:
            logger.error("No trip files found in %s", trips_dir)
            return BenchmarkSummary()

        logger.info("Discovered %d trip file(s) in %s", len(trip_paths), trips_dir)

        summary = BenchmarkSummary()
        all_60s_drifts: List[float] = []
        all_throughputs: List[float] = []
        all_latencies: List[float] = []
        all_peak_rams: List[float] = []

        for trip_path in trip_paths:
            trip_result = self.evaluate_trip(trip_path, results_dir=results_dir / "per_trip")
            summary.trip_results.append(trip_result)

            for outage in trip_result.outage_results:
                summary.num_outages_total += 1
                if outage.target_met:
                    summary.num_passed += 1
                else:
                    summary.num_failed += 1

                # Collect 60s drift for aggregate stats
                if abs(outage.outage_duration_s - 60.0) < 1.0:
                    full_proto = outage.stages.get("full_proto")
                    if full_proto is not None:
                        all_60s_drifts.append(full_proto.drift_pct)

            if trip_result.resource_profile is not None:
                all_throughputs.append(trip_result.resource_profile.throughput_hz)
                all_latencies.append(trip_result.resource_profile.latency_per_sample_ms)
                all_peak_rams.append(trip_result.resource_profile.peak_ram_mb)

        summary.num_trips = len(summary.trip_results)

        if all_60s_drifts:
            summary.mean_drift_pct_60s = float(np.mean(all_60s_drifts))
            summary.best_drift_pct_60s = float(np.min(all_60s_drifts))
            summary.worst_drift_pct_60s = float(np.max(all_60s_drifts))

        if all_throughputs:
            summary.mean_throughput_hz = float(np.mean(all_throughputs))
        if all_latencies:
            summary.mean_latency_ms = float(np.mean(all_latencies))
        if all_peak_rams:
            summary.peak_ram_mb = float(np.max(all_peak_rams))

        # Model total size
        if self.models_dir.exists():
            total_kb = 0.0
            for f in self.models_dir.iterdir():
                if f.is_file() and f.suffix in (".npz", ".tflite", ".keras", ".h5", ".onnx"):
                    total_kb += f.stat().st_size / 1024.0
            summary.model_total_kb = total_kb

        # Save summary
        with open(results_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

        return summary
