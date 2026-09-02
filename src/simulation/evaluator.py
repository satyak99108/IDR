"""
IDR MVP -- GNSS Outage Evaluator
=================================
Evaluates navigation and dead-reckoning accuracy during GNSS blackout periods.

Key Metrics (per MVP.md §16):
    - Drift % = (Position Error at Outage Exit / Distance Travelled) * 100%
      (Target: < 10% drift during GNSS blackout)
    - Absolute Endpoint Position Error (metres)
    - Maximum Position Error (metres)
    - Mean & RMSE Position Error (metres)
    - Velocity Error (MAE, RMSE in m/s)
    - Heading Error (MAE in degrees)

Classes:
    OutageMetrics    -- Computed error metrics for a single blackout window.
    ScenarioMetrics  -- Aggregate metrics for an entire scenario.
    OutageEvaluator  -- Metric calculation engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

from src.ins.integration import haversine_distance_deg


# ---------------------------------------------------------------------------
# Metric Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OutageMetrics:
    """
    Evaluation metrics for a single GNSS blackout window.
    """
    outage_id: int
    label: str
    duration_s: float
    distance_travelled_m: float
    endpoint_error_m: float
    drift_percent: float
    max_error_m: float
    mean_error_m: float
    rmse_error_m: float
    velocity_rmse_ms: Optional[float] = None
    velocity_mae_ms: Optional[float] = None
    heading_mae_deg: Optional[float] = None
    samples_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outage_id": self.outage_id,
            "label": self.label,
            "duration_s": round(self.duration_s, 2),
            "distance_travelled_m": round(self.distance_travelled_m, 2),
            "endpoint_error_m": round(self.endpoint_error_m, 2),
            "drift_percent": round(self.drift_percent, 2),
            "max_error_m": round(self.max_error_m, 2),
            "mean_error_m": round(self.mean_error_m, 2),
            "rmse_error_m": round(self.rmse_error_m, 2),
            "velocity_rmse_ms": round(self.velocity_rmse_ms, 2) if self.velocity_rmse_ms is not None else None,
            "velocity_mae_ms": round(self.velocity_mae_ms, 2) if self.velocity_mae_ms is not None else None,
            "heading_mae_deg": round(self.heading_mae_deg, 2) if self.heading_mae_deg is not None else None,
            "samples_count": self.samples_count,
        }


@dataclass
class ScenarioMetrics:
    """
    Evaluation metrics aggregated across all blackout windows in a scenario.
    """
    scenario_name: str
    description: str
    num_outages: int
    total_outage_time_s: float
    total_outage_distance_m: float
    mean_drift_percent: float
    weighted_drift_percent: float
    overall_max_error_m: float
    overall_rmse_m: float
    outage_details: List[OutageMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "description": self.description,
            "num_outages": self.num_outages,
            "total_outage_time_s": round(self.total_outage_time_s, 2),
            "total_outage_distance_m": round(self.total_outage_distance_m, 2),
            "mean_drift_percent": round(self.mean_drift_percent, 2),
            "weighted_drift_percent": round(self.weighted_drift_percent, 2),
            "overall_max_error_m": round(self.overall_max_error_m, 2),
            "overall_rmse_m": round(self.overall_rmse_m, 2),
            "outages": [m.to_dict() for m in self.outage_details],
        }


# ---------------------------------------------------------------------------
# Outage Evaluator Engine
# ---------------------------------------------------------------------------

class OutageEvaluator:
    """
    Engine to evaluate trajectory estimation accuracy specifically during GNSS outages.
    """

    @staticmethod
    def evaluate_outage_window(
        df_outage: pd.DataFrame,
        window_id: int,
        label: str = "Outage",
        pred_lat_col: str = "pred_lat",
        pred_lon_col: str = "pred_lon",
        ref_lat_col: str = "ref_lat",
        ref_lon_col: str = "ref_lon",
        pred_speed_col: Optional[str] = "pred_speed",
        ref_speed_col: Optional[str] = "ref_speed",
        pred_heading_col: Optional[str] = "pred_heading",
        ref_heading_col: Optional[str] = "ref_heading",
    ) -> OutageMetrics:
        """
        Calculates drift and error metrics for a single blackout slice.
        """
        n_samples = len(df_outage)
        if n_samples == 0:
            return OutageMetrics(
                outage_id=window_id,
                label=label,
                duration_s=0.0,
                distance_travelled_m=0.0,
                endpoint_error_m=0.0,
                drift_percent=0.0,
                max_error_m=0.0,
                mean_error_m=0.0,
                rmse_error_m=0.0,
                samples_count=0,
            )

        # Duration
        if "time_elapsed_s" in df_outage.columns:
            dur_s = float(df_outage["time_elapsed_s"].iloc[-1] - df_outage["time_elapsed_s"].iloc[0])
        elif "timestamp_ms" in df_outage.columns:
            dur_s = float((df_outage["timestamp_ms"].iloc[-1] - df_outage["timestamp_ms"].iloc[0]) / 1000.0)
        else:
            dur_s = float(n_samples * 0.1)

        # Ground truth distance travelled in this window (vectorized)
        ref_lats = df_outage[ref_lat_col].values
        ref_lons = df_outage[ref_lon_col].values
        if n_samples > 1:
            step_dists = haversine_distance_deg(ref_lats[:-1], ref_lons[:-1], ref_lats[1:], ref_lons[1:])
            dist_m = float(np.nansum(step_dists))
        else:
            dist_m = 0.0

        # Step-by-step position errors (vectorized)
        pred_lats = df_outage[pred_lat_col].values
        pred_lons = df_outage[pred_lon_col].values
        pos_errors = haversine_distance_deg(pred_lats, pred_lons, ref_lats, ref_lons)
        pos_errors = np.nan_to_num(pos_errors, nan=0.0)

        endpoint_error = float(pos_errors[-1])
        max_error = float(np.nanmax(pos_errors))
        mean_error = float(np.nanmean(pos_errors))
        rmse_error = float(np.sqrt(np.nanmean(pos_errors**2)))

        # Drift % = (endpoint_error / distance_travelled) * 100%
        if dist_m > 1.0:
            drift_pct = float((endpoint_error / dist_m) * 100.0)
        else:
            drift_pct = 0.0

        # Velocity metrics if available
        vel_rmse = None
        vel_mae = None
        if pred_speed_col in df_outage.columns and ref_speed_col in df_outage.columns:
            p_spd = df_outage[pred_speed_col].values
            r_spd = df_outage[ref_speed_col].values
            valid_spd = ~(np.isnan(p_spd) | np.isnan(r_spd))
            if np.sum(valid_spd) > 0:
                diff_spd = p_spd[valid_spd] - r_spd[valid_spd]
                vel_mae = float(np.mean(np.abs(diff_spd)))
                vel_rmse = float(np.sqrt(np.mean(diff_spd**2)))

        # Heading metrics if available
        heading_mae = None
        if pred_heading_col in df_outage.columns and ref_heading_col in df_outage.columns:
            p_hdg = df_outage[pred_heading_col].values
            r_hdg = df_outage[ref_heading_col].values
            valid_hdg = ~(np.isnan(p_hdg) | np.isnan(r_hdg))
            if np.sum(valid_hdg) > 0:
                diff_hdg = (p_hdg[valid_hdg] - r_hdg[valid_hdg] + 180.0) % 360.0 - 180.0
                heading_mae = float(np.mean(np.abs(diff_hdg)))

        return OutageMetrics(
            outage_id=window_id,
            label=label,
            duration_s=dur_s,
            distance_travelled_m=dist_m,
            endpoint_error_m=endpoint_error,
            drift_percent=drift_pct,
            max_error_m=max_error,
            mean_error_m=mean_error,
            rmse_error_m=rmse_error,
            velocity_rmse_ms=vel_rmse,
            velocity_mae_ms=vel_mae,
            heading_mae_deg=heading_mae,
            samples_count=n_samples,
        )

    @classmethod
    def evaluate_scenario(
        cls,
        df_simulated: pd.DataFrame,
        scenario_name: str = "Outage Scenario",
        description: str = "",
        pred_lat_col: str = "pred_lat",
        pred_lon_col: str = "pred_lon",
        ref_lat_col: str = "ref_lat",
        ref_lon_col: str = "ref_lon",
        pred_speed_col: Optional[str] = "pred_speed",
        ref_speed_col: Optional[str] = "ref_speed",
        pred_heading_col: Optional[str] = "pred_heading",
        ref_heading_col: Optional[str] = "ref_heading",
    ) -> ScenarioMetrics:
        """
        Evaluates all blackout windows present in df_simulated.
        """
        outage_ids = [oid for oid in df_simulated["outage_id"].unique() if oid > 0]
        outage_metrics_list = []

        all_outage_errors = []
        total_dist = 0.0
        total_endpoint_error = 0.0
        total_time = 0.0

        for oid in sorted(outage_ids):
            slice_df = df_simulated[df_simulated["outage_id"] == oid]
            lbl = slice_df["outage_label"].iloc[0] if "outage_label" in slice_df.columns else f"Outage #{oid}"
            m = cls.evaluate_outage_window(
                slice_df,
                window_id=int(oid),
                label=lbl,
                pred_lat_col=pred_lat_col,
                pred_lon_col=pred_lon_col,
                ref_lat_col=ref_lat_col,
                ref_lon_col=ref_lon_col,
                pred_speed_col=pred_speed_col,
                ref_speed_col=ref_speed_col,
                pred_heading_col=pred_heading_col,
                ref_heading_col=ref_heading_col,
            )
            outage_metrics_list.append(m)

            total_dist += m.distance_travelled_m
            total_endpoint_error += m.endpoint_error_m
            total_time += m.duration_s

            # Collect errors across all blackout points (vectorized)
            p_lat = slice_df[pred_lat_col].values
            p_lon = slice_df[pred_lon_col].values
            r_lat = slice_df[ref_lat_col].values
            r_lon = slice_df[ref_lon_col].values
            e_arr = haversine_distance_deg(p_lat, p_lon, r_lat, r_lon)
            valid_e = e_arr[~np.isnan(e_arr)]
            if len(valid_e) > 0:
                all_outage_errors.extend(valid_e.tolist())

        num_outages = len(outage_metrics_list)
        if num_outages > 0:
            mean_drift = float(np.mean([m.drift_percent for m in outage_metrics_list]))
            weighted_drift = float((total_endpoint_error / total_dist) * 100.0) if total_dist > 1.0 else 0.0
            overall_max = float(max(m.max_error_m for m in outage_metrics_list))
            overall_rmse = float(np.sqrt(np.mean(np.array(all_outage_errors)**2))) if len(all_outage_errors) > 0 else 0.0
        else:
            mean_drift = 0.0
            weighted_drift = 0.0
            overall_max = 0.0
            overall_rmse = 0.0

        return ScenarioMetrics(
            scenario_name=scenario_name,
            description=description,
            num_outages=num_outages,
            total_outage_time_s=total_time,
            total_outage_distance_m=total_dist,
            mean_drift_percent=mean_drift,
            weighted_drift_percent=weighted_drift,
            overall_max_error_m=overall_max,
            overall_rmse_m=overall_rmse,
            outage_details=outage_metrics_list,
        )
