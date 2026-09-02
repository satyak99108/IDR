"""
IDR MVP -- GNSS Outage Simulator
=================================
Implements the GNSS Outage Simulator for Phase 4 (MVP.md §8 and TECH_STACK.md §10).

This module generates artificial GNSS blackout/tunnel windows to systematically
evaluate and benchmark dead-reckoning performance without requiring physical
tunnel testing.

Classes:
    OutageWindow         -- Represents an individual GNSS blackout window.
    OutageScenario       -- A collection of outage windows with metadata.
    GNSSOutageSimulator  -- Generates scenarios and applies outage masks to data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Outage Window & Scenario Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OutageWindow:
    """
    Represents an individual GNSS blackout window (simulated tunnel/underpass).

    Attributes:
        outage_id: Unique integer identifier for this blackout window.
        start_time_s: Outage start time in seconds relative to trip start.
        end_time_s: Outage end time in seconds relative to trip start.
        label: Descriptive name (e.g., '30s Highway Tunnel', 'Urban Canyon Drop').
        start_idx: Optional DataFrame row index where outage starts.
        end_idx: Optional DataFrame row index where outage ends.
        start_dist_m: Optional distance travelled (m) at outage start.
        end_dist_m: Optional distance travelled (m) at outage end.
    """
    outage_id: int
    start_time_s: float
    end_time_s: float
    label: str = "Simulated Tunnel"
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    start_dist_m: Optional[float] = None
    end_dist_m: Optional[float] = None

    @property
    def duration_s(self) -> float:
        """Duration of the outage in seconds."""
        return max(0.0, self.end_time_s - self.start_time_s)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outage_id": self.outage_id,
            "start_time_s": round(self.start_time_s, 3),
            "end_time_s": round(self.end_time_s, 3),
            "duration_s": round(self.duration_s, 3),
            "label": self.label,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "start_dist_m": round(self.start_dist_m, 2) if self.start_dist_m is not None else None,
            "end_dist_m": round(self.end_dist_m, 2) if self.end_dist_m is not None else None,
        }


@dataclass
class OutageScenario:
    """
    Represents a full simulation scenario comprising one or more outage windows.

    Attributes:
        name: Short identifier (e.g., 'scenario_30s_tunnel').
        description: User-readable description of the test scenario.
        windows: List of OutageWindow instances.
    """
    name: str
    description: str
    windows: List[OutageWindow] = field(default_factory=list)

    @property
    def total_outage_duration_s(self) -> float:
        """Total time in seconds spent in GNSS blackout across all windows."""
        return sum(w.duration_s for w in self.windows)

    @property
    def num_outages(self) -> int:
        """Number of distinct blackout windows in the scenario."""
        return len(self.windows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "num_outages": self.num_outages,
            "total_outage_duration_s": round(self.total_outage_duration_s, 3),
            "windows": [w.to_dict() for w in self.windows],
        }


# ---------------------------------------------------------------------------
# GNSS Outage Simulator
# ---------------------------------------------------------------------------

class GNSSOutageSimulator:
    """
    Generates GNSS blackout scenarios and applies outage masks to sensor data.

    Features:
    - Time-bounded outage windows (e.g. 0-30s GNSS, 30-60s blackout, 60s+ GNSS).
    - Short underpass / long tunnel presets.
    - Periodic intermittent dropouts.
    - Distance-based / spatial trigger windows.
    - Stochastic / random canyon dropouts with fixed seed.
    - Non-destructive masking: preserves ground-truth while masking GNSS observation.
    """

    @staticmethod
    def create_time_window_scenario(
        name: str,
        description: str,
        windows: List[Tuple[float, float, str]],
    ) -> OutageScenario:
        """
        Builds a scenario from explicit (start_sec, end_sec, label) tuples.

        Args:
            name: Identifier for the scenario.
            description: Description of the test scenario.
            windows: List of tuples (start_time_s, end_time_s, label).

        Returns:
            OutageScenario instance.
        """
        outage_windows = []
        for i, (t_start, t_end, lbl) in enumerate(windows, start=1):
            if t_end <= t_start:
                raise ValueError(f"Window {i}: end_time ({t_end}) must be > start_time ({t_start})")
            outage_windows.append(
                OutageWindow(
                    outage_id=i,
                    start_time_s=float(t_start),
                    end_time_s=float(t_end),
                    label=lbl,
                )
            )
        return OutageScenario(name=name, description=description, windows=outage_windows)

    @classmethod
    def create_standard_mvp_scenario(cls, start_sec: float = 30.0, duration_sec: float = 30.0) -> OutageScenario:
        """
        Creates the canonical MVP 30s tunnel scenario specified in MVP.md §8:
        0–30s GNSS available -> 30–60s GNSS blackout -> 60s+ GNSS available.
        """
        return cls.create_time_window_scenario(
            name="mvp_standard_30s_tunnel",
            description="Canonical MVP 30-second simulated tunnel outage (30s-60s)",
            windows=[(start_sec, start_sec + duration_sec, "30s Highway Tunnel")],
        )

    @classmethod
    def create_short_underpass_scenario(cls, start_sec: float = 40.0, duration_sec: float = 10.0) -> OutageScenario:
        """Creates a short 10-second bridge/underpass blackout scenario (40s-50s while in motion)."""
        return cls.create_time_window_scenario(
            name="short_underpass_10s",
            description="Short 10-second urban bridge/underpass blackout (40s-50s)",
            windows=[(start_sec, start_sec + duration_sec, "10s Bridge Underpass")],
        )

    @classmethod
    def create_long_tunnel_scenario(cls, start_sec: float = 30.0, duration_sec: float = 60.0) -> OutageScenario:
        """Creates an extended 60-second highway/mountain tunnel blackout scenario."""
        return cls.create_time_window_scenario(
            name="long_tunnel_60s",
            description="Extended 60-second mountain tunnel blackout (30s-90s)",
            windows=[(start_sec, start_sec + duration_sec, "60s Mountain Tunnel")],
        )

    @classmethod
    def create_multi_outage_scenario(
        cls,
        windows: Optional[List[Tuple[float, float, str]]] = None,
    ) -> OutageScenario:
        """
        Creates a realistic multi-tunnel / urban canyon corridor scenario
        with 3 sequential blackouts separated by recovery zones.
        """
        if windows is None:
            windows = [
                (20.0, 35.0, "Tunnel 1 (15s)"),
                (55.0, 85.0, "Tunnel 2 (30s)"),
                (110.0, 130.0, "Tunnel 3 (20s)"),
            ]
        return cls.create_time_window_scenario(
            name="multi_tunnel_corridor",
            description="Multi-tunnel corridor with 3 sequential GNSS outages and recovery zones",
            windows=windows,
        )

    @classmethod
    def create_periodic_scenario(
        cls,
        total_duration_s: float,
        outage_duration_s: float = 15.0,
        gap_duration_s: float = 30.0,
        initial_offset_s: float = 20.0,
    ) -> OutageScenario:
        """
        Creates a periodic blackout scenario repeating every (outage + gap) seconds.
        """
        windows = []
        t = initial_offset_s
        outage_idx = 1
        while t + outage_duration_s <= total_duration_s:
            windows.append((t, t + outage_duration_s, f"Periodic Outage #{outage_idx}"))
            t += outage_duration_s + gap_duration_s
            outage_idx += 1

        return cls.create_time_window_scenario(
            name="periodic_blackouts",
            description=f"Periodic {outage_duration_s}s blackouts every {gap_duration_s}s",
            windows=windows,
        )

    @classmethod
    def create_stochastic_scenario(
        cls,
        total_duration_s: float,
        num_outages: int = 4,
        min_duration_s: float = 8.0,
        max_duration_s: float = 25.0,
        min_gap_s: float = 15.0,
        seed: int = 42,
    ) -> OutageScenario:
        """
        Generates pseudo-random intermittent dropouts simulating dense urban canyons.
        """
        rng = np.random.RandomState(seed)
        windows = []
        current_t = 15.0  # Allow initial GNSS lock

        for i in range(1, num_outages + 1):
            if current_t >= total_duration_s - min_duration_s:
                break
            dur = rng.uniform(min_duration_s, max_duration_s)
            end_t = min(current_t + dur, total_duration_s - 5.0)
            if end_t <= current_t:
                break
            windows.append((current_t, end_t, f"Urban Canyon Drop #{i}"))
            gap = rng.uniform(min_gap_s, min_gap_s * 2.0)
            current_t = end_t + gap

        return cls.create_time_window_scenario(
            name="urban_canyon_stochastic",
            description=f"Stochastic urban canyon dropouts (seed={seed})",
            windows=windows,
        )

    # -----------------------------------------------------------------------
    # Mask Application
    # -----------------------------------------------------------------------

    def apply_outage_mask(
        self,
        df: pd.DataFrame,
        scenario: OutageScenario,
        time_col: Optional[str] = None,
        mask_gnss_values: bool = True,
    ) -> Tuple[pd.DataFrame, OutageScenario]:
        """
        Applies scenario blackout windows to a sensor DataFrame.

        Added columns:
            - `is_outage`: bool, True during any simulated blackout window.
            - `gnss_available`: int, 0 during blackout, 1 when GNSS is available.
            - `outage_id`: int, ID of current blackout window, 0 when GNSS available.
            - `outage_label`: str, Label of the current outage window or 'GNSS Active'.
            - `time_elapsed_s`: float, Elapsed time from trip start (seconds).

        Masking behavior:
            When `mask_gnss_values=True`, GNSS observation columns
            (`gnss_lat`, `gnss_lon`, `gnss_speed`, `gnss_heading`, `gnss_accuracy`)
            are replaced with NaN during blackout.
            Ground-truth columns (`ref_lat`, `ref_lon`, `ref_speed`, etc.) remain intact.

        Args:
            df: Input calibrated/preprocessed DataFrame.
            scenario: OutageScenario defining blackout windows.
            time_col: Column name containing timestamps in milliseconds (auto-detected if None).
            mask_gnss_values: Whether to set GNSS measurement columns to NaN in blackout.

        Returns:
            Tuple of (masked_df, updated_scenario_with_indices).
        """
        df_out = df.copy()

        # Auto-detect timestamp column
        if time_col is None:
            if "timestamp" in df_out.columns:
                time_col = "timestamp"
            elif "timestamp_ms" in df_out.columns:
                time_col = "timestamp_ms"
            else:
                time_col = df_out.columns[0]

        # Compute elapsed time in seconds
        t_start_ms = df_out[time_col].iloc[0]
        time_elapsed_s = (df_out[time_col] - t_start_ms) / 1000.0
        df_out["time_elapsed_s"] = time_elapsed_s.values

        # Initialize mask columns
        df_out["is_outage"] = False
        df_out["gnss_available"] = 1
        df_out["outage_id"] = 0
        df_out["outage_label"] = "GNSS Active"

        resolved_windows = []

        # Precompute cumulative distance if ref coordinates exist
        cum_dists = None
        if "ref_lat" in df_out.columns and "ref_lon" in df_out.columns:
            from src.ins.integration import haversine_distance_deg
            lats = df_out["ref_lat"].values
            lons = df_out["ref_lon"].values
            # Vectorized step distance calculation
            step_dists = haversine_distance_deg(lats[:-1], lons[:-1], lats[1:], lons[1:])
            step_dists = np.nan_to_num(step_dists, nan=0.0)
            cum_dists = np.insert(np.cumsum(step_dists), 0, 0.0)

        for window in scenario.windows:
            # Match boolean condition for time window
            mask = (time_elapsed_s >= window.start_time_s) & (time_elapsed_s <= window.end_time_s)
            matching_indices = np.where(mask)[0]

            start_idx = int(matching_indices[0]) if len(matching_indices) > 0 else None
            end_idx = int(matching_indices[-1]) if len(matching_indices) > 0 else None

            start_dist = float(cum_dists[start_idx]) if cum_dists is not None and start_idx is not None else None
            end_dist = float(cum_dists[end_idx]) if cum_dists is not None and end_idx is not None else None

            updated_window = OutageWindow(
                outage_id=window.outage_id,
                start_time_s=window.start_time_s,
                end_time_s=window.end_time_s,
                label=window.label,
                start_idx=start_idx,
                end_idx=end_idx,
                start_dist_m=start_dist,
                end_dist_m=end_dist,
            )
            resolved_windows.append(updated_window)

            if len(matching_indices) > 0:
                df_out.loc[mask, "is_outage"] = True
                df_out.loc[mask, "gnss_available"] = 0
                df_out.loc[mask, "outage_id"] = window.outage_id
                df_out.loc[mask, "outage_label"] = window.label

        # Mask GNSS measurement columns during blackout
        if mask_gnss_values:
            gnss_cols = ["gnss_lat", "gnss_lon", "gnss_speed", "gnss_heading", "gnss_accuracy"]
            for col in gnss_cols:
                if col in df_out.columns:
                    raw_col = f"raw_{col}"
                    if raw_col not in df_out.columns:
                        df_out[raw_col] = df_out[col]
                    df_out.loc[df_out["is_outage"], col] = np.nan

        resolved_scenario = OutageScenario(
            name=scenario.name,
            description=scenario.description,
            windows=resolved_windows,
        )

        return df_out, resolved_scenario
