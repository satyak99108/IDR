"""
IDR MVP -- Map Matching Engine (Phase 10)
==========================================
High-level orchestrator that connects the RoadGraph and HMMMapMatcher
to the fusion engine output.  Provides a one-call ``apply_map_matching``
convenience function for batch DataFrames (MVP.md §14).

Usage:
    from src.map_matching import apply_map_matching

    fusion_df = fusion_engine.run_batch(...)
    result_df = apply_map_matching(fusion_df)
    # result_df now has mm_lat, mm_lon, mm_road_id, mm_confidence, etc.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from .road_graph import RoadGraph
from .hmm_matcher import HMMMapMatcher, MapMatchResult

logger = logging.getLogger(__name__)


class MapMatcherEngine:
    """
    High-level map matching engine.

    Orchestrates:
    1. Loading/caching the road graph for the trajectory's geographic area.
    2. Running the HMM matcher over batch fusion positions.
    3. Producing MapMatchResult objects with corrected coordinates.

    Parameters
    ----------
    road_graph : RoadGraph, optional
        Pre-loaded road graph. If None, a new one is created and loaded
        from the trajectory bounding box.
    matcher : HMMMapMatcher, optional
        Pre-configured matcher. If None, created with default parameters.
    graph_buffer_m : float
        Buffer around trajectory bounding box for graph download (metres).
    sigma_pos_m : float
        Position emission std dev for HMM matcher.
    sigma_heading_deg : float
        Heading emission std dev for HMM matcher.
    beta_transition : float
        Transition probability decay rate.
    search_radius_m : float
        Candidate road search radius (metres).
    max_candidates : int
        Max road candidates per position.
    cache_dir : str, optional
        Directory for cached GraphML files.
    """

    def __init__(
        self,
        road_graph: Optional[RoadGraph] = None,
        matcher: Optional[HMMMapMatcher] = None,
        graph_buffer_m: float = 500.0,
        sigma_pos_m: float = 15.0,
        sigma_heading_deg: float = 25.0,
        beta_transition: float = 5.0,
        search_radius_m: float = 50.0,
        max_candidates: int = 8,
        cache_dir: Optional[str] = None,
    ):
        self.graph_buffer_m = graph_buffer_m
        self.sigma_pos_m = sigma_pos_m
        self.sigma_heading_deg = sigma_heading_deg
        self.beta_transition = beta_transition
        self.search_radius_m = search_radius_m
        self.max_candidates = max_candidates

        self.road_graph = road_graph or RoadGraph(cache_dir=cache_dir)
        self.matcher = matcher

    def _ensure_graph_loaded(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> None:
        """Ensures the road graph is loaded for the trajectory's area."""
        if not self.road_graph.is_loaded:
            logger.info("Auto-loading road graph from trajectory bounding box...")
            self.road_graph.load_from_trajectory(
                lats=lats,
                lons=lons,
                buffer_m=self.graph_buffer_m,
            )

    def _ensure_matcher(self) -> None:
        """Ensures the HMM matcher is initialized."""
        if self.matcher is None:
            self.matcher = HMMMapMatcher(
                road_graph=self.road_graph,
                sigma_pos_m=self.sigma_pos_m,
                sigma_heading_deg=self.sigma_heading_deg,
                beta_transition=self.beta_transition,
                search_radius_m=self.search_radius_m,
                max_candidates=self.max_candidates,
            )

    def match_trajectory(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        headings_deg: Optional[np.ndarray] = None,
        speeds_ms: Optional[np.ndarray] = None,
        timestamps_ms: Optional[np.ndarray] = None,
    ) -> List[MapMatchResult]:
        """
        Matches a trajectory to road segments.

        Loads the road graph if not already loaded.

        Parameters
        ----------
        lats, lons : np.ndarray
            Trajectory coordinates in degrees.
        headings_deg : np.ndarray, optional
            Vehicle heading at each point (degrees).
        speeds_ms : np.ndarray, optional
            Vehicle speed at each point (m/s).
        timestamps_ms : np.ndarray, optional
            Timestamps in milliseconds.

        Returns
        -------
        List of MapMatchResult.
        """
        self._ensure_graph_loaded(lats, lons)
        self._ensure_matcher()

        return self.matcher.match_batch(
            lats=lats,
            lons=lons,
            headings_deg=headings_deg,
            speeds_ms=speeds_ms,
            timestamps_ms=timestamps_ms,
        )

    def match_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies map matching to a fusion output DataFrame.

        Expects columns from ``GNSSINSFusionEngine.run_batch()``:
            - ``lat_deg``: Latitude
            - ``lon_deg``: Longitude
            - ``heading_deg``: Vehicle heading
            - ``speed_ms``: Vehicle speed
            - ``timestamp_ms``: Timestamp

        Returns a new DataFrame with the original columns plus:
            - ``mm_lat``: Map-matched latitude
            - ``mm_lon``: Map-matched longitude
            - ``mm_road_id``: Matched road segment ID
            - ``mm_road_name``: Road name
            - ``mm_road_type``: OSM highway type
            - ``mm_road_bearing_deg``: Road bearing at snap point
            - ``mm_snap_dist_m``: Snap distance (metres)
            - ``mm_heading_diff_deg``: Heading difference (degrees)
            - ``mm_confidence``: Match confidence [0, 1]
            - ``mm_is_matched``: Whether point was matched

        Parameters
        ----------
        df : pd.DataFrame
            Fusion engine output DataFrame.

        Returns
        -------
        pd.DataFrame with map-matching columns appended.
        """
        if df.empty:
            return df.copy()

        # Extract arrays
        lats = df["lat_deg"].values.astype(np.float64)
        lons = df["lon_deg"].values.astype(np.float64)

        headings = (
            df["heading_deg"].values.astype(np.float64)
            if "heading_deg" in df.columns
            else None
        )
        speeds = (
            df["speed_ms"].values.astype(np.float64)
            if "speed_ms" in df.columns
            else None
        )
        timestamps = (
            df["timestamp_ms"].values.astype(np.int64)
            if "timestamp_ms" in df.columns
            else None
        )

        # Run matching
        results = self.match_trajectory(
            lats=lats,
            lons=lons,
            headings_deg=headings,
            speeds_ms=speeds,
            timestamps_ms=timestamps,
        )

        # Append results to DataFrame
        result_df = df.copy()
        mm_data = pd.DataFrame([r.to_dict() for r in results])

        # Only add the map-matching columns (not duplicated raw/timestamp columns)
        mm_cols = [
            "mm_lat", "mm_lon", "mm_road_id", "mm_road_name", "mm_road_type",
            "mm_road_bearing_deg", "mm_snap_dist_m", "mm_heading_diff_deg",
            "mm_confidence", "mm_is_matched",
        ]
        for col in mm_cols:
            if col in mm_data.columns:
                result_df[col] = mm_data[col].values

        return result_df

    def get_summary(self) -> Dict[str, Any]:
        """Returns summary statistics about the map matching process."""
        return {
            "graph_loaded": self.road_graph.is_loaded,
            "num_road_segments": self.road_graph.num_segments,
            "search_radius_m": self.search_radius_m,
            "sigma_pos_m": self.sigma_pos_m,
            "sigma_heading_deg": self.sigma_heading_deg,
            "beta_transition": self.beta_transition,
            "max_candidates": self.max_candidates,
        }


def apply_map_matching(
    fusion_df: pd.DataFrame,
    road_graph: Optional[RoadGraph] = None,
    graph_buffer_m: float = 500.0,
    sigma_pos_m: float = 15.0,
    sigma_heading_deg: float = 25.0,
    beta_transition: float = 5.0,
    search_radius_m: float = 50.0,
    max_candidates: int = 8,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    One-call convenience function for map matching a fusion output DataFrame.

    This is the primary API for Phase 10 map matching integration.

    Parameters
    ----------
    fusion_df : pd.DataFrame
        Output from ``GNSSINSFusionEngine.run_batch()`` with columns:
        ``lat_deg``, ``lon_deg``, ``heading_deg``, ``speed_ms``, ``timestamp_ms``.
    road_graph : RoadGraph, optional
        Pre-loaded road graph. If None, auto-loaded from trajectory.
    graph_buffer_m : float
        Buffer around trajectory bounding box (metres).
    sigma_pos_m : float
        Position emission std dev.
    sigma_heading_deg : float
        Heading emission std dev.
    beta_transition : float
        Transition probability decay rate.
    search_radius_m : float
        Candidate search radius (metres).
    max_candidates : int
        Maximum candidates per position.
    cache_dir : str, optional
        Directory for cached GraphML files.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with map-matching columns appended:
        ``mm_lat``, ``mm_lon``, ``mm_road_id``, ``mm_confidence``,
        ``mm_snap_dist_m``, etc.

    Example
    -------
    >>> from src.fusion import GNSSINSFusionEngine
    >>> from src.map_matching import apply_map_matching
    >>>
    >>> engine = GNSSINSFusionEngine()
    >>> fusion_df = engine.run_batch(processed_df, ai_velocity=ai_vel, outage_mask=mask)
    >>> result_df = apply_map_matching(fusion_df)
    >>> print(result_df[["lat_deg", "lon_deg", "mm_lat", "mm_lon", "mm_confidence"]].head())
    """
    engine = MapMatcherEngine(
        road_graph=road_graph,
        graph_buffer_m=graph_buffer_m,
        sigma_pos_m=sigma_pos_m,
        sigma_heading_deg=sigma_heading_deg,
        beta_transition=beta_transition,
        search_radius_m=search_radius_m,
        max_candidates=max_candidates,
        cache_dir=cache_dir,
    )
    return engine.match_dataframe(fusion_df)
