"""
IDR MVP -- HMM-Based Map Matcher (Phase 10)
=============================================
Matches a sequence of GPS/fusion positions to road segments using a
Hidden Markov Model with Viterbi decoding (MVP.md §14).

The matcher does NOT simply snap every point to the nearest road.
It considers trajectory continuity and heading alignment:

    Emission probability:
        P(observation | road) ∝ exp(-d²/2σ²) × heading_alignment_factor

    Transition probability:
        P(road_j at t | road_i at t-1) ∝ exp(-|Δ_gc - Δ_road| / β)

    Viterbi decoding:
        Finds the globally optimal sequence of road matches.

Key Design Decisions:
    - Roads can be traversed in both directions (heading diff capped at 90°)
    - Adjacent/connected segments have boosted transition probability
    - Stationary points (speed ≈ 0) skip matching to prevent jitter
    - Graceful fallback: if no candidates exist, returns unmatched position
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from .road_graph import RoadGraph, RoadCandidate, RoadSegment

logger = logging.getLogger(__name__)


@dataclass
class MapMatchResult:
    """Result of map matching for a single position.

    Attributes:
        timestamp_ms:    Original timestamp.
        raw_lat:         Input latitude (from fusion).
        raw_lon:         Input longitude (from fusion).
        matched_lat:     Corrected latitude (snapped to road).
        matched_lon:     Corrected longitude (snapped to road).
        road_id:         Matched road segment ID (u, v, key) or None.
        road_name:       Road name if available.
        road_type:       OSM highway type.
        road_bearing_deg: Road bearing at snap point (degrees).
        snap_distance_m: Perpendicular distance from raw to snapped point (m).
        heading_diff_deg: Heading difference between vehicle and road (degrees).
        confidence:      Match confidence score [0.0, 1.0].
        is_matched:      Whether the point was successfully matched to a road.
    """
    timestamp_ms: int = 0
    raw_lat: float = 0.0
    raw_lon: float = 0.0
    matched_lat: float = 0.0
    matched_lon: float = 0.0
    road_id: Optional[Tuple[int, int, int]] = None
    road_name: str = ""
    road_type: str = ""
    road_bearing_deg: float = 0.0
    snap_distance_m: float = 0.0
    heading_diff_deg: float = 0.0
    confidence: float = 0.0
    is_matched: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "raw_lat": self.raw_lat,
            "raw_lon": self.raw_lon,
            "mm_lat": self.matched_lat,
            "mm_lon": self.matched_lon,
            "mm_road_id": str(self.road_id) if self.road_id else "",
            "mm_road_name": self.road_name,
            "mm_road_type": self.road_type,
            "mm_road_bearing_deg": self.road_bearing_deg,
            "mm_snap_dist_m": self.snap_distance_m,
            "mm_heading_diff_deg": self.heading_diff_deg,
            "mm_confidence": self.confidence,
            "mm_is_matched": self.is_matched,
        }


class HMMMapMatcher:
    """
    HMM-based map matcher with Viterbi decoding.

    Matches a trajectory to the most probable sequence of road segments
    by jointly considering position proximity, heading alignment, and
    transition plausibility between consecutive road assignments.

    Parameters
    ----------
    road_graph : RoadGraph
        Loaded road graph with spatial index.
    sigma_pos_m : float
        Position emission standard deviation (metres). Controls how
        strongly distance-to-road affects emission probability.
    sigma_heading_deg : float
        Heading emission standard deviation (degrees). Controls how
        strongly heading alignment affects emission probability.
    beta_transition : float
        Transition probability decay rate. Higher = more tolerant of
        route-distance vs great-circle-distance discrepancy.
    search_radius_m : float
        Maximum search radius for candidate roads (metres).
    max_candidates : int
        Maximum number of road candidates per position.
    min_speed_ms : float
        Minimum speed for matching. Below this, the point is considered
        stationary and the previous match is carried forward.
    heading_weight : float
        Weight of heading alignment in emission probability [0, 1].
        0 = position only, 1 = equal weight to heading and position.
    """

    def __init__(
        self,
        road_graph: RoadGraph,
        sigma_pos_m: float = 15.0,
        sigma_heading_deg: float = 25.0,
        beta_transition: float = 5.0,
        search_radius_m: float = 50.0,
        max_candidates: int = 8,
        min_speed_ms: float = 0.5,
        heading_weight: float = 0.6,
    ):
        self.road_graph = road_graph
        self.sigma_pos_m = max(1.0, sigma_pos_m)
        self.sigma_heading_deg = max(1.0, sigma_heading_deg)
        self.beta_transition = max(0.1, beta_transition)
        self.search_radius_m = max(10.0, search_radius_m)
        self.max_candidates = max(1, max_candidates)
        self.min_speed_ms = max(0.0, min_speed_ms)
        self.heading_weight = np.clip(heading_weight, 0.0, 1.0)

        # Last matched state (for incremental mode)
        self._last_candidates: Optional[List[RoadCandidate]] = None
        self._last_match_idx: Optional[int] = None

    # ------------------------------------------------------------------
    # Emission & Transition Probability
    # ------------------------------------------------------------------

    def emission_probability(
        self,
        candidate: RoadCandidate,
        vehicle_heading_deg: Optional[float] = None,
    ) -> float:
        """
        Computes the emission probability P(observation | road candidate).

        Combines:
        1. Gaussian on perpendicular distance: exp(-d² / 2σ²)
        2. Gaussian on heading difference:     exp(-Δθ² / 2σ_θ²)

        Parameters
        ----------
        candidate : RoadCandidate
            Road candidate with distance and heading info.
        vehicle_heading_deg : float, optional
            Vehicle heading for heading alignment scoring.

        Returns
        -------
        Emission probability (unnormalized, > 0).
        """
        # Position component
        dist_prob = np.exp(
            -0.5 * (candidate.distance_m / self.sigma_pos_m) ** 2
        )

        # Heading component
        if vehicle_heading_deg is not None:
            hdg_diff = self.road_graph._heading_difference(
                vehicle_heading_deg, candidate.bearing_deg
            )
            hdg_prob = np.exp(
                -0.5 * (hdg_diff / self.sigma_heading_deg) ** 2
            )
            # Weighted combination
            prob = (1.0 - self.heading_weight) * dist_prob + self.heading_weight * hdg_prob
        else:
            prob = dist_prob

        # Floor to prevent log(0)
        return max(prob, 1e-30)

    def transition_probability(
        self,
        cand_prev: RoadCandidate,
        cand_curr: RoadCandidate,
        gc_distance_m: float,
    ) -> float:
        """
        Computes the transition probability P(road_curr | road_prev).

        Uses the absolute difference between the great-circle distance
        (between consecutive observations) and the road-network distance
        (between consecutive candidate snap points).

        Topologically connected segments get a bonus.

        Parameters
        ----------
        cand_prev : RoadCandidate
            Previous time step's road candidate.
        cand_curr : RoadCandidate
            Current time step's road candidate.
        gc_distance_m : float
            Great-circle distance between observations at t-1 and t.

        Returns
        -------
        Transition probability (unnormalized, > 0).
        """
        seg_prev = cand_prev.segment
        seg_curr = cand_curr.segment

        # Same segment — very high probability
        if seg_prev.edge_id == seg_curr.edge_id:
            return 1.0

        # Compute road-network distance between snap points
        # For MVP: use great-circle distance between snap points as approximation
        snap_gc = self._great_circle_m(
            cand_prev.snap_lat, cand_prev.snap_lon,
            cand_curr.snap_lat, cand_curr.snap_lon,
        )

        # Route consistency: |gc_observation - gc_snap| should be small
        route_diff = abs(gc_distance_m - snap_gc)
        route_prob = np.exp(-route_diff / self.beta_transition)

        # Connectivity bonus
        if self.road_graph.are_connected(seg_prev, seg_curr):
            route_prob *= 1.5

        return max(route_prob, 1e-30)

    # ------------------------------------------------------------------
    # Batch Viterbi Matching
    # ------------------------------------------------------------------

    def match_batch(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        headings_deg: Optional[np.ndarray] = None,
        speeds_ms: Optional[np.ndarray] = None,
        timestamps_ms: Optional[np.ndarray] = None,
    ) -> List[MapMatchResult]:
        """
        Matches a full trajectory using Viterbi decoding.

        This is the main entry point for batch map matching.

        Parameters
        ----------
        lats, lons : np.ndarray
            Trajectory latitude/longitude arrays (degrees).
        headings_deg : np.ndarray, optional
            Vehicle heading at each point (degrees).
        speeds_ms : np.ndarray, optional
            Vehicle speed at each point (m/s).
        timestamps_ms : np.ndarray, optional
            Timestamps in milliseconds.

        Returns
        -------
        List of MapMatchResult, one per input point.
        """
        n = len(lats)
        if n == 0:
            return []

        # Default arrays
        if headings_deg is None:
            headings_deg = np.full(n, np.nan)
        if speeds_ms is None:
            speeds_ms = np.full(n, np.nan)
        if timestamps_ms is None:
            timestamps_ms = np.arange(n, dtype=np.int64)

        # Step 1: Find candidates for each observation
        all_candidates: List[List[RoadCandidate]] = []
        for i in range(n):
            if np.isnan(lats[i]) or np.isnan(lons[i]):
                all_candidates.append([])
                continue

            hdg = float(headings_deg[i]) if not np.isnan(headings_deg[i]) else None
            cands = self.road_graph.find_nearby_roads(
                lat=float(lats[i]),
                lon=float(lons[i]),
                radius_m=self.search_radius_m,
                max_candidates=self.max_candidates,
                vehicle_heading_deg=hdg,
            )
            all_candidates.append(cands)

        # Step 2: Viterbi forward pass
        # V[t][j] = log-probability of the most likely path ending at candidate j at time t
        # B[t][j] = index of the best predecessor at time t-1
        V: List[Optional[np.ndarray]] = []
        B: List[Optional[np.ndarray]] = []
        candidate_counts: List[int] = []

        for t in range(n):
            cands = all_candidates[t]
            nc = len(cands)
            candidate_counts.append(nc)

            if nc == 0:
                V.append(None)
                B.append(None)
                continue

            hdg = float(headings_deg[t]) if not np.isnan(headings_deg[t]) else None

            if t == 0 or V[t - 1] is None:
                # Initialize: emission probabilities only
                log_probs = np.array([
                    np.log(self.emission_probability(c, hdg))
                    for c in cands
                ])
                V.append(log_probs)
                B.append(np.zeros(nc, dtype=np.int32))
            else:
                # Find the last valid time step
                t_prev = t - 1
                while t_prev >= 0 and V[t_prev] is None:
                    t_prev -= 1

                if t_prev < 0:
                    # No valid predecessor
                    log_probs = np.array([
                        np.log(self.emission_probability(c, hdg))
                        for c in cands
                    ])
                    V.append(log_probs)
                    B.append(np.zeros(nc, dtype=np.int32))
                    continue

                prev_cands = all_candidates[t_prev]
                nc_prev = len(prev_cands)

                # Great-circle distance between consecutive observations
                gc_dist = self._great_circle_m(
                    float(lats[t_prev]), float(lons[t_prev]),
                    float(lats[t]), float(lons[t]),
                )

                log_probs = np.full(nc, -np.inf)
                back_ptrs = np.zeros(nc, dtype=np.int32)

                for j, cand_j in enumerate(cands):
                    emission_lp = np.log(self.emission_probability(cand_j, hdg))
                    best_lp = -np.inf
                    best_prev = 0

                    for i, cand_i in enumerate(prev_cands):
                        trans_lp = np.log(self.transition_probability(
                            cand_i, cand_j, gc_dist
                        ))
                        total_lp = V[t_prev][i] + trans_lp + emission_lp

                        if total_lp > best_lp:
                            best_lp = total_lp
                            best_prev = i

                    log_probs[j] = best_lp
                    back_ptrs[j] = best_prev

                V.append(log_probs)
                B.append(back_ptrs)

        # Step 3: Viterbi backtracking
        best_path: List[int] = [0] * n  # index into all_candidates[t]
        matched_flags: List[bool] = [False] * n

        # Find the last valid time step
        t_last = n - 1
        while t_last >= 0 and V[t_last] is None:
            t_last -= 1

        if t_last >= 0 and V[t_last] is not None:
            best_path[t_last] = int(np.argmax(V[t_last]))
            matched_flags[t_last] = True

            for t in range(t_last - 1, -1, -1):
                if V[t] is None:
                    continue

                # Find the next valid time step
                t_next = t + 1
                while t_next <= t_last and B[t_next] is None:
                    t_next += 1

                if t_next <= t_last and B[t_next] is not None:
                    best_path[t] = int(B[t_next][best_path[t_next]])
                    matched_flags[t] = True
                else:
                    best_path[t] = int(np.argmax(V[t]))
                    matched_flags[t] = True

        # Step 4: Build results
        results: List[MapMatchResult] = []
        last_valid_result: Optional[MapMatchResult] = None

        for t in range(n):
            ts = int(timestamps_ms[t])
            raw_lat = float(lats[t]) if not np.isnan(lats[t]) else 0.0
            raw_lon = float(lons[t]) if not np.isnan(lons[t]) else 0.0

            # Check if stationary
            spd = float(speeds_ms[t]) if not np.isnan(speeds_ms[t]) else float('inf')
            if spd < self.min_speed_ms and last_valid_result is not None:
                # Carry forward previous match
                result = MapMatchResult(
                    timestamp_ms=ts,
                    raw_lat=raw_lat,
                    raw_lon=raw_lon,
                    matched_lat=last_valid_result.matched_lat,
                    matched_lon=last_valid_result.matched_lon,
                    road_id=last_valid_result.road_id,
                    road_name=last_valid_result.road_name,
                    road_type=last_valid_result.road_type,
                    road_bearing_deg=last_valid_result.road_bearing_deg,
                    snap_distance_m=0.0,
                    heading_diff_deg=last_valid_result.heading_diff_deg,
                    confidence=last_valid_result.confidence * 0.95,
                    is_matched=True,
                )
                results.append(result)
                continue

            cands = all_candidates[t]
            if not matched_flags[t] or not cands:
                # Unmatched — return raw position
                result = MapMatchResult(
                    timestamp_ms=ts,
                    raw_lat=raw_lat,
                    raw_lon=raw_lon,
                    matched_lat=raw_lat,
                    matched_lon=raw_lon,
                    is_matched=False,
                    confidence=0.0,
                )
                results.append(result)
                continue

            # Get the best candidate from Viterbi
            best_idx = best_path[t]
            if best_idx >= len(cands):
                best_idx = 0
            best_cand = cands[best_idx]

            # Compute confidence from emission probability
            hdg = float(headings_deg[t]) if not np.isnan(headings_deg[t]) else None
            em_prob = self.emission_probability(best_cand, hdg)
            confidence = float(np.clip(em_prob, 0.0, 1.0))

            result = MapMatchResult(
                timestamp_ms=ts,
                raw_lat=raw_lat,
                raw_lon=raw_lon,
                matched_lat=best_cand.snap_lat,
                matched_lon=best_cand.snap_lon,
                road_id=best_cand.segment.edge_id,
                road_name=best_cand.segment.name,
                road_type=best_cand.segment.road_type,
                road_bearing_deg=best_cand.bearing_deg,
                snap_distance_m=best_cand.distance_m,
                heading_diff_deg=best_cand.heading_diff_deg,
                confidence=confidence,
                is_matched=True,
            )
            results.append(result)
            last_valid_result = result

        logger.info(
            "Map matching complete: %d/%d points matched",
            sum(1 for r in results if r.is_matched),
            n,
        )
        return results

    # ------------------------------------------------------------------
    # Incremental Matching (single-step)
    # ------------------------------------------------------------------

    def match_single(
        self,
        lat: float,
        lon: float,
        heading_deg: Optional[float] = None,
        speed_ms: Optional[float] = None,
        timestamp_ms: int = 0,
    ) -> MapMatchResult:
        """
        Matches a single position incrementally (greedy, not Viterbi).

        Uses emission probability + transition from last match.
        Suitable for real-time/streaming applications.

        Parameters
        ----------
        lat, lon : float
            Position in decimal degrees.
        heading_deg : float, optional
            Vehicle heading.
        speed_ms : float, optional
            Vehicle speed.
        timestamp_ms : int
            Timestamp.

        Returns
        -------
        MapMatchResult
        """
        if np.isnan(lat) or np.isnan(lon):
            return MapMatchResult(
                timestamp_ms=timestamp_ms,
                raw_lat=lat,
                raw_lon=lon,
                matched_lat=lat,
                matched_lon=lon,
                is_matched=False,
            )

        cands = self.road_graph.find_nearby_roads(
            lat=lat,
            lon=lon,
            radius_m=self.search_radius_m,
            max_candidates=self.max_candidates,
            vehicle_heading_deg=heading_deg,
        )

        if not cands:
            return MapMatchResult(
                timestamp_ms=timestamp_ms,
                raw_lat=lat,
                raw_lon=lon,
                matched_lat=lat,
                matched_lon=lon,
                is_matched=False,
            )

        # Score each candidate
        best_score = -np.inf
        best_cand = cands[0]

        for cand in cands:
            em_score = np.log(self.emission_probability(cand, heading_deg))

            trans_score = 0.0
            if (
                self._last_candidates is not None
                and self._last_match_idx is not None
                and self._last_match_idx < len(self._last_candidates)
            ):
                prev_cand = self._last_candidates[self._last_match_idx]
                gc_dist = self._great_circle_m(
                    prev_cand.snap_lat, prev_cand.snap_lon,
                    cand.snap_lat, cand.snap_lon,
                )
                trans_score = np.log(self.transition_probability(
                    prev_cand, cand, gc_dist
                ))

            total = em_score + trans_score
            if total > best_score:
                best_score = total
                best_cand = cand

        # Update state
        self._last_candidates = cands
        self._last_match_idx = cands.index(best_cand)

        confidence = float(np.clip(
            self.emission_probability(best_cand, heading_deg), 0.0, 1.0
        ))

        return MapMatchResult(
            timestamp_ms=timestamp_ms,
            raw_lat=lat,
            raw_lon=lon,
            matched_lat=best_cand.snap_lat,
            matched_lon=best_cand.snap_lon,
            road_id=best_cand.segment.edge_id,
            road_name=best_cand.segment.name,
            road_type=best_cand.segment.road_type,
            road_bearing_deg=best_cand.bearing_deg,
            snap_distance_m=best_cand.distance_m,
            heading_diff_deg=best_cand.heading_diff_deg,
            confidence=confidence,
            is_matched=True,
        )

    def reset(self) -> None:
        """Resets incremental matching state."""
        self._last_candidates = None
        self._last_match_idx = None

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _great_circle_m(
        lat1_deg: float,
        lon1_deg: float,
        lat2_deg: float,
        lon2_deg: float,
    ) -> float:
        """Haversine great-circle distance in metres."""
        R = 6_371_000.0
        lat1 = np.radians(lat1_deg)
        lat2 = np.radians(lat2_deg)
        dlat = lat2 - lat1
        dlon = np.radians(lon2_deg - lon1_deg)

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        return float(2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))
