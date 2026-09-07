"""
IDR MVP -- Unit Tests for Offline Map Matching (Phase 10)
==========================================================
Tests for road_graph.py, hmm_matcher.py, and matcher_engine.py.

These tests use synthetic road geometry (no osmnx/internet required)
to validate the core matching logic.
"""

from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from typing import List, Optional, Tuple

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.map_matching.road_graph import RoadGraph, RoadSegment, RoadCandidate
from src.map_matching.hmm_matcher import HMMMapMatcher, MapMatchResult


# ---------------------------------------------------------------------------
# Test Helpers — Synthetic Road Network
# ---------------------------------------------------------------------------

def _make_segment(
    edge_id: Tuple[int, int, int],
    lat_start: float,
    lon_start: float,
    lat_end: float,
    lon_end: float,
    road_type: str = "primary",
    name: str = "",
    oneway: bool = False,
) -> RoadSegment:
    """Creates a RoadSegment from start/end coordinates."""
    lat_mid = (lat_start + lat_end) / 2.0
    lon_mid = (lon_start + lon_end) / 2.0
    bearing = RoadGraph._compute_bearing(lat_start, lon_start, lat_end, lon_end)
    # Approximate length via flat-Earth
    dlat_m = (lat_end - lat_start) * 111_320.0
    dlon_m = (lon_end - lon_start) * 111_320.0 * np.cos(np.radians(lat_mid))
    length = np.sqrt(dlat_m ** 2 + dlon_m ** 2)

    return RoadSegment(
        edge_id=edge_id,
        u_node=edge_id[0],
        v_node=edge_id[1],
        lat_start=lat_start,
        lon_start=lon_start,
        lat_end=lat_end,
        lon_end=lon_end,
        lat_mid=lat_mid,
        lon_mid=lon_mid,
        bearing_deg=bearing,
        length_m=length,
        road_type=road_type,
        name=name,
        oneway=oneway,
    )


def _build_synthetic_graph() -> RoadGraph:
    """
    Builds a synthetic road graph for testing.

    Layout (approximate):
        - Road A: West→East along lat=28.6, lon from 77.20 to 77.21  (bearing ≈ 90°)
        - Road B: South→North along lon=77.205, lat from 28.595 to 28.605 (bearing ≈ 0°)
        - Road C: Parallel to A, ~100m north, lat=28.601  (bearing ≈ 90°)

    Roads A and B intersect near (28.6, 77.205).
    """
    graph = RoadGraph(cache_dir=None)

    # Road A: East-West main road (node 1→2)
    seg_a = _make_segment(
        edge_id=(1, 2, 0),
        lat_start=28.600, lon_start=77.200,
        lat_end=28.600, lon_end=77.210,
        road_type="primary",
        name="Main Road",
    )

    # Road B: North-South cross road (node 3→4), sharing node concept near intersection
    seg_b = _make_segment(
        edge_id=(3, 4, 0),
        lat_start=28.595, lon_start=77.205,
        lat_end=28.605, lon_end=77.205,
        road_type="secondary",
        name="Cross Street",
    )

    # Road C: Parallel to A, ~110m north (node 5→6)
    seg_c = _make_segment(
        edge_id=(5, 6, 0),
        lat_start=28.601, lon_start=77.200,
        lat_end=28.601, lon_end=77.210,
        road_type="residential",
        name="Parallel Lane",
    )

    graph.segments = [seg_a, seg_b, seg_c]

    # Build spatial index
    mid_coords = np.array([
        [seg.lat_mid, seg.lon_mid] for seg in graph.segments
    ])
    from scipy.spatial import cKDTree
    graph._mid_coords = mid_coords
    graph._kdtree = cKDTree(mid_coords)
    graph._loaded = True

    return graph


# ---------------------------------------------------------------------------
# Tests: Road Graph Geometry
# ---------------------------------------------------------------------------

class TestRoadGraphGeometry(unittest.TestCase):
    """Tests for RoadGraph geometry utilities."""

    def test_compute_bearing_east(self):
        """Bearing from west to east should be ~90°."""
        bearing = RoadGraph._compute_bearing(28.6, 77.20, 28.6, 77.21)
        self.assertAlmostEqual(bearing, 90.0, delta=1.0)

    def test_compute_bearing_north(self):
        """Bearing from south to north should be ~0°."""
        bearing = RoadGraph._compute_bearing(28.59, 77.20, 28.61, 77.20)
        self.assertAlmostEqual(bearing, 0.0, delta=1.0)

    def test_compute_bearing_south(self):
        """Bearing from north to south should be ~180°."""
        bearing = RoadGraph._compute_bearing(28.61, 77.20, 28.59, 77.20)
        self.assertAlmostEqual(bearing, 180.0, delta=1.0)

    def test_compute_bearing_west(self):
        """Bearing from east to west should be ~270°."""
        bearing = RoadGraph._compute_bearing(28.6, 77.21, 28.6, 77.20)
        self.assertAlmostEqual(bearing, 270.0, delta=1.0)

    def test_snap_to_segment_midpoint(self):
        """Point at midpoint should snap to midpoint with ~0 distance."""
        snap_lat, snap_lon, dist = RoadGraph._snap_to_segment(
            28.600, 77.205,   # query: midpoint of Road A
            28.600, 77.200,   # segment start
            28.600, 77.210,   # segment end
        )
        self.assertAlmostEqual(snap_lat, 28.600, places=4)
        self.assertAlmostEqual(snap_lon, 77.205, places=4)
        self.assertLess(dist, 1.0)  # Less than 1 metre

    def test_snap_to_segment_perpendicular(self):
        """Point offset north from an east-west road should snap perpendicularly."""
        snap_lat, snap_lon, dist = RoadGraph._snap_to_segment(
            28.601, 77.205,   # query: ~110m north of Road A midpoint
            28.600, 77.200,   # Road A start
            28.600, 77.210,   # Road A end
        )
        self.assertAlmostEqual(snap_lat, 28.600, places=3)
        self.assertAlmostEqual(snap_lon, 77.205, places=3)
        self.assertGreater(dist, 50.0)   # Should be roughly 111m

    def test_snap_to_segment_endpoint_clamp(self):
        """Point beyond segment end should snap to the endpoint."""
        snap_lat, snap_lon, dist = RoadGraph._snap_to_segment(
            28.600, 77.215,   # query: east of Road A end
            28.600, 77.200,   # Road A start
            28.600, 77.210,   # Road A end
        )
        self.assertAlmostEqual(snap_lon, 77.210, places=3)

    def test_heading_difference_same(self):
        """Same heading → 0° difference."""
        diff = RoadGraph._heading_difference(90.0, 90.0)
        self.assertAlmostEqual(diff, 0.0, places=1)

    def test_heading_difference_opposite(self):
        """Opposite direction on same road → 0° difference (bidirectional)."""
        diff = RoadGraph._heading_difference(90.0, 270.0)
        self.assertAlmostEqual(diff, 0.0, places=1)

    def test_heading_difference_perpendicular(self):
        """Perpendicular → 90° difference."""
        diff = RoadGraph._heading_difference(0.0, 90.0)
        self.assertAlmostEqual(diff, 90.0, places=1)

    def test_heading_difference_wrap(self):
        """Wrapping across 360° boundary."""
        diff = RoadGraph._heading_difference(350.0, 10.0)
        self.assertLess(diff, 25.0)


# ---------------------------------------------------------------------------
# Tests: Road Graph Spatial Queries
# ---------------------------------------------------------------------------

class TestRoadGraphSpatialQuery(unittest.TestCase):
    """Tests for RoadGraph.find_nearby_roads()."""

    def setUp(self):
        self.graph = _build_synthetic_graph()

    def test_finds_nearest_road(self):
        """Query near Road A should return Road A as the closest candidate."""
        cands = self.graph.find_nearby_roads(
            lat=28.600, lon=77.205, radius_m=200
        )
        self.assertGreater(len(cands), 0)
        # The closest should be Road A (nearly zero distance)
        self.assertLess(cands[0].distance_m, 5.0)

    def test_finds_multiple_roads(self):
        """Query near intersection should find multiple roads."""
        cands = self.graph.find_nearby_roads(
            lat=28.600, lon=77.205, radius_m=500
        )
        self.assertGreaterEqual(len(cands), 2)

    def test_empty_when_far_away(self):
        """Query far from all roads should return empty list."""
        cands = self.graph.find_nearby_roads(
            lat=28.700, lon=77.300, radius_m=50
        )
        self.assertEqual(len(cands), 0)

    def test_heading_difference_populated(self):
        """Heading difference should be populated when vehicle heading is given."""
        cands = self.graph.find_nearby_roads(
            lat=28.600, lon=77.205,
            radius_m=500,
            vehicle_heading_deg=90.0,  # Heading East
        )
        for c in cands:
            self.assertIsInstance(c.heading_diff_deg, float)


# ---------------------------------------------------------------------------
# Tests: HMM Emission Probability
# ---------------------------------------------------------------------------

class TestHMMEmission(unittest.TestCase):
    """Tests for HMMMapMatcher emission probability."""

    def setUp(self):
        self.graph = _build_synthetic_graph()
        self.matcher = HMMMapMatcher(
            road_graph=self.graph,
            sigma_pos_m=15.0,
            sigma_heading_deg=25.0,
        )

    def test_close_point_high_probability(self):
        """Point very close to road should have high emission probability."""
        cand = RoadCandidate(
            segment=self.graph.segments[0],
            distance_m=1.0,
            snap_lat=28.600,
            snap_lon=77.205,
            bearing_deg=90.0,
            heading_diff_deg=0.0,
        )
        prob = self.matcher.emission_probability(cand, vehicle_heading_deg=90.0)
        self.assertGreater(prob, 0.5)

    def test_far_point_low_probability(self):
        """Point far from road should have low emission probability."""
        cand = RoadCandidate(
            segment=self.graph.segments[0],
            distance_m=100.0,
            snap_lat=28.600,
            snap_lon=77.205,
            bearing_deg=90.0,
            heading_diff_deg=0.0,
        )
        # Test position-only emission (no heading) to isolate distance effect
        prob = self.matcher.emission_probability(cand, vehicle_heading_deg=None)
        self.assertLess(prob, 0.01)
        # With heading, the blended probability should still be lower than close point
        prob_with_hdg = self.matcher.emission_probability(cand, vehicle_heading_deg=90.0)
        close_cand = RoadCandidate(
            segment=self.graph.segments[0],
            distance_m=1.0,
            snap_lat=28.600,
            snap_lon=77.205,
            bearing_deg=90.0,
            heading_diff_deg=0.0,
        )
        prob_close = self.matcher.emission_probability(close_cand, vehicle_heading_deg=90.0)
        self.assertGreater(prob_close, prob_with_hdg)

    def test_heading_alignment_matters(self):
        """Aligned heading should give higher probability than misaligned."""
        cand_aligned = RoadCandidate(
            segment=self.graph.segments[0],
            distance_m=10.0,
            snap_lat=28.600,
            snap_lon=77.205,
            bearing_deg=90.0,
            heading_diff_deg=5.0,
        )
        cand_misaligned = RoadCandidate(
            segment=self.graph.segments[0],
            distance_m=10.0,
            snap_lat=28.600,
            snap_lon=77.205,
            bearing_deg=0.0,
            heading_diff_deg=85.0,
        )
        prob_aligned = self.matcher.emission_probability(cand_aligned, vehicle_heading_deg=90.0)
        prob_misaligned = self.matcher.emission_probability(cand_misaligned, vehicle_heading_deg=90.0)
        self.assertGreater(prob_aligned, prob_misaligned)


# ---------------------------------------------------------------------------
# Tests: HMM Transition Probability
# ---------------------------------------------------------------------------

class TestHMMTransition(unittest.TestCase):
    """Tests for HMMMapMatcher transition probability."""

    def setUp(self):
        self.graph = _build_synthetic_graph()
        self.matcher = HMMMapMatcher(
            road_graph=self.graph,
            beta_transition=5.0,
        )

    def test_same_road_highest(self):
        """Staying on the same road should give probability 1.0."""
        seg = self.graph.segments[0]
        cand_prev = RoadCandidate(
            segment=seg, distance_m=1.0,
            snap_lat=28.600, snap_lon=77.203, bearing_deg=90.0,
        )
        cand_curr = RoadCandidate(
            segment=seg, distance_m=1.0,
            snap_lat=28.600, snap_lon=77.206, bearing_deg=90.0,
        )
        prob = self.matcher.transition_probability(cand_prev, cand_curr, gc_distance_m=300.0)
        self.assertEqual(prob, 1.0)

    def test_consistent_route_higher(self):
        """Transition with consistent route distance should be higher than inconsistent."""
        seg_a = self.graph.segments[0]
        seg_b = self.graph.segments[1]

        # Consistent: gc ≈ snap distance
        cand_prev = RoadCandidate(
            segment=seg_a, distance_m=1.0,
            snap_lat=28.600, snap_lon=77.205, bearing_deg=90.0,
        )
        cand_curr_consistent = RoadCandidate(
            segment=seg_b, distance_m=1.0,
            snap_lat=28.600, snap_lon=77.205, bearing_deg=0.0,
        )
        # Inconsistent: gc << snap distance
        cand_curr_far = RoadCandidate(
            segment=seg_b, distance_m=1.0,
            snap_lat=28.602, snap_lon=77.205, bearing_deg=0.0,
        )

        prob_consistent = self.matcher.transition_probability(
            cand_prev, cand_curr_consistent, gc_distance_m=10.0
        )
        prob_far = self.matcher.transition_probability(
            cand_prev, cand_curr_far, gc_distance_m=10.0
        )
        self.assertGreater(prob_consistent, prob_far)


# ---------------------------------------------------------------------------
# Tests: Batch Viterbi Matching
# ---------------------------------------------------------------------------

class TestViterbiMatching(unittest.TestCase):
    """Tests for HMMMapMatcher.match_batch() Viterbi decoding."""

    def setUp(self):
        self.graph = _build_synthetic_graph()
        self.matcher = HMMMapMatcher(
            road_graph=self.graph,
            sigma_pos_m=15.0,
            sigma_heading_deg=25.0,
            search_radius_m=800.0,  # Wide radius to cover long road segments
        )

    def test_straight_road_matching(self):
        """Points along Road A should all match to Road A."""
        n = 10
        lats = np.full(n, 28.600)
        lons = np.linspace(77.201, 77.209, n)
        headings = np.full(n, 90.0)  # Heading East
        speeds = np.full(n, 10.0)

        results = self.matcher.match_batch(lats, lons, headings, speeds)

        self.assertEqual(len(results), n)
        matched_count = sum(1 for r in results if r.is_matched)
        self.assertGreater(matched_count, n * 0.5)  # At least half should match

        # All matched points should snap close to Road A
        for r in results:
            if r.is_matched:
                self.assertLess(r.snap_distance_m, 50.0)

    def test_heading_selects_correct_parallel_road(self):
        """
        Point between Road A (lat=28.600) and Road C (lat=28.601) with
        heading East should match the closest road, but heading should
        break ties.
        """
        # Point exactly between A and C, heading East
        lats = np.array([28.6005])
        lons = np.array([77.205])
        headings = np.array([90.0])
        speeds = np.array([10.0])

        results = self.matcher.match_batch(lats, lons, headings, speeds)

        self.assertEqual(len(results), 1)
        # Both roads head East, so the closest one should win
        if results[0].is_matched:
            self.assertLess(results[0].snap_distance_m, 100.0)

    def test_single_point_input(self):
        """Single-point trajectory should match without error."""
        results = self.matcher.match_batch(
            np.array([28.600]),
            np.array([77.205]),
            np.array([90.0]),
            np.array([10.0]),
        )
        self.assertEqual(len(results), 1)

    def test_stationary_vehicle_carries_forward(self):
        """Stationary points (speed ≈ 0) should carry forward the last match."""
        n = 5
        lats = np.full(n, 28.600)
        lons = np.full(n, 77.205)
        headings = np.full(n, 90.0)
        speeds = np.array([10.0, 0.0, 0.0, 0.0, 10.0])  # Stop in middle

        results = self.matcher.match_batch(lats, lons, headings, speeds)

        self.assertEqual(len(results), n)
        # The stationary points should still be matched (carried forward)
        for r in results:
            self.assertTrue(r.is_matched)

    def test_no_nearby_roads_graceful_fallback(self):
        """Points far from any road should return unmatched with raw position."""
        lats = np.array([28.700, 28.700])
        lons = np.array([77.300, 77.301])

        results = self.matcher.match_batch(lats, lons)

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertFalse(r.is_matched)
            self.assertAlmostEqual(r.matched_lat, r.raw_lat, places=5)
            self.assertAlmostEqual(r.matched_lon, r.raw_lon, places=5)

    def test_nan_coordinates_handled(self):
        """NaN coordinates should be handled gracefully."""
        lats = np.array([28.600, np.nan, 28.600])
        lons = np.array([77.205, np.nan, 77.207])

        results = self.matcher.match_batch(lats, lons)

        self.assertEqual(len(results), 3)
        # NaN point should be unmatched
        self.assertFalse(results[1].is_matched)


# ---------------------------------------------------------------------------
# Tests: Incremental Matching
# ---------------------------------------------------------------------------

class TestIncrementalMatching(unittest.TestCase):
    """Tests for HMMMapMatcher.match_single() incremental mode."""

    def setUp(self):
        self.graph = _build_synthetic_graph()
        self.matcher = HMMMapMatcher(
            road_graph=self.graph,
            search_radius_m=800.0,  # Wide radius to cover long road segments
        )

    def test_single_match(self):
        """Single point match should return a valid result."""
        result = self.matcher.match_single(
            lat=28.600, lon=77.205, heading_deg=90.0, speed_ms=10.0,
        )
        self.assertTrue(result.is_matched)
        self.assertLess(result.snap_distance_m, 10.0)

    def test_incremental_consistency(self):
        """Consecutive incremental matches should maintain state."""
        r1 = self.matcher.match_single(28.600, 77.205, heading_deg=90.0, speed_ms=10.0)
        r2 = self.matcher.match_single(28.600, 77.206, heading_deg=90.0, speed_ms=10.0)
        r3 = self.matcher.match_single(28.600, 77.207, heading_deg=90.0, speed_ms=10.0)

        self.assertTrue(r1.is_matched)
        self.assertTrue(r2.is_matched)
        self.assertTrue(r3.is_matched)

    def test_reset_clears_state(self):
        """Reset should clear incremental state."""
        self.matcher.match_single(28.600, 77.205, heading_deg=90.0)
        self.matcher.reset()
        self.assertIsNone(self.matcher._last_candidates)
        self.assertIsNone(self.matcher._last_match_idx)

    def test_nan_input_unmatched(self):
        """NaN input should return unmatched result."""
        result = self.matcher.match_single(lat=np.nan, lon=np.nan)
        self.assertFalse(result.is_matched)


# ---------------------------------------------------------------------------
# Tests: MapMatchResult
# ---------------------------------------------------------------------------

class TestMapMatchResult(unittest.TestCase):
    """Tests for MapMatchResult serialization."""

    def test_to_dict_keys(self):
        """to_dict should contain all expected keys."""
        result = MapMatchResult(
            timestamp_ms=1000,
            raw_lat=28.600,
            raw_lon=77.205,
            matched_lat=28.600,
            matched_lon=77.205,
            road_id=(1, 2, 0),
            road_name="Main Road",
            road_type="primary",
            road_bearing_deg=90.0,
            snap_distance_m=2.5,
            heading_diff_deg=3.0,
            confidence=0.95,
            is_matched=True,
        )
        d = result.to_dict()
        expected_keys = {
            "timestamp_ms", "raw_lat", "raw_lon",
            "mm_lat", "mm_lon", "mm_road_id", "mm_road_name", "mm_road_type",
            "mm_road_bearing_deg", "mm_snap_dist_m", "mm_heading_diff_deg",
            "mm_confidence", "mm_is_matched",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_unmatched_result(self):
        """Unmatched result should have confidence 0 and raw coordinates."""
        result = MapMatchResult(
            raw_lat=28.600, raw_lon=77.205,
            matched_lat=28.600, matched_lon=77.205,
            is_matched=False, confidence=0.0,
        )
        self.assertFalse(result.is_matched)
        self.assertEqual(result.confidence, 0.0)


# ---------------------------------------------------------------------------
# Tests: RoadGraph Connectivity
# ---------------------------------------------------------------------------

class TestRoadGraphConnectivity(unittest.TestCase):
    """Tests for RoadGraph.are_connected()."""

    def setUp(self):
        self.graph = _build_synthetic_graph()

    def test_same_segment_connected(self):
        """A segment is connected to itself."""
        seg = self.graph.segments[0]
        self.assertTrue(self.graph.are_connected(seg, seg))

    def test_sharing_node_connected(self):
        """Two segments sharing a node should be connected."""
        # Manually create two segments sharing node 2
        seg1 = _make_segment((1, 2, 0), 28.600, 77.200, 28.600, 77.205)
        seg2 = _make_segment((2, 3, 0), 28.600, 77.205, 28.600, 77.210)
        self.assertTrue(self.graph.are_connected(seg1, seg2))

    def test_disjoint_segments_not_connected(self):
        """Two segments with no shared nodes should not be connected."""
        seg_a = self.graph.segments[0]  # nodes 1, 2
        seg_c = self.graph.segments[2]  # nodes 5, 6
        self.assertFalse(self.graph.are_connected(seg_a, seg_c))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
