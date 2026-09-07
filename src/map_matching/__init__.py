"""
IDR MVP -- Offline Map Matching Package (Phase 10)
====================================================
Uses offline OpenStreetMap road geometry to constrain the inertial/fusion
trajectory, reducing positional drift during GNSS outages (MVP.md §14).

Pipeline:
    FusionEngine output (lat, lon, heading, speed)
        → RoadGraph (OSMnx + cKDTree spatial index)
        → HMMMapMatcher (emission + transition probabilities, Viterbi)
        → MapMatchResult (corrected lat/lon, road_id, confidence)

Exports:
    RoadGraph          -- OSM road network loader with spatial index
    HMMMapMatcher      -- HMM-based map matcher with Viterbi decoding
    MapMatchResult     -- Matched position container
    MapMatcherEngine   -- High-level orchestrator
    apply_map_matching -- Convenience function for batch DataFrames
"""

from .road_graph import RoadGraph, RoadSegment, RoadCandidate
from .hmm_matcher import HMMMapMatcher, MapMatchResult
from .matcher_engine import MapMatcherEngine, apply_map_matching

__all__ = [
    "RoadGraph",
    "RoadSegment",
    "RoadCandidate",
    "HMMMapMatcher",
    "MapMatchResult",
    "MapMatcherEngine",
    "apply_map_matching",
]
