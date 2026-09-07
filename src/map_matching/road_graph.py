"""
IDR MVP -- Road Graph Loader & Spatial Index (Phase 10)
========================================================
Downloads, caches, and indexes OpenStreetMap road networks for offline
map matching (MVP.md §14, TECH_STACK.md §1).

Uses OSMnx to fetch road graphs and scipy.spatial.cKDTree for sub-
millisecond nearest-edge spatial queries.

Usage:
    graph = RoadGraph()
    graph.load_from_center(lat=48.86, lon=2.35, radius_m=2000)
    candidates = graph.find_nearby_roads(lat=48.861, lon=2.351, radius_m=50)
"""

from __future__ import annotations

import os
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EARTH_RADIUS_M = 6_371_000.0
_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "osm_cache",
)


@dataclass
class RoadSegment:
    """A single road segment (edge) from the OSM graph.

    Attributes:
        edge_id:    Unique identifier for this edge (u, v, key).
        u_node:     Start node OSM ID.
        v_node:     End node OSM ID.
        lat_start:  Start point latitude (degrees).
        lon_start:  Start point longitude (degrees).
        lat_end:    End point latitude (degrees).
        lon_end:    End point longitude (degrees).
        lat_mid:    Midpoint latitude (degrees).
        lon_mid:    Midpoint longitude (degrees).
        bearing_deg: Road bearing at midpoint (0 = North, CW positive, degrees).
        length_m:   Edge length in metres.
        road_type:  OSM highway tag (e.g. 'primary', 'residential').
        name:       Road name if available.
        oneway:     Whether the road is one-way.
    """
    edge_id: Tuple[int, int, int]
    u_node: int
    v_node: int
    lat_start: float
    lon_start: float
    lat_end: float
    lon_end: float
    lat_mid: float
    lon_mid: float
    bearing_deg: float
    length_m: float
    road_type: str = "unclassified"
    name: str = ""
    oneway: bool = False


@dataclass
class RoadCandidate:
    """A candidate road segment near a query point.

    Attributes:
        segment:      The road segment.
        distance_m:   Perpendicular distance from query point to segment (m).
        snap_lat:     Latitude of the closest point on the segment.
        snap_lon:     Longitude of the closest point on the segment.
        bearing_deg:  Road bearing at the snap point (degrees).
        heading_diff_deg: Absolute heading difference from vehicle to road (degrees).
    """
    segment: RoadSegment
    distance_m: float
    snap_lat: float
    snap_lon: float
    bearing_deg: float
    heading_diff_deg: float = 0.0


class RoadGraph:
    """
    Offline OSM road network manager with fast spatial queries.

    Loads an OpenStreetMap road graph via ``osmnx``, extracts edge geometries,
    and builds a ``cKDTree`` over edge midpoints for sub-millisecond
    nearest-neighbour lookups.

    The graph is cached as a GraphML file so subsequent runs are fully offline.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or _DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.graph = None              # networkx.MultiDiGraph from osmnx
        self.segments: List[RoadSegment] = []
        self._kdtree: Optional[cKDTree] = None
        self._mid_coords: Optional[np.ndarray] = None  # (N, 2) lat/lon of midpoints
        self._loaded = False

    # ------------------------------------------------------------------
    # Graph Loading
    # ------------------------------------------------------------------

    def load_from_center(
        self,
        lat: float,
        lon: float,
        radius_m: float = 2000.0,
        network_type: str = "drive",
    ) -> None:
        """
        Loads the OSM road graph centered on (lat, lon) within radius_m.

        First checks for a cached GraphML file; downloads via osmnx if not found.

        Parameters
        ----------
        lat, lon : float
            Center point in decimal degrees.
        radius_m : float
            Search radius in metres (default 2000).
        network_type : str
            OSMnx network type (default 'drive' for drivable roads).
        """
        cache_key = self._cache_key(lat, lon, radius_m, network_type)
        cache_path = self.cache_dir / f"{cache_key}.graphml"

        if cache_path.exists():
            logger.info("Loading cached road graph: %s", cache_path)
            self._load_from_graphml(cache_path)
        else:
            logger.info(
                "Downloading OSM road graph: center=(%.5f, %.5f), radius=%.0fm",
                lat, lon, radius_m,
            )
            self._download_and_cache(lat, lon, radius_m, network_type, cache_path)

        self._extract_segments()
        self._build_spatial_index()
        self._loaded = True
        logger.info(
            "Road graph ready: %d segments, cKDTree built", len(self.segments)
        )

    def load_from_bbox(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        network_type: str = "drive",
    ) -> None:
        """
        Loads the OSM road graph for a geographic bounding box.

        Parameters
        ----------
        north, south, east, west : float
            Bounding box edges in decimal degrees.
        network_type : str
            OSMnx network type.
        """
        cache_key = self._cache_key_bbox(north, south, east, west, network_type)
        cache_path = self.cache_dir / f"{cache_key}.graphml"

        if cache_path.exists():
            logger.info("Loading cached road graph: %s", cache_path)
            self._load_from_graphml(cache_path)
        else:
            logger.info(
                "Downloading OSM road graph: bbox=(N=%.5f, S=%.5f, E=%.5f, W=%.5f)",
                north, south, east, west,
            )
            self._download_bbox_and_cache(north, south, east, west, network_type, cache_path)

        self._extract_segments()
        self._build_spatial_index()
        self._loaded = True
        logger.info(
            "Road graph ready: %d segments, cKDTree built", len(self.segments)
        )

    def load_from_trajectory(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        buffer_m: float = 500.0,
        network_type: str = "drive",
    ) -> None:
        """
        Auto-detects the bounding box from a trajectory and loads the graph.

        Parameters
        ----------
        lats, lons : np.ndarray
            Arrays of latitude/longitude values from the trajectory.
        buffer_m : float
            Extra buffer around the trajectory bounding box in metres.
        network_type : str
            OSMnx network type.
        """
        valid_mask = ~(np.isnan(lats) | np.isnan(lons))
        valid_lats = lats[valid_mask]
        valid_lons = lons[valid_mask]

        if len(valid_lats) == 0:
            raise ValueError("No valid lat/lon values in trajectory")

        # Compute bounding box with buffer
        lat_center = float(np.mean(valid_lats))
        buffer_deg = buffer_m / 111_320.0  # approx metres -> degrees

        north = float(np.max(valid_lats)) + buffer_deg
        south = float(np.min(valid_lats)) - buffer_deg
        east = float(np.max(valid_lons)) + buffer_deg / np.cos(np.radians(lat_center))
        west = float(np.min(valid_lons)) - buffer_deg / np.cos(np.radians(lat_center))

        self.load_from_bbox(north, south, east, west, network_type)

    # ------------------------------------------------------------------
    # Spatial Queries
    # ------------------------------------------------------------------

    def find_nearby_roads(
        self,
        lat: float,
        lon: float,
        radius_m: float = 50.0,
        max_candidates: int = 8,
        vehicle_heading_deg: Optional[float] = None,
    ) -> List[RoadCandidate]:
        """
        Finds road segments near a query point.

        Parameters
        ----------
        lat, lon : float
            Query point in decimal degrees.
        radius_m : float
            Search radius in metres.
        max_candidates : int
            Maximum number of candidates to return.
        vehicle_heading_deg : float, optional
            Vehicle heading for computing heading difference.

        Returns
        -------
        List of RoadCandidate sorted by distance (closest first).
        """
        if not self._loaded or self._kdtree is None:
            return []

        # Convert radius to approximate degrees for cKDTree query
        radius_deg = radius_m / 111_320.0

        # Query cKDTree for nearby midpoints
        query_point = np.array([lat, lon])
        indices = self._kdtree.query_ball_point(query_point, r=radius_deg)

        if not indices:
            return []

        candidates: List[RoadCandidate] = []
        for idx in indices:
            seg = self.segments[idx]

            # Compute perpendicular snap point and distance
            snap_lat, snap_lon, dist_m = self._snap_to_segment(
                lat, lon, seg.lat_start, seg.lon_start, seg.lat_end, seg.lon_end
            )

            if dist_m > radius_m:
                continue

            # Compute bearing at snap point
            bearing = self._compute_bearing(
                seg.lat_start, seg.lon_start, seg.lat_end, seg.lon_end
            )

            # Heading difference
            hdg_diff = 0.0
            if vehicle_heading_deg is not None:
                hdg_diff = self._heading_difference(vehicle_heading_deg, bearing)

            candidates.append(RoadCandidate(
                segment=seg,
                distance_m=dist_m,
                snap_lat=snap_lat,
                snap_lon=snap_lon,
                bearing_deg=bearing,
                heading_diff_deg=hdg_diff,
            ))

        # Sort by distance, take top candidates
        candidates.sort(key=lambda c: c.distance_m)
        return candidates[:max_candidates]

    @property
    def is_loaded(self) -> bool:
        """Whether a road graph has been loaded."""
        return self._loaded

    @property
    def num_segments(self) -> int:
        """Number of road segments in the graph."""
        return len(self.segments)

    # ------------------------------------------------------------------
    # Internal: Download & Cache
    # ------------------------------------------------------------------

    def _download_and_cache(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        network_type: str,
        cache_path: Path,
    ) -> None:
        """Downloads graph from OSM via osmnx and saves to cache."""
        try:
            import osmnx as ox

            ox.settings.use_cache = True
            ox.settings.log_console = False

            self.graph = ox.graph_from_point(
                (lat, lon),
                dist=radius_m,
                network_type=network_type,
                simplify=True,
            )
            ox.save_graphml(self.graph, cache_path)
            logger.info("Cached road graph to: %s", cache_path)

        except ImportError:
            raise ImportError(
                "osmnx is required for map matching. "
                "Install with: pip install osmnx>=1.6.0"
            )

    def _download_bbox_and_cache(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        network_type: str,
        cache_path: Path,
    ) -> None:
        """Downloads graph for a bounding box from OSM via osmnx."""
        try:
            import osmnx as ox

            ox.settings.use_cache = True
            ox.settings.log_console = False

            self.graph = ox.graph_from_bbox(
                bbox=(north, south, east, west),
                network_type=network_type,
                simplify=True,
            )
            ox.save_graphml(self.graph, cache_path)
            logger.info("Cached road graph to: %s", cache_path)

        except ImportError:
            raise ImportError(
                "osmnx is required for map matching. "
                "Install with: pip install osmnx>=1.6.0"
            )

    def _load_from_graphml(self, path: Path) -> None:
        """Loads a cached GraphML road graph."""
        try:
            import osmnx as ox
            self.graph = ox.load_graphml(path)
        except ImportError:
            raise ImportError(
                "osmnx is required for map matching. "
                "Install with: pip install osmnx>=1.6.0"
            )

    # ------------------------------------------------------------------
    # Internal: Segment Extraction
    # ------------------------------------------------------------------

    def _extract_segments(self) -> None:
        """Extracts RoadSegment objects from the loaded networkx graph."""
        if self.graph is None:
            return

        self.segments = []
        nodes = dict(self.graph.nodes(data=True))

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            u_data = nodes.get(u, {})
            v_data = nodes.get(v, {})

            lat_start = float(u_data.get("y", 0.0))
            lon_start = float(u_data.get("x", 0.0))
            lat_end = float(v_data.get("y", 0.0))
            lon_end = float(v_data.get("x", 0.0))

            lat_mid = (lat_start + lat_end) / 2.0
            lon_mid = (lon_start + lon_end) / 2.0

            bearing = self._compute_bearing(lat_start, lon_start, lat_end, lon_end)
            length = float(data.get("length", 0.0))

            highway = data.get("highway", "unclassified")
            if isinstance(highway, list):
                highway = highway[0]

            name = data.get("name", "")
            if isinstance(name, list):
                name = name[0]

            oneway = data.get("oneway", False)
            if isinstance(oneway, str):
                oneway = oneway.lower() in ("true", "yes", "1")

            self.segments.append(RoadSegment(
                edge_id=(u, v, key),
                u_node=u,
                v_node=v,
                lat_start=lat_start,
                lon_start=lon_start,
                lat_end=lat_end,
                lon_end=lon_end,
                lat_mid=lat_mid,
                lon_mid=lon_mid,
                bearing_deg=bearing,
                length_m=length,
                road_type=str(highway),
                name=str(name) if name else "",
                oneway=bool(oneway),
            ))

    def _build_spatial_index(self) -> None:
        """Builds a cKDTree over segment midpoints for fast spatial queries."""
        if not self.segments:
            self._kdtree = None
            self._mid_coords = None
            return

        self._mid_coords = np.array(
            [[seg.lat_mid, seg.lon_mid] for seg in self.segments],
            dtype=np.float64,
        )
        self._kdtree = cKDTree(self._mid_coords)

    # ------------------------------------------------------------------
    # Internal: Geometry Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bearing(
        lat1_deg: float,
        lon1_deg: float,
        lat2_deg: float,
        lon2_deg: float,
    ) -> float:
        """Computes forward azimuth (bearing) from point 1 to point 2 in degrees [0, 360)."""
        lat1 = np.radians(lat1_deg)
        lat2 = np.radians(lat2_deg)
        dlon = np.radians(lon2_deg - lon1_deg)

        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        bearing_rad = np.arctan2(x, y)
        return float(np.degrees(bearing_rad) % 360.0)

    @staticmethod
    def _snap_to_segment(
        lat_q: float,
        lon_q: float,
        lat_a: float,
        lon_a: float,
        lat_b: float,
        lon_b: float,
    ) -> Tuple[float, float, float]:
        """
        Snaps a query point to the nearest point on segment A→B.

        Uses a local flat-Earth approximation for speed (valid for short segments).

        Returns
        -------
        (snap_lat, snap_lon, distance_m)
        """
        # Convert to local metres using flat-Earth approximation
        cos_lat = np.cos(np.radians(lat_q))
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * cos_lat

        ax = (lon_a - lon_q) * m_per_deg_lon
        ay = (lat_a - lat_q) * m_per_deg_lat
        bx = (lon_b - lon_q) * m_per_deg_lon
        by = (lat_b - lat_q) * m_per_deg_lat

        dx = bx - ax
        dy = by - ay
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-10:
            # Degenerate segment (zero length)
            dist_m = np.sqrt(ax * ax + ay * ay)
            return lat_a, lon_a, float(dist_m)

        # Parametric projection t ∈ [0, 1]
        t = max(0.0, min(1.0, (-ax * dx + -ay * dy) / seg_len_sq))
        # Wait — we need the projection of the origin (query point at 0,0) onto line A→B
        # Vector from A to query = (0 - ax, 0 - ay) = (-ax, -ay)
        # t = dot(AQ, AB) / |AB|^2 = (-ax * dx + (-ay) * dy) / seg_len_sq
        t = max(0.0, min(1.0, (-ax * dx + -ay * dy) / seg_len_sq))

        snap_x = ax + t * dx
        snap_y = ay + t * dy

        dist_m = float(np.sqrt(snap_x * snap_x + snap_y * snap_y))

        # Convert snap point back to lat/lon
        snap_lon = lon_q + snap_x / m_per_deg_lon
        snap_lat = lat_q + snap_y / m_per_deg_lat

        return float(snap_lat), float(snap_lon), dist_m

    @staticmethod
    def _heading_difference(heading1_deg: float, heading2_deg: float) -> float:
        """
        Computes the minimum absolute angular difference between two headings.

        Takes into account that roads can be travelled in either direction,
        so the difference is always in [0, 90] — a 180° difference means
        travelling in the opposite direction on the same road, which is valid.

        Returns
        -------
        float: Heading difference in degrees [0, 90].
        """
        diff = abs(heading1_deg - heading2_deg) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        # Roads can be traversed in both directions, so 180° difference = same road
        if diff > 90.0:
            diff = 180.0 - diff
        return float(diff)

    # ------------------------------------------------------------------
    # Internal: Cache Key Generation
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(
        lat: float,
        lon: float,
        radius_m: float,
        network_type: str,
    ) -> str:
        """Generates a deterministic cache key for center-based queries."""
        raw = f"center_{lat:.5f}_{lon:.5f}_{radius_m:.0f}_{network_type}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _cache_key_bbox(
        north: float,
        south: float,
        east: float,
        west: float,
        network_type: str,
    ) -> str:
        """Generates a deterministic cache key for bounding-box queries."""
        raw = f"bbox_{north:.5f}_{south:.5f}_{east:.5f}_{west:.5f}_{network_type}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Adjacency Query (for HMM transition probability)
    # ------------------------------------------------------------------

    def are_connected(self, seg_a: RoadSegment, seg_b: RoadSegment) -> bool:
        """
        Checks whether two road segments share a node (are topologically adjacent).

        Parameters
        ----------
        seg_a, seg_b : RoadSegment

        Returns
        -------
        True if the segments share at least one endpoint node.
        """
        nodes_a = {seg_a.u_node, seg_a.v_node}
        nodes_b = {seg_b.u_node, seg_b.v_node}
        return bool(nodes_a & nodes_b)

    def shortest_path_length(
        self,
        seg_a: RoadSegment,
        seg_b: RoadSegment,
    ) -> Optional[float]:
        """
        Computes the shortest road-network distance between two segments.

        Uses the closest pair of (u/v) nodes and networkx shortest_path_length.

        Returns
        -------
        Distance in metres, or None if no path exists.
        """
        if self.graph is None:
            return None

        import networkx as nx

        best_dist = None
        for n_a in (seg_a.u_node, seg_a.v_node):
            for n_b in (seg_b.u_node, seg_b.v_node):
                if n_a == n_b:
                    return 0.0
                try:
                    d = nx.shortest_path_length(
                        self.graph, n_a, n_b, weight="length"
                    )
                    if best_dist is None or d < best_dist:
                        best_dist = d
                except nx.NetworkXNoPath:
                    continue

        return best_dist
