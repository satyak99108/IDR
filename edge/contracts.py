"""
IDR MVP -- Edge Engine Telemetry Contracts (Phase 15)
=====================================================
Defines the sensor-agnostic standard input and output contracts for the
Intelligent Dead Reckoning (IDR) Edge Engine, strictly adhering to
MVP.md §19 and TECH_STACK.md §14.

Standard Input:
    timestamp_ms, ax, ay, az, gx, gy, gz, mx, my, mz,
    gnss_lat, gnss_lon, gnss_speed, gnss_accuracy

Standard Output:
    timestamp_ms, lat_deg, lon_deg, speed_ms, heading_deg,
    confidence, mode, diagnostics
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EdgeSensorPacket:
    """Standardized single-timestep sensor packet ingested by the Edge Engine (MVP.md §19).
    
    Fields:
        timestamp_ms: Epoch millisecond timestamp of the sensor observation.
        ax, ay, az:   Linear acceleration in vehicle coordinate frame (m/s^2).
        gx, gy, gz:   Angular velocity in vehicle coordinate frame (rad/s).
        mx, my, mz:   Optional 3-axis magnetometer readings (micro-Tesla).
        gnss_lat:     Optional GNSS latitude in WGS-84 decimal degrees.
        gnss_lon:     Optional GNSS longitude in WGS-84 decimal degrees.
        gnss_alt:     Optional GNSS altitude in meters above ellipsoid.
        gnss_speed:   Optional GNSS ground speed in m/s.
        gnss_course:  Optional GNSS ground track heading in degrees [0, 360).
        gnss_accuracy:Optional horizontal 1-sigma dilution/accuracy in meters.
        gnss_valid:   True if GNSS fix is active, healthy, and usable.
        source_id:    Identifier of the sensor source (e.g. 'smartphone', 'external_imu').
    """

    timestamp_ms: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mx: Optional[float] = None
    my: Optional[float] = None
    mz: Optional[float] = None
    gnss_lat: Optional[float] = None
    gnss_lon: Optional[float] = None
    gnss_alt: Optional[float] = None
    gnss_speed: Optional[float] = None
    gnss_course: Optional[float] = None
    gnss_accuracy: Optional[float] = None
    gnss_valid: bool = False
    source_id: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the packet to a serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the packet to a compact JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EdgeSensorPacket:
        """Instantiate an EdgeSensorPacket from a dictionary with field validation."""
        return cls(
            timestamp_ms=int(data["timestamp_ms"]),
            ax=float(data["ax"]),
            ay=float(data["ay"]),
            az=float(data["az"]),
            gx=float(data["gx"]),
            gy=float(data["gy"]),
            gz=float(data["gz"]),
            mx=float(data["mx"]) if data.get("mx") is not None else None,
            my=float(data["my"]) if data.get("my") is not None else None,
            mz=float(data["mz"]) if data.get("mz") is not None else None,
            gnss_lat=float(data["gnss_lat"]) if data.get("gnss_lat") is not None else None,
            gnss_lon=float(data["gnss_lon"]) if data.get("gnss_lon") is not None else None,
            gnss_alt=float(data["gnss_alt"]) if data.get("gnss_alt") is not None else None,
            gnss_speed=float(data["gnss_speed"]) if data.get("gnss_speed") is not None else None,
            gnss_course=float(data["gnss_course"]) if data.get("gnss_course") is not None else None,
            gnss_accuracy=float(data["gnss_accuracy"]) if data.get("gnss_accuracy") is not None else None,
            gnss_valid=bool(data.get("gnss_valid", False)),
            source_id=str(data.get("source_id", "unknown")),
        )

    @classmethod
    def from_json(cls, json_str: str) -> EdgeSensorPacket:
        """Parse an EdgeSensorPacket from a JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class EdgeNavigationOutput:
    """Standardized single-timestep navigation state emitted by the Edge Engine (MVP.md §19).
    
    Fields:
        timestamp_ms:      Epoch millisecond timestamp of the navigation output.
        lat_deg:           Estimated latitude in WGS-84 decimal degrees.
        lon_deg:           Estimated longitude in WGS-84 decimal degrees.
        alt_m:             Estimated altitude in meters.
        speed_ms:          Estimated forward vehicle velocity in m/s.
        heading_deg:       Estimated vehicle azimuth heading in degrees [0, 360).
        confidence:        Estimated state confidence score in [0.0, 1.0].
        mode:              Current operational mode ('GNSS_INS', 'DEAD_RECKONING', 'RECOVERY', 'DEGRADED_GNSS').
        is_gnss_available: Whether GNSS was available at this timestep.
        outage_duration_s: Cumulative duration of ongoing GNSS blackout in seconds.
        diagnostics:       Supplementary metrics (p_n, p_e, ai_speed, filter trace, adapter info).
    """

    timestamp_ms: int
    lat_deg: float
    lon_deg: float
    alt_m: float
    speed_ms: float
    heading_deg: float
    confidence: float
    mode: str
    is_gnss_available: bool
    outage_duration_s: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the output to a serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the output to a compact JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EdgeNavigationOutput:
        """Instantiate an EdgeNavigationOutput from a dictionary."""
        return cls(
            timestamp_ms=int(data["timestamp_ms"]),
            lat_deg=float(data["lat_deg"]),
            lon_deg=float(data["lon_deg"]),
            alt_m=float(data.get("alt_m", 0.0)),
            speed_ms=float(data["speed_ms"]),
            heading_deg=float(data["heading_deg"]),
            confidence=float(data["confidence"]),
            mode=str(data["mode"]),
            is_gnss_available=bool(data["is_gnss_available"]),
            outage_duration_s=float(data.get("outage_duration_s", 0.0)),
            diagnostics=dict(data.get("diagnostics", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> EdgeNavigationOutput:
        """Parse an EdgeNavigationOutput from a JSON string."""
        return cls.from_dict(json.loads(json_str))
