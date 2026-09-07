"""
IDR MVP -- Edge Navigation Subsystem (Phase 15)
==============================================
Exposes the sensor-agnostic Edge Navigation Engine, Telemetry Contracts,
and Streaming Sensor Adapters per MVP.md §19 and TECH_STACK.md §14.
"""

from edge.contracts import EdgeSensorPacket, EdgeNavigationOutput
from edge.core import EdgeNavigationEngine
from edge.adapters.base import BaseSensorAdapter, AdapterStats
from edge.adapters.smartphone_adapter import SmartphoneAdapter
from edge.adapters.external_imu_adapter import ExternalIMUAdapter

__all__ = [
    "EdgeSensorPacket",
    "EdgeNavigationOutput",
    "EdgeNavigationEngine",
    "BaseSensorAdapter",
    "AdapterStats",
    "SmartphoneAdapter",
    "ExternalIMUAdapter",
]
