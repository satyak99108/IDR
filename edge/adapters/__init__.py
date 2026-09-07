"""
IDR MVP -- Edge Engine Adapters Subsystem (Phase 15)
====================================================
Exposes the abstract BaseSensorAdapter, SmartphoneAdapter, and
ExternalIMUAdapter for sensor-agnostic edge navigation.
"""

from edge.adapters.base import BaseSensorAdapter, AdapterStats
from edge.adapters.smartphone_adapter import SmartphoneAdapter
from edge.adapters.external_imu_adapter import ExternalIMUAdapter

__all__ = [
    "BaseSensorAdapter",
    "AdapterStats",
    "SmartphoneAdapter",
    "ExternalIMUAdapter",
]
