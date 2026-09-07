"""
IDR MVP -- End-to-End Navigation Pipeline Package (Phase 11)
============================================================
Exports the unified navigation engine, configuration, and data structures.
"""

from .engine import (
    IDRNavigationEngine,
    PipelineConfig,
    SensorInput,
    NavigationOutput,
)

__all__ = [
    "IDRNavigationEngine",
    "PipelineConfig",
    "SensorInput",
    "NavigationOutput",
]
