"""
IDR MVP -- Evaluation Package (Phase 12)
==========================================
Provides comprehensive benchmarking, resource profiling, and GNSS recovery
analysis for the end-to-end navigation pipeline.

Components:
    ResourceProfiler  -- CPU/RAM/model-size profiling context manager.
    RecoveryAnalyzer  -- GNSS recovery convergence analysis.
    BenchmarkRunner   -- Multi-trip, multi-outage benchmark engine.
"""

from .resource_profiler import ResourceProfiler, ResourceProfile
from .recovery_analyzer import RecoveryAnalyzer, RecoveryMetrics
from .benchmark import BenchmarkRunner, TripResult, BenchmarkSummary

__all__ = [
    "ResourceProfiler",
    "ResourceProfile",
    "RecoveryAnalyzer",
    "RecoveryMetrics",
    "BenchmarkRunner",
    "TripResult",
    "BenchmarkSummary",
]
