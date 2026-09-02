"""
IDR MVP -- Simulation Package
==============================
Provides tools for simulating sensor loss, GNSS outages, and evaluating
dead-reckoning performance under degraded sensor conditions.
"""

from .outage_simulator import (
    OutageWindow,
    OutageScenario,
    GNSSOutageSimulator,
)
from .evaluator import (
    OutageMetrics,
    ScenarioMetrics,
    OutageEvaluator,
)

__all__ = [
    "OutageWindow",
    "OutageScenario",
    "GNSSOutageSimulator",
    "OutageMetrics",
    "ScenarioMetrics",
    "OutageEvaluator",
]
