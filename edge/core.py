"""
IDR MVP -- Edge Navigation Engine Core (Phase 15)
=================================================
Sensor-agnostic, shared edge navigation core engine adhering to MVP.md §19
and TECH_STACK.md §14. Ingests standard EdgeSensorPacket streams from any
adapter (smartphone, external IMU) and emits standard EdgeNavigationOutput.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from edge.contracts import EdgeSensorPacket, EdgeNavigationOutput
from edge.adapters.base import BaseSensorAdapter

from src.pipeline import IDRNavigationEngine, PipelineConfig, SensorInput, NavigationOutput as PipelineOutput

logger = logging.getLogger(__name__)

_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class EdgeNavigationEngine:
    """Sensor-agnostic edge navigation engine.
    
    Acts as the shared core between Android and External IMU systems per MVP.md §19.
    Wraps the unified IDRNavigationEngine pipeline, converting standardized
    EdgeSensorPacket inputs into EdgeNavigationOutput states.
    """

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        use_ai_velocity: bool = True,
        use_map_matching: bool = False,
        dropout_debounce_samples: int = 5,
        recovery_window_s: float = 3.0,
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else _DEFAULT_MODELS_DIR
        self.config = PipelineConfig(
            models_dir=str(self.models_dir),
            use_ai_velocity=use_ai_velocity,
            use_map_matching=use_map_matching,
            dropout_debounce_samples=dropout_debounce_samples,
            recovery_confirm_samples=3,
            recovery_window_s=recovery_window_s,
        )
        self.pipeline = IDRNavigationEngine(config=self.config)
        self.total_packets_processed: int = 0
        self.current_adapter_name: str = "none"
        self.blackout_start_time_s: Optional[float] = None
        self.outage_duration_s: float = 0.0

    @property
    def p_n(self) -> float:
        return float(self.pipeline.fusion_engine.ekf.position_ned[0])

    @property
    def p_e(self) -> float:
        return float(self.pipeline.fusion_engine.ekf.position_ned[1])

    @property
    def heading_deg(self) -> float:
        return float(np.degrees(self.pipeline.fusion_engine.ekf.heading_rad)) % 360.0

    def process_packet(self, packet: EdgeSensorPacket) -> EdgeNavigationOutput:
        """Step the shared navigation engine with a standardized sensor observation.
        
        Args:
            packet: Standardized sensor input packet.
            
        Returns:
            Standardized navigation output packet.
        """
        self.total_packets_processed += 1
        self.current_adapter_name = packet.source_id
        ts_s = packet.timestamp_ms / 1000.0

        if not packet.gnss_valid:
            if self.blackout_start_time_s is None:
                self.blackout_start_time_s = ts_s
            self.outage_duration_s = ts_s - self.blackout_start_time_s
        else:
            self.blackout_start_time_s = None
            self.outage_duration_s = 0.0

        sensor_input = SensorInput(
            timestamp_ms=packet.timestamp_ms,
            ax=packet.ax,
            ay=packet.ay,
            az=packet.az,
            gx=packet.gx,
            gy=packet.gy,
            gz=packet.gz,
            ax_lin=packet.ax,
            ay_lin=packet.ay,
            az_lin=packet.az,
            gnss_lat=packet.gnss_lat if packet.gnss_valid else None,
            gnss_lon=packet.gnss_lon if packet.gnss_valid else None,
            gnss_alt=packet.gnss_alt if packet.gnss_valid else None,
            gnss_speed=packet.gnss_speed if packet.gnss_valid else None,
            gnss_course=packet.gnss_course if packet.gnss_valid else None,
            gnss_accuracy=packet.gnss_accuracy if packet.gnss_valid else None,
        )

        pipe_out: PipelineOutput = self.pipeline.step(sensor_input)

        return EdgeNavigationOutput(
            timestamp_ms=packet.timestamp_ms,
            lat_deg=pipe_out.final_lat if pipe_out.is_map_matched else pipe_out.lat_deg,
            lon_deg=pipe_out.final_lon if pipe_out.is_map_matched else pipe_out.lon_deg,
            alt_m=pipe_out.alt_m,
            speed_ms=round(pipe_out.speed_ms, 3),
            heading_deg=round(pipe_out.heading_deg, 2),
            confidence=round(pipe_out.confidence, 3),
            mode=pipe_out.mode,
            is_gnss_available=packet.gnss_valid,
            outage_duration_s=round(self.outage_duration_s, 2),
            diagnostics={
                "p_n": round(pipe_out.p_n, 3),
                "p_e": round(pipe_out.p_e, 3),
                "ai_speed_ms": round(pipe_out.ai_speed_ms or 0.0, 3),
                "motion_state": pipe_out.motion_state,
                "trust_weight": pipe_out.trust_weight,
                "cov_trace": round(pipe_out.cov_trace, 3),
                "adapter_source": packet.source_id,
            },
        )

    def process_stream(self, adapter: BaseSensorAdapter) -> Iterator[EdgeNavigationOutput]:
        """Stream sensor packets from an adapter continuously through the navigation engine."""
        for packet in adapter.stream():
            yield self.process_packet(packet)
