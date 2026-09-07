"""
IDR MVP -- Base Sensor Adapter Interface (Phase 15)
===================================================
Defines the standard abstract interface for sensor stream adapters
in the Edge Engine, adhering to MVP.md §19 and TECH_STACK.md §14.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Iterator, Optional

from edge.contracts import EdgeSensorPacket


@dataclass
class AdapterStats:
    """Real-time performance and throughput statistics for a sensor adapter."""
    adapter_name: str
    packets_received: int = 0
    packets_dropped: int = 0
    start_time_s: float = field(default_factory=time.time)
    last_packet_time_s: float = 0.0
    sampling_rate_hz: float = 0.0
    mean_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def update_packet(self, ingest_latency_ms: float = 0.0) -> None:
        """Update telemetry metrics on successful packet ingestion."""
        now = time.time()
        self.packets_received += 1
        self.last_packet_time_s = now
        elapsed = now - self.start_time_s
        if elapsed > 0.05:
            self.sampling_rate_hz = self.packets_received / elapsed
        if ingest_latency_ms > 0:
            self.total_latency_ms += ingest_latency_ms
            self.mean_latency_ms = self.total_latency_ms / self.packets_received

    def record_drop(self) -> None:
        """Increment dropped packet count."""
        self.packets_dropped += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to a serializable dictionary."""
        return {
            "adapter_name": self.adapter_name,
            "packets_received": self.packets_received,
            "packets_dropped": self.packets_dropped,
            "sampling_rate_hz": round(self.sampling_rate_hz, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
        }


class BaseSensorAdapter(ABC):
    """Abstract Base Class for all Edge Engine sensor stream adapters.
    
    Subclasses must implement:
        - connect(): Establish socket, open file, or initialize hardware driver.
        - disconnect(): Gracefully release resources.
        - poll_packet(): Non-blocking fetch of the next standardized EdgeSensorPacket.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self._is_connected: bool = False
        self.stats: AdapterStats = AdapterStats(adapter_name=name)

    @property
    def is_connected(self) -> bool:
        """Check if the adapter is currently connected and ready to stream."""
        return self._is_connected

    @abstractmethod
    def connect(self) -> bool:
        """Establish the sensor stream connection.
        
        Returns:
            True if connection was established successfully, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close the stream connection and release underlying resources."""
        pass

    @abstractmethod
    def poll_packet(self) -> Optional[EdgeSensorPacket]:
        """Poll the next sensor packet in a non-blocking or timely manner.
        
        Returns:
            An EdgeSensorPacket instance if data is available, or None.
        """
        pass

    def stream(self, max_packets: Optional[int] = None) -> Iterator[EdgeSensorPacket]:
        """Generator yielding EdgeSensorPackets continuously until exhausted or disconnected.
        
        Args:
            max_packets: Optional maximum number of packets to yield.
            
        Yields:
            EdgeSensorPacket objects adhering to MVP.md §19 standard input.
        """
        if not self._is_connected:
            if not self.connect():
                raise ConnectionError(f"Failed to connect adapter '{self.name}'.")

        count = 0
        while self._is_connected:
            if max_packets is not None and count >= max_packets:
                break
            packet = self.poll_packet()
            if packet is not None:
                count += 1
                yield packet

    def get_stats(self) -> AdapterStats:
        """Retrieve current throughput and packet statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset internal performance counters."""
        self.stats = AdapterStats(adapter_name=self.name)

    def __enter__(self) -> BaseSensorAdapter:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()
