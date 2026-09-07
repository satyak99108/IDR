"""
IDR MVP -- Smartphone Sensor Adapter (Phase 15)
==============================================
Implements the SmartphoneAdapter for ingesting mobile sensor telemetry into
the Edge Engine, supporting both IO-VNBD calibrated dataset replay and live
network streaming over UDP/TCP, per MVP.md §19 and TECH_STACK.md §14.
"""

from __future__ import annotations

import json
from pathlib import Path
import socket
import time
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import pandas as pd

from edge.adapters.base import BaseSensorAdapter
from edge.contracts import EdgeSensorPacket


class SmartphoneAdapter(BaseSensorAdapter):
    """Adapter for smartphone sensor streams (accelerometer, gyroscope, GNSS).
    
    Supports two modes:
    1. **Replay Mode**: Ingests an IO-VNBD synchronized trip DataFrame or CSV file.
       Can pace at real-time (10 Hz) or stream in benchmark mode.
    2. **Network Stream Mode**: Listens on a UDP socket for incoming JSON telemetry
       packets streamed from an Android device running the IDR app.
    """

    def __init__(
        self,
        name: str = "smartphone_adapter",
        data_source: Optional[Union[pd.DataFrame, str, Path]] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        realtime_rate: bool = False,
        target_hz: float = 10.0,
        blackout_start_s: Optional[float] = None,
        blackout_end_s: Optional[float] = None,
    ) -> None:
        super().__init__(name=name)
        self.data_source = data_source
        self.host = host
        self.port = port
        self.realtime_rate = realtime_rate
        self.target_hz = target_hz
        self.blackout_start_s = blackout_start_s
        self.blackout_end_s = blackout_end_s

        # Replay mode state
        self._df: Optional[pd.DataFrame] = None
        self._cursor: int = 0
        self._total_rows: int = 0
        self._first_timestamp_ms: Optional[int] = None

        # Network mode state
        self._sock: Optional[socket.socket] = None

    def connect(self) -> bool:
        """Initialize and connect the data source."""
        self.reset_stats()

        # Network stream mode
        if self.host is not None and self.port is not None:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._sock.bind((self.host, self.port))
                self._sock.settimeout(0.1)  # Non-blocking poll timeout
                self._is_connected = True
                return True
            except Exception as e:
                self._is_connected = False
                raise ConnectionError(f"Failed to bind UDP socket on {self.host}:{self.port} - {e}")

        # Replay mode
        if self.data_source is not None:
            if isinstance(self.data_source, pd.DataFrame):
                self._df = self.data_source.copy().reset_index(drop=True)
            else:
                path = Path(self.data_source)
                if not path.exists():
                    raise FileNotFoundError(f"Smartphone dataset not found: {path}")
                self._df = pd.read_csv(path)
            # Ensure gnss_course exists or compute from ground track / ref_heading
            if not any(c in self._df.columns for c in ["gnss_course", "ref_heading"]):
                lat_col = "ref_lat" if "ref_lat" in self._df.columns else ("gnss_lat" if "gnss_lat" in self._df.columns else None)
                lon_col = "ref_lon" if "ref_lon" in self._df.columns else ("gnss_lon" if "gnss_lon" in self._df.columns else None)
                if lat_col and lon_col:
                    lats = self._df[lat_col].values
                    lons = self._df[lon_col].values
                    dlat = np.diff(lats)
                    dlon = np.diff(lons)
                    dlat = np.append(dlat, dlat[-1] if len(dlat) > 0 else 0.0)
                    dlon = np.append(dlon, dlon[-1] if len(dlon) > 0 else 0.0)
                    crs = np.degrees(np.arctan2(dlon * np.cos(np.radians(lats)), dlat)) % 360.0
                    self._df["gnss_course"] = crs

            self._cursor = 0
            self._total_rows = len(self._df)
            self._first_timestamp_ms = None
            self._is_connected = self._total_rows > 0
            return self._is_connected

        # Empty in-memory queue mode (can accept manual packet injection)
        self._is_connected = True
        return True

    def disconnect(self) -> None:
        """Release underlying socket or dataset handle."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._df = None
        self._is_connected = False

    def poll_packet(self) -> Optional[EdgeSensorPacket]:
        """Fetch the next sensor packet from network socket or replay sequence."""
        if not self._is_connected:
            return None

        t0 = time.perf_counter()

        # 1. Network socket poll
        if self._sock is not None:
            try:
                data, _ = self._sock.recvfrom(4096)
                packet_dict = json.loads(data.decode("utf-8"))
                packet = EdgeSensorPacket.from_dict(packet_dict)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                self.stats.update_packet(ingest_latency_ms=dt_ms)
                return packet
            except socket.timeout:
                return None
            except Exception:
                self.stats.record_drop()
                return None

        # 2. Dataset replay poll
        if self._df is not None:
            if self._cursor >= self._total_rows:
                self._is_connected = False
                return None

            row = self._df.iloc[self._cursor]
            self._cursor += 1

            # Extract timestamp
            ts_ms = int(row.get("timestamp_ms", row.get("timestamp", self._cursor * 100)))
            if self._first_timestamp_ms is None:
                self._first_timestamp_ms = ts_ms

            # Check simulated blackout window
            elapsed_s = (ts_ms - self._first_timestamp_ms) / 1000.0
            in_blackout = False
            if self.blackout_start_s is not None and self.blackout_end_s is not None:
                if self.blackout_start_s <= elapsed_s <= self.blackout_end_s:
                    in_blackout = True

            # Extract IMU measurements (prioritizing calibrated vehicle linear frame)
            ax = float(row.get("ax_veh_lin", row.get("ax_veh", row.get("ax", row.get("acc_x", 0.0)))))
            ay = float(row.get("ay_veh_lin", row.get("ay_veh", row.get("ay", row.get("acc_y", 0.0)))))
            az = float(row.get("az_veh_lin", row.get("az_veh", row.get("az", row.get("acc_z", 0.0)))))
            gx = float(row.get("gx_veh", row.get("gx", row.get("gyro_x", 0.0))))
            gy = float(row.get("gy_veh", row.get("gy", row.get("gyro_y", 0.0))))
            gz = float(row.get("gz_veh", row.get("gz", row.get("gyro_z", 0.0))))

            # Extract GNSS measurements
            gnss_valid = not in_blackout and ("gnss_lat" in row or "lat" in row or "ref_lat" in row)
            gnss_lat = None
            gnss_lon = None
            gnss_speed = None
            gnss_course = None
            gnss_accuracy = None

            if gnss_valid:
                gnss_lat = float(row.get("gnss_lat", row.get("lat", row.get("ref_lat", 0.0))))
                gnss_lon = float(row.get("gnss_lon", row.get("lon", row.get("ref_lon", 0.0))))
                gnss_speed = float(row.get("gnss_speed", row.get("speed", row.get("ref_speed", 0.0))))
                # Prioritize explicit gnss_course, then calibrated ref_heading, then heading
                gnss_course = float(row.get("gnss_course", row.get("ref_heading", row.get("heading", 0.0))))
                gnss_accuracy = float(row.get("gnss_accuracy", 3.0))

            packet = EdgeSensorPacket(
                timestamp_ms=ts_ms,
                ax=ax,
                ay=ay,
                az=az,
                gx=gx,
                gy=gy,
                gz=gz,
                gnss_lat=gnss_lat,
                gnss_lon=gnss_lon,
                gnss_speed=gnss_speed,
                gnss_course=gnss_course,
                gnss_accuracy=gnss_accuracy,
                gnss_valid=gnss_valid,
                source_id="smartphone",
            )

            # Real-time pacing if requested
            if self.realtime_rate:
                time.sleep(1.0 / self.target_hz)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.stats.update_packet(ingest_latency_ms=dt_ms)
            return packet

        return None
