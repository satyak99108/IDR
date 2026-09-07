"""
IDR MVP -- Edge Engine Unit Tests (Phase 15)
============================================
Comprehensive test suite verifying standard contracts, streaming adapters,
rate decimation, state transitions, and adapter interchangeability.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from edge.contracts import EdgeSensorPacket, EdgeNavigationOutput
from edge.adapters.base import AdapterStats
from edge.adapters.smartphone_adapter import SmartphoneAdapter
from edge.adapters.external_imu_adapter import ExternalIMUAdapter
from edge.core import EdgeNavigationEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_trip_df() -> pd.DataFrame:
    """Generate a synthetic 10 Hz calibrated trip DataFrame for testing."""
    n_samples = 200  # 20.0 seconds at 10 Hz
    t = np.arange(n_samples) * 0.1

    # Straight northward movement starting at Singapore (1.3521, 103.8198)
    lat_0 = 1.3521
    lon_0 = 103.8198
    speed = 12.0  # m/s (~43.2 km/h)
    dlat = (speed * t) / 111139.0

    np.random.seed(42)
    vibe = np.random.normal(0.0, 0.2, n_samples).astype(np.float32)

    return pd.DataFrame({
        "timestamp_ms": (t * 1000).astype(np.int64),
        "ax": vibe,
        "ay": vibe,
        "az": vibe,
        "ax_veh_lin": vibe,
        "ay_veh_lin": np.full(n_samples, 0.2, dtype=np.float32) + vibe,
        "az_veh_lin": vibe,
        "gx": vibe * 0.1,
        "gy": vibe * 0.1,
        "gz": vibe * 0.1,  # heading north (0 deg)
        "gx_veh": vibe * 0.1,
        "gy_veh": vibe * 0.1,
        "gz_veh": vibe * 0.1,
        "gnss_lat": lat_0 + dlat,
        "gnss_lon": np.full(n_samples, lon_0),
        "gnss_speed": np.full(n_samples, speed),
        "gnss_course": np.zeros(n_samples),
        "gnss_accuracy": np.full(n_samples, 2.5),
    })


# ---------------------------------------------------------------------------
# 1. Telemetry Contracts Tests
# ---------------------------------------------------------------------------

class TestEdgeContracts:
    """Verify EdgeSensorPacket and EdgeNavigationOutput contracts (MVP.md §19)."""

    def test_sensor_packet_serialization(self):
        packet = EdgeSensorPacket(
            timestamp_ms=1609459200000,
            ax=0.1,
            ay=1.2,
            az=9.81,
            gx=0.01,
            gy=-0.02,
            gz=0.005,
            gnss_lat=1.3521,
            gnss_lon=103.8198,
            gnss_speed=15.5,
            gnss_course=90.0,
            gnss_accuracy=2.0,
            gnss_valid=True,
            source_id="smartphone",
        )
        json_str = packet.to_json()
        assert isinstance(json_str, str)

        restored = EdgeSensorPacket.from_json(json_str)
        assert restored.timestamp_ms == packet.timestamp_ms
        assert abs(restored.ay - 1.2) < 1e-5
        assert restored.gnss_valid is True
        assert restored.source_id == "smartphone"

    def test_navigation_output_serialization(self):
        output = EdgeNavigationOutput(
            timestamp_ms=1609459200100,
            lat_deg=1.3522,
            lon_deg=103.8200,
            alt_m=15.0,
            speed_ms=14.8,
            heading_deg=89.5,
            confidence=0.88,
            mode="DEAD_RECKONING",
            is_gnss_available=False,
            outage_duration_s=4.5,
            diagnostics={"p_n": 10.5, "p_e": 22.1},
        )
        json_str = output.to_json()
        restored = EdgeNavigationOutput.from_json(json_str)

        assert restored.timestamp_ms == 1609459200100
        assert restored.mode == "DEAD_RECKONING"
        assert restored.is_gnss_available is False
        assert restored.diagnostics["p_n"] == 10.5


# ---------------------------------------------------------------------------
# 2. Smartphone Adapter Tests
# ---------------------------------------------------------------------------

class TestSmartphoneAdapter:
    """Verify SmartphoneAdapter functionality, blackout simulation, and stats."""

    def test_smartphone_replay_and_blackout(self, sample_trip_df):
        adapter = SmartphoneAdapter(
            data_source=sample_trip_df,
            blackout_start_s=5.0,
            blackout_end_s=10.0,
        )
        assert adapter.connect() is True

        packets = list(adapter.stream())
        assert len(packets) == len(sample_trip_df)

        # Before 5.0s, GNSS is valid
        assert packets[0].gnss_valid is True
        # During blackout (50th to 100th sample), GNSS is invalid
        blackout_packet = packets[75]
        assert blackout_packet.gnss_valid is False
        assert blackout_packet.gnss_lat is None

        # After blackout (>10.0s), GNSS is restored
        restored_packet = packets[120]
        assert restored_packet.gnss_valid is True
        assert restored_packet.gnss_lat is not None

        stats = adapter.get_stats()
        assert stats.packets_received == len(sample_trip_df)
        adapter.disconnect()
        assert adapter.is_connected is False


# ---------------------------------------------------------------------------
# 3. External IMU Adapter Tests
# ---------------------------------------------------------------------------

class TestExternalIMUAdapter:
    """Verify ExternalIMUAdapter high-rate decimation and anti-aliasing."""

    def test_external_imu_decimation(self, sample_trip_df):
        # Ingest 10 Hz base data, synthesize 100 Hz, decimate back to 10 Hz
        adapter = ExternalIMUAdapter(
            data_source=sample_trip_df,
            native_rate_hz=100.0,
            target_rate_hz=10.0,
        )
        assert adapter.decimation_factor == 10
        assert adapter.connect() is True

        packets = list(adapter.stream())
        # Decimated output packet count should match base 10 Hz dataset length
        assert len(packets) == len(sample_trip_df)
        assert packets[0].source_id == "external_imu"

        # Check that IMU values are valid floats
        assert np.isfinite(packets[0].az)
        assert np.isfinite(packets[0].gz)
        adapter.disconnect()


# ---------------------------------------------------------------------------
# 4. Edge Navigation Engine & Hot-Swapping Tests
# ---------------------------------------------------------------------------

class TestEdgeNavigationEngine:
    """Verify sensor-agnostic navigation core state transitions and throughput."""

    def test_navigation_mode_transitions(self, sample_trip_df):
        engine = EdgeNavigationEngine()
        adapter = SmartphoneAdapter(
            data_source=sample_trip_df,
            blackout_start_s=6.0,
            blackout_end_s=12.0,
        )
        adapter.connect()

        outputs = []
        for packet in adapter.stream():
            out = engine.process_packet(packet)
            outputs.append(out)

        # 1. Initially should be GNSS_INS
        assert outputs[20].mode == "GNSS_INS"
        assert outputs[20].is_gnss_available is True

        # 2. During blackout (at 8.0s = sample 80), should transition to DEAD_RECKONING
        dr_output = outputs[80]
        assert dr_output.mode == "DEAD_RECKONING"
        assert dr_output.is_gnss_available is False
        assert dr_output.outage_duration_s > 0.0

        # 3. Vehicle continues advancing during blackout
        dist_dr = np.sqrt(
            (outputs[110].diagnostics["p_n"] - outputs[70].diagnostics["p_n"]) ** 2
            + (outputs[110].diagnostics["p_e"] - outputs[70].diagnostics["p_e"]) ** 2
        )
        assert dist_dr > 1.0  # Vehicle moved forward in DR mode

        # 4. When GNSS returns (after 12.0s = sample 120), system enters RECOVERY then GNSS_INS
        post_blackout_modes = [o.mode for o in outputs[125:145]]
        assert "RECOVERY" in post_blackout_modes or "GNSS_INS" in post_blackout_modes

    def test_adapter_hot_swapping(self, sample_trip_df):
        """Verify seamless switching between Smartphone and External IMU without state reset."""
        engine = EdgeNavigationEngine()

        adapter_phone = SmartphoneAdapter(data_source=sample_trip_df.iloc[:50])
        adapter_external = ExternalIMUAdapter(data_source=sample_trip_df.iloc[50:], native_rate_hz=50.0)

        adapter_phone.connect()
        adapter_external.connect()

        outputs = []
        # Phase A: Run on Smartphone
        for packet in adapter_phone.stream():
            outputs.append(engine.process_packet(packet))

        assert engine.current_adapter_name == "smartphone"
        last_pn_phone = engine.p_n

        # Phase B: Hot-swap to External IMU
        for packet in adapter_external.stream():
            outputs.append(engine.process_packet(packet))

        assert engine.current_adapter_name == "external_imu"
        # Position should continue advancing smoothly from where phone left off
        assert engine.p_n > last_pn_phone

    def test_navigation_throughput_exceeds_10hz(self, sample_trip_df):
        """MVP.md §18, §19: Edge engine throughput must exceed 10 Hz."""
        import time
        engine = EdgeNavigationEngine()
        adapter = SmartphoneAdapter(data_source=sample_trip_df)
        adapter.connect()

        t0 = time.perf_counter()
        count = 0
        for packet in adapter.stream():
            engine.process_packet(packet)
            count += 1
        elapsed = time.perf_counter() - t0

        throughput_hz = count / elapsed
        assert throughput_hz >= 10.0, f"Throughput was {throughput_hz:.1f} Hz (<10 Hz)"
