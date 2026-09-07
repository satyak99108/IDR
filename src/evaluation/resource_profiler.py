"""
IDR MVP -- Resource Profiler (Phase 12)
========================================
Lightweight CPU/RAM/model-size profiler for benchmarking the navigation
pipeline's computational footprint.

Usage:
    profiler = ResourceProfiler()
    with profiler:
        result = engine.process_trip(df, ...)
    profile = profiler.get_profile()
    print(profile.peak_ram_mb, profile.mean_cpu_pct, profile.wall_time_s)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass
class ResourceProfile:
    """Captured resource usage measurements."""
    wall_time_s: float = 0.0
    peak_ram_mb: float = 0.0
    mean_cpu_pct: float = 0.0
    samples_processed: int = 0
    throughput_hz: float = 0.0
    latency_per_sample_ms: float = 0.0
    model_files: Dict[str, float] = field(default_factory=dict)  # filename -> size_kb

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wall_time_s": round(self.wall_time_s, 3),
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "mean_cpu_pct": round(self.mean_cpu_pct, 1),
            "samples_processed": self.samples_processed,
            "throughput_hz": round(self.throughput_hz, 1),
            "latency_per_sample_ms": round(self.latency_per_sample_ms, 3),
            "model_files": {k: round(v, 2) for k, v in self.model_files.items()},
        }


class ResourceProfiler:
    """
    Context manager that profiles CPU, RAM, and wall-clock time during
    pipeline execution.

    Parameters
    ----------
    models_dir : Path, optional
        Directory containing model files to measure sizes of.
    sampling_interval_s : float
        Interval in seconds between CPU/RAM snapshots (default: 0.1).
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        sampling_interval_s: float = 0.1,
    ):
        self.models_dir = models_dir
        self.sampling_interval_s = sampling_interval_s
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._ram_samples: List[float] = []
        self._cpu_samples: List[float] = []
        self._process: Optional[Any] = None
        self._samples_processed: int = 0

    def __enter__(self) -> "ResourceProfiler":
        if _HAS_PSUTIL:
            self._process = psutil.Process(os.getpid())
            # Prime the cpu_percent measurement (first call always returns 0)
            self._process.cpu_percent(interval=None)
        self._ram_samples.clear()
        self._cpu_samples.clear()
        self._start_time = time.perf_counter()
        # Take initial snapshot
        self._snapshot()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._end_time = time.perf_counter()
        # Take final snapshot
        self._snapshot()

    def snapshot(self) -> None:
        """Manually take a CPU/RAM snapshot during execution."""
        self._snapshot()

    def _snapshot(self) -> None:
        """Internal: record a single CPU/RAM data point."""
        if not _HAS_PSUTIL or self._process is None:
            return
        try:
            mem_info = self._process.memory_info()
            self._ram_samples.append(mem_info.rss / (1024 * 1024))  # MB
            cpu = self._process.cpu_percent(interval=None)
            self._cpu_samples.append(cpu)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def set_samples_processed(self, n: int) -> None:
        """Set the number of samples processed for throughput calculation."""
        self._samples_processed = n

    def get_profile(self) -> ResourceProfile:
        """Build the final ResourceProfile from collected measurements."""
        wall_time = self._end_time - self._start_time if self._end_time > self._start_time else 0.0

        peak_ram = max(self._ram_samples) if self._ram_samples else 0.0
        mean_cpu = float(sum(self._cpu_samples) / len(self._cpu_samples)) if self._cpu_samples else 0.0

        throughput = self._samples_processed / wall_time if wall_time > 0 else 0.0
        latency = (wall_time / self._samples_processed * 1000.0) if self._samples_processed > 0 else 0.0

        # Model file sizes
        model_files: Dict[str, float] = {}
        if self.models_dir and self.models_dir.exists():
            for f in self.models_dir.iterdir():
                if f.is_file() and f.suffix in (".npz", ".tflite", ".keras", ".h5", ".onnx", ".json"):
                    model_files[f.name] = f.stat().st_size / 1024.0  # KB

        return ResourceProfile(
            wall_time_s=wall_time,
            peak_ram_mb=peak_ram,
            mean_cpu_pct=mean_cpu,
            samples_processed=self._samples_processed,
            throughput_hz=throughput,
            latency_per_sample_ms=latency,
            model_files=model_files,
        )
