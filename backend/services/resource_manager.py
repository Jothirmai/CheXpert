"""GPU / RAM / model-cache diagnostics for the Chest Xpert backend.

Records safe aggregate operational metrics. Never exposes case data or
sensitive information.
"""

from __future__ import annotations

import gc
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from backend.config import MODEL_CONFIG, STORAGE_CONFIG


@dataclass
class Snapshot:
    """A point-in-time resource snapshot."""

    timestamp: float = field(default_factory=time.time)
    gpu_name: Optional[str] = None
    gpu_total_mb: Optional[float] = None
    gpu_free_mb: Optional[float] = None
    gpu_used_mb: Optional[float] = None
    torch_allocated_mb: Optional[float] = None
    torch_reserved_mb: Optional[float] = None
    host_ram_used_gb: Optional[float] = None
    host_ram_available_gb: Optional[float] = None


class ResourceMonitor:
    """Collects GPU and host RAM snapshots.

    All metrics are aggregate operational data — no per-case or patient data.
    """

    def __init__(self) -> None:
        self._peak_allocated_mb: Optional[float] = None
        self._gpu_name: Optional[str] = None
        if torch.cuda.is_available():
            try:
                self._gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                self._gpu_name = None

    def snapshot(self, label: str = "") -> Snapshot:
        s = Snapshot()
        s.gpu_name = self._gpu_name

        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                s.gpu_total_mb = round(total / (1024 ** 2), 2)
                s.gpu_free_mb = round(free / (1024 ** 2), 2)
                s.gpu_used_mb = round((total - free) / (1024 ** 2), 2)
                s.torch_allocated_mb = round(
                    torch.cuda.memory_allocated() / (1024 ** 2), 2
                )
                s.torch_reserved_mb = round(
                    torch.cuda.memory_reserved() / (1024 ** 2), 2
                )
                if (
                    self._peak_allocated_mb is None
                    or s.torch_allocated_mb > self._peak_allocated_mb
                ):
                    self._peak_allocated_mb = s.torch_allocated_mb
            except Exception:
                pass

        s.peak_allocated_mb = self._peak_allocated_mb

        # Host RAM via /proc/meminfo (Linux) as a lightweight fallback.
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            mem_total = None
            mem_available = None
            for line in meminfo.splitlines():
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) / 1024 / 1024  # GB
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1]) / 1024 / 1024  # GB
            if mem_total is not None:
                s.host_ram_used_gb = round(mem_total - (mem_available or 0), 2)
                s.host_ram_available_gb = round(mem_available or 0, 2)
        except Exception:
            pass

        return s

    def record_peak(self, allocated_mb: float) -> None:
        if self._peak_allocated_mb is None or allocated_mb > self._peak_allocated_mb:
            self._peak_allocated_mb = allocated_mb

    def cleanup(self) -> None:
        """Run garbage collection and clear PyTorch CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def model_cache_size_mb(self) -> float:
        """Return approximate size of the TorchXRayVision model cache dir."""
        cache_dir = Path.home() / ".torchxrayvision" / "models_data"
        if not cache_dir.exists():
            return 0.0
        total = 0
        for f in cache_dir.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except Exception:
                    pass
        return round(total / (1024 ** 2), 2)

    def temp_dir_size_mb(self) -> float:
        """Return approximate size of the backend temp dir."""
        d = STORAGE_CONFIG.temp_dir
        if not d.exists():
            return 0.0
        total = 0
        for f in d.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except Exception:
                    pass
        return round(total / (1024 ** 2), 2)

    def reports_dir_size_mb(self) -> float:
        """Return approximate size of the reports dir."""
        d = STORAGE_CONFIG.reports_dir
        if not d.exists():
            return 0.0
        total = 0
        for f in d.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except Exception:
                    pass
        return round(total / (1024 ** 2), 2)

    def gpu_process_info(self) -> list[dict]:
        """Return list of processes using GPU memory (via nvidia-smi).

        Safe aggregate info only — command names and memory usage, no user data.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return []
            procs = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    procs.append(
                        {
                            "pid": parts[0],
                            "process_name": parts[1],
                            "gpu_memory_mb": parts[2],
                        }
                    )
            return procs
        except Exception:
            return []

    def disk_usage(self) -> dict:
        """Return disk usage for relevant paths."""
        import shutil

        usage = {}
        for label, path in [
            ("root", Path("/")),
            ("home", Path("/home")),
            ("temp", STORAGE_CONFIG.temp_dir),
            ("reports", STORAGE_CONFIG.reports_dir),
            ("model_cache", Path.home() / ".torchxrayvision" / "models_data"),
        ]:
            try:
                u = shutil.disk_usage(str(path))
                usage[label] = {
                    "total_gb": round(u.total / (1024 ** 3), 2),
                    "used_gb": round(u.used / (1024 ** 3), 2),
                    "free_gb": round(u.free / (1024 ** 3), 2),
                }
            except Exception:
                usage[label] = {"error": "unavailable"}
        return usage


MONITOR = ResourceMonitor()
