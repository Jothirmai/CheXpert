"""Temporary artifact cleanup with dry-run support.

Preserves model weights, user reports, and active job artifacts. Only deletes
expired temp files and stale job artifacts within Chest Xpert-owned directories.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from backend.config import STORAGE_CONFIG
from backend.services.resource_manager import MONITOR


class CleanupService:
    """Safe administrative cleanup of expired application artifacts."""

    def __init__(self) -> None:
        self.temp_dir = STORAGE_CONFIG.temp_dir
        self.reports_dir = STORAGE_CONFIG.reports_dir

    def dry_run(self, ttl_s: Optional[int] = None) -> dict:
        """Return a summary of what would be deleted, without deleting."""
        if ttl_s is None:
            ttl_s = STORAGE_CONFIG.artifact_ttl_s
        now = time.time()
        candidates: list[dict] = []
        total_bytes = 0

        for d in [self.temp_dir, self.reports_dir]:
            if not d.exists():
                continue
            for f in d.rglob("*"):
                if not f.is_file():
                    continue
                try:
                    mtime = f.stat().st_mtime
                    size = f.stat().st_size
                except Exception:
                    continue
                age = now - mtime
                if age > ttl_s:
                    candidates.append(
                        {
                            "path": str(f),
                            "age_s": round(age, 1),
                            "size_mb": round(size / (1024 * 1024), 4),
                        }
                    )
                    total_bytes += size

        return {
            "ttl_s": ttl_s,
            "candidate_count": len(candidates),
            "total_size_mb": round(total_bytes / (1024 * 1024), 4),
            "candidates": candidates,
            "preserved": [
                "~/.torchxrayvision/models_data/ (pretrained weights)",
                "active job artifacts (not yet expired)",
                "user-requested reports (unless retention policy expires them)",
            ],
        }

    def run(self, ttl_s: Optional[int] = None, dry_run: bool = False) -> dict:
        """Execute (or simulate) cleanup. Returns summary."""
        summary = self.dry_run(ttl_s)
        if dry_run:
            return summary

        deleted: list[str] = []
        freed_bytes = 0
        now = time.time()
        if ttl_s is None:
            ttl_s = STORAGE_CONFIG.artifact_ttl_s

        for d in [self.temp_dir, self.reports_dir]:
            if not d.exists():
                continue
            for f in list(d.rglob("*")):
                if not f.is_file():
                    continue
                try:
                    mtime = f.stat().st_mtime
                    size = f.stat().st_size
                except Exception:
                    continue
                age = now - mtime
                if age > ttl_s:
                    try:
                        f.unlink()
                        deleted.append(str(f))
                        freed_bytes += size
                    except Exception:
                        pass

        summary["deleted_count"] = len(deleted)
        summary["freed_mb"] = round(freed_bytes / (1024 * 1024), 4)
        summary["deleted"] = deleted
        return summary

    def cache_sizes(self) -> dict:
        """Return sizes of the various caches (read-only)."""
        return {
            "torchxrayvision_model_weights_mb": MONITOR.model_cache_size_mb(),
            "temp_artifacts_mb": MONITOR.temp_dir_size_mb(),
            "reports_mb": MONITOR.reports_dir_size_mb(),
        }


cleanup_service = CleanupService()
