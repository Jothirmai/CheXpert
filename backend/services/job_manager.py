"""Background job manager with concurrency limiting and TTL cleanup."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

from backend.config import STORAGE_CONFIG
from backend.services.inference_service import InferenceService, JobRecord, JobStatus


class JobManager:
    """Tracks analysis jobs and enqueues pipeline work.

    Concurrency is bounded by the semaphore in InferenceService so GPU-heavy
    jobs are serialized. Jobs older than the TTL are candidates for cleanup.
    """

    def __init__(self, service: InferenceService) -> None:
        self._service = service

    def create(
        self,
        analysis_id: str,
        image_path: str,
        request_data: dict,
        generate_grad_cam: bool = False,
    ) -> JobRecord:
        job = self._service.create_job(analysis_id, image_path)
        # Fire-and-forget background task; the semaphore serializes execution.
        asyncio.create_task(
            self._service.run_analysis(
                analysis_id,
                image_path,
                request_data,
                generate_grad_cam,
            )
        )
        return job

    def get(self, analysis_id: str) -> Optional[JobRecord]:
        return self._service.get_job(analysis_id)

    def active_count(self) -> int:
        return sum(
            1
            for j in self._service.list_jobs()
            if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
        )

    def queued_count(self) -> int:
        return sum(1 for j in self._service.list_jobs() if j.status == JobStatus.QUEUED)

    def cleanup_expired(self, ttl_s: Optional[int] = None) -> list[str]:
        """Remove expired jobs and their temp artifacts. Returns cleaned IDs."""
        if ttl_s is None:
            ttl_s = STORAGE_CONFIG.artifact_ttl_s
        now = time.time()
        cleaned: list[str] = []
        for aid, job in list(self._service.list_jobs() and self._service._jobs.items()):
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                age = now - (job.completed_at or job.created_at)
                if age > ttl_s:
                    # Delete temp artifacts (Grad-CAM sheet, upload copy).
                    for p in [job.image_path, job.grad_cam_path]:
                        if p:
                            try:
                                Path(p).unlink(missing_ok=True)
                            except Exception:
                                pass
                    del self._service._jobs[aid]
                    cleaned.append(aid)
        return cleaned
