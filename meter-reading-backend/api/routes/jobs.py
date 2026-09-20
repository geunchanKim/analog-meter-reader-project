"""
api/routes/jobs.py

Polling endpoint -- what a client calls repeatedly after POST /gauges
until status is "done" (then it goes fetch GET /gauges/{gauge_id}),
"failed", or "retake".
"""

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db.models import Job
from db.session import get_session

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobStatusResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    detail: str | None = None
    gauge_id: uuid.UUID | None = None


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: uuid.UUID) -> JobStatusResponse:
    with get_session() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        # job.gauges is ordered by created_at (see models.py) -- the last
        # entry is the most recent detection attempt, which is the one a
        # client cares about once status is "done". Most jobs only ever
        # have exactly one (a retake means Stage 1 never got far enough
        # to create a Gauge row in the first place -- see torch_worker.py),
        # so this is usually just "the" gauge, not "the latest of many".
        latest_gauge = job.gauges[-1] if job.gauges else None

        return JobStatusResponse(
            job_id=job.id, status=job.status, detail=job.detail,
            gauge_id=latest_gauge.id if latest_gauge else None,
        )