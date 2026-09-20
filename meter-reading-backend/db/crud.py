"""
db/crud.py

The actual save/query functions -- this is what fills in every
"# TODO: write ... to the ... table" comment left in torch_worker.py and
paddle_worker.py. Keeping these as named functions here (rather than
writing raw session.add(...) calls directly inside the workers) means
the workers don't need to know anything about SQLAlchemy session
mechanics -- they just call e.g. save_stage3_results(...) and move on.
"""

import uuid

from sqlalchemy.orm import Session

from db.models import Candidate, Gauge, Job, Keypoints, Selection


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    """
    Celery serializes task arguments as JSON, and JSON has no UUID type --
    an ID that started life as a uuid.UUID object (e.g. gauge.id) arrives
    in the NEXT task as a plain string. This converts it back to a real
    uuid.UUID before it's used as a primary/foreign key value, which is
    the type the Uuid-typed columns in models.py actually expect.
    Safe to call on a value that's already a uuid.UUID (returns it as-is).
    """
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def create_job(session: Session) -> Job:
    """Creates a new job with status='queued'. Called when a photo is first uploaded."""
    job = Job(status="queued")
    session.add(job)
    session.flush()  # assigns job.id without needing a full commit yet
    return job


def update_job_status(
    session: Session, job_id: uuid.UUID | str, status: str, detail: str | None = None
) -> None:
    """
    Updates a job's status (and optionally why) -- this is the function
    behind every "write job_id's status to the jobs table" TODO in the
    workers.
    """
    job = session.get(Job, _as_uuid(job_id))
    if job is not None:
        job.status = status
        if detail is not None:
            job.detail = detail


def save_stage12_results(
    session: Session,
    job_id: uuid.UUID | str,
    bbox: tuple[int, int, int, int],
    keypoints_global: dict[str, tuple[float, float]],
    confidences: dict[str, float],
    qc_status: str,
    qc_reasons: str = "",
) -> Gauge:
    """
    Saves Stage 1's bbox + Stage 2's keypoints/confidences + the QC
    gate's verdict, all in one call -- this is what torch_worker.py's
    process_stage12 calls right before handing off to Stage 3.

    Creates one Gauge row and one Keypoints row, linked together.
    """
    x, y, w, h = bbox

    gauge = Gauge(
        job_id=_as_uuid(job_id),
        bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
        qc_status=qc_status,
        qc_reasons=qc_reasons,
    )
    session.add(gauge)
    session.flush()  # need gauge.id before creating the Keypoints row below

    keypoints = Keypoints(
        gauge_id=gauge.id,
        center_x=keypoints_global["center"][0], center_y=keypoints_global["center"][1],
        tip_x=keypoints_global["tip"][0], tip_y=keypoints_global["tip"][1],
        min_x=keypoints_global["min"][0], min_y=keypoints_global["min"][1],
        max_x=keypoints_global["max"][0], max_y=keypoints_global["max"][1],
        center_conf=confidences["center"], tip_conf=confidences["tip"],
        min_conf=confidences["min"], max_conf=confidences["max"],
    )
    session.add(keypoints)

    return gauge


def save_stage3_results(
    session: Session,
    gauge_id: uuid.UUID | str,
    pos_a_candidates: list[dict],
    pos_b_candidates: list[dict],
) -> None:
    """
    Saves Stage 3's top-N candidate lists for both positions, one row per
    candidate -- this is what paddle_worker.py's process_stage3 calls
    right before marking the job "done".

    pos_a_candidates / pos_b_candidates are the CandidateGroup dicts
    pipeline_stage3.py produces (already JSON-safe by the time they get
    here, per paddle_worker.py's _make_json_safe step).
    """
    gauge_id = _as_uuid(gauge_id)

    for rank, candidate in enumerate(pos_a_candidates, start=1):
        session.add(Candidate(
            gauge_id=gauge_id, position="A", rank=rank,
            text=candidate["text"], parsed_value=candidate["parsed_value"],
            composite_score=candidate.get("composite_score"),
            is_forced_zero=candidate.get("forced", False),
        ))

    for rank, candidate in enumerate(pos_b_candidates, start=1):
        session.add(Candidate(
            gauge_id=gauge_id, position="B", rank=rank,
            text=candidate["text"], parsed_value=candidate["parsed_value"],
            composite_score=candidate.get("composite_score"),
            is_forced_zero=candidate.get("forced", False),
        ))


def save_selection(
    session: Session,
    gauge_id: uuid.UUID | str,
    chosen_min_value: float,
    chosen_max_value: float,
    swapped: bool,
    computed_value: float,
    actual_value: float | None = None,
    judgment: str | None = None,
    manual_correction: str | None = None,
    note: str | None = None,
) -> Selection:
    """
    Saves the human's final min/max choice + the resulting computed value.
    This is what POST /gauges/{gauge_id}/selection will call.

    actual_value is the real value the person read off the gauge by eye
    -- expected to be filled in only when judgment="incorrect" (the
    frontend enforces that; this function itself doesn't care whether
    the two are consistent, since that's a UI-level policy, not a data
    integrity one). The error itself (actual_value - computed_value) is
    intentionally NOT computed or stored here -- see models.py's comment
    on why.
    """
    selection = Selection(
        gauge_id=_as_uuid(gauge_id),
        chosen_min_value=chosen_min_value,
        chosen_max_value=chosen_max_value,
        swapped=swapped,
        computed_value=computed_value,
        actual_value=actual_value,
        judgment=judgment,
        manual_correction=manual_correction,
        note=note,
    )
    session.add(selection)
    return selection


def get_job_with_gauges(session: Session, job_id: uuid.UUID | str) -> Job | None:
    """Fetches a job along with every gauge attempt under it (for GET /jobs/{job_id})."""
    return session.get(Job, _as_uuid(job_id))