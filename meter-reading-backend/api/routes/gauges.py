"""
api/routes/gauges.py

Endpoints for uploading a photo, fetching a gauge's Stage 3 candidates,
and submitting the human's final min/max choice.
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from db import crud
from db.models import Candidate, Gauge, Keypoints, Selection
from db.session import get_session
from pointer_math.angle_utils import compute_value
from workers.celery_app import celery_app
from agent.llm_client import CandidateExplainer

router = APIRouter(prefix="/gauges", tags=["gauges"])


class EnqueueResponse(BaseModel):
    job_id: uuid.UUID
    status: str


@router.post("", response_model=EnqueueResponse)
async def enqueue_gauge(file: UploadFile = File(...)) -> EnqueueResponse:
    """
    Uploads a photo, creates a job row, and enqueues Stage 1 processing.

    Deliberately does NOT import torch_worker.py (which would load a YOLO
    model into every API server process just to reference a task name --
    exactly the kind of cross-contamination the torch/paddle process
    split exists to prevent). celery_app.send_task() enqueues the task by
    its string name instead, the same trick torch_worker.py itself uses
    to hand off to paddle_worker.py without importing it.
    """
    image_bytes = await file.read()

    with get_session() as session:
        job = crud.create_job(session)
        job_id = job.id

    celery_app.send_task(
        "workers.torch_worker.process_stage12",
        args=[str(job_id), image_bytes],
        queue="torch_queue",
    )

    return EnqueueResponse(job_id=job_id, status="queued")


class CandidateOut(BaseModel):
    text: str
    parsed_value: float
    rank: int
    is_forced_zero: bool = False


class GaugeResultResponse(BaseModel):
    gauge_id: uuid.UUID
    image_url: str | None  # None if the photo upload to storage failed or hasn't run
    bbox: list[int] | None  # [x, y, w, h] from Stage 1, in ORIGINAL image pixel coords
    keypoints: dict | None
    pos_a_candidates: list[CandidateOut]
    pos_b_candidates: list[CandidateOut]


@router.get("/{gauge_id}", response_model=GaugeResultResponse)
def get_gauge_result(gauge_id: uuid.UUID) -> GaugeResultResponse:
    """
    Fetches a gauge's crop bbox, keypoints, and Stage 3 candidates --
    what the frontend loads once GET /jobs/{job_id} reports
    status="done", to show the candidate-selection + swap screen.

    bbox is included so the frontend can visually crop the ORIGINAL photo
    down to just the detected gauge (Stage 1's output) without the
    backend needing to store a second, separately-cropped image file --
    the frontend already has the full photo client-side (as an object
    URL created at upload time) and can window into it using bbox alone.
    """
    with get_session() as session:
        gauge = session.get(Gauge, gauge_id)
        keypoints = session.query(Keypoints).filter_by(gauge_id=gauge_id).first()
        candidates = (
            session.query(Candidate)
            .filter_by(gauge_id=gauge_id)
            .order_by(Candidate.position, Candidate.rank)
            .all()
        )

        if gauge is None and keypoints is None and not candidates:
            raise HTTPException(status_code=404, detail="gauge not found")

        pos_a = [CandidateOut(text=c.text, parsed_value=c.parsed_value, rank=c.rank, is_forced_zero=c.is_forced_zero)
                 for c in candidates if c.position == "A"]
        pos_b = [CandidateOut(text=c.text, parsed_value=c.parsed_value, rank=c.rank, is_forced_zero=c.is_forced_zero)
                 for c in candidates if c.position == "B"]

        bbox_out = None
        if gauge is not None:
            bbox_out = [gauge.bbox_x, gauge.bbox_y, gauge.bbox_w, gauge.bbox_h]

        kp_out = None
        if keypoints is not None:
            kp_out = {
                "center": [keypoints.center_x, keypoints.center_y],
                "tip": [keypoints.tip_x, keypoints.tip_y],
                "min": [keypoints.min_x, keypoints.min_y],
                "max": [keypoints.max_x, keypoints.max_y],
            }

        return GaugeResultResponse(
            gauge_id=gauge_id,
            image_url=gauge.image_url if gauge is not None else None,
            bbox=bbox_out, keypoints=kp_out,
            pos_a_candidates=pos_a, pos_b_candidates=pos_b,
        )


# Loaded once per API server process, on first use -- not at import time,
# since GROQ_API_KEY may not be configured yet in every environment (e.g.
# local dev without agent work happening), and failing at import would
# take down every OTHER endpoint in this file too.
_explainer: CandidateExplainer | None = None


def _get_explainer() -> CandidateExplainer:
    global _explainer
    if _explainer is None:
        _explainer = CandidateExplainer()
    return _explainer


class ExplainResponse(BaseModel):
    position_a: str | None
    position_b: str | None


@router.post("/{gauge_id}/explain", response_model=ExplainResponse)
def explain_candidates(gauge_id: uuid.UUID) -> ExplainResponse:
    """
    Asks the LLM agent to comment on which OCR candidate looks most
    plausible for each position -- advisory only, same principle as the
    rest of this pipeline: the agent never picks a value for the human,
    it only explains. See agent/llm_client.py's system prompt for the
    exact rules that keep it from drifting into making the decision.
    """
    with get_session() as session:
        candidates = (
            session.query(Candidate)
            .filter_by(gauge_id=gauge_id)
            .order_by(Candidate.position, Candidate.rank)
            .all()
        )

        if not candidates:
            raise HTTPException(status_code=404, detail="no candidates found for this gauge")

        pos_a = [
            {"rank": c.rank, "text": c.text, "parsed_value": c.parsed_value, "is_forced_zero": c.is_forced_zero}
            for c in candidates if c.position == "A"
        ]
        pos_b = [
            {"rank": c.rank, "text": c.text, "parsed_value": c.parsed_value, "is_forced_zero": c.is_forced_zero}
            for c in candidates if c.position == "B"
        ]

    result = _get_explainer().explain(pos_a, pos_b)
    return ExplainResponse(**result)


class SelectionRequest(BaseModel):
    chosen_min_value: float
    chosen_max_value: float
    # True if the keypoint model's "min"-labeled point actually turned out
    # to be the MAX side (Position A/B were backwards) -- see qc_gate.py
    # and pipeline_stage3.py's docstrings on why this is never decided
    # automatically.
    swapped: bool = False
    judgment: str | None = None          # "correct" / "incorrect" / "unclear"
    # The real value read off the gauge by eye -- expected only when
    # judgment="incorrect". The frontend is what enforces that pairing;
    # this endpoint just stores whatever it's given.
    actual_value: float | None = None
    manual_correction: str | None = None
    note: str | None = None


class SelectionResponse(BaseModel):
    gauge_id: uuid.UUID
    computed_value: float
    swapped: bool


class PreviewRequest(BaseModel):
    chosen_min_value: float
    chosen_max_value: float
    swapped: bool = False


class PreviewResponse(BaseModel):
    computed_value: float


def _compute_value_for_gauge(
    session, gauge_id: uuid.UUID, chosen_min_value: float, chosen_max_value: float, swapped: bool
) -> float:
    """
    Shared by both /preview and /selection -- the actual math is just
    one call to pointer_math.angle_utils.compute_value(); this helper's
    job is assembling that call's arguments correctly from what's stored
    in the `keypoints` table plus what swapped means:

      - swapped=False: the model's own min/max labels were correct --
        use keypoints.min_x/y as the min point, keypoints.max_x/y as max.
      - swapped=True: they were backwards -- use keypoints.max_x/y as the
        min point (since that's the physical location the tip's angle
        should be measured against for the min side) and vice versa.
    """
    keypoints = session.query(Keypoints).filter_by(gauge_id=gauge_id).first()
    if keypoints is None:
        raise HTTPException(status_code=404, detail="gauge has no keypoints yet")

    center = (keypoints.center_x, keypoints.center_y)
    tip = (keypoints.tip_x, keypoints.tip_y)

    if not swapped:
        min_point = (keypoints.min_x, keypoints.min_y)
        max_point = (keypoints.max_x, keypoints.max_y)
    else:
        min_point = (keypoints.max_x, keypoints.max_y)
        max_point = (keypoints.min_x, keypoints.min_y)

    return compute_value(center, tip, min_point, max_point, chosen_min_value, chosen_max_value)


@router.post("/{gauge_id}/preview", response_model=PreviewResponse)
def preview_value(gauge_id: uuid.UUID, body: PreviewRequest) -> PreviewResponse:
    """
    Computes what the reading WOULD be for a given min/max choice --
    without saving anything. This is what lets a person see the
    calculated value BEFORE deciding whether to judge it correct/
    incorrect: judging "incorrect" only makes sense once you've actually
    seen the number that's supposedly wrong, so a person needs a way to
    look at it first without that look being treated as a final answer.
    """
    with get_session() as session:
        computed_value = _compute_value_for_gauge(
            session, gauge_id, body.chosen_min_value, body.chosen_max_value, body.swapped
        )
    return PreviewResponse(computed_value=computed_value)


@router.post("/{gauge_id}/selection", response_model=SelectionResponse)
def submit_selection(gauge_id: uuid.UUID, body: SelectionRequest) -> SelectionResponse:
    """
    Takes the human's final min/max choice (+ judgment, + actual_value
    if incorrect), computes the actual gauge reading, and saves
    everything. In practice, the frontend already showed this same
    computed_value via /preview before the person picked a judgment --
    this call recomputes it (rather than trusting a client-sent value)
    so the persisted number can't drift from what the keypoints/inputs
    actually produce, and saves the judgment alongside it.
    """
    with get_session() as session:
        computed_value = _compute_value_for_gauge(
            session, gauge_id, body.chosen_min_value, body.chosen_max_value, body.swapped
        )

        crud.save_selection(
            session, gauge_id,
            chosen_min_value=body.chosen_min_value,
            chosen_max_value=body.chosen_max_value,
            swapped=body.swapped,
            computed_value=computed_value,
            actual_value=body.actual_value,
            judgment=body.judgment,
            manual_correction=body.manual_correction,
            note=body.note,
        )

    return SelectionResponse(gauge_id=gauge_id, computed_value=computed_value, swapped=body.swapped)


class HistoryItem(BaseModel):
    gauge_id: uuid.UUID
    image_url: str | None
    created_at: datetime
    chosen_min_value: float
    chosen_max_value: float
    computed_value: float
    actual_value: float | None
    judgment: str | None
    swapped: bool


@router.get("", response_model=list[HistoryItem])
def list_history(limit: int = 50, offset: int = 0) -> list[HistoryItem]:
    """
    Lists finished judgments, most recent first -- the raw material for
    "how many gauges have I read, how many were correct" and (later)
    error analysis (actual_value - computed_value), computed on demand
    rather than stored, per models.py's Selection.actual_value comment.

    Joined against `gauges` (not just `selections` alone) specifically
    for image_url -- now that torch_worker.py actually uploads photos to
    Supabase Storage, this is what lets the history sidebar show a real
    thumbnail instead of numbers only.
    """
    with get_session() as session:
        rows = (
            session.query(Selection, Gauge.image_url)
            .join(Gauge, Gauge.id == Selection.gauge_id)
            .order_by(Selection.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return [
            HistoryItem(
                gauge_id=sel.gauge_id,
                image_url=image_url,
                created_at=sel.created_at,
                chosen_min_value=sel.chosen_min_value,
                chosen_max_value=sel.chosen_max_value,
                computed_value=sel.computed_value,
                actual_value=sel.actual_value,
                judgment=sel.judgment,
                swapped=sel.swapped,
            )
            for sel, image_url in rows
        ]