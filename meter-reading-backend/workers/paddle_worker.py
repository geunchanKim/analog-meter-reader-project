"""
workers/paddle_worker.py

Celery tasks that run inside the "paddle worker" process: Stage 3
(the OCR search over the min/max keypoint patches).

THIS FILE IS THE ONLY PLACE paddleocr gets imported. torch_worker.py must
never be imported here (directly or indirectly) -- same reasoning as
torch_worker.py's own docstring, just in the opposite direction.

The OCR engine is loaded ONCE at module import time -- once when this
worker process starts, not once per task. See OcrRunner's own docstring
for why (loading PaddleOCR involves real I/O and isn't something you want
to repeat on every request).
"""

import os

import cv2
import numpy as np

from workers.celery_app import celery_app
from stage3.ocr_runner import OcrRunner
from stage3.pipeline_stage3 import process_gauge
from db.session import get_session
from db import crud

# TODO: move to an environment variable once .env wiring is set up
OCR_DEVICE = os.environ.get("OCR_DEVICE", "cpu")

# Loaded once per worker process -- see docstring above.
ocr_runner = OcrRunner(device=OCR_DEVICE)


def _make_json_safe(value):
    """
    Recursively convert a value into something Celery's default JSON
    serializer can actually store as a task result.

    This exists ONLY because of Celery's requirement -- candidate_scoring.py
    intentionally uses a `set` for `distinct_sources` (that's the right data
    structure for "track which sources this value was seen in, with no
    duplicates"), and that's still the right choice for the algorithm
    itself. The set just can't survive a trip through JSON, so the
    conversion belongs here, at the task boundary, rather than forcing
    the core Stage 3 logic to use a JSON-friendly-but-clunkier structure
    it doesn't actually need.
    """
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(value)
    return value


@celery_app.task(name="workers.paddle_worker.process_stage3")
def process_stage3(job_id: str, gauge_id: str, image_bytes: bytes, bbox, keypoints_global) -> dict:
    """
    Runs Stage 3 for one job.

    job_id, gauge_id, image_bytes, bbox, and keypoints_global are exactly
    what torch_worker.py's process_stage12 passed into send_task() -- this
    is the "other side" of that handoff. gauge_id arrives as a plain
    string (see torch_worker.py's docstring on why a uuid.UUID can't be
    passed through Celery/Redis directly) -- db.crud's functions convert
    it back to a real uuid.UUID internally via _as_uuid(), so it's passed
    straight through here without any conversion of our own.

    Re-decodes image_bytes into a numpy array independently, rather than
    receiving an already-decoded array -- see torch_worker.py's docstring
    for why a numpy array can't be passed through Celery/Redis directly.
    """
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        with get_session() as session:
            crud.update_job_status(session, job_id, "fail", detail="image_decode_error")
        return {
            "job_id": job_id,
            "status": "fail",
            "stage": "stage3",
            "reason": "image_decode_error",
        }

    stage3_result = _make_json_safe(process_gauge(ocr_runner, image, bbox, keypoints_global))

    with get_session() as session:
        crud.save_stage3_results(
            session, gauge_id,
            stage3_result["pos_a_candidates"], stage3_result["pos_b_candidates"],
        )
        crud.update_job_status(session, job_id, "done")

    return {"job_id": job_id, "status": "done", "stage3": stage3_result}