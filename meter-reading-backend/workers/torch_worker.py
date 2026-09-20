"""
workers/torch_worker.py

Celery tasks that run inside the "torch worker" process: Stage 1 (gauge
detection), Stage 2 (keypoint model), and the QC gate.

THIS FILE IS THE ONLY PLACE ultralytics/YOLO gets imported. paddle_worker.py
must never be imported here (directly or indirectly) -- doing so would
pull paddleocr into this process and reproduce the crash confirmed in
test_torch_paddle_coexist.py. When this task needs Stage 3 to run, it
hands off via celery_app.send_task() using a STRING task name, never a
direct Python import of paddle_worker.py.

The keypoint model is loaded ONCE at module import time -- i.e. once when
this worker process starts up, not once per task. A Celery worker process
stays alive and handles many tasks one after another, so loading the model
per-task would reload it on every single request for no reason.
"""

import logging
import os

from workers.celery_app import celery_app
from stage1_2.gauge_detection import gauge_detection_pipeline_from_bytes
from stage1_2.key_points import KeypointModel, map_points_to_original
from stage1_2.qc_gate import run_qc_gate
from db.session import get_session
from db import crud
from db.storage import upload_gauge_photo

logger = logging.getLogger(__name__)

# TODO: move to an environment variable once .env wiring is set up.
# Expects the repo-root-relative path models/<name>/best.pt -- see the
# "models/ folder" note in the project README for why weights live here
# and not inside stage1_2/ itself.
KEYPOINT_MODEL_PATH = os.environ.get("KEYPOINT_MODEL_PATH", "models/meter_keypoint_v3/best.pt")

# Loaded once per worker process -- see docstring above.
keypoint_model = KeypointModel(KEYPOINT_MODEL_PATH, device="cpu")


@celery_app.task(name="workers.torch_worker.process_stage12")
def process_stage12(job_id: str, image_bytes: bytes) -> dict:
    """
    Runs Stage 1 -> Stage 2 -> QC gate for one job.

    job_id must already exist as a real row in the `jobs` table (created
    via db.crud.create_job() -- normally by the API when a photo is first
    uploaded) -- this task only ever UPDATES an existing job, it never
    creates one, since "a job exists" is what makes GET /jobs/{job_id}
    meaningful to poll in the first place.

    Every early exit below (Stage 1 fail/retake, Stage 2 fail, QC gate
    reject) updates the job's status/detail in the DB and returns
    immediately WITHOUT calling Stage 3.
    """
    # --- Stage 1 ---
    detection_result = gauge_detection_pipeline_from_bytes(image_bytes)

    if detection_result["status"] != "ok":
        with get_session() as session:
            crud.update_job_status(
                session, job_id, detection_result["status"],
                detail=detection_result.get("reason"),
            )
        return {
            "job_id": job_id,
            "status": detection_result["status"],
            "stage": "stage1",
            "reason": detection_result.get("reason"),
        }

    bbox = detection_result["candidate"]["bbox"]
    cropped_image = detection_result["cropped_image"]
    offset = detection_result["offset"]
    original_image = detection_result["original_image"]

    # --- Stage 2 ---
    kp_result = keypoint_model.extract_keypoints(cropped_image)

    if kp_result["status"] != "ok":
        with get_session() as session:
            crud.update_job_status(session, job_id, "fail", detail=kp_result.get("reason"))
        return {
            "job_id": job_id,
            "status": "fail",
            "stage": "stage2",
            "reason": kp_result.get("reason"),
        }

    keypoints_global = map_points_to_original(kp_result["points"], offset)

    # --- QC gate ---
    qc_result = run_qc_gate(keypoints_global, kp_result["confidences"])

    if qc_result["status"] != "ok":
        with get_session() as session:
            crud.update_job_status(session, job_id, "retake", detail=", ".join(qc_result["reasons"]))
        return {
            "job_id": job_id,
            "status": "retake",
            "stage": "qc_gate",
            "reason": ", ".join(qc_result["reasons"]),
        }

    # --- Save Stage 1+2 results, then hand off to Stage 3 ---
    with get_session() as session:
        gauge = crud.save_stage12_results(
            session, job_id=job_id, bbox=bbox,
            keypoints_global=keypoints_global, confidences=kp_result["confidences"],
            qc_status=qc_result["status"],
        )
        gauge_id = gauge.id          # a real uuid.UUID, valid until the `with` block exits

        # Best-effort: a failed upload (bad credentials, Supabase hiccup,
        # not configured yet) should NOT block the actual reading -- only
        # the history sidebar's photo display depends on this succeeding.
        try:
            gauge.image_url = upload_gauge_photo(image_bytes, gauge_id)
        except Exception:
            logger.warning("failed to upload gauge photo to storage", exc_info=True)

        crud.update_job_status(session, job_id, "stage3_queued")

    # Passing str(gauge_id) rather than the uuid.UUID object itself --
    # same reason as passing raw image_bytes instead of a numpy array:
    # Celery serializes task arguments as JSON, and a uuid.UUID isn't
    # JSON-serializable any more than a numpy array or a set was.
    # paddle_worker.py converts it back via db.crud._as_uuid() before use.
    celery_app.send_task(
        "workers.paddle_worker.process_stage3",
        args=[job_id, str(gauge_id), image_bytes, bbox, keypoints_global],
        queue="paddle_queue",
    )

    return {"job_id": job_id, "status": "stage12_done", "stage": "stage1_2"}