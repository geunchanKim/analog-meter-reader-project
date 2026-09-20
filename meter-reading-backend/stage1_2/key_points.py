"""
key_points.py

Stage 2 of the analog meter reading pipeline: given a CROPPED gauge image
(the output of Stage 1's crop_gauge), locate the four keypoints needed for
the angle-based value calculation: center, tip, min, max.

This module wraps a single YOLOv8-pose model (meter_keypoint_v3). Same
philosophy as gauge_detection.py: it is detection-only. It reports what it
found -- including per-keypoint confidence -- but does not decide whether
the result is trustworthy enough to proceed. That judgment belongs to
qc_gate.py, which will consume this module's output (points + confidences)
alongside Stage 1's output.

IMPORTANT: this module must ONLY ever run on the CROPPED gauge image
(Stage 1's output), never the full raw photo -- running the pose model on
the full photo was a real, measured accuracy regression during
development, not a theoretical concern.
"""

import logging
from typing import TypedDict

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Keypoint order must match the order the model was TRAINED on. This is not
# just a display label list -- index 0 of the model's raw output IS
# "center" only because that's what the training labels said. Changing this
# list without retraining silently reassigns which coordinate means what.
KP_ORDER = ["center", "tip", "min", "max"]

Point = tuple[float, float]


class KeypointResult(TypedDict, total=False):
    status: str                      # "ok" or "fail"
    reason: str                      # present when status == "fail"
    points: dict[str, Point]         # e.g. {"center": (x, y), "tip": (x, y), ...}
    confidences: dict[str, float]    # per-keypoint confidence, same keys as points


class KeypointModel:
    """
    Thin wrapper around a single loaded YOLO pose model.

    Wrapped in a class (rather than a bare module-level `kp_model` variable,
    which is what the exploration notebooks used) so a backend process can
    load the model ONCE at startup and reuse it across many requests.
    Loading a YOLO checkpoint involves real disk I/O and weight
    deserialization -- reloading it per request would make every API call
    slow for no reason, and a module-level global makes it easy to
    accidentally reload it in more than one place as the codebase grows.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        device defaults to "cpu" to match the production deployment
        assumption (GPU is acceptable for local dev/testing only, per the
        deployment plan -- the CPU-vs-GPU distinction matters more for
        Stage 3's OCR calls, but the same default is used here for
        consistency and to make local dev/prod parity easier to reason about).
        """
        self.model = YOLO(model_path)
        self.device = device
        logger.info("key_points: loaded model from %s (device=%s)", model_path, device)

    def extract_keypoints(self, cropped_image: np.ndarray) -> KeypointResult:
        """
        Run the pose model on an already-cropped gauge image and return the
        four keypoints in CROP-LOCAL pixel coordinates, along with each
        one's confidence score.

        Returns {'status': 'fail', 'reason': ...} if:
            - the model found no pose at all ("no_pose_detected"), or
            - it found a pose but not exactly len(KP_ORDER) keypoints
            ("unexpected_keypoint_count") -- this can happen on a
            malformed/partial detection and should be treated as
            untrustworthy rather than silently truncated or zero-padded.
        """
        result = self.model.predict(cropped_image, verbose=False, device=self.device)[0]

        if result.keypoints is None or result.keypoints.xy.shape[0] == 0:
            logger.info("key_points: no pose detected")
            return {"status": "fail", "reason": "no_pose_detected"}

        xy = result.keypoints.xy[0].cpu().numpy()      # shape: (n_keypoints, 2)
        conf = result.keypoints.conf[0].cpu().numpy()  # shape: (n_keypoints,)

        if xy.shape[0] != len(KP_ORDER):
            logger.warning(
                "key_points: expected %d keypoints, model returned %d",
                len(KP_ORDER), xy.shape[0],
            )
            return {"status": "fail", "reason": "unexpected_keypoint_count"}

        points = {name: (float(xy[i][0]), float(xy[i][1])) for i, name in enumerate(KP_ORDER)}
        confidences = {name: float(conf[i]) for i, name in enumerate(KP_ORDER)}

        logger.debug("key_points: extracted points with confidences=%s", confidences)
        return {"status": "ok", "points": points, "confidences": confidences}


def map_points_to_original(points: dict[str, Point], offset: tuple[int, int]) -> dict[str, Point]:
    """
    Translate crop-local keypoints back into the ORIGINAL (pre-crop) image's
    coordinate space, using the (offset_x, offset_y) returned by
    gauge_detection.crop_gauge(). Needed anywhere keypoints must line up
    with the original photo rather than the crop -- e.g. drawing a
    visualization on the original image, or any future step that needs
    coordinates relative to the un-cropped frame.
    """
    offset_x, offset_y = offset
    return {name: (x + offset_x, y + offset_y) for name, (x, y) in points.items()}