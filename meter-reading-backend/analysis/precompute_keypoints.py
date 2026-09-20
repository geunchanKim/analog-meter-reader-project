"""
precompute_keypoints.py

Regenerates precomputed_keypoints.json using the modularized Stage 1/2
code (gauge_detection.py, key_points.py) instead of the inline notebook
functions from test_pytorch.ipynb.

The one real change from the original notebook version: each entry now
also stores per-keypoint CONFIDENCE, not just coordinates. The original
precomputed_keypoints.json only saved coordinates, which meant
qc_gate.py's check_keypoint_confidence() had nothing real to validate
against -- this script produces the data qc_gate's threshold (currently
an estimate, MIN_KEYPOINT_CONFIDENCE=0.5) needs to actually be checked
against.

Meant to be run locally (needs real images + the YOLO checkpoint on
disk) -- this sandbox has neither, so this script is written and
syntax-checked but not executed here.

Usage:
    python precompute_keypoints.py
"""

import glob
import json
import logging
import os
import random

import cv2

from gauge_detection import gauge_detection_pipeline
from key_points import KeypointModel, map_points_to_original

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config -- adjust paths to match your local layout -------------------
ANALOG_CROPPED_DIR = "../../data/raw/analog_cropped_v2/test/images"
EXTERNAL_TEST_DIR = "../../data/raw/external_test/needle_base_tip_min_max/test"
KEYPOINT_MODEL_PATH = "../../data/models/meter_keypoint_v3/weights/best.pt"
OUTPUT_PATH = "precomputed_keypoints.json"
N_SAMPLES_PER_SOURCE = 100
SEED = 42


def list_images(directory: str) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(files)


def process_one_image(model: KeypointModel, img_path: str) -> dict | None:
    """
    Run Stage 1 -> Stage 2 on a single image and return everything
    precompute_keypoints.py needs to save, or None (with a logged reason)
    if either stage failed.
    """
    detection_result = gauge_detection_pipeline(img_path)

    if detection_result["status"] != "ok":
        logger.info(
            "skip %s: gauge_detection status=%s reason=%s",
            os.path.basename(img_path), detection_result["status"],
            detection_result.get("reason"),
        )
        return None

    bbox = detection_result["candidate"]["bbox"]
    offset = detection_result["offset"]
    cropped_image = detection_result["cropped_image"]

    kp_result = model.extract_keypoints(cropped_image)
    if kp_result["status"] != "ok":
        logger.info(
            "skip %s: key_points status=%s reason=%s",
            os.path.basename(img_path), kp_result["status"], kp_result.get("reason"),
        )
        return None

    keypoints_global = map_points_to_original(kp_result["points"], offset)

    return {
        "bbox": bbox,
        "offset": offset,
        "keypoints_global": keypoints_global,
        "confidences": kp_result["confidences"],
    }


def main() -> None:
    random.seed(SEED)

    analog_images = list_images(ANALOG_CROPPED_DIR)
    external_images = list_images(EXTERNAL_TEST_DIR)
    logger.info("analog_cropped_v2 images: %d", len(analog_images))
    logger.info("external_test images: %d", len(external_images))

    sample_analog = random.sample(analog_images, min(N_SAMPLES_PER_SOURCE, len(analog_images)))
    sample_external = random.sample(external_images, min(N_SAMPLES_PER_SOURCE, len(external_images)))

    all_sample = (
        [{"path": p, "source": "analog_cropped_v2"} for p in sample_analog]
        + [{"path": p, "source": "external_test"} for p in sample_external]
    )

    model = KeypointModel(KEYPOINT_MODEL_PATH, device="cpu")

    precomputed = []
    for item in all_sample:
        result = process_one_image(model, item["path"])
        if result is None:
            continue

        precomputed.append({
            "path": item["path"],
            "source": item["source"],
            "bbox": list(result["bbox"]),
            "offset": list(result["offset"]),
            "keypoints_global": {k: list(v) for k, v in result["keypoints_global"].items()},
            "confidences": result["confidences"],
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(precomputed, f, indent=2)

    logger.info("saved %d / %d entries to %s", len(precomputed), len(all_sample), OUTPUT_PATH)


if __name__ == "__main__":
    main()