"""
scripts/visualize_keypoints.py

Runs the REAL Stage 1 (crop) + Stage 2 (keypoint model) on a photo, and
draws all 4 keypoints (center, tip, min, max) as red dots with text
labels on top of the cropped gauge -- for presentation slides.

Reuses gauge_detection.py and key_points.py directly (no reimplemented
logic), so the slide image shows exactly what the real pipeline sees.

Usage (from the backend repo root):
    python3 scripts/visualize_keypoints.py path/to/photo.jpg [model_path]

model_path defaults to $KEYPOINT_MODEL_PATH (same env var torch_worker.py
uses), or "models/meter_keypoint_v3/best.pt" if that's not set either.

Produces, next to the input file:
    photo_keypoints.jpg   -- the cropped gauge with all 4 keypoints marked
"""

import os
import sys

# Same reasoning as visualize_detection.py: makes this runnable regardless
# of current working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from stage1_2.gauge_detection import gauge_detection_pipeline
from stage1_2.key_points import KeypointModel

RED = (0, 0, 255)       # OpenCV uses BGR
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DOT_RADIUS = 8
LABEL_OFFSET = (12, 4)


def draw_keypoints(image, points: dict[str, tuple[float, float]]):
    """
    Draws every keypoint as a solid red dot with a thin white ring
    (so it stays visible against both light and dark gauge faces), plus
    a text label -- black outline + white fill, so it reads clearly
    regardless of what's behind it in the photo.
    """
    annotated = image.copy()

    for name, (x, y) in points.items():
        x, y = int(round(x)), int(round(y))

        cv2.circle(annotated, (x, y), DOT_RADIUS, RED, -1)
        cv2.circle(annotated, (x, y), DOT_RADIUS, WHITE, 2)

        label_pos = (x + LABEL_OFFSET[0], y + LABEL_OFFSET[1])
        # Draw the label twice -- a thick black outline pass first, then a
        # thinner white pass on top -- so it's legible over any background.
        cv2.putText(annotated, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 3, cv2.LINE_AA)
        cv2.putText(annotated, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

    return annotated


def main(image_path: str, model_path: str) -> None:
    detection = gauge_detection_pipeline(image_path)

    if detection["status"] != "ok":
        print(f"Stage 1에서 실패했어요: {detection['status']} / {detection.get('reason')}")
        return

    cropped = detection["cropped_image"]

    model = KeypointModel(model_path, device="cpu")
    kp_result = model.extract_keypoints(cropped)

    if kp_result["status"] != "ok":
        print(f"Stage 2에서 실패했어요: {kp_result.get('reason')}")
        return

    annotated = draw_keypoints(cropped, kp_result["points"])

    base, ext = os.path.splitext(image_path)
    out_path = f"{base}_keypoints{ext}"
    cv2.imwrite(out_path, annotated)

    print(f"keypoint 결과 저장: {out_path}")
    print("confidences:", {k: round(v, 3) for k, v in kp_result["confidences"].items()})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 scripts/visualize_keypoints.py path/to/photo.jpg [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = (
        sys.argv[2] if len(sys.argv) > 2
        else os.environ.get("KEYPOINT_MODEL_PATH", "models/meter_keypoint_v3/best.pt")
    )
    main(image_path, model_path)