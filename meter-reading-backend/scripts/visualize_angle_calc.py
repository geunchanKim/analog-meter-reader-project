"""
scripts/visualize_angle_calc.py

Runs the REAL Stage 1 + Stage 2 pipeline to get actual keypoints, then
draws the exact geometry pointer_math.angle_utils.resolve_reading_fraction()
uses to compute a reading: the chosen scale arc (major or minor, drawn
thick/teal), radial lines from center to min/max/tip, and the 4 keypoints
-- directly on the real photo. For presentation slides.

Note: cv2.ellipse's angle convention (0 deg = positive x-axis, increasing
clockwise in image coordinates) matches this project's own angle
convention (see angle_utils.py's module docstring) exactly, so the same
degree values compute_angle_deg() produces can be handed to cv2.ellipse
directly with no conversion.

The block computing `chosen_max_used` mirrors resolve_reading_fraction's
own internal major/minor derivation -- duplicated here (not imported)
only because that internal value isn't part of the function's public
return signature, and this script needs it purely to know which
direction to draw the arc, not to make any actual reading decision.

Usage (from the backend repo root):
    python3 scripts/visualize_angle_calc.py path/to/photo.jpg [min_val max_val] [model_path]

    min_val/max_val are optional -- if given, the final computed value is
    also printed, using the same math as angle_utils.compute_value.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from pointer_math.angle_utils import compute_angle_deg, resolve_reading_fraction
from stage1_2.gauge_detection import gauge_detection_pipeline
from stage1_2.key_points import KeypointModel, map_points_to_original

GREEN = (0, 200, 0)      # min
AMBER = (0, 165, 255)    # max (OpenCV is BGR)
RED = (0, 0, 255)        # tip / needle
GRAY = (150, 150, 150)
TEAL = (170, 140, 0)     # the chosen scale arc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_angle_geometry(image, center, tip, min_pt, max_pt):
    annotated = image.copy()

    min_angle = compute_angle_deg(center, min_pt)
    max_angle = compute_angle_deg(center, max_pt)
    tip_angle = compute_angle_deg(center, tip)

    fraction, arc_span, used_minor = resolve_reading_fraction(center, tip, min_pt, max_pt)

    clockwise_delta = (max_angle - min_angle) % 360
    if clockwise_delta >= 180:
        major_max_used = min_angle + clockwise_delta
        minor_max_used = min_angle - (360 - clockwise_delta)
    else:
        major_max_used = min_angle - (360 - clockwise_delta)
        minor_max_used = min_angle + clockwise_delta
    chosen_max_used = minor_max_used if used_minor else major_max_used

    radius = int((
        np.hypot(min_pt[0] - center[0], min_pt[1] - center[1])
        + np.hypot(max_pt[0] - center[0], max_pt[1] - center[1])
    ) / 2)
    center_i = (int(center[0]), int(center[1]))

    cv2.ellipse(annotated, center_i, (radius, radius), 0, 0, 360, GRAY, 1, cv2.LINE_AA)
    cv2.ellipse(annotated, center_i, (radius, radius), 0, min_angle, chosen_max_used, TEAL, 5, cv2.LINE_AA)

    for pt, color in ((min_pt, GREEN), (max_pt, AMBER), (tip, RED)):
        cv2.line(annotated, center_i, (int(pt[0]), int(pt[1])), color, 2, cv2.LINE_AA)

    for pt, color, label in (
        (center, WHITE, "center"), (min_pt, GREEN, "min"), (max_pt, AMBER, "max"), (tip, RED, "tip"),
    ):
        p = (int(pt[0]), int(pt[1]))
        cv2.circle(annotated, p, 8, color, -1)
        cv2.circle(annotated, p, 8, BLACK, 2)
        cv2.putText(annotated, label, (p[0] + 12, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 3, cv2.LINE_AA)
        cv2.putText(annotated, label, (p[0] + 12, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

    info = {
        "min_angle": min_angle, "max_angle": max_angle, "tip_angle": tip_angle,
        "arc_span": arc_span, "used_minor": used_minor, "fraction": fraction,
    }
    return annotated, info


def main(image_path: str, min_val, max_val, model_path: str) -> None:
    detection = gauge_detection_pipeline(image_path)
    if detection["status"] != "ok":
        print(f"Stage 1에서 실패했어요: {detection.get('reason')}")
        return

    model = KeypointModel(model_path, device="cpu")
    kp_result = model.extract_keypoints(detection["cropped_image"])
    if kp_result["status"] != "ok":
        print(f"Stage 2에서 실패했어요: {kp_result.get('reason')}")
        return

    points = map_points_to_original(kp_result["points"], detection["offset"])

    annotated, info = draw_angle_geometry(
        detection["original_image"], points["center"], points["tip"], points["min"], points["max"]
    )

    base, ext = os.path.splitext(image_path)
    out_path = f"{base}_angle{ext}"
    cv2.imwrite(out_path, annotated)

    print(f"결과 이미지 저장: {out_path}\n")
    print(f"min_angle : {info['min_angle']:.1f}도")
    print(f"max_angle : {info['max_angle']:.1f}도")
    print(f"tip_angle : {info['tip_angle']:.1f}도")
    print(f"선택된 arc: {'minor' if info['used_minor'] else 'major'} ({info['arc_span']:.1f}도)")
    print(f"fraction  : {info['fraction']:.3f}")

    if min_val is not None and max_val is not None:
        value = min_val + info["fraction"] * (max_val - min_val)
        print(f"\n최종 계산값 (min={min_val}, max={max_val}): {value:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 scripts/visualize_angle_calc.py path/to/photo.jpg [min_val max_val] [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    min_val = max_val = None
    model_path = os.environ.get("KEYPOINT_MODEL_PATH", "models/meter_keypoint_v3/best.pt")

    rest = sys.argv[2:]
    if len(rest) >= 2:
        try:
            min_val, max_val = float(rest[0]), float(rest[1])
            rest = rest[2:]
        except ValueError:
            pass
    if rest:
        model_path = rest[0]

    main(image_path, min_val, max_val, model_path)