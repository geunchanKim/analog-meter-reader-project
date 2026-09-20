"""
scripts/inspect_qc_gate.py

Runs the REAL Stage 1 + Stage 2 + qc_gate on a photo, and prints every
value qc_gate.py actually checks -- per-keypoint confidence, which arc
(major/minor) was chosen and its span in degrees, and the final
pass/reject verdict. For pulling real numbers into a presentation table
(run this on a few photos, paste the printed output back).

Usage (from the backend repo root):
    python3 scripts/inspect_qc_gate.py path/to/photo.jpg [model_path]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointer_math.angle_utils import resolve_reading_fraction
from stage1_2.gauge_detection import gauge_detection_pipeline
from stage1_2.key_points import KeypointModel, map_points_to_original
from stage1_2.qc_gate import MIN_ARC_SPAN_DEG, MAX_ARC_SPAN_DEG, MIN_KEYPOINT_CONFIDENCE, run_qc_gate


def main(image_path: str, model_path: str) -> None:
    detection = gauge_detection_pipeline(image_path)

    if detection["status"] != "ok":
        print(f"Stage 1에서 실패했어요: {detection['status']} / {detection.get('reason')}")
        return

    model = KeypointModel(model_path, device="cpu")
    kp_result = model.extract_keypoints(detection["cropped_image"])

    if kp_result["status"] != "ok":
        print(f"Stage 2에서 실패했어요: {kp_result.get('reason')}")
        return

    points = map_points_to_original(kp_result["points"], detection["offset"])
    confidences = kp_result["confidences"]

    print(f"=== {os.path.basename(image_path)} ===\n")

    print(f"-- Keypoint confidence (threshold: {MIN_KEYPOINT_CONFIDENCE}) --")
    for name, conf in confidences.items():
        verdict = "PASS" if conf >= MIN_KEYPOINT_CONFIDENCE else "FAIL"
        print(f"  {name:8s}: {conf:.3f}  [{verdict}]")

    fraction, arc_span, used_minor = resolve_reading_fraction(
        points["center"], points["tip"], points["min"], points["max"]
    )
    arc_ok = MIN_ARC_SPAN_DEG <= arc_span <= MAX_ARC_SPAN_DEG

    print(f"\n-- Arc span (valid range: {MIN_ARC_SPAN_DEG}-{MAX_ARC_SPAN_DEG} deg) --")
    print(f"  arc used  : {'minor' if used_minor else 'major'}")
    print(f"  arc span  : {arc_span:.1f} deg  [{'PASS' if arc_ok else 'FAIL'}]")
    print(f"  fraction  : {fraction:.3f}  (needle's position along that arc, 0=min, 1=max)")

    qc_result = run_qc_gate(points, confidences)
    print(f"\n-- Final verdict --")
    print(f"  status : {qc_result['status']}")
    print(f"  reasons: {qc_result['reasons'] or 'none'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 scripts/inspect_qc_gate.py path/to/photo.jpg [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = (
        sys.argv[2] if len(sys.argv) > 2
        else os.environ.get("KEYPOINT_MODEL_PATH", "models/meter_keypoint_v3/best.pt")
    )
    main(image_path, model_path)