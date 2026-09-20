"""
scripts/visualize_ocr.py

Runs the REAL Stage 3 progressive OCR search (same stage_specs and
early-stop rule as candidate_scoring.get_top_n_groups /
search_strategy.py) for one keypoint position (min or max), saving one
image PER rotation actually attempted -- with every detected numeric
OCR box drawn in red -- plus a printed summary of the final top-3
candidates. For presentation slides.

Note: draws boxes by calling the OCR engine's own .predict() directly
rather than through ocr_runner.run_with_boxes(), since that function
only returns a detection's center + height (enough for the real
candidate-scoring logic), not the full box polygon needed to draw an
accurate outline. The actual candidate collection/scoring below still
goes through the real collect_candidates_for_angles/get_top_n_groups
functions, unmodified.

Usage (from the backend repo root):
    python3 scripts/visualize_ocr.py path/to/photo.jpg <min|max> [model_path]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from stage1_2.gauge_detection import gauge_detection_pipeline
from stage1_2.key_points import KeypointModel, map_points_to_original
from stage3.candidate_scoring import collect_candidates_for_angles, get_top_n_groups
from stage3.ocr_runner import OcrRunner
from stage3.preprocessing import (
    ANGLES_PRIMARY,
    ANGLES_SECONDARY,
    compute_dynamic_crop_ratio,
    crop_around_point,
    rotate_patch_and_point,
    upscale_and_binarize,
    upscale_only,
)

RED = (0, 0, 255)


def draw_ocr_boxes(engine, patch):
    """Every detected text region's exact quadrilateral, drawn in red,
    labeled with the recognized text + confidence."""
    annotated = patch.copy()
    result = engine.predict(patch)
    if not result:
        return annotated

    page = result[0]
    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])
    polys = page.get("rec_polys", [])

    for text, score, poly in zip(texts, scores, polys):
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=RED, thickness=2)
        label_x, label_y = int(poly[0][0]), int(max(12, poly[0][1] - 8))
        cv2.putText(
            annotated, f"{text} ({score:.2f})", (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA,
        )

    return annotated


def main(image_path: str, position: str, model_path: str) -> None:
    detection = gauge_detection_pipeline(image_path)
    if detection["status"] != "ok":
        print(f"Stage 1에서 실패했어요: {detection.get('reason')}")
        return

    keypoint_model = KeypointModel(model_path, device="cpu")
    kp_result = keypoint_model.extract_keypoints(detection["cropped_image"])
    if kp_result["status"] != "ok":
        print(f"Stage 2에서 실패했어요: {kp_result.get('reason')}")
        return

    original_points = map_points_to_original(kp_result["points"], detection["offset"])

    x, y, w, h = detection["candidate"]["bbox"]
    radius = (w + h) / 4
    crop_ratio = compute_dynamic_crop_ratio(original_points["min"], original_points["max"], radius)

    target_point = original_points[position]
    patch_crop, offset = crop_around_point(detection["original_image"], target_point, radius, crop_ratio)
    local_point = (target_point[0] - offset[0], target_point[1] - offset[1])

    patch_raw, _ = upscale_only(patch_crop)
    patch_binary, _ = upscale_and_binarize(patch_crop)

    ocr_runner = OcrRunner(device="cpu")

    # Identical stage_specs to candidate_scoring/search_strategy's real
    # progressive search -- see pipeline_stage3.py for the production
    # version of this exact loop.
    stage_specs = [
        (patch_raw, ANGLES_PRIMARY, "raw", 1),
        (patch_raw, ANGLES_SECONDARY, "raw", 2),
        (patch_binary, ANGLES_PRIMARY, "binary", 3),
        (patch_binary, ANGLES_SECONDARY, "binary", 4),
    ]

    base, ext = os.path.splitext(image_path)
    all_seen = []
    tile_paths = []

    for patch, angles, source_name, stage_num in stage_specs:
        for angle in angles:
            rotated_patch, _ = rotate_patch_and_point(patch, local_point, angle)
            annotated = draw_ocr_boxes(ocr_runner.engine, rotated_patch)

            tile_path = f"{base}_{position}_s{stage_num}_{source_name}_{angle}deg{ext}"
            cv2.imwrite(tile_path, annotated)
            tile_paths.append(tile_path)

        all_seen.extend(
            collect_candidates_for_angles(ocr_runner, patch, local_point, angles, source_name, stage_num)
        )
        distinct_seen = len({c["parsed_value"] for c in all_seen})
        print(f"[stage {stage_num}, {source_name}] distinct values so far: {distinct_seen}")

        if distinct_seen >= 3:
            print(f" -> stage {stage_num}에서 조기 종료 (3개 이상 모임)\n")
            break

    top_groups = get_top_n_groups(all_seen, n=3)

    print(f"=== {position.upper()} top-3 candidates ===")
    for i, g in enumerate(top_groups, 1):
        print(
            f"  {i}. \"{g['text']}\" -> {g['parsed_value']}  "
            f"(composite_score={g['composite_score']:.3f}, seen {g['occurrences']}x)"
        )

    print(f"\n생성된 회전별 이미지 {len(tile_paths)}개:")
    for p in tile_paths:
        print(" -", p)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python3 scripts/visualize_ocr.py path/to/photo.jpg <min|max> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    position = sys.argv[2]
    if position not in ("min", "max"):
        print("두 번째 인자는 min 또는 max여야 해요.")
        sys.exit(1)

    model_path = (
        sys.argv[3] if len(sys.argv) > 3
        else os.environ.get("KEYPOINT_MODEL_PATH", "models/meter_keypoint_v3/best.pt")
    )
    main(image_path, position, model_path)