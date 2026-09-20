"""
scripts/inspect_candidate_scores.py

Runs the REAL Stage 3 OCR collection (all 4 stages, no early stop --
unlike production, since this is for showing a rich example table, not
for speed) for one keypoint position, then prints EVERY distinct-value
candidate group with its full score breakdown -- representative_score,
value_bonus, consistency_bonus, decimal_bonus, composite_score -- using
the real compute_composite_scores_v2 function, unmodified.

Usage (from the backend repo root):
    python3 scripts/inspect_candidate_scores.py path/to/photo.jpg <min|max> [model_path]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1_2.gauge_detection import gauge_detection_pipeline
from stage1_2.key_points import KeypointModel, map_points_to_original
from stage3.candidate_scoring import collect_candidates_for_angles, compute_composite_scores_v2
from stage3.ocr_runner import OcrRunner
from stage3.preprocessing import (
    ANGLES_PRIMARY,
    ANGLES_SECONDARY,
    compute_dynamic_crop_ratio,
    crop_around_point,
    upscale_and_binarize,
    upscale_only,
)


def print_table(groups: list[dict]) -> None:
    groups_sorted = sorted(groups, key=lambda g: -g["composite_score"])

    header = (
        f"{'#':<3} {'text':<8} {'value':<8} {'rep_score':<10} {'value_b':<9} "
        f"{'consist_b':<10} {'decimal_b':<10} {'composite':<10} {'seen':<5} {'sources'}"
    )
    print(header)
    print("-" * len(header))

    for i, g in enumerate(groups_sorted, 1):
        sources = "+".join(sorted(g["distinct_sources"]))
        print(
            f"{i:<3} {g['text']:<8} {g['parsed_value']:<8} "
            f"{g['representative_score']:<10.3f} {g['value_bonus']:<9.3f} "
            f"{g['consistency_bonus']:<10.3f} {g['decimal_bonus']:<10.3f} "
            f"{g['composite_score']:<10.3f} {g['occurrences']:<5} {sources}"
        )


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

    # No early stop here (unlike production/search_strategy.py) -- we
    # want every candidate found across all 4 stages, for a fuller
    # example table, not the fastest path to 3 distinct values.
    stage_specs = [
        (patch_raw, ANGLES_PRIMARY, "raw", 1),
        (patch_raw, ANGLES_SECONDARY, "raw", 2),
        (patch_binary, ANGLES_PRIMARY, "binary", 3),
        (patch_binary, ANGLES_SECONDARY, "binary", 4),
    ]

    all_candidates = []
    for patch, angles, source_name, stage_num in stage_specs:
        all_candidates.extend(
            collect_candidates_for_angles(ocr_runner, patch, local_point, angles, source_name, stage_num)
        )

    if not all_candidates:
        print(f"'{position}' 위치에서 숫자 후보를 하나도 못 찾았어요.")
        return

    groups = compute_composite_scores_v2(all_candidates)

    print(f"=== {position.upper()} — every distinct-value candidate found ===\n")
    print_table(groups)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python3 scripts/inspect_candidate_scores.py path/to/photo.jpg <min|max> [model_path]")
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