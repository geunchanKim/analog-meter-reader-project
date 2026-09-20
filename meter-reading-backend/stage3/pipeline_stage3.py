"""
pipeline_stage3.py

Stage 3 assembly: given a full (uncropped) original image, Stage 1's
bbox, and Stage 2's global keypoints, run the min/max OCR search on both
positions and return top-candidate lists for a human (or an API client)
to pick from.

Ported from test_paddle9.ipynb's process_entry() / build_min_candidates_with_forced_zero().
One real adaptation from the notebook version: the notebook's process_entry
took an `entry` dict tied to precomputed_keypoints.json (a batch-testing
artifact with a file path on disk) and called cv2.imread(entry["path"])
itself. A backend request has the image in memory already (Stage 1 already
decoded it), so this version takes the image array + bbox + keypoints
directly as arguments instead of a path to re-read from disk.

NO AUTO-SWAP: Position A is wherever the keypoint model labeled "min",
Position B is wherever it labeled "max" -- both are OCR'd independently
and returned as-is. Which one is actually used as min/max for the final
value calculation is a decision left to the caller (human verifier or a
future automated policy), not made here. See geometry_unwarping.md for
why automatic swapping was tried (v6-v7) and abandoned (v9).
"""

import os
from typing import TypedDict

import cv2
import numpy as np

from stage3.candidate_scoring import Candidate, CandidateGroup
from stage3.ocr_runner import OcrRunner
from stage3.preprocessing import (
    Point,
    compute_dynamic_crop_ratio,
    crop_around_point,
    upscale_and_binarize,
    upscale_only,
)
from stage3.search_strategy import DEFAULT_TOP_N, get_top_candidates


class Stage3Result(TypedDict):
    bbox: tuple[int, int, int, int]
    keypoints_global: dict[str, Point]
    crop_ratio_used: float

    center_point: Point
    tip_point: Point

    pos_a_point: Point                       # wherever the keypoint model labeled "min"
    pos_a_all_candidates: list[Candidate]
    pos_a_stage: int
    pos_a_candidates: list[CandidateGroup]   # top-N, with "0" forced in as priority #1

    pos_b_point: Point                       # wherever the keypoint model labeled "max"
    pos_b_all_candidates: list[Candidate]
    pos_b_stage: int
    pos_b_candidates: list[CandidateGroup]   # top-N, no forcing


def build_min_candidates_with_forced_zero(
    top_groups: list[CandidateGroup], top_n: int = DEFAULT_TOP_N
) -> list[CandidateGroup]:
    """
    Ensure "0" is always priority #1 in the min-position candidate list,
    since gauges overwhelmingly start their scale at zero and OCR
    occasionally fails to read the (often small/faint) "0" label even
    when it's clearly there.

    - If "0" was actually found among the OCR'd groups, it's promoted to
      the front and its display text is normalized to exactly "0" (an
      OCR'd zero-valued group might otherwise show as "00", "0.0", etc.
      even though the parsed value is the same).
    - If "0" was NOT found at all, a synthetic placeholder group is
      inserted instead (marked "forced": True so a caller/UI can visually
      distinguish "the model actually saw this" from "we assumed this").
    """
    zero_group = None
    others = []

    for g in top_groups:
        if g["parsed_value"] == 0:
            zero_group = g
        else:
            others.append(g)

    if zero_group is None:
        zero_group = {
            "parsed_value": 0.0, "text": "0", "representative_score": None,
            "occurrences": 0, "distinct_sources": set(), "is_decimal": False,
            "example": None, "all_occurrences": [], "composite_score": None,
            "forced": True,
        }
    else:
        zero_group["text"] = "0"
        zero_group["forced"] = False

    return ([zero_group] + others)[:top_n]


def process_gauge(
    ocr_runner: OcrRunner,
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    keypoints_global: dict[str, Point],
    base_crop_ratio: float = 0.6,
    min_crop_ratio: float = 0.25,
    top_n: int = DEFAULT_TOP_N,
) -> Stage3Result:
    """
    Run Stage 3 for one gauge image.

    `image` is the FULL original image (not the Stage 1 crop) and
    `keypoints_global` are keypoints already mapped into that same
    original-image coordinate space (key_points.map_points_to_original) --
    this matters because `bbox` (from Stage 1) is used here only to
    estimate the gauge's overall radius, and that radius needs to be in
    the same coordinate space as the keypoints for the crop math to line up.
    """
    x, y, w, h = bbox
    radius = (w + h) / 4

    pos_a_point = keypoints_global["min"]
    pos_b_point = keypoints_global["max"]

    crop_ratio = compute_dynamic_crop_ratio(
        pos_a_point, pos_b_point, radius, base_ratio=base_crop_ratio, min_ratio=min_crop_ratio
    )

    # Position A -- wherever the keypoint model labeled "min"
    a_patch_crop, (aox, aoy) = crop_around_point(image, pos_a_point, radius, crop_ratio=crop_ratio)
    a_patch_raw, a_scale = upscale_only(a_patch_crop)
    a_patch_binary, _ = upscale_and_binarize(a_patch_crop)
    a_local = ((pos_a_point[0] - aox) * a_scale, (pos_a_point[1] - aoy) * a_scale)
    a_top, a_all, a_stage = get_top_candidates(ocr_runner, a_patch_raw, a_patch_binary, a_local, n=top_n)
    a_candidates_final = build_min_candidates_with_forced_zero(a_top, top_n=top_n)

    # Position B -- wherever the keypoint model labeled "max"
    b_patch_crop, (bx, by) = crop_around_point(image, pos_b_point, radius, crop_ratio=crop_ratio)
    b_patch_raw, b_scale = upscale_only(b_patch_crop)
    b_patch_binary, _ = upscale_and_binarize(b_patch_crop)
    b_local = ((pos_b_point[0] - bx) * b_scale, (pos_b_point[1] - by) * b_scale)
    b_top, b_all, b_stage = get_top_candidates(ocr_runner, b_patch_raw, b_patch_binary, b_local, n=top_n)

    return {
        "bbox": bbox,
        "keypoints_global": keypoints_global,
        "crop_ratio_used": crop_ratio,

        "center_point": keypoints_global["center"],
        "tip_point": keypoints_global["tip"],

        "pos_a_point": pos_a_point,
        "pos_a_all_candidates": a_all,
        "pos_a_stage": a_stage,
        "pos_a_candidates": a_candidates_final,

        "pos_b_point": pos_b_point,
        "pos_b_all_candidates": b_all,
        "pos_b_stage": b_stage,
        "pos_b_candidates": b_top,
    }


def process_gauge_from_path(
    ocr_runner: OcrRunner,
    image_path: str,
    bbox: tuple[int, int, int, int],
    keypoints_global: dict[str, Point],
    **kwargs,
) -> Stage3Result:
    """
    Convenience wrapper matching the notebooks' batch-testing usage
    pattern (image on disk, e.g. re-running against precomputed_keypoints.json
    for evaluation) -- reads the image once, then delegates to process_gauge.
    Not the expected path for a live API request (see process_gauge's
    docstring); kept for offline evaluation scripts.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"could not read image at {image_path}")
    return process_gauge(ocr_runner, image, bbox, keypoints_global, **kwargs)