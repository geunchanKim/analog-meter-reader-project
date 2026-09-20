"""
gauge_detection.py

Stage 1 of the analog meter reading pipeline: locate a single gauge in a
raw photo using classical OpenCV techniques (no trained model), decide
whether the photo is usable, and crop the gauge region if so.

This module is detection-only. It does NOT decide whether the result
"looks wrong" beyond what find_main_gauge already checks (no gauge found /
multiple gauges found). Higher-level sanity checks (e.g. is the detected
circle cut off by the frame edge, is keypoint confidence too low) live in
qc_gate.py, which consumes this module's output rather than modifying it.

Two ways to run the pipeline, depending on where the image comes from:
    - gauge_detection_pipeline(path)        : image already exists on disk
    - gauge_detection_pipeline_from_bytes(b): image arrived as raw bytes
      (e.g. an HTTP file upload) -- avoids writing a temp file just to
      read it back with cv2.imread.

Usage:
    from gauge_detection import find_main_gauge, crop_gauge

    image = cv2.imread(path)
    result = find_main_gauge(image)

    if result['status'] == 'ok':
        cropped = crop_gauge(image, result['candidate']['bbox'])
    elif result['status'] == 'retake':
        # ask the user to retake the photo (ambiguous scene: multiple gauges)
        ...
    else:  # 'fail'
        # no gauge could be located at all
        ...
"""

import logging
from typing import Optional, TypedDict

import cv2
import numpy as np

# Module-level logger. In a notebook, "why did this get flagged as retake"
# is visible on screen; in a backend process nobody is watching stdout, so
# these decisions need to be logged instead of printed, otherwise they're
# unrecoverable once the request has finished.
logger = logging.getLogger(__name__)

# A bounding box in (x, y, width, height) pixel coordinates.
BBox = tuple[int, int, int, int]


class Candidate(TypedDict, total=False):
    bbox: BBox
    shape: str          # "circular" or "rectangular"
    area: int
    coverage_ratio: float  # only present on the 'main' candidate returned by find_main_gauge


def detect_circles(image: np.ndarray) -> list[Candidate]:
    """
    Detect circular gauge candidates using HoughCircles.

    Parameters are scaled relative to image size (not fixed pixel values),
    so this works for both small gauges in a large frame and gauges that
    fill most of the frame. Parameters are tuned to be strict (minDist,
    param2 set high) to minimize duplicate detections of the same physical
    circle, which HoughCircles is otherwise prone to.
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1,
        minDist=int(min_dim * 0.5),
        param1=50,
        param2=60,
        minRadius=int(min_dim * 0.15),
        maxRadius=int(min_dim * 0.6)
    )

    results: list[Candidate] = []
    if circles is not None:
        for x, y, r in circles[0, :]:
            x, y, r = int(x), int(y), int(r)
            bbox = (max(0, x - r), max(0, y - r), 2 * r, 2 * r)
            area = (2 * r) ** 2
            results.append({'bbox': bbox, 'shape': 'circular', 'area': area})

    return results


def detect_rectangles(image: np.ndarray, min_area_ratio: float = 0.05) -> list[Candidate]:
    """
    Detect rectangular gauge candidates using contour + polygon approximation.

    Uses RETR_EXTERNAL (outermost contours only) to avoid nested duplicate
    detections. Returns all plausible candidates above a minimum area
    threshold, not just the single largest one.
    """
    h, w = image.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results: list[Candidate] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * min_area_ratio:
            continue  # skip tiny noise contours

        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        if 4 <= len(approx) <= 6:  # roughly rectangular shape
            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / float(bh)
            if 0.3 < aspect_ratio < 3.0:  # exclude extremely elongated shapes
                results.append({'bbox': (x, y, bw, bh), 'shape': 'rectangular', 'area': bw * bh})

    return results


def compute_overlap_ratio(box1: BBox, box2: BBox) -> float:
    """
    Compute how much of the SMALLER box's area is covered by the
    intersection with the other box. Used to detect duplicate/overlapping
    detections of the same physical gauge, even when box sizes differ
    (e.g. the outer bezel vs. the inner dial face detected separately).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    smaller_area = min(w1 * h1, w2 * h2)
    return inter_area / smaller_area if smaller_area > 0 else 0


def merge_overlapping_candidates(candidates: list[Candidate], overlap_threshold: float = 0.1) -> list[Candidate]:
    """
    Merge candidates whose bounding boxes overlap significantly, keeping
    only the LARGEST one per overlapping group. overlap_threshold=0.1 is
    intentionally conservative (i.e. even slight overlap triggers a merge),
    since duplicate detections of one gauge (bezel vs. dial vs. inner ring)
    are far more common in this dataset than genuinely separate gauges
    that happen to be close together.
    """
    if len(candidates) <= 1:
        return candidates

    sorted_candidates = sorted(candidates, key=lambda c: c['area'], reverse=True)

    kept: list[Candidate] = []
    for cand in sorted_candidates:
        is_duplicate = any(
            compute_overlap_ratio(cand['bbox'], k['bbox']) > overlap_threshold
            for k in kept
        )
        if not is_duplicate:
            kept.append(cand)

    return kept


def resize_for_detection(image: np.ndarray, max_dimension: int = 1000) -> tuple[np.ndarray, float]:
    """
    Downscale large images before running detection. HoughCircles and
    contour detection scale poorly with resolution -- a raw smartphone
    photo (e.g. 3024x4032) can take dramatically longer to process than
    a modestly-sized image, with no meaningful gain in detection accuracy
    (gauge shapes remain clearly visible even at reduced resolution).

    Returns (resized_image, scale). scale is 1.0 if no resize was needed
    (image was already smaller than max_dimension), otherwise it's the
    factor the image was shrunk by -- divide any bbox coordinates
    computed on the resized image by this scale to map them back to the
    original image's coordinate space.
    """
    h, w = image.shape[:2]
    scale = max_dimension / max(h, w)

    if scale >= 1.0:
        return image, 1.0

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def find_main_gauge(
    image: np.ndarray,
    size_ratio_threshold: float = 0.6,
    overlap_threshold: float = 0.1,
    max_dimension: int = 1000,
) -> dict:
    """
    Main entry point: locate a single gauge in the image.

    The image is downscaled internally (if larger than max_dimension) for
    speed, and the resulting bbox coordinates are scaled back up to match
    the ORIGINAL image passed in -- callers never need to think about the
    internal resizing, bbox values returned are always in the original
    image's coordinate space.

    Decision logic:
      - 0 candidates found -> {'status': 'fail', 'reason': 'no_gauge_detected'}
      - After merging duplicates, if a second candidate remains close in
        size to the largest one -> likely two distinct gauges in frame ->
        {'status': 'retake', 'reason': 'multiple_candidates', 'candidates': [...]}
      - Otherwise -> {'status': 'ok', 'candidate': {...}}

    The returned 'candidate' dict has keys: 'bbox' (x, y, w, h), 'shape'
    ('circular' or 'rectangular'), 'area', and 'coverage_ratio' (the
    candidate's area as a fraction of the full ORIGINAL image's area).
    coverage_ratio is exposed specifically so qc_gate.py can flag frames
    where the gauge is too small/too large relative to the photo, without
    this module needing to know anything about that policy.
    """
    original_h, original_w = image.shape[:2]
    original_area = original_h * original_w

    detection_image, scale = resize_for_detection(image, max_dimension)

    all_candidates = detect_circles(detection_image) + detect_rectangles(detection_image)

    if len(all_candidates) == 0:
        logger.info("gauge_detection: no candidates found (status=fail)")
        return {'status': 'fail', 'reason': 'no_gauge_detected'}

    merged_candidates = merge_overlapping_candidates(all_candidates, overlap_threshold)
    sorted_candidates = sorted(merged_candidates, key=lambda c: c['area'], reverse=True)

    # Scale bboxes (and area) back up to the original image's coordinate space
    for cand in sorted_candidates:
        x, y, w, h = cand['bbox']
        cand['bbox'] = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
        cand['area'] = cand['bbox'][2] * cand['bbox'][3]

    main = sorted_candidates[0]
    main['coverage_ratio'] = main['area'] / original_area

    if len(sorted_candidates) > 1:
        second = sorted_candidates[1]
        if second['area'] / main['area'] > size_ratio_threshold:
            logger.info(
                "gauge_detection: multiple similarly-sized candidates found "
                "(status=retake, main_area=%d, second_area=%d)",
                main['area'], second['area'],
            )
            return {
                'status': 'retake',
                'reason': 'multiple_candidates',
                'candidates': sorted_candidates[:3],
            }

    logger.debug(
        "gauge_detection: found gauge (shape=%s, coverage_ratio=%.3f)",
        main['shape'], main['coverage_ratio'],
    )
    return {'status': 'ok', 'candidate': main, 'all_candidates': sorted_candidates}


def crop_gauge(image: np.ndarray, bbox: BBox, padding_ratio: float = 0.1) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Crop the gauge region with proportional padding, so the gauge is not
    cut off right at the edge. Returns (cropped_image, (x1, y1)) where
    (x1, y1) is the top-left offset of the crop in the original image's
    coordinate space -- useful for translating keypoint coordinates
    (e.g. ground-truth labels, or Stage 2 model predictions) into the
    cropped frame.
    """
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]

    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    return image[y1:y2, x1:x2], (x1, y1)


def gauge_detection_pipeline(image_path: str, padding_ratio: float = 0.1, max_dimension: int = 1000) -> dict:
    """
    Convenience wrapper for images that already exist on disk: load from
    image_path, run find_main_gauge, and crop it if the result is 'ok'.
    Returns the same dict as find_main_gauge, with 'cropped_image' and
    'offset' added when status is 'ok'. Cropping is always done on the
    full-resolution original image -- max_dimension only affects the
    internal detection step.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.warning("gauge_detection: failed to load image at %s", image_path)
        return {'status': 'fail', 'reason': 'image_load_error', 'path': image_path}

    return _run_pipeline_on_image(image, padding_ratio, max_dimension)


def gauge_detection_pipeline_from_bytes(
    image_bytes: bytes, padding_ratio: float = 0.1, max_dimension: int = 1000
) -> dict:
    """
    Same as gauge_detection_pipeline, but for images that arrive as raw
    bytes rather than a filesystem path -- the normal case for a backend
    API receiving an uploaded file. This decodes the bytes directly with
    cv2.imdecode, so the caller never has to write a temp file just to
    read it back with cv2.imread.

    A corrupted or non-image upload decodes to None, same failure mode as
    a bad path in gauge_detection_pipeline -- both return the same
    {'status': 'fail', 'reason': ...} shape, so callers can handle either
    entry point identically regardless of how the image arrived.
    """
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        logger.warning("gauge_detection: failed to decode uploaded image bytes")
        return {'status': 'fail', 'reason': 'image_decode_error'}

    return _run_pipeline_on_image(image, padding_ratio, max_dimension)


def _run_pipeline_on_image(image: np.ndarray, padding_ratio: float, max_dimension: int) -> dict:
    """
    Shared logic between the path- and bytes-based entry points above.

    Also stashes the full ORIGINAL (uncropped) image on the result as
    'original_image' when status is 'ok' -- Stage 3 (pipeline_stage3.py)
    needs the original image, not just the crop, since it re-crops small
    regions around the min/max keypoints using original-image-space
    coordinates. Without this, a caller (orchestrator.py) would have to
    decode the same bytes/path a second time just to get back to the
    original image.
    """
    result = find_main_gauge(image, max_dimension=max_dimension)

    if result['status'] == 'ok':
        cropped, offset = crop_gauge(image, result['candidate']['bbox'], padding_ratio)
        result['cropped_image'] = cropped
        result['offset'] = offset
        result['original_image'] = image

    return result