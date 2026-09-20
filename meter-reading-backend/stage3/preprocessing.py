"""
preprocessing.py

Stage 3 preprocessing helpers: crop a small region around a keypoint
(min or max), prepare it for OCR (upscale, optionally binarize), and
rotate a patch to any angle for the multi-angle OCR search in
search_strategy.py.

Ported from test_paddle9.ipynb (the current end state of the Stage 3
exploration -- see geometry_unwarping.md for the full version history).
Logic is unchanged from that notebook; only added type hints, docstrings,
and a shared Patch/OffsetPoint naming convention so the other stage3
modules can be written against a consistent contract.
"""

from typing import Optional

import cv2
import numpy as np

Point = tuple[float, float]

# The two fixed rotation sets tried by search_strategy.py's progressive
# search (stages 1-2 use ANGLES_PRIMARY, stages 3-4 add ANGLES_SECONDARY).
# Kept here rather than in search_strategy.py since they're really a
# property of "how rotate_patch_and_point is meant to be swept", not of
# the search/stopping logic itself.
ANGLES_PRIMARY = [0, 90, 180, 270]
ANGLES_SECONDARY = [45, 135, 225, 315]


def compute_dynamic_crop_ratio(
    min_pt: Point,
    max_pt: Point,
    radius: float,
    base_ratio: float = 0.6,
    min_ratio: float = 0.25,
    safety_factor: float = 0.9,
) -> float:
    """
    Shrink crop_ratio when the min and max keypoints are physically close
    together, so the min-crop and max-crop regions never overlap. Without
    this, a narrow-scale gauge could end up with both crops covering the
    same physical area, causing the true value to appear in only one of
    them while the other picks up noise from the wrong region (the bug
    this was written to fix in v8 of the exploration notebooks).

    Returns base_ratio unchanged when the keypoints are far enough apart
    that overlap isn't a risk; only shrinks toward min_ratio as they get
    closer.
    """
    dist = np.hypot(max_pt[0] - min_pt[0], max_pt[1] - min_pt[1])
    if radius <= 0:
        return base_ratio

    max_half_size = (dist / 2) * safety_factor
    ratio_from_distance = max_half_size / radius
    return max(min_ratio, min(base_ratio, ratio_from_distance))


def crop_around_point(
    image: np.ndarray, point: Point, radius: float, crop_ratio: float = 0.6
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Crop a square region centered on `point`, sized relative to `radius`
    (the gauge's overall radius, so the crop scales with how large the
    gauge is in the photo, not a fixed pixel size).

    Returns (patch, (x0, y0)) -- the offset is needed to map any point
    found within the patch (e.g. an OCR box's center) back to this crop's
    parent image coordinate space.
    """
    px, py = point
    half_size = radius * crop_ratio

    x0, y0 = int(px - half_size), int(py - half_size)
    x1, y1 = int(px + half_size), int(py + half_size)

    h, w = image.shape[:2]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)

    return image[y0:y1, x0:x1], (x0, y0)


def upscale_only(
    patch: np.ndarray, target_min_dim: int = 250, interpolation: int = cv2.INTER_CUBIC
) -> tuple[np.ndarray, float]:
    """
    Upscale a small crop so its shorter side reaches target_min_dim,
    without any binarization -- kept as a plain color/grayscale image for
    OCR to try directly. This is the "raw" source in the two-source
    (raw vs. binary) search in search_strategy.py.

    Returns (patch, 1.0) unchanged if the patch is empty, so callers don't
    need to special-case empty crops before calling this.
    """
    if patch.size == 0:
        return patch, 1.0

    h, w = patch.shape[:2]
    scale = max(target_min_dim / min(h, w), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    upscaled = cv2.resize(patch, (new_w, new_h), interpolation=interpolation)
    return upscaled, scale


def upscale_and_binarize(
    patch: np.ndarray, target_min_dim: int = 250, interpolation: int = cv2.INTER_CUBIC
) -> tuple[np.ndarray, float]:
    """
    Upscale, then denoise + Otsu-threshold to clean black-on-white, then a
    small morphological opening to remove leftover speckle. This is the
    "binary" source in the two-source search -- helps on noisy/grainy
    source images, though it can occasionally mangle bold digits or thin
    decimal points (which is exactly why the raw version above is tried
    in parallel rather than relying on binarization alone).
    """
    if patch.size == 0:
        return patch, 1.0

    h, w = patch.shape[:2]
    scale = max(target_min_dim / min(h, w), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    upscaled = cv2.resize(patch, (new_w, new_h), interpolation=interpolation)

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 5)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR), scale


def rotate_patch_and_point(
    patch: np.ndarray, local_point: Point, angle_deg: float
) -> tuple[np.ndarray, Point]:
    """
    Rotate a patch by an arbitrary angle (used with ANGLES_PRIMARY /
    ANGLES_SECONDARY for the multi-angle OCR search), and carry a
    reference point (e.g. the keypoint's own position within the patch)
    along into the rotated frame using the same rotation matrix -- so
    candidate positions found by OCR on the rotated patch can still be
    compared against where the original keypoint landed.

    White border fill (255, 255, 255) is used for corners exposed by the
    rotation, matching the light background most gauge photos have, so
    the fill doesn't introduce a dark artifact PaddleOCR could latch onto.
    """
    h, w = patch.shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(patch, rotation_matrix, (w, h), borderValue=(255, 255, 255))

    px, py = local_point
    new_point = rotation_matrix @ np.array([px, py, 1.0])
    return rotated, (float(new_point[0]), float(new_point[1]))