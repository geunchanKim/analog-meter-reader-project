"""
qc_gate.py

Combined Stage 1 + Stage 2 quality gate. Runs AFTER gauge detection
(gauge_detection.py) and keypoint extraction (key_points.py), BEFORE
Stage 3 OCR -- Stage 3 is the expensive step (multiple PaddleOCR passes
per image, measured at ~14-30s/image on CPU during development), so
rejecting an unusable image here avoids paying that cost on an image that
was never going to produce a trustworthy reading.

This module makes no detection decisions of its own. It only reads the
outputs of the two earlier stages -- points/confidences (key_points.py)
-- and applies policy thresholds on top of them. That separation matters
for maintenance: if a threshold needs tuning after looking at more
failure cases (as happened repeatedly with Stage 3's scoring weights
across v1-v9), there is exactly one place to look, instead of thresholds
being scattered across the two detection modules.

NOTE: the bounds check (was here, checked whether Stage 1's bbox touched
the frame edge) was removed -- not needed for the current demo scope
(no cut-off test photos are being used), and it had a real accuracy
problem besides: gauge_detection.py's Hough-circle fitting frequently
clamps a coordinate to exactly 0 via max(0, x-r) even on photos that
aren't actually cropped, which made the check fire on normal photos, not
just genuinely cut-off ones. Can be reintroduced later with a more
robust signal than raw bbox-vs-frame-edge distance.

Depends on pointer_math.angle_utils for the arc-span check -- that module
must exist and be importable for run_qc_gate() to work.
"""

import logging
from typing import TypedDict

from pointer_math.angle_utils import resolve_reading_fraction

logger = logging.getLogger(__name__)

# --- Tunable thresholds --------------------------------------------------
# Kept as module-level constants, not buried inside function bodies, so
# they're easy to find and adjust in one place after reviewing more
# real-world failures -- the same way Stage 3's composite-scoring weights
# kept changing release over release.
MIN_KEYPOINT_CONFIDENCE = 0.5   # per-keypoint confidence floor
MIN_ARC_SPAN_DEG = 60           # narrowest plausible min-to-max scale sweep
MAX_ARC_SPAN_DEG = 330          # widest plausible min-to-max scale sweep


class QcResult(TypedDict, total=False):
    status: str            # "ok" or "reject"
    reasons: list[str]     # empty when status == "ok"; one entry per failed check


def check_keypoint_confidence(
    confidences: dict[str, float],
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
) -> bool:
    """
    True only if EVERY keypoint clears min_confidence.

    A single weak keypoint is enough to reject -- a confident center/tip
    paired with a weak min is exactly the kind of partial detection seen
    in the earlier wrong-gauge review case (#16: min/max candidates were
    '0'/'3' vs '1111'). The model was detecting something, just not the
    right thing, and low confidence on at least one point is usually the
    tell.
    """
    return all(c >= min_confidence for c in confidences.values())


def check_arc_span(points: dict[str, tuple[float, float]]) -> bool:
    """
    True if the ACTUAL angular sweep the needle appears to use (whichever
    of the major/minor arc between min and max the tip cleanly fits into
    -- see resolve_reading_fraction) falls within a physically plausible
    range for an analog gauge scale.

    This used to always check the major arc (>=180 degrees), based on the
    assumption that gauges sweep 200-300 degrees. That assumption broke on
    a real panel ammeter, whose scale genuinely only sweeps ~94 degrees
    (a minor arc) -- resolve_reading_fraction now picks whichever arc the
    tip actually fits into, so this check follows that same choice rather
    than blindly measuring the major arc every time. MIN_ARC_SPAN_DEG is
    lowered to 60 to accommodate compact-swing panel meters like that one,
    while MAX_ARC_SPAN_DEG still catches keypoints that clearly aren't on
    the same physical dial (reflections, a second gauge in frame, garbage
    detections).
    """
    _, arc_span, _ = resolve_reading_fraction(
        points["center"], points["tip"], points["min"], points["max"]
    )
    return MIN_ARC_SPAN_DEG <= arc_span <= MAX_ARC_SPAN_DEG


def run_qc_gate(
    points: dict[str, tuple[float, float]],
    confidences: dict[str, float],
) -> QcResult:
    """
    Run all checks and combine them into a single accept/reject decision.

    Every check runs, even after an earlier one has already failed --
    intentionally NOT short-circuiting on the first failure -- so a
    rejected image's log entry (and eventually its API response) can
    report every reason at once.
    """
    reasons: list[str] = []

    if not check_keypoint_confidence(confidences):
        reasons.append("low_keypoint_confidence")

    if not check_arc_span(points):
        reasons.append("implausible_arc_span")

    if reasons:
        logger.info("qc_gate: rejecting image, reasons=%s", reasons)
        return {"status": "reject", "reasons": reasons}

    logger.debug("qc_gate: image passed all checks")
    return {"status": "ok", "reasons": []}