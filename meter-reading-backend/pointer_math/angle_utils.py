"""
angle_utils.py

Angle math shared across the pipeline: given the four keypoints
(center, tip, min, max) produced by Stage 2, compute the final gauge
reading.

This logic previously existed ONLY in JavaScript, inside the interactive
HTML verification report (test_paddle9.ipynb / min_max_ocr_no_autoswap.ipynb).
This module is the Python port needed so the backend can compute the same
value the report shows, without a browser in the loop. Keeping the two
implementations in sync matters -- if this module and the report's JS ever
disagree, the API's answer and what a human verifier saw would silently
diverge.

Angle convention used throughout this module: 0 degrees points along the
positive x-axis (3 o'clock), and angles increase CLOCKWISE, because image
coordinates have y increasing downward. This matches the convention used
in the exploration notebooks (test_unwarp.ipynb, test_original.ipynb).

All functions here are pure (no I/O, no model calls) so they're cheap to
unit test directly against known keypoint coordinates and expected values --
worth doing before wiring this into the API, since a wrong angle
convention would silently produce plausible-looking but incorrect values
rather than an obvious crash.
"""

import math

Point = tuple[float, float]


def compute_angle_deg(center: Point, point: Point) -> float:
    """
    Angle from `center` to `point`, in degrees, in the range [0, 360).

    Uses atan2(dy, dx) rather than atan2(dx, dy) or similar variants --
    several of the exploration notebooks (test_patch.ipynb in particular)
    used a different convention (0 degrees = 12 o'clock). This module
    picks ONE convention and every function below is written consistently
    against it; don't mix in a helper copied from a different notebook
    without re-deriving it against this convention first.
    """
    cx, cy = center
    px, py = point
    return math.degrees(math.atan2(py - cy, px - cx)) % 360


def resolve_sweep_direction(min_angle: float, max_angle: float) -> tuple[float, float]:
    """
    Given the raw angles of the min and max keypoints, decide which of the
    two possible sweep directions (clockwise or counterclockwise from min
    to max) is the gauge's actual scale, and return:

        (max_angle_used, arc_span)

    where max_angle_used is max_angle adjusted so that
    (max_angle_used - min_angle) == arc_span exactly (no modulo needed by
    the caller), and arc_span is always the MAJOR arc (>= 180 degrees).

    Rationale: analog gauges physically sweep the LARGER of the two arcs
    between their min and max labels (typically 200-300 degrees) -- the
    smaller arc is the "dead zone" behind the scale where no markings
    exist. Always picking the major arc is the same rule used earlier in
    test_unwarp.ipynb's unwarp_gauge_clean() and test_patch.ipynb's
    resolve_sweep_direction(); this is the canonical version those should
    have been importing from, had this module existed at the time.
    """
    clockwise_delta = (max_angle - min_angle) % 360

    if clockwise_delta >= 180:
        return min_angle + clockwise_delta, clockwise_delta

    # The clockwise direction was the minor arc -- the major arc goes the
    # other way, past 0/360, so max_angle_used ends up LESS than min_angle.
    arc_span = 360 - clockwise_delta
    return min_angle - arc_span, arc_span


def _best_fraction(min_angle: float, max_angle_used: float, tip_angle: float) -> tuple[float, float]:
    """
    Given one resolved arc (min_angle -> max_angle_used, in either
    direction), finds the tip's angle's best-fit fraction along it and
    how far outside [0, 1] that fraction falls (0.0 if it fits cleanly).

    Tries all three ways the tip's angle could wrap relative to this arc
    (tip_angle - 360, tip_angle, tip_angle + 360) -- see compute_value's
    docstring on why wrapping needs to be tried in all three directions.
    """
    directed_span = max_angle_used - min_angle
    best_fraction = None
    best_distance = float("inf")

    for candidate_angle in (tip_angle - 360, tip_angle, tip_angle + 360):
        fraction = (candidate_angle - min_angle) / directed_span
        distance = 0.0 if 0.0 <= fraction <= 1.0 else min(abs(fraction), abs(fraction - 1.0))
        if distance < best_distance:
            best_distance = distance
            best_fraction = fraction

    return best_fraction, best_distance


def resolve_reading_fraction(
    center: Point, tip: Point, min_point: Point, max_point: Point
) -> tuple[float, float, bool]:
    """
    Decides whether the gauge's true scale sweep is the MAJOR arc between
    min and max (>=180 degrees -- the right assumption for most round
    pressure/temperature gauges, which sweep 200-300 degrees) or the MINOR
    arc (<180 degrees -- true for some panel meters, like the ammeter that
    exposed this: its scale genuinely only sweeps ~94 degrees).

    Earlier versions of this module always assumed the major arc. That
    broke on the ammeter case: its needle sat well inside the true (minor)
    arc, but the code forced the major-arc interpretation, treated the
    needle's real position as the "empty gap" behind the dial, and
    gap-snapped it to 0 -- a confidently wrong answer despite every
    keypoint being detected correctly.

    The fix: compute the tip's best-fit fraction under BOTH
    interpretations (major and minor), and use whichever one the tip
    actually fits into without needing to clamp. If neither (or both) fit
    cleanly -- genuinely ambiguous cases -- default to the major arc,
    since that remains correct for the large majority of round gauges;
    ambiguity shouldn't flip the common case to the rarer minor-arc reading.

    Returns (clamped_fraction, arc_span_used, used_minor_arc) -- arc_span_used
    is a magnitude (always positive), and used_minor_arc lets a caller
    (e.g. qc_gate.py) know which interpretation was actually chosen.
    """
    min_angle = compute_angle_deg(center, min_point)
    max_angle = compute_angle_deg(center, max_point)
    tip_angle = compute_angle_deg(center, tip)

    clockwise_delta = (max_angle - min_angle) % 360

    if clockwise_delta >= 180:
        major_max_used, major_span = min_angle + clockwise_delta, clockwise_delta
        minor_max_used, minor_span = min_angle - (360 - clockwise_delta), 360 - clockwise_delta
    else:
        major_max_used, major_span = min_angle - (360 - clockwise_delta), 360 - clockwise_delta
        minor_max_used, minor_span = min_angle + clockwise_delta, clockwise_delta

    major_fraction, major_distance = _best_fraction(min_angle, major_max_used, tip_angle)
    minor_fraction, minor_distance = _best_fraction(min_angle, minor_max_used, tip_angle)

    if minor_distance < major_distance:
        chosen_fraction, chosen_span, used_minor = minor_fraction, minor_span, True
    else:
        chosen_fraction, chosen_span, used_minor = major_fraction, major_span, False

    clamped_fraction = min(1.0, max(0.0, chosen_fraction))
    return clamped_fraction, chosen_span, used_minor


def compute_value(
    center: Point,
    tip: Point,
    min_point: Point,
    max_point: Point,
    min_val: float,
    max_val: float,
) -> float:
    """
    Compute the gauge's reading from its four keypoints and the two scale
    endpoint values (min_val, max_val -- typically OCR'd from the min/max
    labels in Stage 3, or supplied by a human verifier).

    The actual arc-choice and gap-snapping logic lives in
    resolve_reading_fraction() -- this function just turns the resulting
    fraction into a value via linear interpolation:
    value = min_val + fraction * (max_val - min_val).

    THE GAP-SNAPPING FIX: because angles wrap around at 360 degrees, and
    because a gauge's true sweep can be either the major or minor arc
    (see resolve_reading_fraction), the tip's angle needs to be checked
    against multiple wrapped representations and both arc candidates.
    When the needle sits very close to a scale boundary, keypoint noise
    can occasionally push its angle just outside the true arc -- clamping
    to whichever boundary is closest (rather than always defaulting to
    max) fixed ~40% of "incorrect" judgments in the 200-image verification
    run.
    """
    fraction, _, _ = resolve_reading_fraction(center, tip, min_point, max_point)
    return min_val + fraction * (max_val - min_val)