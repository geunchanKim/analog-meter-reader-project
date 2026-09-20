"""
candidate_scoring.py

Stage 3 candidate collection and scoring: sweep a patch across a set of
rotation angles, collect every numeric OCR detection (filtering out
spec/rating labels), group them by distinct value, and score each group
with the composite formula that reached its current form in v7-v9 of the
exploration notebooks.

Ported from test_paddle9.ipynb. collect_candidates_for_angles now takes an
OcrRunner instance explicitly (rather than calling a bare module-level
`run_ocr_with_boxes`), since ocr_runner.py's OCR engine is loaded once and
passed in rather than referenced as a global -- see ocr_runner.py's
docstring for why.
"""

from typing import TypedDict

import numpy as np

from stage3.ocr_runner import OcrRunner, contains_spec_keyword, parse_number
from stage3.preprocessing import Point, rotate_patch_and_point


class Candidate(TypedDict):
    text: str
    score: float
    is_number: bool
    center: tuple[float, float]
    box_height: float
    parsed_value: float
    rotation_deg: float
    source: str
    stage: int
    is_decimal: bool


class CandidateGroup(TypedDict):
    parsed_value: float
    text: str
    representative_score: float
    occurrences: int
    distinct_sources: set
    is_decimal: bool
    example: Candidate
    all_occurrences: list[Candidate]
    # present only after compute_composite_scores_v2 has run:
    value_bonus: float
    consistency_bonus: float
    decimal_bonus: float
    composite_score: float


def collect_candidates_for_angles(
    ocr_runner: OcrRunner,
    patch: np.ndarray,
    local_point: Point,
    angles: list[float],
    source_name: str,
    stage_num: int,
    label_proximity_factor: float = 1.5,
) -> list[Candidate]:
    """
    Try OCR at every angle in `angles`, keep only detections that parse as
    numbers, and drop any number sitting too close to a spec/unit label
    detected in the SAME rotation pass (e.g. a "2.5" glued to a "CL" label
    is a rating, not the gauge's scale value).

    The proximity threshold scales with the larger of the label's and the
    number's own box height (times label_proximity_factor), rather than a
    fixed pixel distance -- a fixed distance was tried first and ended up
    filtering out real scale numbers that happened to sit near a label at
    a different physical scale/resolution.

    `source_name` and `stage_num` are just tags carried along on each
    candidate so downstream scoring/logging can tell where a given
    reading came from (which preprocessing source, which stage of the
    progressive search) -- they don't affect the OCR itself.
    """
    candidates: list[Candidate] = []

    for angle in angles:
        rotated_patch, rotated_point = rotate_patch_and_point(patch, local_point, angle)
        ocr_results = ocr_runner.run_with_boxes(rotated_patch)

        labels_in_pass = [r for r in ocr_results if contains_spec_keyword(r["text"])]

        for r in ocr_results:
            if not r["is_number"]:
                continue
            parsed = parse_number(r["text"])
            if parsed is None:
                continue

            cx, cy = r["center"]
            near_label = False
            for lbl in labels_in_pass:
                lx, ly = lbl["center"]
                threshold = max(lbl["box_height"], r["box_height"]) * label_proximity_factor
                if np.hypot(cx - lx, cy - ly) < threshold:
                    near_label = True
                    break
            if near_label:
                continue

            candidates.append({
                **r,
                "parsed_value": parsed,
                "rotation_deg": angle,
                "source": source_name,
                "stage": stage_num,
                "is_decimal": "." in r["text"],
            })

    return candidates


def group_by_distinct_value(candidates: list[Candidate]) -> list[CandidateGroup]:
    """
    Collapse repeated OCR readings of the SAME numeric value (seen across
    different angles/sources) into one group per distinct value. Each
    group's representative text/score is taken from whichever occurrence
    had the highest confidence, and the group tracks how many times the
    value appeared and from how many distinct sources (raw vs binary) --
    both used as scoring signals below.
    """
    groups: dict[float, CandidateGroup] = {}

    for c in candidates:
        key = c["parsed_value"]
        if key not in groups:
            groups[key] = {
                "parsed_value": key,
                "text": c["text"],
                "representative_score": c["score"],
                "occurrences": 0,
                "distinct_sources": set(),
                "is_decimal": False,
                "example": c,
                "all_occurrences": [],
            }

        g = groups[key]
        g["occurrences"] += 1
        g["distinct_sources"].add(c["source"])
        g["is_decimal"] = g["is_decimal"] or c["is_decimal"]
        g["all_occurrences"].append(c)

        if c["score"] > g["representative_score"]:
            g["representative_score"] = c["score"]
            g["text"] = c["text"]
            g["example"] = c

    return list(groups.values())


def compute_composite_scores_v2(
    candidates: list[Candidate],
    value_weight: float = 0.02,
    same_source_repeat_weight: float = 0.01,
    cross_source_weight: float = 0.03,
    decimal_bonus_amount: float = 0.01,
    max_bonus_cap: float = 0.06,
) -> list[CandidateGroup]:
    """
    Score each distinct-value group as:

        composite = representative_score + value_bonus + consistency_bonus + decimal_bonus

    Confidence (representative_score) dominates the ranking. The other
    three terms are small, ADDITIVE, and individually/jointly capped
    (max_bonus_cap) -- deliberately too small to let a large-but-wrong
    misread outrank a correct, more confident smaller value (the "100"
    misread as "1100" failure that motivated switching away from a
    multiplicative value bonus in earlier versions).

    - value_bonus: mild rank-style nudge based on where this value falls
      in the observed range (relative_value in [0, 1]).
    - consistency_bonus: rewards a value reappearing across multiple
      angles/sources -- reappearing in a DIFFERENT source (raw vs binary)
      counts more (cross_source_weight) than repeating within the same
      source at a different rotation (same_source_repeat_weight), since
      cross-source agreement is stronger evidence it's a real printed
      number rather than an artifact of one particular rotation/threshold.
    - decimal_bonus: small flat nudge if the value was ever read with a
      decimal point (a decimal reading among the candidates is often a
      signal the gauge's true scale is decimal, e.g. "2.5" next to a
      misread "25").
    """
    groups = group_by_distinct_value(candidates)
    if not groups:
        return groups

    values = [g["parsed_value"] for g in groups]
    max_val, min_val = max(values), min(values)
    val_range = max_val - min_val if max_val != min_val else 1

    for g in groups:
        relative_value = (g["parsed_value"] - min_val) / val_range
        value_bonus = value_weight * relative_value

        same_source_repeats = max(0, g["occurrences"] - 1)
        cross_source_bonus = cross_source_weight if len(g["distinct_sources"]) > 1 else 0
        consistency_bonus = min(
            same_source_repeat_weight * same_source_repeats + cross_source_bonus,
            max_bonus_cap,
        )

        decimal_bonus = decimal_bonus_amount if g["is_decimal"] else 0

        g["value_bonus"] = value_bonus
        g["consistency_bonus"] = consistency_bonus
        g["decimal_bonus"] = decimal_bonus
        g["composite_score"] = g["representative_score"] + value_bonus + consistency_bonus + decimal_bonus

    return groups


def get_top_n_groups(candidates: list[Candidate], n: int = 3) -> list[CandidateGroup]:
    """Score all candidates and return the top `n` distinct-value groups by composite score."""
    groups = compute_composite_scores_v2(candidates)
    return sorted(groups, key=lambda g: -g["composite_score"])[:n]