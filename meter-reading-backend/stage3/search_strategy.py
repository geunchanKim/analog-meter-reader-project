"""
search_strategy.py

Stage 3's progressive 4-stage search: expand the pool of angles/sources
tried, stopping as soon as enough distinct candidate values have been
seen, instead of always running the full (and most expensive) sweep.

Stage order, cumulative:
  1. raw patch,    ANGLES_PRIMARY   (0/90/180/270)
  2. raw patch,    ANGLES_SECONDARY (45/135/225/315)  -- added on top of stage 1
  3. binary patch, ANGLES_PRIMARY                     -- added on top of stages 1-2
  4. binary patch, ANGLES_SECONDARY                    -- added on top of stages 1-3

After each stage, if the cumulative pool already contains `n` distinct
numeric values, the search stops there and scores what it has -- no need
to pay for the remaining (increasingly expensive) rotations if there are
already enough candidates for a human to choose from. If stage 4 finishes
with fewer than `n` distinct values, whatever was found (even 0, 1, or 2)
is returned as-is.

Ported from test_paddle9.ipynb's get_top_candidates(), refactored from
four near-identical copy-pasted stage blocks into one loop over a stage
list -- behavior is unchanged (same stages, same order, same early-stop
condition), just without four repeated blocks of the same code.
"""

import numpy as np

from stage3.candidate_scoring import (
    Candidate,
    CandidateGroup,
    collect_candidates_for_angles,
    get_top_n_groups,
)
from stage3.ocr_runner import OcrRunner
from stage3.preprocessing import ANGLES_PRIMARY, ANGLES_SECONDARY, Point

DEFAULT_TOP_N = 3


def get_top_candidates(
    ocr_runner: OcrRunner,
    patch_raw: np.ndarray,
    patch_binary: np.ndarray,
    local_point: Point,
    n: int = DEFAULT_TOP_N,
) -> tuple[list[CandidateGroup], list[Candidate], int]:
    """
    Run the progressive 4-stage search for a single keypoint (min or max).

    Returns (top_groups, all_candidates_seen, stage_reached):
      - top_groups: the top `n` distinct-value groups by composite score
        (what a human verifier picks from).
      - all_candidates_seen: every individual OCR detection collected
        across however many stages ran, before grouping -- kept around
        for debugging/logging, e.g. an interactive report showing the
        full per-rotation candidate log.
      - stage_reached: which stage the search stopped at (1-4). Useful for
        monitoring how often the search needs to go all the way to stage 4
        -- a rising stage-4 rate over time could signal the earlier stages
        need retuning, or that harder images are coming through.
    """
    stage_specs = [
        (patch_raw, ANGLES_PRIMARY, "raw", 1),
        (patch_raw, ANGLES_SECONDARY, "raw", 2),
        (patch_binary, ANGLES_PRIMARY, "binary", 3),
        (patch_binary, ANGLES_SECONDARY, "binary", 4),
    ]

    all_seen: list[Candidate] = []
    stage_reached = 1

    for patch, angles, source_name, stage_num in stage_specs:
        stage_reached = stage_num
        all_seen.extend(
            collect_candidates_for_angles(ocr_runner, patch, local_point, angles, source_name, stage_num)
        )

        distinct_values_seen = len({c["parsed_value"] for c in all_seen})
        if distinct_values_seen >= n:
            break

    return get_top_n_groups(all_seen, n), all_seen, stage_reached