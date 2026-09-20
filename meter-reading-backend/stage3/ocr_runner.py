"""
ocr_runner.py

Stage 3 OCR execution: wraps a single loaded PaddleOCR engine, runs it on
a patch, and filters/parses the results into a consistent shape for
candidate_scoring.py and search_strategy.py to consume.

Ported from test_paddle9.ipynb. Same reasoning as key_points.py's
KeypointModel: the notebooks used a bare module-level `ocr` variable,
loaded once per notebook session -- fine there, but in a backend process
that pattern invites accidentally reloading the engine (slow) or sharing
mutable global state across requests in a way that's hard to reason
about. Wrapped in a class so the API server can load it once at startup.
"""

import logging
import re
from typing import TypedDict

import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

NUMBER_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")

# Text near these words is almost always part of a spec/rating label
# (e.g. "CL 2.5" = accuracy class), not the gauge's actual scale value.
# Checked by contains_spec_keyword() below, and used together with a
# box's own text height (in candidate_scoring.py's proximity filter) to
# decide whether a nearby number is a spec label rather than a real
# min/max reading.
SPEC_LABEL_KEYWORDS = {"cl", "psi", "bar", "kpa", "mpa", "kg", "class", "cm2"}


class OcrDetection(TypedDict):
    text: str
    score: float
    is_number: bool
    center: tuple[float, float]
    box_height: float


def is_number_like(text: str) -> bool:
    """True if `text` is a plain number (int or decimal, optionally negative)."""
    cleaned = text.strip().replace(",", "")
    return bool(NUMBER_PATTERN.match(cleaned))


def parse_number(text: str) -> float | None:
    """Parse `text` as a float, or None if it isn't a valid number."""
    try:
        return float(text)
    except ValueError:
        return None


def contains_spec_keyword(text: str) -> bool:
    """True if `text` contains one of SPEC_LABEL_KEYWORDS (case-insensitive)."""
    text_lower = text.strip().lower()
    return any(kw in text_lower for kw in SPEC_LABEL_KEYWORDS)


class OcrRunner:
    """Thin wrapper around a single loaded PaddleOCR engine."""

    def __init__(self, lang: str = "en", device: str = "cpu"):
        """
        device defaults to "cpu" to match the production deployment
        assumption. Note: the exploration notebooks used
        `use_angle_cls=True`, which PaddleOCR's own runtime logs flagged
        as deprecated ("please use use_textline_orientation instead") --
        switched here since this is going into code meant to actually run
        in production, not a one-off notebook.
        """
        self.engine = PaddleOCR(use_textline_orientation=True, lang=lang, device=device)
        logger.info("ocr_runner: PaddleOCR loaded (lang=%s, device=%s)", lang, device)

    def run_with_boxes(self, patch: np.ndarray) -> list[OcrDetection]:
        """
        Run OCR on a single patch and return every detected text box as a
        structured dict: text, confidence score, whether it parses as a
        plain number, the box's center (patch-local pixel coordinates),
        and the box's height (used elsewhere to scale the spec-label
        proximity filter to this particular text's own size, rather than
        a fixed pixel distance).

        Returns [] for an empty/None patch rather than raising, so callers
        (e.g. a loop trying many rotated patches) don't need to
        special-case empty crops.
        """
        if patch is None or patch.size == 0:
            return []

        result = self.engine.predict(patch)
        if not result:
            return []

        page = result[0]
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        polys = page.get("rec_polys", [])

        found: list[OcrDetection] = []
        for text, score, poly in zip(texts, scores, polys):
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            found.append({
                "text": text,
                "score": float(score),
                "is_number": is_number_like(text),
                "center": (float(np.mean(xs)), float(np.mean(ys))),
                "box_height": float(max(ys) - min(ys)),
            })

        return found