"""
scripts/visualize_detection.py

Draws Stage 1's actual detected candidates (circular gauges as circles,
rectangular gauges as rectangles) in red on top of the original photo,
and separately saves the final cropped result -- for presentation slides.

Reuses the real gauge_detection.py functions directly (find_main_gauge,
crop_gauge) rather than reimplementing detection logic, so the slide
images show exactly what the actual pipeline does, not an approximation.

Usage (from the backend repo root):
    python3 scripts/visualize_detection.py path/to/photo.jpg

Produces, next to the input file:
    photo_detected.jpg   -- original photo with red outlines on every candidate
    photo_cropped.jpg    -- the final cropped gauge (only if status == "ok")
"""

import os
import sys

# Lets this script be run directly (python3 scripts/visualize_detection.py ...)
# regardless of the current working directory, by putting the backend
# repo root (this script's parent's parent) on sys.path -- without this,
# `from stage1_2...` only works if you happen to be `cd`'d into the repo
# root already.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from stage1_2.gauge_detection import crop_gauge, find_main_gauge

RED = (0, 0, 255)  # OpenCV uses BGR, not RGB
LINE_THICKNESS = 3


def draw_candidates(image, candidates):
    """
    Draws each candidate's bbox in red -- as a circle outline for
    'circular' candidates (reconstructing center/radius from the bbox,
    since that's all gauge_detection.py stores), as a rectangle outline
    for 'rectangular' ones.
    """
    annotated = image.copy()
    for cand in candidates:
        x, y, w, h = cand["bbox"]
        if cand["shape"] == "circular":
            cx, cy, r = x + w // 2, y + h // 2, w // 2
            cv2.circle(annotated, (cx, cy), r, RED, LINE_THICKNESS)
        else:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), RED, LINE_THICKNESS)
    return annotated


def main(image_path: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 못 읽었어요: {image_path}")
        return

    result = find_main_gauge(image)

    if result["status"] == "fail":
        print("게이지를 하나도 못 찾았어요:", result.get("reason"))
        return

    # 'all_candidates' only exists on status=='ok'; status=='retake' uses
    # 'candidates' instead (see find_main_gauge's docstring) -- either way,
    # this shows every candidate actually found, not just the chosen one.
    candidates = result.get("all_candidates") or result.get("candidates") or []
    annotated = draw_candidates(image, candidates)

    base, ext = os.path.splitext(image_path)
    annotated_path = f"{base}_detected{ext}"
    cv2.imwrite(annotated_path, annotated)
    print(f"검출 결과(빨간 윤곽선) 저장: {annotated_path}  ({len(candidates)}개 후보)")

    if result["status"] == "ok":
        cropped, offset = crop_gauge(image, result["candidate"]["bbox"])
        cropped_path = f"{base}_cropped{ext}"
        cv2.imwrite(cropped_path, cropped)
        print(f"크롭 결과 저장: {cropped_path}")
    else:
        print("여러 후보(retake 상황)라 크롭은 생략했어요 -- 후보들은 다 표시됐어요.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python3 scripts/visualize_detection.py path/to/photo.jpg")
        sys.exit(1)
    main(sys.argv[1])