"""
test_torch_paddle_coexist.py

Checks whether ultralytics (PyTorch) and PaddleOCR (PaddlePaddle) can be
imported and run in the SAME process without crashing. This was reported
as a hard conflict during the exploration phase (see geometry_unwarping.md
/ the main handoff doc) -- this script re-verifies whether that's still
true on this machine/environment, since it decides whether the backend
needs a single-process architecture or a two-process (queue + separate
workers) one.

Run this INSIDE your venv:
    python test_torch_paddle_coexist.py

Read the printed output at each step -- if it crashes, note EXACTLY which
step number it died at and paste the full traceback back for the next step.
"""

import sys
import traceback


def step(n, description):
    print(f"\n{'='*60}")
    print(f"STEP {n}: {description}")
    print('='*60)


def main():
    step(1, "Import ultralytics (PyTorch) alone")
    try:
        from ultralytics import YOLO
        print("OK -- ultralytics imported successfully")
    except Exception:
        print("FAILED at step 1 -- ultralytics itself won't import. Fix this first, unrelated to paddle.")
        traceback.print_exc()
        sys.exit(1)

    step(2, "Import paddleocr (PaddlePaddle) alone, AFTER ultralytics")
    try:
        from paddleocr import PaddleOCR
        print("OK -- paddleocr imported successfully alongside ultralytics")
    except Exception:
        print("FAILED at step 2 -- import-time conflict between the two libraries.")
        traceback.print_exc()
        sys.exit(1)

    step(3, "Load the YOLO keypoint model (no inference yet)")
    try:
        # Replace with your actual model path
        MODEL_PATH = "../../data/models/meter_keypoint_v3/weights/best.pt"
        yolo_model = YOLO(MODEL_PATH)
        print("OK -- YOLO model loaded")
    except Exception:
        print("FAILED at step 3 -- YOLO model failed to load (check MODEL_PATH).")
        traceback.print_exc()
        sys.exit(1)

    step(4, "Load the PaddleOCR engine (no inference yet), AFTER YOLO is loaded")
    try:
        ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en", device="cpu")
        print("OK -- PaddleOCR engine loaded alongside a loaded YOLO model")
    except Exception:
        print("FAILED at step 4 -- PaddleOCR failed to initialize once YOLO was already loaded.")
        print("This is the conflict the notebooks hit. Note the traceback below.")
        traceback.print_exc()
        sys.exit(1)

    step(5, "Run YOLO inference on a real cropped gauge image")
    try:
        import cv2
        import numpy as np
        # Replace with a real cropped gauge image path from your dataset
        TEST_IMAGE = "../../data/raw/analog_cropped_v2/test/images/pressureGauge_421_jpg.rf.60b0188c5b82b72f503e747ff691c196.jpg"
        image = cv2.imread(TEST_IMAGE)
        if image is None:
            print(f"SKIPPED -- couldn't load test image at {TEST_IMAGE}, adjust the path and rerun")
        else:
            result = yolo_model.predict(image, verbose=False, device="cpu")
            print(f"OK -- YOLO inference ran, found keypoints: {result[0].keypoints is not None}")
    except Exception:
        print("FAILED at step 5 -- YOLO inference itself broke after PaddleOCR was loaded.")
        traceback.print_exc()
        sys.exit(1)

    step(6, "Run PaddleOCR inference on the same image, right after YOLO inference")
    try:
        ocr_result = ocr_engine.predict(image)
        print(f"OK -- PaddleOCR inference ran, got {len(ocr_result)} result(s)")
    except Exception:
        print("FAILED at step 6 -- PaddleOCR inference broke after running YOLO inference in the same process.")
        print("This is the most realistic reproduction of the actual pipeline's usage pattern.")
        traceback.print_exc()
        sys.exit(1)

    step(7, "Run BOTH again, interleaved, 3 times (stress the shared process)")
    try:
        for i in range(3):
            yolo_model.predict(image, verbose=False, device="cpu")
            ocr_engine.predict(image)
        print("OK -- interleaved repeated calls did not crash")
    except Exception:
        print(f"FAILED at step 7 (iteration may vary) -- conflict shows up only under repeated/interleaved use.")
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*60)
    print("ALL STEPS PASSED.")
    print("No crash was reproduced -- torch + paddle appear to coexist in one")
    print("process on THIS machine/environment. This doesn't guarantee the same")
    print("holds on a different OS/dependency version, but it's a strong signal")
    print("a single-process backend architecture is viable here.")
    print("="*60)


if __name__ == "__main__":
    main()