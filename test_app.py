"""Verification and diagnostic script for AI Real-Time Object Detection Web App."""

from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from detector import ObjectDetector
from utils.metrics import compute_detection_summary, format_latency_and_fps


def test_detector_inference() -> bool:
    print("[1/3] Testing ObjectDetector and YOLO model inference...")
    try:
        detector = ObjectDetector(
            model_name=config.MODEL_NAME,
            conf_threshold=0.25,
        )
        print(f"  [OK] Model loaded on device: {detector.device}")
        print(f"  [OK] Recognized classes: {len(detector.class_names)}")

        # Create synthetic test image
        img = Image.new("RGB", (640, 640), color=(200, 200, 200))
        detections, latency = detector.detect(img)
        print(f"  [OK] Inference ran in {latency*1000:.1f}ms. Detections count: {len(detections)}")

        # Annotate
        annotated = detector.annotate_image(img, detections)
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == (640, 640, 3)
        print("  [OK] Annotation function executed cleanly")
        return True
    except Exception as e:
        print(f"  [FAIL] Detector test failed: {e}")
        return False


def test_metrics_utility() -> bool:
    print("[2/3] Testing metrics utilities...")
    try:
        mock_detections = [
            {"class_id": 0, "class_name": "person", "confidence": 0.95, "bbox": [10, 10, 100, 200]},
            {"class_id": 0, "class_name": "person", "confidence": 0.90, "bbox": [150, 20, 250, 220]},
            {"class_id": 2, "class_name": "car", "confidence": 0.85, "bbox": [300, 50, 500, 300]},
        ]
        summary = compute_detection_summary(mock_detections)
        assert summary.total_count == 3
        assert summary.class_counts["person"] == 2
        assert summary.class_counts["car"] == 1
        assert round(summary.average_confidence, 2) == 0.90

        latency_ms, fps = format_latency_and_fps(0.025)
        assert latency_ms == 25.0
        assert fps == 40.0
        print("  [OK] Summary and FPS computations accurate")
        return True
    except Exception as e:
        print(f"  [FAIL] Metrics test failed: {e}")
        return False


def test_sample_generation() -> bool:
    print("[3/3] Generating sample preview image for screenshots directory...")
    try:
        detector = ObjectDetector(model_name=config.MODEL_NAME, conf_threshold=0.25)
        # Create a sample frame with two clear color patches
        test_img = np.full((600, 900, 3), 240, dtype=np.uint8)
        import cv2
        cv2.putText(test_img, "AI Object Detection Streamlit Preview", (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
        mock_dets = [
            {"class_id": 0, "class_name": "person", "confidence": 0.96, "bbox": [120, 140, 380, 520]},
            {"class_id": 67, "class_name": "cell phone", "confidence": 0.91, "bbox": [480, 200, 640, 440]},
            {"class_id": 39, "class_name": "bottle", "confidence": 0.84, "bbox": [700, 250, 820, 480]},
        ]
        annotated = detector.annotate_image(test_img, mock_dets)
        out_path = config.SCREENSHOTS_DIR / "sample_web_preview.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"  [OK] Saved preview visual to {out_path.name}")
        return True
    except Exception as e:
        print(f"  [FAIL] Sample generation failed: {e}")
        return False


if __name__ == "__main__":
    t1 = test_detector_inference()
    t2 = test_metrics_utility()
    t3 = test_sample_generation()
    if t1 and t2 and t3:
        print("\nALL BACKEND & METRICS TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED!")
        sys.exit(1)
